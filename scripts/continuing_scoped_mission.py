"""Static public-roster sensor preflight and bounded mechanism execution with independent post-task grading."""
import argparse
import fcntl
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image

from experiments.flexible_execution.adapters.native_views import close_pose
from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.live_planning import Costs
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import digest,read,save,write_table
from experiments.flexible_execution.timeline import Timeline,read_trace_rows
from research.flexible_scheduling.grouped_regions import PROMPT,answers
from research.flexible_scheduling.scoped_execution import execute,prepare
from scripts.continuing_crossworld_views import assert_snapshot,copied_renderer
from scripts.continuing_sensor_visibility import bodies,camera


ROOT=Path("outputs/continuing/scoped-mission-dev01")
NAV=ROOT.parent/"local-view-navigation01"
REGIONS=ROOT.parent/"scope-closure-dev01-parserfix/rosters.json"
CONFIG=Path("config/scoped_mission_dev01.json")


def public_missions():
    doc=read(REGIONS)
    result=[]
    for c in doc["cases"]:
        photo=Path(c["origin"])/"private/robot"/(c["source_frame"]+".jpg")
        if digest(photo)!=c["ref_sha256"]:
            raise ValueError("Public source photo has changed")
        cells=[dict(id=x["id"],bbox=x["bbox"],may_photograph_identity_before_result=True) for x in c["regions"]]
        result.append(dict(mission=dict(id="scoped-crack-"+c["source_case"],static=True,reference_sha256=c["ref_sha256"],
                                        regions=cells),source_case=c["source_case"],reference=str(photo),settings=c["settings"]))
    return result


def register():
    config=read(CONFIG)
    cases=public_missions()
    chosen=min(cases,key=lambda c:sha256((config["selection_salt"]+c["source_case"]).encode()).digest())
    files=[Path(p) for p in config["runtime_files"]]
    code=sha256(b"".join(p.as_posix().encode()+b"\0"+p.read_bytes() for p in files)).hexdigest()
    from experiments.flexible_execution.revision_jobs import source_id
    reg=dict(config_sha256=digest(CONFIG),cases=cases,mechanism_case=chosen["source_case"],runtime_sha256=code,
             joint_base_runtime=source_id(),rosters_sha256=digest(REGIONS),bounded_online_order=config["online_order"])
    ROOT.mkdir(exist_ok=True,parents=True)
    assert_snapshot(ROOT/"registration.json",reg)
    return config,reg


def preflight(reg):
    results=[]
    for case in reg["cases"]:
        m=case["mission"]
        old=Path(case["reference"]).read_bytes()
        for arm in ("radial_same_distance","frontal_same_distance"):
            ep=NAV/case["source_case"]/arm
            initial=ep/"onboard_sensor_smoke.jpg"
            obs=read(ep/"navigation.json")
            if digest(initial)!=obs["calibration_rgb"]:
                raise ValueError("Local route and RGB differ")
            g=read(ep/"sensor_geometry.json")
            live=dict(data=initial.read_bytes(),ros_stamp=g["camera"]["ros_stamp"])
            roster,cells,matches=prepare(m,old,live,g)
            results.append(dict(case=case["source_case"],route=arm,source=digest(initial),complete_roster_alignment=roster is not None,
                public_regions=len(m["regions"]),registered_regions=len(cells),
                regions_with_measured_view=sum(bool(k.get("proposals")) for k in matches),
                minimum_region_alignment=min((k["registration"]["ncc"] for k in matches if k.get("registration")),default=None),
                unresolved=matches[0].get("id") if roster is None else None))
    write_table(ROOT/"actual_initial_views.csv",results)
    counts=dict(camera_resets=len(results),layouts=len(reg["cases"]),all_regions_registered=sum(r["complete_roster_alignment"] for r in results),
         public_region_exposures=sum(r["public_regions"] for r in results),registered_region_exposures=sum(r["registered_regions"] for r in results),
         regions_with_measured_view=sum(r["regions_with_measured_view"] for r in results),new_model_requests=0,
         note="Initial onboard RGBs stored during the eight completed navigation resets. Candidate geometry requires a later live map check. No semantic answers, model-latency observations or full task success are claimed.")
    save(ROOT/"preflight.json",counts)
    print(counts)


def run_one(config,reg):
    from uav_bridge.chatgpt_auth import CredentialStore
    from experiments.task_attack.model import ModelClient,EXTERNAL_RESPONSE_CODES
    from experiments.task_attack_integrity.run import main as simulate
    with (ROOT.parent/"online.lock").open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if unresolved_service_limits(ROOT.parent):
            raise SystemExit("Recorded plan restriction still applies")
        assay=ROOT.parent/"same-frame-region-assay01/summary.json"
        if not assay.is_file() or not all(c["group_all_correct"] is True and c["single_all_correct"] is True
                                          for c in read(assay)["cases"]) or len(read(assay)["cases"])!=2:
            raise SystemExit("Source-level paired region readings have not met the registered admission criterion")
        case=next(c for c in reg["cases"] if c["source_case"]==reg["mechanism_case"])
        for method in config["online_order"]:
            ep=ROOT/case["source_case"]/method
            receipt=ep.with_suffix(".receipt.json")
            if receipt.is_file():
                if digest(ep/"scoped_attempt.json")!=read(receipt)["attempt_sha256"]:
                    raise ValueError("Completed mechanism outcome changed")
                continue
            if ep.exists():
                raise SystemExit("Preserve and inspect original unsettled run: "+str(ep))
            ep.parent.mkdir(exist_ok=True)
            args=["--policy","clean","--backend","cpu","--headless","--no-rviz","--sensor-smoke","--ros-port","11854",
                  "--gazebo-port","11855","--episode-timeout",str(config["task_deadline_s"]),"--output",str(ep)]
            for k in ("family","seed","layout_seed","equipment_count","source_profile"):
                args+=["--"+k.replace("_","-"),str(case["settings"][k])]
            def probe(backend,obs,folder,event):
                client=ModelClient(folder/"model_calls.jsonl",timeout=config["per_call_s"],
                        credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),
                        request_budget=RequestBudget(folder/"sends.jsonl",1))
                costs=Costs(**read(Path(config["costs"]))["costs"])
                begin=monotonic()
                try:
                    outcome=execute(backend,client,case["mission"],Path(case["reference"]).read_bytes(),folder,
                                    Timeline(folder/"scoped_events.jsonl"),costs,method,begin+config["task_deadline_s"])
                    outcome["endpoint_block"]=None
                except Exception as ex:
                    code=getattr(ex,"code",type(ex).__name__)
                    outcome=dict(report=None,gaps=[["mission",str(code)]],sends=client.calls,
                                 endpoint_block=str(code) if code in EXTERNAL_RESPONSE_CODES or getattr(ex,"status",None) in (401,403,429) else None)
                outcome.update(task_s=monotonic()-begin,method=method,registration_sha256=digest(ROOT/"registration.json"))
                save(folder/"scoped_attempt.json",outcome)
            simulate(args,sensor_exercise=probe,archive_external_rgb=False)
            record=read(ep/"scoped_attempt.json")
            save(receipt,dict(attempt_sha256=digest(ep/"scoped_attempt.json"),sends=record["sends"],endpoint_block=record["endpoint_block"]))
            if record["endpoint_block"]:
                raise SystemExit("External service interruption retained")
            break


def grade(reg):
    c=next(x for x in reg["cases"] if x["source_case"]==reg["mechanism_case"])
    scored=[]
    for method in reg["bounded_online_order"]:
        ep=ROOT/c["source_case"]/method
        receipt=ep.with_suffix(".receipt.json")
        if not receipt.is_file():
            raise SystemExit("Await all registered mechanism receipts before opening the grader")
        if read(receipt)["attempt_sha256"]!=digest(ep/"scoped_attempt.json"):
            raise ValueError("Original report changed")
    for method in reg["bounded_online_order"]:
        ep=ROOT/c["source_case"]/method
        r=read(ep/"scoped_attempt.json")
        log=ep/"model_calls.jsonl"
        calls=read_trace_rows(log)[0] if log.is_file() else []
        ledger=read_trace_rows(ep/"sends.jsonl")[0] if (ep/"sends.jsonl").exists() else []
        sent=[x for x in calls if x.get("dispatched_monotonic_seconds") is not None]
        if len(ledger)!=len(sent) or len(sent)!=r["sends"] or len(sent)>1:
            raise ValueError("Transport ledger inconsistent")
        final=r["report"]
        good=False
        if final is not None and len(sent)==1:
            call=sent[0]; h=digest(ep/"scope-overview.jpg")
            fixed=read(ep/"live_roster.json");ret=read(ep/"measured_return.json")
            start=read(ep/"scope-overview.json")
            public,regions,_=prepare(c["mission"],Path(c["reference"]).read_bytes(),{**start,"data":(ep/"scope-overview.jpg").read_bytes()},
                                     read(ep/"scope-overview_geometry.json"))
            if public is None or public.scope_hash!=fixed["scope"] or [v.public() for v in regions]!=fixed["regions"]:
                raise ValueError("Live roster differs from the registered public mission")
            raw=answers(call["input"],call["image_sha256"],call.get("output"))
            if not (call["model"]=="gpt-6.1-terra" and call["reasoning_effort"]=="max" and ledger[0]["image_sha256"]==[h]==call["image_sha256"]
                    and call["instructions_sha256"]==sha256(PROMPT.encode()).hexdigest() and call["input"]["regions"]==fixed["regions"]
                    and call["input"]["photo_sha256"]==h==fixed["reference"] and fixed["scope"]==final["scope_sha256"] and raw
                    and close_pose(ret["initial"],ret["final"]) and ret["returned"]):
                raise ValueError("Proposed certificate is not bound to a genuine overview, reply and return")
            dev=read(ep/"private/truth.json")["devices"]
            with copied_renderer(ep) as (scene,renderer):
                names=bodies(scene,renderer)
                def source(frame):
                    meta=read(ep/(frame+".json"))
                    if meta["sha256"]!=digest(ep/(frame+".jpg")):
                        raise ValueError("Frame changed after measurement")
                    rgb,mask=camera(renderer,meta)
                    with Image.open(ep/(frame+".jpg")) as original:
                        align=float((np.abs(np.asarray(original.convert("RGB"),dtype=float)-rgb).max(axis=2)<=18).mean())
                    return mask,align
                mask,alignment=source("scope-overview")
                expected={}
                physical_ids=[]
                reads={e["step"]:e for e in read_trace_rows(ep/"scoped_events.jsonl")[0] if e["event"]=="scoped_local_read"}
                observed={v["region"]:v for v in final["coverage"]};found={v["region"]:v for v in final["findings"]}
                good=alignment>=.90 and len(observed)==len(fixed["regions"]) and len(final["coverage"])==len(observed)
                for cell in fixed["regions"]:
                    b=cell["bbox"];k=cell["id"];ids=[]
                    for i,d in enumerate(dev,1):
                        part=names[("equipment_station_%02d"%i,"reference_identity")]
                        yy,xx=np.where(mask==part)
                        if len(xx)>=36 and b[0]<=xx.mean()/mask.shape[1]<=b[2] and b[1]<=yy.mean()/mask.shape[0]<=b[3]:
                            ids.append((d,part))
                    if len(ids)!=1 or k not in observed:
                        good=False;continue
                    d,part=ids[0]; expected[k]=d["identity"] if d["cracked"] else None
                    physical_ids.append(d["identity"])
                    good &= (raw.get(k)==("positive" if d["cracked"] else "negative") and observed[k]["cracked"] is d["cracked"]
                             and observed[k]["image_sha256"]==h and observed[k]["bbox"]==b)
                    if d["cracked"] and k in found:
                        v=found[k]
                        image,quality=source(v["photo"])
                        measured=reads.get(v["photo"],{})
                        xy=measured.get("measurement",{}).get("center")
                        on_plate=bool(xy and 0<=xy[0]<1 and 0<=xy[1]<1 and image[int(xy[1]*image.shape[0]),int(xy[0]*image.shape[1])]==part)
                        good &= (v["asset_id"]==d["identity"] and v["image_sha256"]==digest(ep/(v["photo"]+".jpg"))
                                 and measured.get("region")==k and measured.get("source_sha256")==v["image_sha256"]
                                 and measured.get("measurement",{}).get("value")==v["asset_id"] and on_plate
                                 and int((image==part).sum())>=144 and quality>=.90)
                good &= {k for k,v in expected.items() if v is not None}==set(found) and len(found)==len(final["findings"])
                good &= len(set(physical_ids))==len(fixed["regions"])
        scored.append(dict(method=method,reported=final is not None,strict=bool(good),sends=r["sends"],wall_s=r["task_s"],
                           gaps=r["gaps"],external=r["endpoint_block"]))
    save(ROOT/"mechanism_results.json",dict(role="exposed-closed-roster-mechanism-only",scored=scored))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("preflight","run-next","grade"),default="preflight",nargs="?")
    cfg,reg=register()
    action=parser.parse_args().action
    if action=="preflight":preflight(reg)
    elif action=="run-next":run_one(cfg,reg)
    else:grade(reg)

if __name__=="__main__":
    main()
