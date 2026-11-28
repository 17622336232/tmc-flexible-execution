"""Replay each sealed first decision once on a fresh simulator and score only after all captures."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

import numpy as np
from PIL import Image

from experiments.flexible_execution.adapters.native_views import close_pose
from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.conditional_view_move import perform
from research.flexible_scheduling.scoped_execution import match_source
from scripts.continuing_crossworld_views import assert_snapshot, copied_renderer
from scripts.continuing_first_views import ROOT, register, seal, slots
from scripts.continuing_navigation_assay import geometry_hash
from scripts.continuing_sensor_visibility import bodies, camera

DEST=ROOT/"replay"
SOURCE=Path(__file__)


def pose(v):
    return {**v["position"],"yaw":v["yaw"]}


def photo(folder,name,obs):
    path=folder/(name+".jpg")
    path.write_bytes(obs["data"])
    save(folder/(name+".json"),{**{k:v for k,v in obs.items() if k!="data"},"sha256":digest(path)})


def local_attempt(case,method,raw_folder,ep):
    from experiments.task_attack_integrity.run import main as simulate
    log=raw_folder/"model_calls.jsonl"
    raw=read_trace_rows(log)[0][0]
    reply=raw.get("output")
    args=["--policy","clean","--backend","cpu","--headless","--no-rviz","--sensor-smoke",
          "--ros-port","11890","--gazebo-port","11891","--episode-timeout","80","--output",str(ep)]
    for key,value in case["settings"].items():
        args+=["--"+key.replace("_","-"),str(value)]
    def exercise(backend,obs,folder,event):
        start=monotonic()
        backend.execution_deadline_monotonic=start+80
        events=[]
        def record(kind,**data):
            events.append(dict(event=kind,elapsed_s=monotonic()-start,**data))
        out=dict(case=case["id"],method=method,raw_log_sha256=digest(log),events=events,
                 model_requests=0,matching_world=geometry_hash(folder)==case["matching_world"])
        try:
            reference=Path(case["reference"])
            if digest(reference)!=case["source_sha256"] or not reply:
                raise ValueError("Source or response unavailable")
            # One sensor pair supplies both camera data and the associated measured geometry.
            obs,geo=backend.observe()
            photo(folder,"before",obs)
            out["matching_pose"]=close_pose(pose(obs),pose(read(reference.with_suffix(".json"))))
            bboxes={"original":case["original_approach"]["bbox"]}
            action=reply.get("action") or {}
            if action.get("type")=="approach":
                bboxes["requested"]=action["bbox"]
            matched={}
            for key,box in bboxes.items():
                fit=match_source(reference.read_bytes(),obs["data"],box)
                if not fit or fit["ncc"]<.90 or max(abs(z) for z in fit["delta_pixels"])>8:
                    raise ValueError("Initial source region does not match: "+key)
                # Native execution always receives the unmodified frozen reply.
                dx,dy=fit["delta_pixels"]
                w,h=obs["width"],obs["height"]
                matched[key]={"box":[box[0]+dx/w,box[1]+dy/h,box[2]+dx/w,box[3]+dy/h],**fit}
            out["matched"]=matched
            if not out["matching_world"] or not out["matching_pose"]:
                raise ValueError("Original geometry or camera pose is not restored")
            try:
                out["action"]=perform(backend,obs,geo,reply,"onboard_0001",method=="contingent",record)
            except (ValueError,RuntimeError) as ex:
                out["action_error"]=dict(code=getattr(ex,"code",type(ex).__name__),message=str(ex)[:200])
            after_ros=backend.ros.Time.now().to_sec()
            until=monotonic()+3
            while monotonic()<until:
                fresh,_=backend.observe()
                if fresh["ros_stamp"]>after_ros:
                    photo(folder,"after",fresh)
                    out["fresh_after_action"]=True
                    out["final_pose"]=pose(fresh)
                    break
                sleep(.025)
            if not out.get("fresh_after_action"):
                raise ValueError("No independently timestamped after-action frame")
        except Exception as ex:
            out["failure"]=dict(code=getattr(ex,"code",type(ex).__name__),message=str(ex)[:220])
        finally:
            out["simulation_elapsed_s"]=monotonic()-start
            save(folder/"attempt.json",out)
            backend.execution_deadline_monotonic=None
    simulate(args,sensor_exercise=exercise,archive_external_rgb=False)


def files(ep):
    if not (ep/"attempt.json").is_file() or not (ep/"result.json").is_file():
        raise ValueError("Preserve unsettled physical attempt "+str(ep))
    if list(ep.rglob("model_calls.jsonl")) or list(ep.rglob("sends.jsonl")):
        raise ValueError("A replay must never contain a new model call")
    return {p.name:digest(p) for name in ("attempt.json","result.json","before.jpg","before.json","after.jpg","after.json")
            if (p:=ep/name).is_file()}


def pixels(ep,case,method):
    attempt=read(ep/"attempt.json")
    action=attempt.get("action") or {}
    out=dict(case=case["id"],method=method,source_bound=bool(attempt.get("matching_world") and attempt.get("matching_pose") and attempt.get("matched")),
        kind=action.get("kind"),primary_refusal=action.get("primary_refusal"),fallback=bool(action.get("fallback")),
        reached=bool(action.get("reached")),action_error=attempt.get("action_error"),failure=attempt.get("failure"),
        timings_s=attempt["simulation_elapsed_s"],source_pixels={},limiting_ratio=None,witness=False)
    if not out["source_bound"] or not attempt.get("fresh_after_action"):
        return out
    devices=read(ep/"private/truth.json")["devices"]
    equipment=case["settings"]["family"]=="cracks"
    with copied_renderer(ep) as (scene,renderer):
        ids=bodies(scene,renderer)
        masks={};qualities={}
        for when in ("before","after"):
            meta=read(ep/(when+".json"))
            if digest(ep/(when+".jpg"))!=meta["sha256"]:
                raise ValueError("Scoring image changed")
            rgb,masks[when]=camera(renderer,meta)
            with Image.open(ep/(when+".jpg")) as measured:
                qualities[when]=float((np.abs(np.asarray(measured.convert("RGB"),dtype=float)-rgb).max(2)<=18).mean())
        out["render_alignment"]=qualities
        def station(i):
            return ("equipment" if equipment else "warehouse")+"_station_%02d"%i
        def plate(i):
            return ids[(station(i),"reference_identity")] if equipment else ids[("panel_%02d"%i,"front")]
        def assignments(key):
            l,t,r,b=attempt["matched"][key]["box"]
            mask=masks["before"];h,w=mask.shape
            options=[]
            for i in range(1,len(devices)+1):
                yy,xx=np.where(mask==plate(i))
                if len(xx)>=36 and l<=xx.mean()/w<=r and t<=yy.mean()/h<=b:
                    options.append(i)
            return options
        original=assignments("original")
        out["unique_source"]=len(original)==1
        if len(original)!=1:
            return out
        i=original[0]
        out["scored_subject"]=devices[i-1]["identity"]
        same=assignments("requested")==original if "requested" in attempt["matched"] else None
        out["requested_same_subject"]=same
        parts={"label":plate(i)}
        names=["door_surface"] if equipment else ["rack_post_-0.67","rack_post_0.67","rack_shelf_0.23","rack_shelf_0.82",
            "rack_shelf_1.42",*["cargo_%d"%n for n in range(devices[i-1]["box_count"])]]
        parts.update({n:ids[(station(i),n)] for n in names})
        minimum={}
        for k,mask in masks.items():
            counts={name:int((mask==v).sum()) for name,v in parts.items()}
            out["source_pixels"][k]=counts
            minimum[k]=min(counts.values())
        out["limiting_ratio"]=minimum["after"]/minimum["before"] if minimum["before"]>0 else None
        real_goal=bool(action.get("goal") and close_pose(attempt["final_pose"],action["goal"]))
        out["witness"]=bool(method=="contingent" and action.get("fallback") and real_goal and same and
            min(qualities.values())>=.90 and minimum["after"]>=144 and minimum["before"]>0
            and minimum["after"]>=1.25*minimum["before"])
    return out


def grade(reg):
    tasks=slots(reg)
    sealed={c["id"]+"/"+m:files(DEST/c["id"]/m) for c,m,_ in tasks}
    assert_snapshot(DEST/"sealed_measurements.json",sealed)
    rows=[pixels(DEST/c["id"]/m,c,m) for c,m,_ in tasks]
    result=dict(role="source-matched single-decision physical replay",
        sealed_responses_sha256=digest(ROOT/"sealed_replies.json"),sealed_measurements_sha256=digest(DEST/"sealed_measurements.json"),
        actual_attempts=len(rows),new_model_requests=0,source_bound=sum(r["source_bound"] for r in rows),
        reached=sum(r["reached"] for r in rows),guarded_movements=sum(r["fallback"] for r in rows),
        registered_witnesses=sum(r["witness"] for r in rows),cases=rows)
    save(ROOT/"replay_summary.json",result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("run","grade","worker"))
    p.add_argument("--case")
    p.add_argument("--method")
    a=p.parse_args()
    reg=register()
    seal(reg)
    if a.mode=="worker":
        item,method,raw=next((c,m,f) for c,m,f in slots(reg) if (c["id"],m)==(a.case,a.method))
        ep=DEST/item["id"]/method
        rc=ep.with_suffix(".receipt.json")
        if ep.exists() or not rc.is_file() or read(rc)["state"] not in ("starting","running"):
            raise SystemExit("Worker needs a fresh replay reservation")
        local_attempt(item,method,raw,ep)
        return
    if a.mode=="run":
        DEST.mkdir(exist_ok=True)
        with (ROOT.parent/"online.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for c,m,raw in slots(reg):
                ep=DEST/c["id"]/m;rc=ep.with_suffix(".receipt.json")
                key=dict(case=c["id"],method=m,code_sha256=digest(SOURCE),model_log=digest(raw/"model_calls.jsonl"))
                if rc.exists():
                    old=read(rc)
                    if any(old.get(k)!=v for k,v in key.items()):
                        raise ValueError("Replay version or input changed")
                    seal_files=files(ep)
                    if old.get("files",seal_files)!=seal_files:
                        raise ValueError("Original physical measurements changed")
                    if old["state"]!="settled":
                        save(rc,{**old,"state":"settled","files":seal_files})
                    continue
                if ep.exists():
                    raise SystemExit("Unreserved existing output "+str(ep))
                ep.parent.mkdir(parents=True,exist_ok=True)
                save(rc,{**key,"state":"starting"})
                with ep.with_suffix(".log").open("a") as sink:
                    proc=subprocess.Popen([sys.executable,str(SOURCE),"worker","--case",c["id"],"--method",m],stdout=sink,stderr=subprocess.STDOUT)
                    save(rc,{**key,"state":"running","worker":proc.pid})
                    proc.wait()
                if proc.returncode:
                    raise SystemExit("Preserve and inspect original worker "+str(ep))
                save(rc,{**key,"state":"settled","files":files(ep)})
                print(c["id"],m,"settled",flush=True)
    out=grade(reg)
    print({k:v for k,v in out.items() if k!="cases"})


if __name__=="__main__":
    main()
