"""Collect and seal same-photo matched service/recognition measurements for the real room study."""
import argparse
import fcntl
from hashlib import sha256
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.public_audit import save_segmentation,_grade_segmentation
from experiments.flexible_execution.public_task import MIN_OBJECT_PIXELS
from experiments.flexible_execution.public_visual import _house,open_controller,camera_jpeg
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import read,save,digest
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.task_attack.model import ModelClient,EXTERNAL_RESPONSE_CODES
from research.flexible_scheduling.scoped_room_query import PROMPT,request
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_room_study import worker_command

CFG=Path("config/matched_event_rooms_dev01.json")
ROOT=Path("outputs/continuing/matched-event-rooms-dev01")
CAL=ROOT/"calibration"


def prepare():
    cfg=read(CFG);catalog=read(Path(cfg["runtime_catalog"]))
    known={v["official_house_index"] for v in catalog["cases"]}
    if not {v["house"] for v in cfg["calibration"]+cfg["cases"]}<=known:
        raise ValueError("Only registered previously exposed development houses are supported")
    CAL.mkdir(parents=True,exist_ok=True)
    reg=dict(config_sha256=digest(CFG),catalog_sha256=digest(Path(cfg["runtime_catalog"])),
             prompt_sha256=sha256(PROMPT.encode()).hexdigest(),query_sha256=digest(Path("research/flexible_scheduling/scoped_room_query.py")))
    assert_snapshot(ROOT/"registration.json",reg)
    return cfg,catalog["runtime"]


def launch_local(runtime,case,script,args,log):
    cmd,env=worker_command(runtime,case,"camera",False,ROOT)
    worker=cmd[:cmd.index(sys.executable)+1]+[str(script),*args]
    with log.open("a") as out:
        return subprocess.Popen(worker,env=env,stdout=out,stderr=subprocess.STDOUT)


def capture(runtime,case):
    folder=CAL/case["id"]/"camera"
    folder.mkdir(parents=True,exist_ok=False)
    house,world=_house(Path(runtime["archive"]),case["house"])
    controller=open_controller(house,Path(runtime["build_dir"]),"opengl",os.environ["DISPLAY"],segmentation=True)
    try:
        event=controller.last_event
        (folder/"front.jpg").write_bytes(camera_jpeg(event))
        # Score artifacts stay outside the robot-side input. Predictions are sealed before scoring.
        for i,cat in enumerate(case["categories"]):
            save_segmentation(folder,"q%d"%i,event,cat,MIN_OBJECT_PIXELS)
        save(folder/"source.json",dict(house=case["house"],house_sha256=world,photo=digest(folder/"front.jpg"),
                                      actual_pose=event.metadata["agent"]))
    finally:
        controller.stop()


def slots(cfg):
    for c in cfg["calibration"]:
        for mode in c["order"]:
            qs=[("q%d"%i,cat) for i,cat in enumerate(c["categories"]) if mode=="joint" or (mode=="first")== (i==0)]
            yield c,mode,qs,CAL/c["id"]/mode


def receipt(case,mode,qs,folder):
    log=folder/"model_calls.jsonl";calls=read_trace_rows(log)[0]
    if len(calls)!=1:raise ValueError("Expected exactly one inference attempt")
    c=calls[0]
    f=CAL/case["id"]/"camera/front.jpg";h=digest(f)
    expected=dict(image="front",sha256=h,questions=[dict(id=i,image="front",category=q) for i,q in qs])
    if (c["input"]!=expected or c["image_sha256"]!=[h] or c["model"]!="gpt-6.1-terra" or
            c["instructions_sha256"]!=sha256(PROMPT.encode()).hexdigest() or c["reasoning_effort"]!="max"):
        raise ValueError("Same-image service binding mismatch")
    start=c.get("dispatched_monotonic_seconds")
    if start is not None:
        matches=[v for v in read_trace_rows(CAL/"sends.jsonl")[0] if v["reservation"]==c["logical_job_attempt"]]
        if len(matches)!=1 or matches[0]["image_sha256"]!=[h]:raise ValueError("Dispatch ledger mismatch")
    err=c.get("error") or {}
    external=err.get("code") in EXTERNAL_RESPONSE_CODES or err.get("status") in (401,403,429)
    data=dict(case=case["id"],method=mode,sent=start is not None,error=err.get("code"),
              endpoint_block=(err.get("code") or "external") if external else None,
              service_s=c["return_monotonic_seconds"]-start if start is not None else None,
              log_sha256=digest(log),source_sha256=h,outcome=c.get("output"))
    assert_snapshot(folder.with_suffix(".receipt.json"),data)
    return data


def run(cfg,runtime):
    from uav_bridge.chatgpt_auth import CredentialStore
    with (ROOT.parent/"online.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in cfg["calibration"]:
            site=CAL/c["id"];site.mkdir(exist_ok=True)
            if (site/"camera/source.json").is_file():
                if read(site/"camera/source.json")["photo"]!=digest(site/"camera/front.jpg"):
                    raise ValueError("Initial camera digest differs")
                continue
            if (site/"camera").exists():raise RuntimeError("Inspect preserved partial camera reset "+str(site))
            proc=launch_local(runtime,c,Path(__file__),["capture","--case",c["id"]],site/"camera.log")
            if proc.wait()!=0:raise RuntimeError("Inspect preserved original camera run "+str(site))
        sources={c["id"]:read(CAL/c["id"]/"camera/source.json")["photo"] for c in cfg["calibration"]}
        assert_snapshot(CAL/"registered_photos.json",sources)
        budget=RequestBudget(CAL/"sends.jsonl",6)
        for c,m,qs,folder in slots(cfg):
            if (folder/"model_calls.jsonl").is_file():
                old=receipt(c,m,qs,folder)
                if old["endpoint_block"]:raise SystemExit("Original service interruption "+str(folder))
                continue
            if folder.exists():raise SystemExit("Preserve incomplete or already dispatched request "+str(folder))
            if unresolved_service_limits(ROOT.parent):raise SystemExit("Authorized service has a recorded restriction")
            folder.mkdir()
            model=ModelClient(folder/"model_calls.jsonl",timeout=120,
                credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),request_budget=budget)
            try:request(model,"front",(CAL/c["id"]/"camera/front.jpg").read_bytes(),qs)
            except Exception:
                if not (folder/"model_calls.jsonl").exists():raise
            record=receipt(c,m,qs,folder)
            print(c["id"],m,record["sent"],record["error"],flush=True)
            if record["endpoint_block"] or not record["sent"]:raise SystemExit("Service failure at "+str(folder))
        return grade(cfg)


def grade(cfg):
    observations=[(c,m,q,f,receipt(c,m,q,f)) for c,m,q,f in slots(cfg)]
    assert_snapshot(CAL/"sealed_predictions.json",{c["id"]+"/"+m:r["log_sha256"] for c,m,_,_,r in observations})
    rows=[];service={1:[],2:[]}
    for c,m,qs,folder,r in observations:
        if r["service_s"] is not None:service[len(qs)].append(r["service_s"])
        for key,cat in qs:
            expected=_grade_segmentation(CAL/c["id"]/"camera/scoring",key)
            out=r["outcome"]
            valid=isinstance(out,dict) and set(out)=={k for k,_ in qs} and all(v in ("positive","negative","uncertain") for v in out.values())
            answer=out.get(key) if valid else None
            rows.append(dict(case=c["id"],method=m,field=key,category=cat,answer=answer,expected=expected,
                             correct=answer==expected,source_sha256=r["source_sha256"],service_s=r["service_s"]))
    if len(service[1])!=4 or len(service[2])!=2:raise ValueError("Incomplete registered clock measurements")
    result=dict(cases=rows,actual_sends=6,single_s=median(service[1]),pair_s=median(service[2]),
                singleton_correct=sum(r["correct"] for r in rows if r["method"]!="joint"),
                grouped_correct=sum(r["correct"] for r in rows if r["method"]=="joint"),
                sealed_sha256=digest(CAL/"sealed_predictions.json"))
    save(CAL/"summary.json",result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("capture","run","grade","status"))
    p.add_argument("--case");a=p.parse_args()
    cfg,runtime=prepare()
    if a.mode=="capture":
        capture(runtime,next(c for c in cfg["calibration"] if c["id"]==a.case))
    elif a.mode=="status":
        print("sealed", (CAL/"sealed_predictions.json").is_file(),"receipts",len(list(CAL.glob("*/*.receipt.json"))))
    else:
        result=run(cfg,runtime) if a.mode=="run" else grade(cfg)
        print({k:v for k,v in result.items() if k!="cases"})


if __name__=="__main__":
    main()
