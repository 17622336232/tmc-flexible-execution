"""Run one preregistered six-arm room block, or a separate camera-only smoke. Use the pinned display."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from time import time

from experiments.flexible_execution.revision_audit import digest,read,save
from research.flexible_scheduling.room_inspection import run,model_plan


CONFIG=Path("config/room_workflow_dev01.json")


def paths(root,case,method,smoke):
    return root/("smoke" if smoke else "live")/case["id"]/method


def worker_command(runtime,case,method,smoke,root):
    xvfb=Path(runtime["xvfb_run"])
    parent=root/"display"
    envpaths={key:parent/sub for key,sub in (("XDG_CACHE_HOME","cache"),("XDG_CONFIG_HOME","config"),("XDG_RUNTIME_DIR","runtime"))}
    for path in envpaths.values():
        path.mkdir(parents=True,exist_ok=True,mode=0o700)
    env={**os.environ,"PATH":str(xvfb.parent)+os.pathsep+os.environ["PATH"],
         "PYTHONPATH":os.pathsep.join((runtime["packages"],str(Path("bridge").resolve()),str(Path.cwd()))),
         "LIBGL_ALWAYS_SOFTWARE":"1","GALLIUM_DRIVER":"llvmpipe","LP_NUM_THREADS":"2",
         **{k:str(p.resolve()) for k,p in envpaths.items()}}
    cmd=[str(xvfb),"-a","-n","95","-s","-screen 0 640x480x24 -nolisten tcp -noreset",sys.executable,
         "scripts/continuing_room_study.py","worker","--case",case["id"],"--method",method]
    if smoke:
        cmd.append("--smoke")
    return cmd,env


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("smoke","run","worker","status"))
    parser.add_argument("--case",default="conditional-room")
    parser.add_argument("--method")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    config=read(CONFIG)
    case=next(c for c in config["cases"] if c["id"]==args.case)
    root=Path("outputs/continuing")/config["id"]
    root.mkdir(parents=True,exist_ok=True)
    if args.action=="worker":
        if not args.smoke and args.method not in case["order"]:
            raise ValueError("Unregistered arm")
        result=run(config,case,args.method,paths(root,case,args.method,args.smoke),os.environ["DISPLAY"],args.smoke)
        print(result,flush=True)
        return
    if args.action=="status":
        for p in sorted(root.glob("*/*/*/result.json")):
            d=read(p)
            print(str(p.parent.relative_to(root)),d.get("strict"),d["sends"],round(d["seconds"],2),d["failure"])
        return
    smoke=args.action=="smoke"
    methods=["camera-check"] if smoke else case["order"]
    if not smoke:
        for c in config["cases"]:
            sample=paths(root,c,"camera-check",True)/"result.json"
            if not sample.exists() or not read(sample).get("restored") or read(sample).get("failure"):
                raise SystemExit("Development camera calibration has not succeeded for "+c["id"])
        plans=[model_plan(case,m,config) for m in methods]
        if len({(p["joint"],p["overlap"]) for p in plans})<2:
            raise SystemExit("Identical registered policies on this mission; keep for mechanism audit.")
    runtime=read(Path(config["runtime_catalog"]))["runtime"]
    guard_file=(root/"smoke.lock") if smoke else (root.parent/"online.lock")
    with guard_file.open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not smoke:
            from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
            if unresolved_service_limits(root.parent):
                raise SystemExit("A recorded limit on the authorized model plan is still pending.")
        for m in methods:
            ep=paths(root,case,m,smoke)
            receipt=ep.with_name(ep.name+".receipt.json")
            if receipt.exists() or ep.exists():
                if (ep/"result.json").exists():
                    d=read(ep/"result.json")
                    if d.get("endpoint_block") or d["failure"] and not d["sends"]:
                        raise SystemExit("Preserved external or startup failure in "+str(ep))
                    if receipt.exists() and read(receipt).get("result_sha256") not in (None,digest(ep/"result.json")):
                        raise ValueError("Settled result changed")
                    continue
                raise SystemExit("Unsettled room attempt requires inspection: "+str(ep))
            ep.parent.mkdir(parents=True,exist_ok=True)
            cmd,env=worker_command(runtime,case,m,smoke,root)
            meta=dict(state="starting",start_wall=time(),case=case["id"],method=m,config_sha256=digest(CONFIG),
                      source_sha256=digest(Path("research/flexible_scheduling/room_inspection.py")),
                      solver_sha256=digest(Path("research/flexible_scheduling/workflow.py")),controller_pid=os.getpid())
            save(receipt,meta)
            with ep.with_name(m+".log").open("a") as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                save(receipt,{**meta,"state":"running","worker_pid":proc.pid})
                while proc.poll() is None:
                    ledger=ep/"requests.jsonl"
                    save(root/("smoke_heartbeat.json" if smoke else "heartbeat.json"),dict(state="running",wall=time(),
                         case=case["id"],method=m,controller_pid=os.getpid(),worker_pid=proc.pid,
                         sends=len(ledger.read_bytes().splitlines()) if ledger.exists() else 0))
                    try:
                        proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        pass
            if not (ep/"result.json").exists():
                save(receipt,{**meta,"state":"interrupted","exit_code":proc.returncode})
                raise SystemExit("Worker failed; preserve attempt and inspect "+str(ep))
            d=read(ep/"result.json")
            save(receipt,{**meta,"state":"settled","end_wall":time(),"result_sha256":digest(ep/"result.json"),
                          "strict":d.get("strict"),"sends":d["sends"],"error":d["failure"]})
            print(case["id"],m,d.get("strict"),d["sends"],d["seconds"],d["failure"],flush=True)
            if d["endpoint_block"] or d["failure"] and not d["sends"]:
                raise SystemExit("Room block stopped at an external or startup error.")
        save(root/("smoke_heartbeat.json" if smoke else "heartbeat.json"),dict(state="idle",wall=time(),workers=[]))


if __name__=="__main__":
    main()
