"""Execute the second travel protocol with measured return waypoints and the original robot mapper."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import monotonic

from experiments.flexible_execution.adapters.native_views import close_pose
from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES
from research.flexible_scheduling.inspection_travel import execute, schedule
from research.flexible_scheduling.observed_return import ObservedTrail
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_inspection_travel import Platform, pose, score_live, sealed_attempt, utc
from scripts.continuing_itinerary_sources import grade_frames
from scripts.continuing_navigation_assay import geometry_hash
from scripts.continuing_travel_optics import ROOT, grade as grade_optics, measure, register

SOURCE=Path(__file__)


class ReturnPlatform(Platform):
    def __init__(self,backend,start,root,deadline):
        super().__init__(backend,start,root,deadline)
        self.trail=ObservedTrail(self.start)

    def move(self,key,goal):
        stop=Event()
        failures=[]
        def sample():
            try:
                while not stop.is_set():
                    self.trail.record(pose(self.backend.adapter.snapshot()))
                    stop.wait(.15)
            except Exception as ex:
                failures.append(type(ex).__name__)
        recorder=Thread(target=sample)
        recorder.start()
        try:
            super().move(key,goal)
        finally:
            stop.set()
            recorder.join()
            self.event("recorded_outward_path",stop=key,points=self.trail.points,sampling_errors=failures,
                       capacity_reached=self.trail.saturated)
        if failures:
            raise RuntimeError("Could not record actual outward odometry")

    def home(self):
        self.remaining()
        def step(index,goal):
            self.remaining()
            self.event("return_leg_start",index=index,goal=goal)
            reached=pose(self.backend.move(goal))
            self.event("return_leg_end",index=index,goal=goal,measured=reached)
            if not close_pose(reached,goal):
                raise RuntimeError("An admitted return segment did not reach its measured waypoint")
            return reached
        self.event("travel_start",stop="home",goal=self.start)
        self.trail.return_home(pose(self.backend.adapter.snapshot()),self.backend.check_goal,step,self.event)
        reached=pose(self.backend.adapter.snapshot())
        self.event("travel_end",stop="home",measured=reached,goal=self.start)
        if not close_pose(reached,self.start):
            raise RuntimeError("Return did not restore the departure position and heading")


def simulator(c,item,method,ep):
    from experiments.task_attack_integrity.run import main as simulate
    args=["--policy","clean","--backend","cpu","--headless","--no-rviz","--sensor-smoke","--ros-port","11886",
          "--gazebo-port","11887","--output",str(ep),"--episode-timeout","240"]
    for key,v in item["settings"].items():
        args+=["--"+key.replace("_","-"),str(v)]

    def exercise(backend,initial,folder,event):
        begin=monotonic()
        deadline=begin+240
        platform=ReturnPlatform(backend,initial,folder,deadline)
        backend.execution_deadline_monotonic=deadline
        out=dict(start=begin,role=method,case=c["id"],current_home=platform.start,
                 registration_sha256=digest(ROOT/"registration.json"),matching_world=geometry_hash(folder)==item["matching_world"])
        try:
            if not out["matching_world"]:
                raise ValueError("Registered physical layout differs")
            platform.event("task_start",method=method)
            if method=="calibration":
                for key,s in zip(("a","b"),item["stops"]):
                    platform.move(key,s["goal"])
                    platform.take(key)
                platform.home()
                out["returned"]=True
            else:
                from experiments.task_attack.model import ModelClient
                from uav_bridge.chatgpt_auth import CredentialStore
                client=ModelClient(folder/"model_calls.jsonl",timeout=120,
                    credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),
                    request_budget=RequestBudget(folder/"sends.jsonl",2))
                client.deadline_monotonic=deadline
                out.update(execute(c,item["stops"],platform,client,method,deadline))
            platform.remaining()
        except Exception as ex:
            code=getattr(ex,"code",type(ex).__name__)
            out["failure"]=dict(code=code,message=str(ex)[:240])
            if code in EXTERNAL_RESPONSE_CODES or getattr(ex,"status",None) in (401,403,429):
                out["endpoint_block"]=str(code)
        finally:
            out["elapsed_s"]=monotonic()-begin
            out["finished"]=monotonic()
            ledger=folder/"sends.jsonl"
            out["sends"]=len(ledger.read_bytes().splitlines()) if ledger.exists() else 0
            platform.event("task_finish",returned=out.get("returned",False))
            save(folder/"mission.json",out)
            backend.execution_deadline_monotonic=None
    simulate(args,sensor_exercise=exercise,archive_external_rgb=False)


def grade_attempt(c,item,ep,allow_rotated):
    out=read(ep/"mission.json")
    reads=out.get("readings",{})
    grade=grade_frames(ep,item,c["field"],reads)
    intr=read(ep/"sensor_geometry.json")["camera"]
    for key,stop in zip(("a","b"),item["stops"]):
        if key not in grade:
            continue
        g=grade[key]
        opt=measure(Path(stop["reference"]),ep/(key+".jpg"),intr)
        g["optics"]=opt
        g["raw_verified"]=g["verified"]
        if "part_pixels" not in g:
            continue
        parts_ok=min(g["part_pixels"].values())>=(144 if c["field"]=="cracked" else 36)
        if allow_rotated and opt["registered"] and g["rgb_agreement"]>=.90 and g["label_pixels"]>=144 and parts_ok:
            g["verified"]=True
        claim=reads.get(key)
        g["correct"]=bool(g["verified"] and g["label_box_verified"] and claim and claim["id"]==stop["target"]
                          and claim["source"]==key and claim["sha256"]==g["source_sha256"]
                          and type(claim["value"]) is type(g["actual"]) and claim["value"]==g["actual"])
    events=read_trace_rows(ep/"mission_events.jsonl")[0]
    starts={v["stop"]:v["clock"] for v in events if v["event"]=="travel_start"}
    ends={v["stop"]:v["clock"] for v in events if v["event"]=="travel_end"}
    returns={v["index"]:v["clock"] for v in events if v["event"]=="return_leg_start"}
    legs=[(starts[k],ends[k]) for k in ("a","b") if k in ends]
    legs.extend((returns[v["index"]],v["clock"]) for v in events if v["event"]=="return_leg_end")
    final=[v for v in events if v["event"]=="travel_end" and v["stop"]=="home"]
    route_ok=bool(not out.get("failure") and out.get("returned") and out["matching_world"] and
                  list(ends)==["a","b","home"] and final and close_pose(final[0]["measured"],out["current_home"]))
    captured={v["stop"] for v in events if v["event"]=="camera"}
    row=dict(case=c["id"],method=ep.name,route_ok=route_ok,elapsed_s=out["elapsed_s"],sends=out["sends"],
             frames=grade,physical_segment_s=[b-a for a,b in legs],return_segments=len(returns),
             refusal_reasons=[v["map_reason"] for v in events if v["event"]=="return_admission" and not v["accepted"]],
             failure=out.get("failure"),endpoint_block=out.get("endpoint_block"),attempt_sha256=digest(ep/"mission.json"))
    if ep.name=="calibration":
        row["admitted"]=bool(route_ok and captured=={"a","b"} and all(grade[k]["verified"] for k in ("a","b")))
    else:
        row.update(score_live(c,ep,out,grade,legs,events,route_ok,captured))
    return row


def eligible(config,group):
    if group=="calibration":
        return {c["id"]:["calibration"] for c in config["cases"]},{}
    cal=read(ROOT/"calibration/summary.json")["cases"]
    if len(cal)!=len(config["cases"]):
        raise ValueError("Both registered calibration attempts must settle")
    active={r["case"] for r in cal if r["admitted"]}
    slots={};aliases={}
    for c in config["cases"]:
        if c["id"] not in active:
            continue
        seen={};slots[c["id"]]=[]
        for method in c["order"]:
            graph=tuple(sorted(schedule(method,c["b_capture"]).items()))
            if graph in seen:
                aliases[c["id"]+"/"+method]=seen[graph]
            else:
                seen[graph]=method
                slots[c["id"]].append(method)
    return slots,aliases


def assess(config,roster,group):
    slots,aliases=eligible(config,group)
    eps=[(ROOT/group/k/m) for k,methods in slots.items() for m in methods]
    seals={str(ep.relative_to(ROOT)):sealed_attempt(ep) for ep in eps}
    assert_snapshot(ROOT/group/"sealed_attempts.json",seals)
    cases={v["id"]:v for v in config["cases"]};items={v["id"]:v for v in roster["cases"]}
    optics=read(ROOT/"optics.json")
    rows=[grade_attempt(cases[ep.parent.name],items[ep.parent.name],ep,optics["allow_rotated"]) for ep in eps]
    result=dict(role=group,registered=slots,equivalent_policies_without_independent_runs=aliases,
                actual_sends=sum(r["sends"] for r in rows),cases=rows,worker_sha256=digest(SOURCE),
                optics_grader_sha256=optics["grader_sha256"])
    save(ROOT/group/"summary.json",result)
    return result


def run(config,roster,group):
    grade_optics()
    target=ROOT/group
    target.mkdir(exist_ok=True)
    slots,_=eligible(config,group)
    with (ROOT.parent/"online.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key,methods in slots.items():
            for m in methods:
                ep=target/key/m
                rc=ep.with_suffix(".receipt.json")
                bind=dict(case=key,method=m,source_sha256=digest(SOURCE),registration_sha256=digest(ROOT/"registration.json"))
                if rc.exists():
                    old=read(rc)
                    if any(old.get(k)!=v for k,v in bind.items()):
                        raise ValueError("Version differs at "+str(rc))
                    seal=sealed_attempt(ep)
                    if old.get("files",seal)!=seal:
                        raise ValueError("Previous attempt files changed")
                    d=read(ep/"mission.json")
                    if old["state"]!="settled":
                        save(rc,{**old,"files":seal,"state":"settled","end":utc(),"actual_sends":d["sends"],
                                 "endpoint_block":d.get("endpoint_block")})
                    if d.get("endpoint_block"):
                        raise SystemExit("Preserved external restriction at "+str(rc))
                    continue
                if ep.exists():
                    raise SystemExit("Inspect the existing attempt "+str(ep))
                if group=="live" and unresolved_service_limits(ROOT.parent):
                    raise SystemExit("The current model service has an unresolved limit")
                ep.parent.mkdir(parents=True,exist_ok=True)
                meta=dict(**bind,coordinator=os.getpid(),start=utc())
                save(rc,{**meta,"state":"starting"})
                with ep.with_suffix(".log").open("a") as log:
                    cmd=[sys.executable,str(SOURCE),"worker","--case",key,"--method",m]
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    save(rc,{**meta,"state":"running","worker":proc.pid})
                    proc.wait()
                if proc.returncode or not (ep/"mission.json").is_file():
                    save(rc,{**meta,"state":"interrupted","exit":proc.returncode})
                    raise SystemExit("Preserve and inspect original failure "+str(rc))
                d=read(ep/"mission.json")
                save(rc,{**meta,"state":"settled","files":sealed_attempt(ep),"end":utc(),
                         "actual_sends":d["sends"],"endpoint_block":d.get("endpoint_block")})
                if d.get("endpoint_block"):
                    raise SystemExit("Original service interruption at "+str(rc))
        return assess(config,roster,group)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("calibration","live","grade","status","worker"))
    p.add_argument("--group",choices=("live","calibration"),default="live")
    p.add_argument("--case")
    p.add_argument("--method")
    args=p.parse_args()
    _,config,roster,_=register()
    if args.action=="worker":
        group="calibration" if args.method=="calibration" else "live"
        slots,_=eligible(config,group)
        if args.method not in slots.get(args.case,[]):
            raise SystemExit("Unregistered or closed worker")
        ep=ROOT/group/args.case/args.method
        rc=ep.with_suffix(".receipt.json")
        if ep.exists() or not rc.is_file() or read(rc)["state"] not in ("running","starting"):
            raise SystemExit("No fresh coordinator reservation")
        case=next(c for c in config["cases"] if c["id"]==args.case)
        item=next(c for c in roster["cases"] if c["id"]==args.case)
        simulator(case,item,args.method,ep)
    elif args.action=="status":
        for rc in sorted(ROOT.glob("*/*/*.receipt.json")):
            s=read(rc);pid=s.get("worker");print(rc.relative_to(ROOT),s["state"],
                bool(pid and Path("/proc/%d"%pid).exists()),s.get("actual_sends"))
    else:
        res=assess(config,roster,args.group) if args.action=="grade" else run(config,roster,args.action)
        for row in res["cases"]:
            print(row["case"],row["method"],row.get("strict",row.get("admitted")),round(row["elapsed_s"],2),row["sends"])


if __name__=="__main__":
    main()
