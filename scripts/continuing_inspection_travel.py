"""Run the pre-registered two-station inspection with original local navigation and metered model use."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from experiments.flexible_execution.adapters.native_views import close_pose
from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES
from experiments.task_attack.runtime import append_json
from research.flexible_scheduling.inspection_travel import Capture, execute, positive, schedule
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_itinerary_sources import CFG, ROOT, grade_frames, register
from scripts.continuing_navigation_assay import geometry_hash


def utc():
    return datetime.now(timezone.utc).isoformat()


def pose(s):
    return {**s["position"], "yaw":s["yaw"]}


class Platform:
    def __init__(self, backend, start, root, deadline):
        self.backend = backend
        self.start = pose(start)
        self.root = root
        self.deadline = deadline

    def event(self, name, **values):
        append_json(self.root/"mission_events.jsonl", dict(event=name,clock=monotonic(),**values))

    def remaining(self):
        if monotonic()>=self.deadline:
            raise TimeoutError("Registered complete-task deadline")

    def move(self, key, goal):
        self.remaining()
        self.backend.check_goal(goal)
        self.event("travel_start", stop=key, goal=goal)
        reached = self.backend.move(goal)
        measured = pose(reached)
        self.event("travel_end",stop=key,measured=measured,goal=goal)
        if not close_pose(measured,goal):
            raise RuntimeError("Original navigation did not reach registered stop "+key)

    def home(self):
        self.move("home",self.start)

    def take(self, key):
        self.remaining()
        obs,_ = self.backend.observe()
        self.remaining()
        frame = Capture(key,obs["data"])
        (self.root/(key+".jpg")).write_bytes(frame.data)
        save(self.root/(key+".json"),dict(**{k:v for k,v in obs.items() if k!="data"},sha256=frame.sha))
        self.event("camera",stop=key,sha256=frame.sha,pose=pose(obs))
        return frame


def worker(config,c,item,method,folder):
    from experiments.task_attack_integrity.run import main as simulate
    instrument = method=="calibration"
    argv=["--policy","clean","--backend","cpu","--headless","--no-rviz","--sensor-smoke",
          "--ros-port","11884","--gazebo-port","11885","--output",str(folder),"--episode-timeout","240"]
    for k,v in item["settings"].items():
        argv += ["--"+k.replace("_","-"),str(v)]
    def probe(backend,obs,ep,event):
        begin=monotonic()
        end=begin+240
        platform=Platform(backend,obs,ep,end)
        backend.execution_deadline_monotonic=end
        out=dict(start=begin,role=method,registration_sha256=digest(ROOT/"registration.json"),case=c["id"],
                 current_home=platform.start,matching_world=geometry_hash(ep)==item["matching_world"])
        try:
            if not out["matching_world"]:
                raise ValueError("Station locations belong to a different static layout")
            platform.event("task_start",method=method)
            if instrument:
                for k,s in zip(("a","b"),item["stops"]):
                    platform.move(k,s["goal"])
                    platform.take(k)
                platform.home()
                out["returned"]=True
            else:
                from experiments.task_attack.model import ModelClient
                from uav_bridge.chatgpt_auth import CredentialStore
                client=ModelClient(ep/"model_calls.jsonl",timeout=120,
                    credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),
                    request_budget=RequestBudget(ep/"sends.jsonl",2))
                client.deadline_monotonic=end
                out.update(execute(c,item["stops"],platform,client,method,end))
            platform.remaining()
        except Exception as ex:
            code=getattr(ex,"code",type(ex).__name__)
            out["failure"]=dict(code=code,message=str(ex)[:240])
            if code in EXTERNAL_RESPONSE_CODES or getattr(ex,"status",None) in (401,403,429):
                out["endpoint_block"]=str(code)
        finally:
            out["elapsed_s"]=monotonic()-begin
            out["finished"]=monotonic()
            out["sends"]=len((ep/"sends.jsonl").read_bytes().splitlines()) if (ep/"sends.jsonl").exists() else 0
            platform.event("task_finish",returned=out.get("returned",False))
            save(ep/"mission.json",out)
            backend.execution_deadline_monotonic=None
    simulate(argv,sensor_exercise=probe,archive_external_rgb=False)


def sealed_attempt(ep):
    files=[ep/n for n in ("mission.json","mission_events.jsonl","result.json","model_calls.jsonl","sends.jsonl",
                          "a.jpg","a.json","b.jpg","b.json") if (ep/n).exists()]
    if not (ep/"mission.json").exists() or not (ep/"result.json").exists():
        raise ValueError("Incomplete original attempt: "+str(ep))
    return {p.name:digest(p) for p in files}


def collect(config,reg,mode):
    instrument=mode=="calibration"
    group=ROOT/mode
    cases={v["id"]:v for v in reg["cases"]}
    attempts={}
    for c in config["cases"]:
        for ep in sorted((group/c["id"]).glob("*")):
            if ep.is_dir():
                attempts[(c["id"],ep.name)] = sealed_attempt(ep)
    if instrument and len(attempts)!=len(cases):
        raise SystemExit("All registered local attempts must finish before inspecting calibration truth")
    seal={k+"/"+m:v for (k,m),v in attempts.items()}
    assert_snapshot(group/"sealed_attempts.json",seal)
    summary=[]
    for (key,m),hashes in attempts.items():
        c=next(v for v in config["cases"] if v["id"]==key)
        ep=group/key/m
        d=read(ep/"mission.json")
        rows=grade_frames(ep,cases[key],c["field"],d.get("readings",{}))
        path=read_trace_rows(ep/"mission_events.jsonl")[0]
        legs={v["stop"]:v["clock"] for v in path if v["event"]=="travel_start"}
        motion=[(legs[v["stop"]],v["clock"]) for v in path if v["event"]=="travel_end"]
        photo_keys={v["stop"] for v in path if v["event"]=="camera"}
        route_ok=bool(not d.get("failure") and d.get("returned") and d["matching_world"] and len(motion)==3
                      and list(legs)==["a","b","home"])
        record=dict(case=key,method=m,elapsed_s=d["elapsed_s"],sends=d["sends"],route_ok=route_ok,frames=rows,
                    motion_s=[b-a for a,b in motion],endpoint_block=d.get("endpoint_block"),failure=d.get("failure"),
                    attempt_sha256=hashes["mission.json"])
        if instrument:
            record["admitted"]=route_ok and photo_keys=={"a","b"} and all(rows[k]["verified"] for k in ("a","b"))
        else:
            record.update(score_live(c,ep,d,rows,motion,path,route_ok,photo_keys))
        summary.append(record)
    save(group/"summary.json",dict(role=mode,registrations=digest(ROOT/"registration.json"),cases=summary))
    return summary


def score_live(c,ep,out,graded,motion,events,route_ok,photos):
    raw = read_trace_rows(ep/"model_calls.jsonl")[0] if (ep/"model_calls.jsonl").exists() else []
    reservations = read_trace_rows(ep/"sends.jsonl")[0] if (ep/"sends.jsonl").exists() else []
    expected = {r["reservation"]:r for r in reservations}
    calls={}
    for r in raw:
        if "dispatched_monotonic_seconds" not in r:
            continue
        key=r["input"]["image_id"]
        h=r["input"]["sha256"]
        if (r["logical_job_attempt"] not in expected or expected[r["logical_job_attempt"]]["image_sha256"]!=[h] or
                r["image_sha256"]!=[h] or h!=digest(ep/(key+".jpg")) or key in calls
                or r["model"]!="gpt-6.1-terra" or r["reasoning_effort"]!="max"):
            raise ValueError("Unverifiable camera/transport binding")
        calls[key]=(r["dispatched_monotonic_seconds"],r["return_monotonic_seconds"])
    if len(calls)!=len(reservations):
        raise ValueError("Every actual send must have its original response or error")
    times={v["stop"]:v["clock"] for v in events if v["event"]=="camera"}
    true_required="a" in graded and positive({"value":graded["a"]["actual"]},c["field"])
    actual_active=positive(out.get("readings",{}).get("a"),c["field"])
    permitted=(c["b_capture"]=="mandatory-archive" or "b" not in times or
               (actual_active and "a" in calls and times["b"]>=calls["a"][1]))
    permitted &= ("b" not in calls or (actual_active and "a" in calls and calls["b"][0]>=calls["a"][1]))
    used={"a","b"} if true_required else {"a"}
    expected_photos={"a","b"} if c["b_capture"]=="mandatory-archive" or true_required else {"a"}
    correct=bool(route_ok and permitted and used==set(calls) and photos==expected_photos and
                 all(graded.get(k,{}).get("correct",False) for k in used))
    service=sum(b-a for a,b in calls.values())
    overlap=sum(max(0.,min(b,y)-max(a,x)) for a,b in calls.values() for x,y in motion)
    return dict(strict=correct,restricted_s=out["elapsed_s"] if correct else 240.,
        condition_positive_actual=true_required,condition_positive_reported=actual_active,
        permissions_verified=bool(permitted),service_s=service,overlap_s=overlap,service_without_motion_s=service-overlap)


def run(config,reg,mode):
    group=ROOT/mode
    group.mkdir(exist_ok=True)
    item_by_id={v["id"]:v for v in reg["cases"]}
    instrument=mode=="calibration"
    accepted={}
    if not instrument:
        cal=read(ROOT/"calibration/summary.json")
        accepted={r["case"]:r["admitted"] for r in cal["cases"]}
        if len(accepted)!=len(config["cases"]):
            raise ValueError("Incomplete local calibration")
    aliases={}
    with (ROOT.parent/"online.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case in config["cases"]:
            key=case["id"]
            if not instrument and not accepted[key]:
                continue
            seen={}
            for method in (["calibration"] if instrument else case["order"]):
                if not instrument:
                    graph=str(sorted(schedule(method,case["b_capture"]).items()))
                    if graph in seen:
                        aliases[key+"/"+method]=seen[graph]
                        continue
                    seen[graph]=method
                ep=group/key/method
                receipt=ep.with_suffix(".receipt.json")
                base=dict(method=method,case=key,registration_sha256=digest(ROOT/"registration.json"),
                          worker_sha256=digest(Path(__file__)))
                if receipt.exists():
                    old=read(receipt)
                    if any(old.get(k)!=v for k,v in base.items()):
                        raise ValueError("Existing attempt has a different version or registration")
                    if (ep/"mission.json").is_file() and (ep/"result.json").is_file():
                        seal=sealed_attempt(ep)
                        if old.get("files",seal)!=seal:
                            raise ValueError("Existing attempt files changed")
                        if read(ep/"mission.json").get("endpoint_block"):
                            raise SystemExit("Recorded external service restriction at "+str(receipt))
                        if old.get("state")!="settled":
                            save(receipt,{**old,"state":"settled","files":seal,"end":utc()})
                        continue
                    raise SystemExit("Inspect the preserved unsettled attempt: "+str(ep))
                if ep.exists():
                    raise SystemExit("An unregistered existing attempt requires inspection: "+str(ep))
                if not instrument and unresolved_service_limits(ROOT.parent):
                    raise SystemExit("The authorized model service is currently restricted")
                ep.parent.mkdir(parents=True,exist_ok=True)
                meta=dict(**base,start=utc(),state="starting",coordinator=os.getpid())
                save(receipt,meta)
                cmd=[sys.executable,"scripts/continuing_inspection_travel.py","worker","--case",key,"--method",method]
                with ep.with_suffix(".log").open("a") as log:
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    save(receipt,{**meta,"state":"running","worker":proc.pid})
                    proc.wait()
                if proc.returncode or not (ep/"mission.json").exists():
                    save(receipt,{**meta,"state":"interrupted","exit":proc.returncode})
                    raise SystemExit("Preserve failed process at "+str(ep))
                out=read(ep/"mission.json")
                save(receipt,{**meta,"state":"settled","files":sealed_attempt(ep),"end":utc(),
                              "endpoint_block":out.get("endpoint_block"),"actual_sends":out["sends"]})
                if out.get("endpoint_block"):
                    raise SystemExit("Service restriction: "+str(receipt))
        if not instrument:
            assert_snapshot(group/"equivalent_policies.json",aliases)
        return collect(config,reg,mode)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("register","calibration","live","worker","grade","status"))
    parser.add_argument("--group",choices=("calibration","live"),default="live")
    parser.add_argument("--case")
    parser.add_argument("--method")
    args=parser.parse_args()
    config,reg=register()
    if args.mode=="register":
        print([(r["id"],r["stops"][0]["target"],r["stops"][1]["target"]) for r in reg["cases"]])
    elif args.mode=="worker":
        c=next(v for v in config["cases"] if v["id"]==args.case)
        item=next(v for v in reg["cases"] if v["id"]==args.case)
        mode="calibration" if args.method=="calibration" else "live"
        if mode=="live":
            if args.method not in c["order"] or not any(r["case"]==c["id"] and r["admitted"]
                    for r in read(ROOT/"calibration/summary.json")["cases"]):
                raise SystemExit("Live case is outside the calibration or method registration")
        folder=ROOT/mode/c["id"]/args.method
        rc=folder.with_suffix(".receipt.json")
        if folder.exists() or not rc.is_file() or read(rc)["state"] not in ("starting","running"):
            raise SystemExit("Worker requires its fresh coordinator reservation")
        worker(config,c,item,args.method,folder)
    elif args.mode=="status":
        for r in sorted(ROOT.glob("*/*/*.receipt.json")):
            d=read(r);p=d.get("worker")
            print(r.relative_to(ROOT),d["state"],bool(p and Path("/proc/%d"%p).exists()),d.get("actual_sends"))
    else:
        rows=collect(config,reg,args.group) if args.mode=="grade" else run(config,reg,args.mode)
        for v in rows:
            print(v["case"],v["method"],v.get("admitted",v.get("strict")),round(v["elapsed_s"],2),v["sends"])


if __name__=="__main__":
    main()

