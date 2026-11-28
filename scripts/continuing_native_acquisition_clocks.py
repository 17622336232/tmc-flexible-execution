"""Measure authorized capture and return once per exposed heading. No model client or recognition."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

from experiments.flexible_execution.adapters.native_views import close_pose
from experiments.flexible_execution.revision_audit import digest, read, save
from research.flexible_scheduling.ordinary_frontier import camera_goals
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_first_view_replay import photo, pose
from scripts.continuing_native_action_binding import OUT as BIND
from scripts.continuing_native_section_study import CFG
from scripts.continuing_navigation_assay import geometry_hash

ROOT = BIND.parent / "native-acquisition-clocks-dev01"
SCRIPT = Path(__file__)


def register():
    rows = []
    binding = read(BIND / "eligibility.json")
    for case in read(CFG)["cases"]:
        ep = Path(case["source"])
        data = read(ep / "clean_preflight.json")
        original, = [r for r in binding["rows"] if r["case"]==case["id"] and r["method"]=="original"]
        rows.append(dict(case=case["id"], source=case["source"], direction=case["direction"], angle_degrees=case["angle_degrees"],
            ordinary_first_eligible=original["eligible"], world=geometry_hash(ep),
            settings={k:data[k] for k in ("family","layout_seed","seed","equipment_count","source_profile")}))
    ROOT.mkdir(exist_ok=True)
    assert_snapshot(ROOT/"registration.json", dict(eligibility=digest(BIND/"eligibility.json"), cases=rows,
        worker=digest(SCRIPT), test="Zero semantic requests. At a measured front take one user-authorized turn, capture the side and return once. The return to the front in both calibration runs measures cost even when the original job finishes at the side. Preserve all failures. These are action clocks, not task outcomes."))
    return rows


def fresh(backend, after, goal, timeout):
    start=monotonic()
    while monotonic()-start<timeout:
        obs, geo=backend.observe()
        if obs["ros_stamp"]>after and close_pose(pose(obs),goal):
            return obs,geo
        sleep(.025)
    raise ValueError("No fresh source-matched image in the registered time")


def worker(case):
    from experiments.task_attack_integrity.run import main as simulate
    target=ROOT/case["case"]
    args=["--policy","clean","--backend","cpu","--headless","--no-rviz","--sensor-smoke",
          "--ros-port","11894","--gazebo-port","11895","--episode-timeout","80","--output",str(target)]
    for k,v in case["settings"].items():
        args+=["--"+k.replace("_","-"),str(v)]
    def capture(backend, first, folder, event):
        begin=monotonic()
        backend.execution_deadline_monotonic=begin+80
        record=dict(world_matches=geometry_hash(folder)==case["world"], phases=[])
        try:
            if not record["world_matches"]:
                raise ValueError("Mismatched scene")
            front,g=backend.observe()
            photo(folder,"front",front)
            goals=camera_goals(case,front["position"],front["yaw"])
            for view in ("side","front"):
                start=monotonic()
                arrived=backend.move(goals[view])
                motion=monotonic()-start
                obs,_=fresh(backend,backend.ros.Time.now().to_sec(),goals[view],5)
                elapsed=monotonic()-start
                photo(folder,"end-"+view,obs)
                record["phases"].append(dict(view=view,motion_s=motion,after_move_capture_s=elapsed-motion,
                    to_fresh_source_s=elapsed,actual_yaw=obs["yaw"],arrival=pose(arrived),sha256=digest(folder/("end-"+view+".jpg"))))
        except Exception as ex:
            record["error"]=dict(code=getattr(ex,"code",type(ex).__name__),message=str(ex)[:220])
        finally:
            backend.execution_deadline_monotonic=None
            save(folder/"measures.json",record)
    simulate(args,sensor_exercise=capture,archive_external_rgb=False)


def receipt(folder):
    if list(folder.rglob("model_calls.jsonl")) or list(folder.rglob("sends.jsonl")):
        raise ValueError("Calibration has a model call")
    files=[folder/"measures.json",folder/"result.json",*sorted(folder.glob("end-*")),folder/"front.jpg",folder/"front.json"]
    return {p.name:digest(p) for p in files if p.is_file()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("run","worker","summary"))
    parser.add_argument("--case")
    args=parser.parse_args();cases=register()
    if args.mode=="worker":
        item=next(c for c in cases if c["case"]==args.case)
        if (ROOT/args.case).exists() or not (ROOT/(args.case+".start.json")).is_file():
            raise ValueError("Only one fresh calibration is allowed")
        worker(item)
        return
    if args.mode=="run":
        with (ROOT.parent/"online.lock").open("a+") as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for case in cases:
                name=case["case"];folder=ROOT/name
                start=ROOT/(name+".start.json")
                if not start.exists():
                    if folder.exists():raise ValueError("Existing calibration directory without receipt")
                    save(start,dict(registration=digest(ROOT/"registration.json"),case=name))
                    with (ROOT/(name+".log")).open("a") as stream:
                        proc=subprocess.Popen([sys.executable,str(SCRIPT),"worker","--case",name],stdout=stream,stderr=subprocess.STDOUT)
                        save(ROOT/(name+".worker.json"),dict(pid=proc.pid))
                        proc.wait()
                    if proc.returncode:raise SystemExit("Preserve physical worker failure "+name)
                if not (folder/"measures.json").is_file() or not (folder/"result.json").is_file():
                    raise SystemExit("Preserve incomplete physical calibration "+name)
                assert_snapshot(ROOT/(name+".receipt.json"),receipt(folder))
    rows=[]
    for case in cases:
        folder=ROOT/case["case"]
        if read(folder.with_suffix(".receipt.json"))!=receipt(folder):raise ValueError("Clock or image changed")
        rows.append(dict(case=case["case"],**read(folder/"measures.json")))
    save(ROOT/"summary.json",dict(new_model_requests=0,role="physical clocks",cases=rows))
    print([(r["case"],[round(v["to_fresh_source_s"],3) for v in r["phases"]],r.get("error")) for r in rows])


if __name__=="__main__":
    main()

