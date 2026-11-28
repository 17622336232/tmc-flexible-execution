"""Run and independently score a capped set of complete native closed-loop development tasks."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

from experiments.flexible_execution.adapters.original_controller import run as original
from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import digest,read,save
from experiments.flexible_execution.timeline import Timeline,read_trace_rows
from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES,ModelClient
from research.flexible_scheduling.native_joint_execution import run_joint
from research.flexible_scheduling.natural_frontier import discover,source_map,released
from scripts.continuing_crossworld_views import assert_snapshot,copied_renderer
from scripts.continuing_native_action_binding import visual_instances
from scripts.continuing_navigation_assay import geometry_hash

CFG=Path("config/native_joint_execution_dev01.json")
ROOT=Path("outputs/continuing/native-joint-execution-dev01")
SOURCE=Path(__file__).resolve().relative_to(Path.cwd().resolve())


def register():
    cfg=read(CFG)
    clock=Path(cfg["capture_calibration"])
    parent=ROOT.parent
    first=read(parent/"native-natural-frontier-dev01/summary.json")
    follow=read(parent/"native-natural-followup-dev01/summary.json")
    cross=read(parent/"cross-source-assay-dev02/summary.json")
    motion=read(clock/"summary.json")
    if not follow["independent_correct_combination"]:
        raise ValueError("Prior native evidence/permission bridge did not pass")
    cases=[]
    for case in cfg["cases"]:
        plan=discover(case["instruction"])
        if plan is None or set(case["score_predicates"])!={f.name for f in plan.fields}:
            raise ValueError("A declared native task or its independent scoring catalogue is incomplete")
        i=case["id"]
        original=next(v for v in read(clock/"registration.json")["cases"] if v["case"]==i)
        measure=next(v for v in motion["cases"] if v["case"]==i)
        if not measure["world_matches"] or not measure["phases"] or measure["phases"][0]["view"]!="side":
            raise ValueError("No measured native command for this development layout")
        service=next(v["service_s"] for v in first["rows"] if v["case"]==i)
        d=measure["phases"][0]
        prices=dict(travel_s=[[0,d["motion_s"]],[d["motion_s"],0]],photo_s=d["after_move_capture_s"],
                    single_s=service,same_s=follow["live_service_s"],cross_s=max(
                        c["observations"]["joint"]["service_s"] for c in cross["cases"]),cross_enabled=True)
        cases.append({**case,"settings":original["settings"],"world":original["world"],"prices":prices})
    core=Path("research/flexible_scheduling")
    files=[SOURCE,Path("experiments/task_attack/agent.py"),Path("experiments/flexible_execution/adapters/original_controller.py"),
           *(core/(n+".py") for n in ("native_joint_execution","natural_frontier","native_frontier_adapter",
              "event_workflow","event_policy","event_lookahead","cross_source_workflow","priced_cross_source") if (core/(n+".py")).is_file())]
    ROOT.mkdir(exist_ok=True)
    reg=dict(config=digest(CFG),cases=cases,code={str(f):digest(f) for f in files},
             sources={str(c):digest(c) for c in (clock/"summary.json",parent/"native-natural-frontier-dev01/summary.json",
                  parent/"native-natural-followup-dev01/summary.json",parent/"cross-source-assay-dev02/summary.json")})
    assert_snapshot(ROOT/"registration.json",reg)
    return cfg,reg


def execute(cfg,case,method,ep):
    from experiments.task_attack_integrity.run import main as simulate
    from uav_bridge.chatgpt_auth import CredentialStore
    args=["--policy","clean","--backend","cpu","--headless","--no-rviz","--ros-port","11912","--gazebo-port","11913",
          "--episode-timeout",str(cfg["deadline_s"]),"--output",str(ep)]
    for k,v in case["settings"].items():
        args += ["--"+k.replace("_","-"),str(v)]
    def controller(backend, ignored_task, folder,event,timeout):
        # The controller sees only the independently registered natural-language request.
        if geometry_hash(ep)!=case["world"]:
            raise ValueError("The native world differs from the paired setup")
        trace=Timeline(folder/"timeline.jsonl")
        class MeteredModel(ModelClient):
            def request(self,instructions,data,images=(),attempt_limit=1):
                return super().request(instructions,data,images,attempt_limit=1)
        client=MeteredModel(folder/"model_calls.jsonl",timeout=cfg["per_send_s"],
            credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),
            request_budget=RequestBudget(folder/"sends.jsonl",cfg["per_arm_max_sends"]))
        from time import monotonic
        started=monotonic()
        try:
            if method=="original-gpt":
                return original(backend,case["instruction"],folder,event,timeout,model=client,trace=trace)
            return run_joint(backend,case["instruction"],folder,event,timeout,model=client,trace=trace,
                             mode="joint" if method=="joint-late-shutter" else method,prices=case["prices"],
                             retain_pending_photo=cfg.get("retain_pending_photo",False)
                             and method!="joint-late-shutter")
        finally:
            state=backend.adapter.snapshot()
            save(ep/"joint_terminal.json",dict(task_s=monotonic()-started,position=state["position"],yaw=state["yaw"],
                calls=client.calls,endpoint_block=getattr(client.external_error,"code",None)))
    simulate(args,agent_runner=controller,archive_external_rgb=False)


def signature(ep):
    files=[ep/"result.json",ep/"joint_terminal.json",ep/"private/truth.json",ep/"inspection.world",
           *((ep/"private/robot").glob("*.json*")),
           *((ep/"private/robot").glob("onboard_*.jpg"))]
    return {p.relative_to(ep).as_posix():digest(p) for p in files if p.is_file()}


def slots(reg):
    for case in reg["cases"]:
        for method in case["order"]:
            yield case,method,ROOT/case["id"]/method


def collect(cfg,reg):
    with (ROOT.parent/"online.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case,method,ep in slots(reg):
            receipt=ep.with_suffix(".receipt.json")
            start=ep.with_suffix(".start.json")
            if receipt.exists():
                r=read(receipt)
                if r["files"]!=signature(ep):
                    raise ValueError("Preserved episode changed")
                if r["endpoint_block"]:
                    raise SystemExit("Preserved original service restriction for "+method)
                continue
            if (ep/"result.json").is_file() and start.is_file():
                restored=True
            else:
                restored=False
                if ep.exists() or start.exists() or unresolved_service_limits(ROOT.parent):
                    raise SystemExit("Inspect unsettled episode or recorded service restriction before sending")
                ep.parent.mkdir(exist_ok=True)
                save(start,dict(case=case["id"],method=method,registration=digest(ROOT/"registration.json")))
                with ep.with_suffix(".log").open("a") as out:
                    command=[sys.executable,str(SOURCE),"worker","--study",cfg["id"],
                             "--case",case["id"],"--method",method]
                    p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
                    save(ep.with_suffix(".worker.json"),dict(pid=p.pid))
                    p.wait()
                if not (ep/"result.json").is_file():
                    raise SystemExit("Keep the original episode startup failure")
            terminal=read(ep/"joint_terminal.json") if (ep/"joint_terminal.json").is_file() else {}
            result=read(ep/"result.json")
            error=result.get("error") or {}
            blocker=terminal.get("endpoint_block") or (error.get("code") if error.get("code") in EXTERNAL_RESPONSE_CODES else None)
            save(receipt,dict(case=case["id"],method=method,files=signature(ep),endpoint_block=blocker,
                calls=terminal.get("calls",0),restored=restored))
            print(case["id"],method,"sealed",terminal.get("calls",0),error.get("message"),flush=True)
            if blocker:
                raise SystemExit("Existing service reported a restriction")
    return grade(cfg,reg)


def evidence_claims(records):
    for row in records:
        if isinstance(row,dict) and isinstance(row.get("facts"),dict):
            name,frame,value=row.get("object_id"),row.get("evidence_image_id"),row["facts"].get("visible")
            if isinstance(name,str) and isinstance(frame,str) and type(value) is bool:
                yield name,frame,value


def runtime_bindings(ep, plan, poses):
    robot=ep/"private/robot"
    vals={}
    accepted=set()
    errs=[]
    steps=read_trace_rows(robot/"model_calls.jsonl")[0] if (robot/"model_calls.jsonl").is_file() else []
    for ordinal,c in enumerate(steps,1):
        if "dispatched_monotonic_seconds" not in c:
            continue
        inp=c["input"]
        if c["model"]!="gpt-6.1-terra" or c["reasoning_effort"]!="max":
            raise ValueError("The experiment must keep the same native model")
        frames=[*[v["image_id"] for v in inp["recent_images"]],inp["current_image_id"]]
        if [digest(robot/(i+".jpg")) for i in frames]!=c["image_sha256"]:
            raise ValueError("Model attachment does not match recorded native JPEG")
        ready=released(plan,vals)
        reply=c.get("output") or {}
        returned=list(reply.get("records") or [])+list((reply.get("report") or {}).get("records") or [])
        staged={}
        for name,image,value in evidence_claims(returned):
            claim=(name,image,value)
            if claim in accepted:
                continue
            matching=[i for i,f in enumerate(plan.fields) if f.name==name and f.view==poses.get(image)]
            if len(matching)!=1 or image not in frames:
                errs.append(dict(turn=ordinal,field=name,photo=image,kind="unbound-source"))
                continue
            i=matching[0]
            if i in vals:
                if vals[i]!=value:
                    errs.append(dict(turn=ordinal,field=name,photo=image,kind="changed-confirmed-value"))
                else:
                    accepted.add(claim)
                continue
            if i not in ready:
                errs.append(dict(turn=ordinal,field=name,photo=image,kind="not-newly-authorized"))
                continue
            if i in staged and staged[i]!=value:
                errs.append(dict(turn=ordinal,field=name,kind="same-response-conflict"))
            staged[i]=value
            accepted.add(claim)
        vals.update(staged)
    return accepted,errs,steps


def score(cfg,case,method,ep):
    result=read(ep/"result.json")
    terminal=read(ep/"joint_terminal.json") if (ep/"joint_terminal.json").is_file() else {}
    robot=ep/"private/robot"
    files=sorted(robot.glob("onboard_[0-9]*.json"))
    row=dict(case=case["id"],method=method,sends=terminal.get("calls",0),whole_s=terminal.get("task_s"),
             strict=False,restricted_s=cfg["deadline_s"],error=result.get("error"))
    ledger=read_trace_rows(robot/"sends.jsonl")[0] if (robot/"sends.jsonl").is_file() else []
    if len(ledger)!=row["sends"]:
        raise ValueError("Native send reservation and terminal count differ")
    if not files:
        return row
    plan=discover(case["instruction"])
    frames=[read(p) for p in files]
    poses=source_map(plan,frames[0],frames)
    final_pose=source_map(plan,frames[0],[dict(image_id="final",**{k:terminal[k] for k in ("position","yaw")})]).get("final") if terminal else None
    bound,issues,steps=runtime_bindings(ep,plan,poses)
    if sum("dispatched_monotonic_seconds" in c for c in steps)!=len(ledger):
        raise ValueError("Native transport dispatch count differs from reservation")
    expected=[]
    values={}
    verifiers={}
    report=result.get("report") or {}
    claimed=list(evidence_claims(report.get("records") or []))
    with copied_renderer(ep) as (scene,renderer):
        truth=read(ep/"private/truth.json")["devices"]
        def assess(image, field):
            if image not in verifiers:
                photo=robot/(image+".jpg")
                verifiers[image]=visual_instances(scene,renderer,read(photo.with_suffix(".json")),photo,truth)
            alignment,pixels=verifiers[image]
            category=case["score_predicates"][field]
            value=max(pixels.get(category,{}).values(),default=0)>=36
            return value if alignment>=.90 else None
        for i,f in enumerate(plan.fields):
            if i not in released(plan,values):
                continue
            actual=next((v for v in claimed if v[0]==f.name and poses.get(v[1])==f.view),None)
            frame=actual[1] if actual else next((j for j,v in poses.items() if v==f.view),None)
            check=assess(frame,f.name) if frame else None
            values[i]=check
            expected.append(dict(field=f.name,view=f.view,expected=check,
                                 bound_and_correct=bool(actual and actual in bound and actual[-1] is check)))
    expected_names={v["field"] for v in expected}
    strict=(report.get("outcome")=="accepted" and final_pose==plan.finish and not issues
            and all(v["bound_and_correct"] for v in expected) and len(expected_names)==len(claimed)
            and len(report.get("records") or [])==len(claimed))
    timeline=read_trace_rows(robot/"timeline.jsonl")[0] if (robot/"timeline.jsonl").is_file() else []
    moves=[e for e in timeline if e["event"]=="joint_motion_during_request"]
    from scripts.continuing_native_report import length
    request_intervals=[(c["dispatched_monotonic_seconds"],c["return_monotonic_seconds"])
                       for c in steps if "dispatched_monotonic_seconds" in c]
    physical_starts={}
    actions=[]
    for e in timeline:
        key=(e["step"],e.get("action"),e.get("purpose"))
        if e["event"]=="action_started":
            physical_starts[key]=e["monotonic_seconds"]
        elif e["event"]=="action_finished" and key in physical_starts:
            actions.append((physical_starts.pop(key),e["monotonic_seconds"]))
    overlap=length([(max(a,c),min(b,d)) for a,b in actions for c,d in request_intervals if min(b,d)>max(a,c)])
    early_captures=[]
    for e in timeline:
        if e["event"]!="joint_pending_capture":
            continue
        image=next((f["image_id"] for f in frames if f["sha256"]==e["image_sha256"]
                    and f["ros_stamp"]==e["capture_ros"]),None)
        truly_early=any(a<=e["started_monotonic_seconds"]<=e["monotonic_seconds"]<b
                        for a,b in request_intervals)
        early_captures.append(dict(image_id=image,completed_with_request_pending=truly_early,
                                   later_attached=any(e["image_sha256"] in c["image_sha256"] and
                                                      c.get("dispatched_monotonic_seconds",0)>e["monotonic_seconds"]
                                                      for c in steps),
                                   used_in_report=any(v[1]==image for v in claimed)))
    return {**row,"strict":strict,"restricted_s":row["whole_s"] if strict else cfg["deadline_s"],
            "actual_service_s":length(request_intervals),"service_with_action_overlap_s":overlap,
            "motion_while_pending":len(moves),"permission_violations":issues,"requested_fields":[
                c["input"].get("this_request_scope",{}).get("open_for_reading") for c in steps],
            "final_pose_ok":final_pose==plan.finish,"expected":expected,"pending_captures":early_captures,
            "submission_sources":[list(v) for v in claimed]}


def grade(cfg,reg):
    runs=list(slots(reg))
    receipts={}
    for case,method,ep in runs:
        p=ep.with_suffix(".receipt.json")
        if read(p)["files"]!=signature(ep):
            raise ValueError("Cannot independently grade a changed native run")
        receipts[case["id"]+"/"+method]=digest(p)
    assert_snapshot(ROOT/"sealed.json",receipts)
    rows=[score(cfg,*v) for v in runs]
    out=dict(rows=rows,strict=sum(r["strict"] for r in rows),actual_sends=sum(r["sends"] for r in rows),
             registration_sha256=digest(ROOT/"registration.json"),sealed_sha256=digest(ROOT/"sealed.json"))
    save(ROOT/"summary.json",out)
    print([(r["method"],r["strict"],r["whole_s"],r["sends"]) for r in rows])
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("run","worker","grade","status"))
    p.add_argument("--study",choices=("native-joint-execution-dev01","native-pending-photo-dev02"),
                   default="native-joint-execution-dev01")
    p.add_argument("--case")
    p.add_argument("--method")
    a=p.parse_args()
    if a.study!="native-joint-execution-dev01":
        ROOT=ROOT.parent/a.study
        CFG=Path("config")/(a.study.replace("-","_")+".json")
    cfg,reg=register()
    if a.mode=="run":
        collect(cfg,reg)
    elif a.mode=="worker":
        case,method,ep=next(v for v in slots(reg) if (v[0]["id"],v[1])==(a.case,a.method))
        if ep.exists() or not ep.with_suffix(".start.json").is_file():
            raise ValueError("A fresh reservation is required")
        execute(cfg,case,method,ep)
    elif a.mode=="grade":
        grade(cfg,reg)
    else:
        print([(c["id"],m,f.with_suffix(".receipt.json").is_file()) for c,m,f in slots(reg)])
