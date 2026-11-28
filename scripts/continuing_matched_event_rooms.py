"""Six matched real-camera development tasks with common permissions, costs, and strict deferred grading."""
import argparse
from collections import defaultdict
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
from time import monotonic,time

from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.public_audit import _grade_segmentation,save_segmentation
from experiments.flexible_execution.public_task import Room,MIN_OBJECT_PIXELS
from experiments.flexible_execution.public_visual import _house,open_controller
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import read,save,digest
from experiments.flexible_execution.timeline import Timeline,read_trace_rows
from experiments.task_attack.model import ModelClient,EXTERNAL_RESPONSE_CODES
from research.flexible_scheduling.event_workflow import ALWAYS,UNKNOWN,EventModel
from research.flexible_scheduling.live_event_room import mission,run_mission,time_reference
from research.flexible_scheduling.scoped_room_query import PROMPT
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_native_report import length
from scripts.continuing_room_event_calibration import CAL,ROOT,launch_local,prepare


LIVE=ROOT/"live"
SOURCE=Path(__file__)


def protocol():
    cfg,runtime=prepare()
    cal=read(CAL/"summary.json")
    if cal["actual_sends"]!=6 or cal["sealed_sha256"]!=digest(CAL/"sealed_predictions.json"):
        raise ValueError("Incomplete calibration block")
    costs={c["id"]:time_reference(c,cal,ROOT.parent/"room-workflow-dev01/smoke") for c in cfg["cases"]}
    core=Path("research/flexible_scheduling")
    files=[SOURCE,*[core/(name+".py") for name in ("event_workflow","event_policy","event_lookahead","live_event_room","scoped_room_query")]]
    registration=dict(parent=digest(ROOT/"registration.json"),calibration=digest(CAL/"summary.json"),costs=costs,
                      implementation={str(f):digest(f) for f in files},orders={c["id"]:c["order"] for c in cfg["cases"]})
    LIVE.mkdir(exist_ok=True)
    assert_snapshot(LIVE/"registration.json",registration)
    return cfg,runtime,costs


def worker(case,method,ep,runtime,cost):
    from uav_bridge.chatgpt_auth import CredentialStore
    ep.mkdir(parents=True,exist_ok=False)
    trace=Timeline(ep/"timeline.jsonl")
    order=mission(case,cost)
    client=ModelClient(ep/"model_calls.jsonl",timeout=120,
        credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),
        request_budget=RequestBudget(ep/"sends.jsonl",len(order.queries)))
    start=monotonic();begin=None;finished=None;controller=room=None
    result=dict(case=case["id"],method=method,registration_sha256=digest(LIVE/"registration.json"))
    try:
        house,world=_house(Path(runtime["archive"]),case["house"])
        result["house_sha256"]=world
        controller=open_controller(house,Path(runtime["build_dir"]),"opengl",os.environ["DISPLAY"],segmentation=True)
        room=Room(controller,ep,trace,start)
        begin=monotonic()
        result["report"]=run_mission(room,case,method,cost,client,ep,begin)
        if monotonic()-begin>240:raise TimeoutError("Late report")
    except Exception as ex:
        code=getattr(ex,"code",type(ex).__name__)
        result["failure"]=dict(code=code,detail=str(ex)[:180])
        if code in EXTERNAL_RESPONSE_CODES or getattr(ex,"status",None) in (401,403,429):
            result["endpoint_block"]=str(code)
    finally:
        finished=monotonic()
        try:
            if room:
                for i,q in enumerate(order.queries):
                    name=order.names[q.sites[0]]
                    if name in room.events:
                        save_segmentation(ep,"q%d"%i,room.events[name],q.label,MIN_OBJECT_PIXELS)
        finally:
            if controller:
                try:controller.stop()
                except Exception as ex:result["teardown_error"]=type(ex).__name__
            result.update(startup_s=(begin or finished)-start,start_clock=begin,finish_clock=finished,
                task_s=finished-begin if begin is not None else None,post_task_archive_and_teardown_s=monotonic()-finished,
                actual_sends=client.calls,peak_worker_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save(ep/"result.json",result)


def signature(ep):
    files=[p for p in ep.iterdir() if p.is_file() and p.suffix in (".json",".jpg",".jsonl")]
    files+=list((ep/"scoring").glob("*"))
    return {str(p.relative_to(ep)):digest(p) for p in files}


def tasks(cfg):
    return [(c,m,LIVE/c["id"]/m) for c in cfg["cases"] for m in c["order"]]


def validate_calls(case,ep,cost):
    graph=EventModel(mission(case,cost));m=graph.m
    vals=[UNKNOWN]*len(m.queries);claims={}
    events=read_trace_rows(ep/"timeline.jsonl")[0]
    cameras={e["step"]:e for e in events if e["event"]=="camera_observation"}
    req=ep/"model_calls.jsonl"
    calls=read_trace_rows(req)[0] if req.exists() else []
    ledger=read_trace_rows(ep/"sends.jsonl")[0]
    seen=[];asked=set();timings=[];bindings=[];last=0.;legal=True
    for c in calls:
        start=c.get("dispatched_monotonic_seconds")
        if start is None:continue
        inp=c["input"];src=inp["image"];h=inp["sha256"]
        if not (start>=last and src in cameras and cameras[src]["monotonic_seconds"]<=start and
                cameras[src]["image_sha256"]==h==digest(ep/(src+".jpg")) and c["image_sha256"]==[h]
                and c["model"]=="gpt-6.1-terra" and c["reasoning_effort"]=="max"
                and c["instructions_sha256"]==sha256(PROMPT.encode()).hexdigest()
                and c["logical_job_attempt"]==len(seen)+1 and ledger[len(seen)]["image_sha256"]==[h]):
            raise ValueError("Failed source, request chronology or transport binding")
        actual=c.get("output");subset=inp["questions"];guards=graph.guards(tuple(vals))
        keys=[q["id"] for q in subset]
        valid=isinstance(actual,dict) and set(actual)==set(keys) and all(v in ("positive","negative","uncertain") for v in actual.values())
        for q in subset:
            i=int(q["id"][1:]);atom=m.queries[i]
            if not (q["id"]=="q%d"%i and q["category"]==atom.label and q["image"]==src==m.names[atom.sites[0]]
                    and guards[i]==ALWAYS and q["id"] not in asked):
                legal=False
            asked.add(q["id"])
        if len(set(keys))!=len(keys) or len(subset) not in (1,2):legal=False
        for q in subset:
            if valid:
                key=q["id"];answer=actual[key]
                if answer in ("positive","negative"):
                    vals[int(key[1:])]=int(answer=="positive")
                    claims[key]=dict(value=answer,source=src,sha256=h)
        last=c["return_monotonic_seconds"];seen.append(c["logical_job_attempt"])
        timings.append((start,last))
        canonical=sha256(json.dumps(inp,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        bindings.append(dict(input_sha256=canonical,questions=keys,service_s=last-start,
                             output=actual if valid else None,clock=[start,last]))
    if len(calls)!=len(ledger) or len(seen)!=len(ledger):raise ValueError("Unmetered call")
    starts={};physical=[]
    for e in events:
        name=e["event"]
        for k in ("action","observation"):
            if name==k+"_started":starts[(k,e["step"])]=e["monotonic_seconds"]
            if name==k+"_finished":
                physical.append((starts.pop((k,e["step"])),e["monotonic_seconds"]))
    intersection=[(max(a,c),min(b,d)) for a,b in physical for c,d in timings if min(b,d)>max(a,c)]
    return claims,legal,bindings,length(timings),length(intersection),sum(e.get("computation_s",0) for e in events)


def grade_case(case,method,ep,cost):
    d=read(ep/"result.json")
    m=mission(case,cost)
    graph=EventModel(m)
    truth=[UNKNOWN]*len(m.queries);expected={};missing=[]
    for i,q in enumerate(m.queries):
        if graph.guards(tuple(truth))[i]!=ALWAYS:continue
        key="q%d"%i;site=m.names[q.sites[0]]
        if not (ep/"scoring"/(key+".json")).exists():
            missing.append(key);continue
        value=_grade_segmentation(ep/"scoring",key)
        truth[i]=int(value=="positive")
        expected[key]=dict(value=value,source=site,sha256=digest(ep/(site+".jpg")))
    actual,perm,bindings,service,over,cpu=validate_calls(case,ep,cost)
    rep=d.get("report") or {}
    correct=bool(not missing and not d.get("failure") and not d.get("teardown_error") and perm
                 and rep.get("fields")==expected and rep.get("final_pose_checked") is True
                 and all(actual.get(k)==v for k,v in expected.items()) and d["task_s"]<=240)
    return dict(case=case["id"],method=method,strict=correct,task_s=d["task_s"],
        restricted_s=d["task_s"] if correct else 240.,startup_s=d["startup_s"],actual_sends=d["actual_sends"],
        source_and_permissions_verified=perm,missing_scores=missing,expected_fields=expected,report=rep,
        failure=d.get("failure"),return_without_submission=not bool(rep),bindings=bindings,model_service_s=service,
        real_motion_or_shutter_overlap_s=over,serialized_fixed_outputs_preserving_idle_s=d["task_s"]+over if d["task_s"] is not None else None,
        planner_cpu_wall_s=cpu,result_sha256=digest(ep/"result.json"))


def grade(cfg,costs):
    entries=tasks(cfg)
    sealed={c["id"]+"/"+m:signature(ep) for c,m,ep in entries}
    assert_snapshot(LIVE/"sealed_outcomes.json",sealed)
    rows=[grade_case(c,m,ep,costs[c["id"]]) for c,m,ep in entries]
    repeated=defaultdict(list)
    for r in rows:
        for b in r["bindings"]:
            repeated[(r["case"],b["input_sha256"])].append(dict(method=r["method"],service_s=b["service_s"],output=b["output"]))
    dup=[dict(case=k[0],input_sha256=k[1],observations=v) for k,v in repeated.items() if len(v)>1]
    result=dict(role="matched complete development simulator tasks",cases=rows,identical_request_bindings=dup,
        registered=len(entries),strict=sum(v["strict"] for v in rows),actual_sends=sum(v["actual_sends"] for v in rows),
        source=digest(LIVE/"sealed_outcomes.json"))
    save(LIVE/"summary.json",result)
    return result


def controller(cfg,runtime,costs):
    with (ROOT.parent/"online.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case,method,ep in tasks(cfg):
            rc=ep.with_suffix(".receipt.json")
            key=dict(registration_sha256=digest(LIVE/"registration.json"),case=case["id"],method=method)
            if rc.exists():
                old=read(rc)
                if any(old.get(k)!=v for k,v in key.items()) or not (ep/"result.json").is_file():
                    raise SystemExit("Inspect version or live original worker at "+str(rc))
                sig=signature(ep)
                if old.get("files",sig)!=sig:raise ValueError("Previous attempt has changed")
                outcome=read(ep/"result.json")
                if old["state"]!="settled":save(rc,{**old,"files":sig,"state":"settled","end":time(),"endpoint_block":outcome.get("endpoint_block")})
                if outcome.get("endpoint_block"):raise SystemExit("Original service restriction "+str(rc))
                continue
            if ep.exists():raise SystemExit("Unreserved existing attempt "+str(ep))
            if unresolved_service_limits(ROOT.parent):raise SystemExit("Authorized service has an unresolved limit")
            ep.parent.mkdir(parents=True,exist_ok=True)
            meta={**key,"start":time(),"controller":os.getpid()}
            save(rc,{**meta,"state":"starting"})
            proc=launch_local(runtime,case,SOURCE,["worker","--case",case["id"],"--method",method],ep.with_suffix(".log"))
            save(rc,{**meta,"state":"running","worker":proc.pid})
            proc.wait()
            if proc.returncode or not (ep/"result.json").exists():
                raise SystemExit("Preserve incomplete worker at "+str(ep))
            result=read(ep/"result.json")
            save(rc,{**meta,"state":"settled","end":time(),"files":signature(ep),"actual_sends":result["actual_sends"],
                     "endpoint_block":result.get("endpoint_block")})
            print(case["id"],method,result["actual_sends"],result["task_s"],result.get("failure"),flush=True)
            if result.get("endpoint_block"):raise SystemExit("Service restriction "+str(rc))
        return grade(cfg,costs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("run","grade","worker","status"))
    p.add_argument("--case");p.add_argument("--method");a=p.parse_args()
    cfg,runtime,costs=protocol()
    if a.mode=="worker":
        c,m,ep=next((c,m,ep) for c,m,ep in tasks(cfg) if (c["id"],m)==(a.case,a.method))
        rc=ep.with_suffix(".receipt.json")
        if ep.exists() or not rc.is_file() or read(rc)["state"] not in ("starting","running"):
            raise SystemExit("Worker needs an unconsumed coordinator reservation")
        worker(c,m,ep,runtime,costs[c["id"]])
    elif a.mode=="status":
        for c,m,ep in tasks(cfg):
            rc=ep.with_suffix(".receipt.json");print(c["id"],m,read(rc)["state"] if rc.exists() else "unsent")
    else:
        result=controller(cfg,runtime,costs) if a.mode=="run" else grade(cfg,costs)
        print({k:v for k,v in result.items() if k not in ("cases","identical_request_bindings")})


if __name__=="__main__":
    main()
