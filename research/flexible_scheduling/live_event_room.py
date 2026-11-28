"""Schedule legal camera work and genuine single-frame semantic requests at real completion events."""
from concurrent.futures import FIRST_COMPLETED,ThreadPoolExecutor,wait
from math import ceil
from pathlib import Path
from statistics import median
from time import monotonic

from experiments.flexible_execution.public_task import Room,TURNS,_same_pose
from experiments.flexible_execution.revision_audit import read,save
from experiments.flexible_execution.timeline import read_trace_rows
from .event_workflow import ALWAYS,UNKNOWN,NO_MOTION,State,EventModel,Mission,indices
from .event_lookahead import LookaheadPlanner
from .scoped_room_query import request


def public_queries(case):
    from .event_workflow import Query
    yes=lambda i:(((i,True),),)
    if case["structure"]=="paired-screen":
        locations=(0,0,1,1)
        guards=(ALWAYS,ALWAYS,yes(0),yes(1))
    elif case["structure"]=="relay-inspection":
        locations=(0,1,1,0,1)
        guards=(ALWAYS,ALWAYS,yes(0),yes(1),(((2,True),),((3,True),)))
    else:
        raise ValueError("Unknown public work order")
    if len(guards)!=len(case["categories"]):
        raise ValueError("Missing declared class")
    return tuple(Query(cat,(s,),g) for cat,s,g in zip(case["categories"],locations,guards))


def mission(case,cost):
    qs=public_queries(case);tick=lambda s:max(1,round(10*s))
    return Mission(("front","side"),qs,((0,tick(cost["outward"])),(tick(cost["return"]),0)),
                   (tick(cost["photo"]),tick(cost["photo"])),(True,True),0,1,
                   (0,tick(cost["single"]),tick(cost["pair"])),0,0 if case["structure"]=="paired-screen" else 1,1)


def time_reference(case,calibration,old_smokes):
    sample="conditional-room" if case["direction"]=="right" else "archive-and-screen"
    path=Path(old_smokes)/sample/"camera-check/timeline.jsonl"
    rows=read_trace_rows(path)[0];starts={};turns={};photos=[]
    for r in rows:
        k,step,t=r["event"],r["step"],r["monotonic_seconds"]
        if k in ("action_started","observation_started"):starts[(k.split("_")[0],step)]=t
        elif k=="action_finished":turns[step]=t-starts[("action",step)]
        elif k=="observation_finished":photos.append(t-starts[("observation",step)])
    out,back,_=TURNS[case["direction"]]
    return dict(outward=turns[out],return_s=turns[back],photo=median(photos),single=calibration["single_s"],pair=calibration["pair_s"],
                **{"return":turns[back]})


def run_mission(room,case,method,cost,client,out,started):
    model=EventModel(mission(case,cost));m=model.m
    planner=LookaheadPlanner(model,ablation="serial" if method=="serial" else None)
    trace=room.trace
    frames={0:room.take("front",room.controller.last_event)}
    site=0
    values=[UNKNOWN]*len(m.queries);claims={}
    physical=None;semantic=None
    deadline=started+240
    client.deadline_monotonic=deadline
    turn,restore,degrees=TURNS[case["direction"]]
    def act(old,target,capture):
        event=room.controller.last_event
        if old!=target:
            event=room.turn(turn if target==1 else restore,degrees if target==1 else 0)
        return room.take(m.names[target],event) if capture else None
    def rest(future):
        return max(1,ceil(future["nominal"]-10*(monotonic()-future["start"])))
    def journal():
        save(out/"partial.json",dict(claims=claims,raw={m.names[s]:f.digest for s,f in frames.items()},site=m.names[site]))
    with ThreadPoolExecutor(max_workers=1) as motor, ThreadPoolExecutor(max_workers=1) as service:
        while True:
            if monotonic()>=deadline:raise TimeoutError("Registered task clock exhausted")
            if physical and physical["future"].done():
                frame=physical["future"].result()
                site=physical["dest"]
                if frame:
                    if site in frames:raise ValueError("Duplicate physical source acquisition")
                    frames[site]=frame
                physical=None
                journal()
            if semantic and semantic["future"].done():
                result=semantic["future"].result()
                src=semantic["source"]
                for i in indices(semantic["mask"]):
                    key="q%d"%i
                    if result[key]=="uncertain":
                        trace.event("uncertain_field",key,source=m.names[src])
                        raise ValueError("Unresolved required visual field "+key)
                    values[i]=int(result[key]=="positive")
                    claims[key]=dict(value=result[key],source=m.names[src],sha256=frames[src].digest)
                semantic=None
                journal()
            motion=(physical["dest"],physical["capture"],rest(physical)) if physical else NO_MOTION
            jobs=((semantic["source"],semantic["mask"],rest(semantic)),) if semantic else ()
            s=State(tuple(values),sum(1<<i for i in frames),site,motion,jobs)
            if model.complete(s):
                required={"q%d"%i for i,g in enumerate(model.guards(s.values)) if g==ALWAYS}
                reported={k:claims[k] for k in sorted(required)}
                final=room.controller.last_event.metadata["agent"]
                expected=degrees if m.finish==1 else 0
                return dict(fields=reported,ended_at=m.names[site],final_pose_checked=_same_pose(room.initial_pose,final,expected),
                            source_digests={m.names[i]:f.digest for i,f in frames.items()})
            before=monotonic()
            if method=="strong-prefetch":
                expected,a=planner.strong(s)
                proof=False
            else:
                solved=planner.solve(s,budget=256)
                a=solved["action"];expected=solved["value_ticks"];proof=solved["proved"]
            trace.event("event_plan",len(claims),computation_s=monotonic()-before,observed=list(values),
                        raw=s.raw,pose=site,model_pending=bool(semantic),motion_pending=bool(physical),
                        motion=a.motion,reads=a.reads,nominal_remaining_s=expected/10,proved=proof)
            if any(job and job["future"].done() for job in (physical,semantic)):
                continue
            if a not in planner.options(s):
                raise ValueError("Dispatch is outside graph permissions")
            if a.reads:
                if semantic or len(a.reads)!=1:raise ValueError("Service capacity exceeded")
                src,mask=a.reads[0]
                frame=frames[src]
                qs=[("q%d"%i,m.queries[i].label) for i in indices(mask)]
                begin=monotonic()
                semantic=dict(source=src,mask=mask,start=begin,nominal=m.service[len(qs)],
                              future=service.submit(request,client,frame.id,frame.payload,qs))
            if a.motion:
                if physical:raise ValueError("Robot cannot execute two moves")
                dest,capture=a.motion
                begin=monotonic()
                physical=dict(dest=dest,capture=capture,start=begin,
                              nominal=m.travel[site][dest]+(m.capture[dest] if capture else 0),
                              future=motor.submit(act,site,dest,capture))
            tasks=[j["future"] for j in (physical,semantic) if j]
            if not tasks:raise RuntimeError("Incomplete task has no pending work")
            wait(tasks,timeout=max(0,deadline-monotonic()),return_when=FIRST_COMPLETED)
