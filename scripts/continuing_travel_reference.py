"""Enumerate all resource-feasible event orders for two mandatory stations, using measured costs."""
from itertools import combinations
from pathlib import Path
from statistics import median

from experiments.flexible_execution.revision_audit import read,save
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_itinerary_sources import ROOT as SOURCE_ROOT
from scripts.continuing_travel_optics import ROOT


def reference(times, active, grant):
    # Condition results become visible only at completion of read-a, even on negative paths.
    jobs={"to-a":("robot",set()),"photo-a":("robot",{"to-a"}),"read-a":("model",{"photo-a"}),
          "to-b":("robot",{"photo-a"})}
    must_photo=active or grant=="mandatory-archive"
    if must_photo:
        jobs["photo-b"]=("robot",{"to-b"} if grant=="mandatory-archive" else {"to-b","read-a"})
    if active:
        jobs["read-b"]=("model",{"read-a","photo-b"})
    jobs["home"]=("robot",{"photo-b"} if must_photo else {"to-b","read-a"})
    schedules=0
    best=float("inf")
    def visit(now,done,pending):
        nonlocal schedules,best
        if len(done)==len(jobs):
            schedules+=1
            best=min(best,now)
            return
        busy={jobs[j][0] for j in pending}
        ready=[j for j,(res,pre) in jobs.items() if pre<=done and j not in done and j not in pending and res not in busy]
        for size in range(len(ready)+1):
            for selected in combinations(ready,size):
                if len({jobs[j][0] for j in selected})!=size:
                    continue
                underway={**pending,**{j:now+times[j] for j in selected}}
                if not underway:
                    continue
                nxt=min(underway.values())
                finished={j for j,end in underway.items() if end==nxt}
                visit(nxt,done|finished,{j:v for j,v in underway.items() if j not in finished})
    visit(0.,set(),{})
    return best,schedules


def analytic(t,active,grant):
    start=t["to-a"]+t["photo-a"]
    a=start+t["read-a"]
    b=start+t["to-b"]
    if grant=="mandatory-archive":
        image=b+t["photo-b"]
        return max(image+t["home"],max(a,image)+t["read-b"] if active else a)
    if active:
        photo=max(a,b)+t["photo-b"]
        return max(photo+t["home"],photo+t["read-b"])
    return max(a,b)+t["home"]


def main():
    reg=read(SOURCE_ROOT/"registration.json")
    cal=read(ROOT/"calibration/summary.json")
    rows=[]
    for r in cal["cases"]:
        # Complete real physical legs are required for an estimate. Never fill a missed home segment.
        if not r["route_ok"]:
            continue
        item=next(c for c in reg["cases"] if c["id"]==r["case"])
        ep=ROOT/"calibration"/r["case"]/"calibration"
        events=read_trace_rows(ep/"mission_events.jsonl")[0]
        def clock(kind,key):
            return next(v["clock"] for v in events if v["event"]==kind and v.get("stop")==key)
        t={"to-"+k:clock("travel_end",k)-clock("travel_start",k) for k in ("a","b")}
        t.update({"photo-"+k:clock("camera",k)-clock("travel_end",k) for k in ("a","b")})
        t["home"]=clock("travel_end","home")-clock("travel_start","home")
        # Old whole-task calls are a proxy for an unmeasured new single-image service.
        calls=read_trace_rows(Path(item["source_episode"])/"private/robot/model_calls.jsonl")[0]
        dur=[c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"] for c in calls if c.get("output")]
        t["read-a"]=t["read-b"]=median(dur)
        for grant in ("mandatory-archive","after-positive-only"):
            for positive in (False,True):
                exact,orders=reference(t,positive,grant)
                scheduled=analytic(t,positive,grant)
                if abs(exact-scheduled)>1e-6:
                    raise ValueError("Independent exhaustive schedule disagrees")
                serial=t["to-a"]+t["photo-a"]+t["read-a"]+t["to-b"]+t["home"]
                if positive or grant=="mandatory-archive":serial+=t["photo-b"]
                if positive:serial+=t["read-b"]
                rows.append(dict(case=r["case"],grant=grant,demand=positive,exact_s=exact,compiled_s=scheduled,
                                 eager_s=scheduled,serial_s=serial,legal_schedules=orders,component_times=t))
    result=dict(rows=rows,model_requests=0,note="Descriptive resource schedule under anchored time assumptions. All three physical legs were actually measured. Both predicate outcomes are enumerated; historical normal-request service is only a proxy. These are not measured model-driven mission times.")
    save(ROOT/"schedule_reference.json",result)
    print([(r["case"],r["grant"],r["demand"],round(r["exact_s"],2),r["legal_schedules"]) for r in rows])


if __name__=="__main__":
    main()
