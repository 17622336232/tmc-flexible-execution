"""Test hard wall-clock scheduling budgets with a common, fully feasible incumbent at interruption."""
import argparse
from math import inf
from pathlib import Path
import signal
from statistics import median
from time import perf_counter

from experiments.flexible_execution.revision_audit import digest,read,save,write_table
from research.flexible_scheduling.event_workflow import EventModel
from research.flexible_scheduling.event_lookahead import LookaheadPlanner
from research.flexible_scheduling.event_policy import order
from scripts.continuing_concurrent_workflow import ROOT as PRIOR,mission
from scripts.continuing_crossworld_views import assert_snapshot

CFG=Path("config/event_compute_dev02.json");ROOT=PRIOR.parent/"event-compute-dev02"
MODES=("strong-prefetch","generic","joint")


class Deadline(Exception):
    pass


def limited(m,mode,ms):
    record={"value":None,"action":None,"source":"default","proved":False}
    counts={"nodes":0,"cache_hits":0,"root_candidates":0}
    def timeout(signum,frame):raise Deadline()
    def improve(value,action,source):
        incumbent=record["value"]
        if incumbent is None or value<incumbent-1e-8 or (abs(value-incumbent)<=1e-8 and order(action)<order(record["action"])):
            record.update(value=value,action=action,source=source)
    previous=signal.signal(signal.SIGALRM,timeout)
    start=perf_counter();stop=None
    try:
        signal.setitimer(signal.ITIMER_REAL,ms/1000.)
        e=EventModel(m);p=LookaheadPlanner(e,quotient=mode!="generic")
        initial=e.initial()
        for style in ("collect","prepare","cost"):
            val=p.roll(initial,style)
            improve(val,p.greedy(initial,style),"rollout")
        options=[]
        for a in p.options(initial):
            dt,branch=p.trans(initial,a)
            lb=dt+sum(prob*p.cost_lb(y) for prob,y in branch)
            options.append((lb,order(a),a,dt,branch))
        options.sort(key=lambda it:(it[0],it[1]))
        for _,_,a,dt,branch in options:
            value=dt+sum(prob*p.one_step(y)[0] for prob,y in branch)
            counts["root_candidates"]+=1
            improve(value,a,"two-dispatch")
        if mode!="strong-prefetch":
            cache={}
            def exact(s):
                if e.complete(s):return 0.
                key=p.key(s)
                if key in cache:
                    counts["cache_hits"]+=1
                    return cache[key]
                counts["nodes"]+=1
                best,_=p.incumbent(s)
                alternatives=[]
                for command in p.options(s):
                    duration,states=p.trans(s,command)
                    low=duration+sum(pr*p.cost_lb(y) for pr,y in states)
                    alternatives.append((low,order(command),duration,states))
                alternatives.sort(key=lambda v:(v[0],v[1]))
                for low,_,duration,states in alternatives:
                    if low>=best-1e-8:break
                    best=min(best,duration+sum(pr*exact(y) for pr,y in states))
                cache[key]=best
                return best
            for lb,_,a,dt,branch in options:
                if lb>=record["value"]-1e-8:break
                value=dt+sum(pr*exact(y) for pr,y in branch)
                improve(value,a,"completed-root-continuation")
            record["proved"]=True
    except Deadline:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        stop=perf_counter()
        signal.signal(signal.SIGALRM,previous)
    actual_ms=(stop-start)*1000
    if record["value"] is None:
        # Evaluate the declared common default after the selection deadline, without revising it.
        e=EventModel(m);p=LookaheadPlanner(e)
        s=e.initial()
        record.update(value=p.roll(s,"collect"),action=p.greedy(s,"collect"))
    action=record["action"]
    return dict(expected_s=record["value"]/10,first=dict(motion=action.motion,reads=action.reads),
                certified=record["proved"],last_complete_incumbent=record["source"],actual_ms=actual_ms,
                overrun_ms=max(0.,actual_ms-ms),**counts)


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("run","summary"));p.add_argument("--limit",type=int)
    a=p.parse_args();cfg=read(CFG)
    older=read(PRIOR/"registration.json")
    ROOT.mkdir(exist_ok=True)
    reg=dict(parent=digest(PRIOR/"registration.json"),study=digest(CFG),driver=digest(Path(__file__)),
             stronger=digest(Path("research/flexible_scheduling/event_lookahead.py")))
    assert_snapshot(ROOT/"registration.json",reg)
    folder=ROOT/"groups";folder.mkdir(exist_ok=True);added=0
    for name in cfg["cases"]:
        for n in (cfg["scale_roots"] if name=="fan-in-scale" else [2]):
            for domain in cfg["service_tables"]:
                for k in cfg["slots"]:
                    m=mission(name,domain,k,older["clocks"],n)
                    for ms in cfg["search_budgets_ms"]:
                        for rep in range(cfg["repetitions"]):
                            f=folder/f"{name}_{n}_{domain}_{k}_{ms}_{rep}.json"
                            if a.mode!="run" or f.exists() or (a.limit and added>=a.limit):continue
                            rows=[]
                            offset=(rep+(k==2)+(domain=="campus"))%3
                            for mode in MODES[offset:]+MODES[:offset]:
                                row=limited(m,mode,ms)
                                rows.append(dict(case=name,roots=n,domain=domain,slots=k,budget_ms=ms,repeat=rep,mode=mode,**row))
                            save(f,rows);added+=1
    results=[row for f in sorted(folder.glob("*.json")) for row in read(f)]
    if results:write_table(ROOT/"timed_planners.csv",results)
    by={}
    for row in results:
        key=(row["case"],row["roots"],row["domain"],row["slots"],row["budget_ms"])
        by.setdefault(key,{}).setdefault(row["mode"],[]).append(row)
    paired=[]
    for key,md in by.items():
        if any(len(md.get(mode,[]))!=cfg["repetitions"] for mode in MODES):continue
        item=dict(zip(("case","roots","domain","slots","budget_ms"),key))
        for mode in MODES:
            rows=md[mode]
            item[mode+"_median_s"]=median(r["expected_s"] for r in rows)
            item[mode+"_certificates"]=sum(r["certified"] for r in rows)
            item[mode+"_worst_overrun_ms"]=max(r["overrun_ms"] for r in rows)
        paired.append(item)
    if paired:write_table(ROOT/"fixed_wall_pairs.csv",paired)
    summary=dict(trials=len(results),matched_cells=len(paired),cost_improved_against_generic=sum(r["joint_median_s"]<r["generic_median_s"]-1e-7 for r in paired),
        cost_tied_with_generic=sum(abs(r["joint_median_s"]-r["generic_median_s"])<=1e-7 for r in paired),
        cost_worse_than_generic=sum(r["joint_median_s"]>r["generic_median_s"]+1e-7 for r in paired),
        improved_against_strong=sum(r["joint_median_s"]<r["strong-prefetch_median_s"]-1e-7 for r in paired),
        worse_than_strong=sum(r["joint_median_s"]>r["strong-prefetch_median_s"]+1e-7 for r in paired),
        more_certificates_than_generic=sum(r["joint_certificates"]>r["generic_certificates"] for r in paired),
        fewer_certificates_than_generic=sum(r["joint_certificates"]<r["generic_certificates"] for r in paired),
        max_observed_overrun_ms=max((r["overrun_ms"] for r in results),default=0),
        median_observed_overrun_ms=median(r["overrun_ms"] for r in results) if results else None,
        zero_request=True)
    save(ROOT/"summary.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    main()
