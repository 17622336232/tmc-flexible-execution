"""Rerun the matched timing study with actual prices and a preselected common fallback."""
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.flexible_execution.revision_audit import digest,read,save
from research.flexible_scheduling.anytime_replan import limited_replan
from research.flexible_scheduling.cross_source_workflow import CrossSourceWorkflow,PermissionCache
from research.flexible_scheduling.priced_cross_source import PricedCrossSourcePlanner
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_permission_cache import DEST,task,ticks

ROOT = DEST/"budget-v2-guarded"
METHODS = ("generic","residual","permissions","future-values","strong")


def trial(reg,n,shape,budget,repetition):
    profiles = reg["profiles"]
    base = task(n,shape,tuple(profiles[0][1]),profiles[0][2])
    output = []
    order = METHODS[repetition:]+METHODS[:repetition]
    for mode in order:
        cache = PermissionCache(base,True,residual=mode in ("residual","permissions"))
        for name,service,travel,cross in profiles:
            began = perf_counter()
            wall_limit = began+budget/1000
            m = replace(base,service=tuple(service),travel=((0,ticks(travel)),(ticks(travel),0)))
            graph = CrossSourceWorkflow(m,ticks(cross),permission_cache=cache)
            planner = PricedCrossSourcePlanner(graph,quotient=mode in ("residual","future-values"))
            initial = graph.initial()
            # All methods determine the same feasible collect action before evaluating search candidates.
            fallback = planner.greedy(initial,"collect")
            first_recorded_ms = (perf_counter()-began)*1000
            result,source,proved,_ = limited_replan(planner,initial,budget,prefetch_only=mode=="strong",
                                                  absolute_deadline=wall_limit-.0002)
            returned_ms = (perf_counter()-began)*1000
            # A callback or scheduler can still return late. Score the preselected
            # common action in that case and report the miss, even if an improved incumbent exists.
            late = returned_ms>budget and first_recorded_ms<=budget
            timely = result is not None and not late
            if not timely:
                clean = PricedCrossSourcePlanner(CrossSourceWorkflow(m,ticks(cross)))
                result = clean.roll(initial,"collect"),fallback
                source = "deadline-fallback"
            output.append(dict(stage=name,method=mode,predicted_s=result[0]/100,
                source=source,proved=bool(proved and timely),first_action_ms=first_recorded_ms,
                returned_ms=returned_ms,timing_miss=late,missing_timely_action=first_recorded_ms>budget,
                cache_entries=len(cache.entries),interrupted_overrun_ms=max(0,returned_ms-budget)))
    return output


def main():
    parent = DEST/"registration.json"
    reg = read(parent)
    ROOT.mkdir(exist_ok=True)
    core = Path("research/flexible_scheduling")
    dependencies = [Path(__file__),Path("scripts/continuing_permission_cache.py"),
                    *(core/(n+".py") for n in ("anytime_replan","cross_source_workflow","priced_cross_source",
                       "event_workflow","event_policy","event_lookahead"))]
    settings = dict(parent=digest(parent),methods=list(METHODS),budgets=[10,30,80],repeats=4,
            dependencies={str(p):digest(p) for p in dependencies},
            measure="Starts before model and planner construction; count common first action and actual return; "
                    "reserve 0.20ms for returning; late advanced selections receive their preselected baseline value. "
                    "Untimed fallback valuation only.")
    assert_snapshot(ROOT/"registration.json",settings)
    cells = []
    for n,shape,budget in product(reg["roots"],reg["dependence"],settings["budgets"]):
        raw = []
        for rep in range(settings["repeats"]):
            path = ROOT/f"{n}-{shape}-{budget}-{rep}.json"
            if not path.exists():
                save(path,trial(reg,n,shape,budget,rep))
            raw.extend(read(path))
        for stage,*_ in reg["profiles"]:
            methods = {}
            for mode in METHODS:
                rows = [v for v in raw if v["method"]==mode and v["stage"]==stage]
                methods[mode] = dict(median_s=median(r["predicted_s"] for r in rows),
                    fallback=sum(r["source"]=="deadline-fallback" for r in rows),timing_miss=sum(r["timing_miss"] for r in rows),
                    missing_action=sum(r["missing_timely_action"] for r in rows),proofs=sum(r["proved"] for r in rows),
                    median_legal_cache_entries=median(r["cache_entries"] for r in rows),
                    max_first_action_ms=max(r["first_action_ms"] for r in rows),max_overrun_ms=max(r["interrupted_overrun_ms"] for r in rows))
            cells.append(dict(roots=n,structure=shape,budget_ms=budget,stage=stage,methods=methods))
    results = {}
    for mode in METHODS[1:]:
        gaps = [c["methods"]["generic"]["median_s"]-c["methods"][mode]["median_s"] for c in cells]
        results[mode] = dict(better=sum(g>1e-6 for g in gaps),tied=sum(abs(g)<=1e-6 for g in gaps),
                             worse=sum(g<-1e-6 for g in gaps),maximum_gain_s=max(gaps))
    metrics = [v for c in cells for v in c["methods"].values()]
    result = dict(cells=len(cells),trials=len(cells)*len(METHODS)*settings["repeats"],comparison_to_generic=results,
                  total_timing_miss=sum(m["timing_miss"] for m in metrics),missed_first_action=sum(m["missing_action"] for m in metrics),
                  max_first_action_ms=max(m["max_first_action_ms"] for m in metrics),max_overrun_ms=max(m["max_overrun_ms"] for m in metrics),
                  new_model_requests=0,details=cells)
    save(ROOT/"summary.json",result)
    print({k:v for k,v in result.items() if k!="details"})


if __name__=="__main__":
    main()
