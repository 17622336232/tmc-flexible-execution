"""Assess repeated planning at the same wall time, with both controls retaining legal options."""
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.flexible_execution.revision_audit import digest, read, save
from research.flexible_scheduling.anytime_replan import limited_replan
from research.flexible_scheduling.cross_source_workflow import CrossSourcePlanner, CrossSourceWorkflow, PermissionCache
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_permission_cache import DEST, task, ticks

ROOT = DEST/"wall"


def trial(reg, roots, structure, ms, rep):
    modes = ("generic","strong","residual")
    offset = rep%len(modes)
    profiles = reg["profiles"]
    shared = task(roots,structure,tuple(profiles[0][1]),profiles[0][2])
    rows = []
    for method in modes[offset:]+modes[:offset]:
        residual = method=="residual"
        cache = PermissionCache(shared,True,residual=residual)
        for name,cost,travel,cross in profiles:
            began = perf_counter()
            m = replace(shared,travel=((0,ticks(travel)),(ticks(travel),0)),service=tuple(cost))
            e = CrossSourceWorkflow(m,ticks(cross),permission_cache=cache)
            planner = CrossSourcePlanner(e,quotient=residual)
            setup_ms = (perf_counter()-began)*1000
            pending = max(.001,ms-setup_ms)
            initial = e.initial()
            choice,source,certificate,spent = limited_replan(planner,initial,pending,prefetch_only=method=="strong")
            if choice is None:
                # Identical fixed fallback. Its offline valuation must not populate the warm cache.
                plain = CrossSourceWorkflow(m,ticks(cross))
                common = CrossSourcePlanner(plain)
                start = perf_counter()
                action = common.greedy(initial,"collect")
                setup_ms += (perf_counter()-start)*1000
                choice = common.roll(initial,"collect"),action
            value,action = choice
            rows.append(dict(stage=name,mode=method,source=source,cost_s=value/100,proved=certificate,
                             actual_ms=spent+setup_ms,first=dict(motion=action.motion,reads=action.reads),
                             snapshots=len(cache.entries),cache_hits=cache.hits))
    return rows


def main():
    reg = read(DEST/"registration.json")
    ROOT.mkdir(exist_ok=True)
    settings = dict(parent=digest(DEST/"registration.json"),repeats=3,wall_ms=[10,30,80],
        planner=digest(Path("research/flexible_scheduling/anytime_replan.py")),driver=digest(Path(__file__)),
        common_default="collect; only valuation after deadline, first fixed action selection reported in actual_ms")
    assert_snapshot(ROOT/"registration.json",settings)
    cells = []
    for n,structure,ms in product(reg["roots"],reg["dependence"],settings["wall_ms"]):
        raw = []
        for rep in range(settings["repeats"]):
            receipts = ROOT/(f"{n}-{structure}-{ms}-{rep}.json")
            if not receipts.is_file():
                save(receipts,trial(reg,n,structure,ms,rep))
            raw.extend(read(receipts))
        for step in reg["profiles"]:
            groups = {mode:[r for r in raw if r["stage"]==step[0] and r["mode"]==mode]
                      for mode in ("generic","strong","residual")}
            cells.append(dict(roots=n,structure=structure,budget_ms=ms,stage=step[0],
              methods={m:dict(median_feasible_s=median(v["cost_s"] for v in rows),
                              full_proofs=sum(v["proved"] for v in rows),
                              fallbacks=sum(v["source"]=="common-collect" for v in rows),
                              maximum_overrun_ms=max(0,max(v["actual_ms"] for v in rows)-ms))
                       for m,rows in groups.items()}))
    def gap(cell, comparator):
        m=cell["methods"]
        return m[comparator]["median_feasible_s"]-m["residual"]["median_feasible_s"]
    result = dict(cells=len(cells),trials=len(cells)*settings["repeats"]*3,
                  residual_better=sum(gap(v,"generic")>1e-6 for v in cells),
                  residual_tied=sum(abs(gap(v,"generic"))<=1e-6 for v in cells),
                  residual_worse=sum(gap(v,"generic")<-1e-6 for v in cells),
                  better_than_strong=sum(gap(v,"strong")>1e-6 for v in cells),
                  worse_than_strong=sum(gap(v,"strong")<-1e-6 for v in cells),
                  worst_overrun_ms=max(r["maximum_overrun_ms"] for v in cells for r in v["methods"].values()),
                  no_model_requests=True,comparisons=cells)
    save(ROOT/"summary.json",result)
    print({k:v for k,v in result.items() if k!="comparisons"})


if __name__=="__main__":
    main()
