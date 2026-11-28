"""Separate reuse of legal actions from reuse of remaining outcome-equivalent subproblems."""
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.flexible_execution.revision_audit import digest,read,save
from research.flexible_scheduling.anytime_replan import limited_replan
from research.flexible_scheduling.cross_source_workflow import CrossSourcePlanner,CrossSourceWorkflow,PermissionCache
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_permission_cache import DEST,task,ticks

ROOT = DEST/"ablation"
ARMS = (("both",True,True),("neither",False,False),("permissions",True,False),("future-values",False,True))


def measure(reg,n,structure,budget,rep):
    profiles = reg["profiles"]
    shared = task(n,structure,tuple(profiles[0][1]),profiles[0][2])
    output = []
    offset = rep%len(ARMS)
    for name,permissions,future in ARMS[offset:]+ARMS[:offset]:
        cache = PermissionCache(shared,True,residual=permissions)
        for stage,cost,motion,cross in profiles:
            start = perf_counter()
            m = replace(shared,travel=((0,ticks(motion)),(ticks(motion),0)),service=tuple(cost))
            model = CrossSourceWorkflow(m,ticks(cross),permission_cache=cache)
            planner = CrossSourcePlanner(model,quotient=future)
            s = model.initial()
            overhead = (perf_counter()-start)*1000
            pick,source,proved,elapsed = limited_replan(planner,s,max(.001,budget-overhead))
            if pick is None:
                plain = CrossSourcePlanner(CrossSourceWorkflow(m,ticks(cross)))
                began = perf_counter()
                action = plain.greedy(s,"collect")
                overhead += (perf_counter()-began)*1000
                pick = plain.roll(s,"collect"),action
            output.append(dict(method=name,stage=stage,seconds=pick[0]/100,proved=proved,
                               overrun_ms=max(0,elapsed+overhead-budget),cache_entries=len(cache.entries),source=source))
    return output


def main():
    reg = read(DEST/"registration.json")
    old = read(DEST/"wall/registration.json")
    ROOT.mkdir(exist_ok=True)
    registration = dict(parent=digest(DEST/"wall/registration.json"),driver=digest(Path(__file__)),
                        repeats=4,wall_ms=old["wall_ms"],arms=[list(row) for row in ARMS],
                        default_policy=old["common_default"])
    assert_snapshot(ROOT/"registration.json",registration)
    cells = []
    for n,structure,ms in product(reg["roots"],reg["dependence"],registration["wall_ms"]):
        rows = []
        for rep in range(registration["repeats"]):
            p = ROOT/f"{n}-{structure}-{ms}-{rep}.json"
            if not p.is_file():
                save(p,measure(reg,n,structure,ms,rep))
            rows.extend(read(p))
        for stage,*_ in reg["profiles"]:
            comparisons = {name:[v for v in rows if v["method"]==name and v["stage"]==stage] for name,_,_ in ARMS}
            cell = {name:dict(median_s=median(v["seconds"] for v in observations),
                         median_cache_entries=median(v["cache_entries"] for v in observations),
                         cert=sum(v["proved"] for v in observations),fallback=sum(v["source"]=="common-collect" for v in observations),
                         max_overrun_ms=max(v["overrun_ms"] for v in observations))
                    for name,observations in comparisons.items()}
            cells.append(dict(roots=n,structure=structure,budget_ms=ms,stage=stage,methods=cell))
    counts = {}
    for name,_,_ in ARMS:
        deltas = [row["methods"]["neither"]["median_s"]-row["methods"][name]["median_s"] for row in cells]
        counts[name] = dict(better=sum(d>1e-6 for d in deltas),tied=sum(abs(d)<=1e-6 for d in deltas),
                            worse=sum(d<-1e-6 for d in deltas),largest_gain_s=max(deltas))
    supported = dict(cells=len(cells),trials=len(cells)*registration["repeats"]*len(ARMS),results=counts,
        max_overrun_ms=max(r["max_overrun_ms"] for row in cells for r in row["methods"].values()),comparisons=cells,
        note="Timing repetitions use the same topology and price scenarios as the prior development benchmark. "
             "Four arms all retain legal options, evaluate revised prices and use the same recursive search and legal actions.")
    save(ROOT/"summary.json",supported)
    print({k:v for k,v in supported.items() if k!="comparisons"})


if __name__=="__main__":
    main()
