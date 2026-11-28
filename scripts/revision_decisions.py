"""Independent finite reference for initial decisions on the symbolic development set.

No model return, material value, or injected failure is provided to the reference.
It optimizes a nominal successful-capture model at the discrete decision epochs.
"""
from functools import lru_cache
from itertools import product
from math import inf
from time import perf_counter

from experiments.flexible_execution.executor import async_action, terminal_action
from experiments.flexible_execution.live_planning import select
from experiments.flexible_execution.revision_audit import OUT, save, write_table
from experiments.flexible_execution.simulation import Episode
from scripts.revision_mechanisms import TYPES, fixture


def reference(task, initial):
    qs = task.queries
    scenarios = tuple(product(*(q.outcomes for q in qs)))
    need = {r.id:r for r in task.requirements}
    routes = {(a,b):cost for a,b,cost in task.travel}
    def travel(a,b):
        return 0. if a==b else routes.get((a,b),inf)

    def at_epoch(at, site, have, worlds):
        # All branches that have generated the same replies still share a decision.
        groups={}
        for i in worlds:
            key=tuple(scenarios[i][j] if at>=q.estimated_ready else None for j,q in enumerate(qs))
            groups.setdefault(key,[]).append(i)
        return sum(len(bucket)*solve(at,site,have,tuple(bucket))[0] for bucket in groups.values())/len(worlds)

    @lru_cache(None)
    def solve(at,site,have,worlds):
        known={q.id:scenarios[worlds[0]][j] for j,q in enumerate(qs) if q.estimated_ready<=at}
        left=[q.estimated_ready for q in qs if q.id not in known]
        possible=[{q.id:value for q,value in zip(qs,scenarios[i])} for i in worlds]
        required={r.id for r in need.values() if any(r.applies(world) for world in possible)}
        missing=required-set(have)
        costs={}
        if not missing:
            finish=max([at+travel(site,task.terminal)]+left)+task.report_seconds
            if finish<=task.deadline:
                costs["handoff"]=finish
        if left:
            costs["wait"]=at_epoch(min(left),site,have,worlds)
        for skill in task.skills:
            if not skill.permitted(known) or not missing.intersection(skill.covers):
                continue
            finish=at+travel(site,skill.site)+skill.seconds
            if finish+task.report_seconds>task.deadline:
                continue
            costs[skill.id]=at_epoch(finish,skill.site,tuple(sorted(set(have)|set(skill.covers))),worlds)
        return (min(costs.values(),default=inf),costs)

    return solve(0.,task.start,tuple(sorted(initial)),tuple(range(len(scenarios))))


def main():
    rows=[]
    for typ,latency,travel,granted in product(TYPES,(2,7,13),(.5,1.5),(False,True)):
        instance=fixture(typ,latency,travel,granted)
        assert all(v is None for _,v in instance.initial_material)
        optimum,actions=reference(instance.task,[k for k,_ in instance.initial_material])
        driver=Episode(instance,"ASYNC")
        driver.initialize()
        state=driver.snapshot()
        assert not state.results
        for method in ("conditional-async","conditional-prefetch","conditional-planner","adaptive-progress"):
            ordinary=async_action(instance.task,state)
            first=ordinary
            t=perf_counter()
            if first is None:
                choice=select(instance.task,state,method,lambda *a,**k:None)
                first=choice.prepare[0] if choice else None
            elapsed=perf_counter()-t
            action="wait" if first is None else first.skill.id if first.skill is not None else "handoff"
            # A move that is observationally redundant can safely be worse than waiting.
            value=actions.get(action)
            rows.append(dict(case=instance.id,type=typ,method=method,latency_s=latency,travel_s=travel,
                        granted=granted,proposed=action,optimal_first_actions="|".join(a for a,v in actions.items() if abs(v-optimum)<1e-7),
                        optimal_forecast_s=optimum,policy_first_forecast_s=value,
                        reference_optimal=value is not None and abs(value-optimum)<1e-7,
                        regret_s=value-optimum if value is not None else None,policy_compute_s=elapsed))
    write_table(OUT/"finite_decision_reference.csv",rows)
    summary={m:dict(n=sum(r["method"]==m for r in rows),
                    optimal=sum(r["method"]==m and r["reference_optimal"] for r in rows),
                    unmapped=sum(r["method"]==m and r["policy_first_forecast_s"] is None for r in rows))
             for m in sorted({r["method"] for r in rows})}
    save(OUT/"finite_decision_reference.json",summary)
    print(summary)


if __name__=="__main__":
    main()
