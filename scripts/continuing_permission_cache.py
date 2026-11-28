"""Compare generic and residual permission reuse over the same changing cost scenarios."""
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.flexible_execution.revision_audit import digest, read, save
from research.flexible_scheduling.cross_source_workflow import CrossSourcePlanner, CrossSourceWorkflow, PermissionCache
from research.flexible_scheduling.event_workflow import Mission, Query, State, UNKNOWN
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_online_cost_replay import CROSS, OUT, ticks

DEST = OUT/"permission-cache"


def task(roots, structure, cost, travel):
    queries = [Query("screen "+str(i),(i%2,)) for i in range(roots)]
    if structure=="shared-followup":
        queries.append(Query("shared",(1,),tuple(((i,True),) for i in range(roots))))
    else:
        for parity in (0,1):
            indices = range(parity,roots,2)
            queries.append(Query("linked "+str(parity),(1-parity,),tuple(((i,True),) for i in indices)))
    return Mission(("front","side"),tuple(queries),((0,ticks(travel)),(ticks(travel),0)),
                   (1,1),(True,True),0,1,cost,0,1,1)


def compare(roots,structure,budget,profiles,first):
    base = task(roots,structure,profiles[0][1],profiles[0][2])
    groups = {}
    for residual in (first,not first):
        cache = PermissionCache(base,True,residual=residual)
        results = []
        for p,cost,travel,cross in profiles:
            start = perf_counter()
            m = replace(base,travel=((0,ticks(travel)),(ticks(travel),0)),service=cost)
            e = CrossSourceWorkflow(m,ticks(cross),permission_cache=cache)
            planner = CrossSourcePlanner(e,quotient=residual)
            # This comparison starts at the original amount of material, while allowing all results
            # in a contingency tree. It never fills a model unknown with a simulated sensor value.
            state = State((UNKNOWN,)*len(m.queries),1,0)
            hit = cache.hits
            answer = planner.solve(state,budget=budget)
            results.append(dict(profile=p,cost_s=answer["value_ticks"]/100,proved=answer["proved"],
                wall_ms=(perf_counter()-start)*1000,permissions=len(cache.entries),permission_hits=cache.hits-hit,
                exact_completion_bound_s=answer["root_lower_ticks"]/100,expansions=answer["expanded"],
                common_strong_cost_s=answer["strong_ticks"]/100))
        groups["residual" if residual else "generic"] = results
    return groups


def main():
    scan = read(OUT/"clock_audit.json")
    from scripts.continuing_online_cost_replay import BASE
    cal = read(BASE/"calibration/summary.json")
    cross = read(CROSS)
    local, transfer = scan["branch"][1]["trip_s"],scan["branch"][0]["trip_s"]
    fast, slow = [case["observations"]["joint"]["service_s"] for case in cross["cases"]]
    initial = (0,ticks(cal["single_s"]),ticks(cal["pair_s"]))
    faster_single = (0,ticks(scan["actual"]["joint"]["calls"][0]["service_s"]),initial[2])
    profiles = [("transferred",initial,transfer,fast),("same-site",initial,local,fast),
                ("semantics-updated",faster_single,local,fast),("slow-cross-service",faster_single,local,slow)]
    DEST.mkdir(exist_ok=True)
    assert_snapshot(DEST/"registration.json", dict(clock=digest(OUT/"clock_audit.json"),cross=digest(CROSS),
        code=digest(Path(__file__)),roots=[4,5],dependence=["shared-followup","linked-followups"],
        expansions=[12,64],profiles=[[p,list(c),t,x] for p,c,t,x in profiles],
        note="Both controls retain legal options across cost changes. Each stage has the same new cost information. "
             "Expansion budgets exclude the common fully feasible two-dispatch incumbent. Timing includes it and is descriptive."))
    rows = []
    for index,(roots,structure,budget) in enumerate(product((4,5),("shared-followup","linked-followups"),(12,64))):
        groups = compare(roots,structure,budget,profiles,index%2==0)
        for j,g in zip(groups["residual"],groups["generic"]):
            if j["proved"] and g["proved"] and abs(j["cost_s"]-g["cost_s"])>1e-6:
                raise ValueError("Certified costs must agree")
            rows.append(dict(roots=roots,structure=structure,budget=budget,profile=j["profile"],residual=j,generic=g))
    answer = dict(pairs=len(rows),same_feasible=sum(abs(v["residual"]["cost_s"]-v["generic"]["cost_s"])<1e-6 for v in rows),
        residual_lower=sum(v["residual"]["cost_s"]<v["generic"]["cost_s"]-1e-6 for v in rows),
        generic_lower=sum(v["generic"]["cost_s"]<v["residual"]["cost_s"]-1e-6 for v in rows),
        residual_certificates=sum(v["residual"]["proved"] for v in rows), generic_certificates=sum(v["generic"]["proved"] for v in rows),
        median_residual_over_generic_wall=median(v["residual"]["wall_ms"]/v["generic"]["wall_ms"] for v in rows),
        median_residual_over_generic_permissions=median(v["residual"]["permissions"]/v["generic"]["permissions"] for v in rows),
        new_model_requests=0,rows=rows)
    save(DEST/"summary.json",answer)
    print({k:v for k,v in answer.items() if k!="rows"})


if __name__=="__main__":
    main()
