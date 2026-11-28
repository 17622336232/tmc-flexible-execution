"""Determine which first choices survive the full observed service-cost envelope."""
from dataclasses import asdict, replace

from experiments.flexible_execution.revision_audit import digest, read, save
from research.flexible_scheduling.cross_source_workflow import CrossSourcePlanner,CrossSourceWorkflow
from research.flexible_scheduling.event_workflow import Dispatch,UNKNOWN
from research.flexible_scheduling.live_event_room import mission
from scripts.continuing_online_cost_replay import BASE,CASE,CFG,CROSS,OUT,ticks


def candidates(case, cost, cross):
    m = mission(case,cost)
    m = replace(m,travel=((0,ticks(cost["outward"])),(ticks(cost["return"]),0)),
                capture=(ticks(cost["photo"]),)*2,service=(0,ticks(cost["single"]),ticks(cost["pair"])))
    e = CrossSourceWorkflow(m,ticks(cross))
    p = CrossSourcePlanner(e,quotient=False)
    start = e.initial()

    def after(state,dispatch):
        if dispatch not in p.options(state):
            raise ValueError("Expected a permitted prospective action")
        dt, branches = p.trans(state,dispatch)
        value = dt
        for prob, child in branches:
            opt = p.solve(child,budget=100000)
            if not opt["proved"]:
                raise ValueError("Incomplete small numerical search")
            value += prob*opt["value_ticks"]
        return value/100

    arrive = Dispatch((1,True))
    dt, outcomes = p.trans(start,arrive)
    _, state = outcomes[0]
    if len(outcomes)!=1 or state.values!=(UNKNOWN,)*len(m.queries):
        raise ValueError("No semantic outcomes should be learned from the first capture")
    root = p.solve(start,budget=100000)
    if not root["proved"]:
        raise ValueError("Incomplete exact comparison")
    return dict(optimal=root["value_ticks"]/100,first=asdict(root["action"]),
        ask_during_trip=after(start,Dispatch((1,True),((0,1),))),
        side_read_first=dt/100+after(state,Dispatch((),((1,2),))),
        collect_then_merge=dt/100+after(state,Dispatch((),((e.cross_source,3),))),
        collect_then_choose=after(start,arrive))


def main():
    costs = read(BASE/"live/registration.json")["costs"][CASE]
    audit = read(OUT/"clock_audit.json")
    cross = read(CROSS)
    target = next(c for c in read(CFG)["cases"] if c["id"]==CASE)
    sources = {k["house"]:k["observations"]["joint"]["service_s"] for k in cross["cases"]}
    sites = {"transferred":costs["outward"],
             "same-site":audit["branch"][1]["trip_s"]-costs["photo"]}
    rows = []
    for plan, outward in sites.items():
        for calibration_source, latency in sources.items():
            got = candidates(target,{**costs,"outward":outward},latency)
            rows.append(dict(motion_profile=plan,calibration_house=calibration_source,cross_service_s=latency,**got))
    options = ("ask_during_trip","side_read_first","collect_then_merge","collect_then_choose")
    scored = {}
    for opts in options:
        scored[opts] = max(r[opts]-r["optimal"] for r in rows if r["motion_profile"]=="same-site")
    # The two costs are levels, not estimated probabilities. In each scenario the
    # continuation is replanned with that cost under the same observed conditional outcomes.
    scope="Exact first-decision sensitivity with optimal cost-aware recourse. 'Collect then choose' assumes the " \
          "cost profile can be resolved at that checkpoint; a real capture by itself reveals no semantic-service " \
          "latency. The true robust attainable cost can be higher. No population risk probabilities estimated."
    save(OUT/"choice_envelope.json",dict(cross_calibration=digest(CROSS),clock_audit=digest(OUT/"clock_audit.json"),
        rows=rows,worst_local_regret_s=scored,new_model_requests=0,scope=scope))
    print(rows)
    print("Worst scenario regret after shared new motion history",scored)


if __name__=="__main__":
    main()
