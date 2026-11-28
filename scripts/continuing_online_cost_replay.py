"""Compare choices with public cost information and no lookahead to a model answer."""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.cross_source_workflow import CausalMeasurements, CrossSourcePlanner, CrossSourceWorkflow
from research.flexible_scheduling.event_workflow import State, UNKNOWN
from research.flexible_scheduling.live_event_room import mission
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_room_clock_replay import BASE, CASE, OUT

CFG = Path("config/matched_event_rooms_dev01.json")
CROSS = BASE.parent / "cross-source-assay-dev02" / "summary.json"


def ticks(seconds):
    return max(1, round(100 * seconds))


def decision(case, cost, cross, *, raw=1, pose=0, values=None):
    base = mission(case, cost)
    m = replace(base, travel=((0,ticks(cost["outward"])),(ticks(cost["return"]),0)),
                capture=(ticks(cost["photo"]),)*2, service=(0,ticks(cost["single"]),ticks(cost["pair"])))
    graph = CrossSourceWorkflow(m, ticks(cross) if cross is not None else None)
    state = State(tuple(values) if values is not None else (UNKNOWN,)*len(m.queries), raw, pose)
    values = {}
    for method, quotient, ablation in (("joint",True,None),("generic",False,None),("serial",False,"serial")):
        start = perf_counter()
        planner = CrossSourcePlanner(graph, quotient=quotient, ablation=ablation)
        solved = planner.solve(state,budget=100000)
        if not solved["proved"]:
            raise ValueError("Expected an exact certificate for the small source-bound task")
        values[method] = dict(expected_s=solved["value_ticks"]/100, first=asdict(solved["action"]),
                              states=solved["expanded"], clock_s=perf_counter()-start,
                              common_prefetch_expected_s=solved["strong_ticks"]/100)
    if abs(values["generic"]["expected_s"] - values["joint"]["expected_s"]) > 1e-6:
        raise ValueError("Same-information exact planners disagree")
    return values


def observed_schedule(case, cost, cross, actual_values):
    """Post-seal replay on the known path only, never look at a hidden or nonreturned field."""
    m = mission(case,cost)
    m = replace(m, travel=((0,ticks(cost["outward"])),(ticks(cost["return"]),0)),
                capture=(ticks(cost["photo"]),)*2, service=(0,ticks(cost["single"]),ticks(cost["pair"])))
    graph = CrossSourceWorkflow(m,ticks(cross) if cross is not None else None)
    engine = CrossSourcePlanner(graph,quotient=False)
    plan_times = {}
    if graph.cross_ticks is None:
        variants = ("optimal","front-during-turn","side-before-ask","ask-front-then-turn")
    else:
        variants = ("optimal","front-during-turn","side-before-ask","ask-front-then-turn","collect-then-batch-roots")
    for label in variants:
        state, now, steps, requests = graph.initial(), 0, [], 0
        while not graph.complete(state):
            actions = engine.options(state)
            if label == "front-during-turn" and not steps:
                action, = [a for a in actions if a.motion==(1,True) and a.reads==((0,1),)]
            elif label in ("side-before-ask","collect-then-batch-roots") and not steps:
                action, = [a for a in actions if a.motion==(1,True) and not a.reads]
            elif label == "side-before-ask" and len(steps)==1:
                action, = [a for a in actions if not a.motion and a.reads==((1,2),)]
            elif label == "collect-then-batch-roots" and len(steps)==1:
                action, = [a for a in actions if not a.motion and a.reads==((graph.cross_source,3),)]
            elif label == "ask-front-then-turn" and not steps:
                action, = [a for a in actions if not a.motion and a.reads==((0,1),)]
            else:
                action = engine.solve(state,budget=100000)["action"]
            duration, branches = graph.successors(state,action)
            compatible = [b for _,b in branches if all(v==old or (i in actual_values and v==actual_values[i])
                                                        for i,(v,old) in enumerate(zip(b.values,state.values)))]
            if len(compatible) != 1:
                raise ValueError("Schedule requires a nonreturned observation on the replay branch")
            steps.append(dict(at_s=now/100, action=asdict(action), time_to_next_event_s=duration/100,
                              prior_results=list(state.values)))
            requests += len(action.reads)
            state, now = compatible[0], now+duration
        plan_times[label] = dict(estimated_task_s=now/100, calls=requests, steps=steps)
    return plan_times


def main():
    source = OUT / "clock_audit.json"
    audit = read(source)
    cost = read(BASE/"live/registration.json")["costs"][CASE]
    case = next(c for c in read(CFG)["cases"] if c["id"]==CASE)
    models = {"transferred":cost, "before_task_site_history":{**cost,
              "outward":audit["branch"][1]["trip_s"]-cost["photo"]}}
    measure = CausalMeasurements()
    for r in audit["prior_same_site"]:
        measure.record("turn",available_at=-1,seconds=r["s"],delivered=True)
    original = audit["actual"]["joint"]
    for call in original["calls"]:
        measure.record("single",available_at=call["reply_s"],seconds=call["service_s"],delivered=call["output"] is not None)
    # The side image is already available on this trace by the first semantic return.
    # Record its elapsed turn+shutter bundle only after the original fused evidence action completed.
    ready = read_trace_rows(BASE/"live"/CASE/"joint/timeline.jsonl")[0]
    first = read(BASE/"live"/CASE/"joint/result.json")["start_clock"]
    side_time = next(v["monotonic_seconds"]-first for v in ready if v["event"]=="camera_observation" and v["step"]=="side")
    measure.record("turn+capture",available_at=side_time,seconds=original["travel_and_capture_s"],delivered=True)
    remaining = [UNKNOWN]*5
    snapshots = []
    for c in original["calls"]:
        for name, value in c["output"].items():
            remaining[int(name[1:])] = int(value=="positive")
        when = c["reply_s"]
        info = {key:measure.describe(key,when,default) for key,default in (("turn",cost["outward"]),
                         ("turn+capture",cost["outward"]+cost["photo"]),("single",cost["single"]))}
        entry = dict(at_s=when, causal=info, actual_already_returned=list(remaining))
        if c!=original["calls"][-1]:
            adaptive = {**models["before_task_site_history"],"single":info["single"]["successful_median_s"],
                        "outward":max(.01,info["turn+capture"]["successful_median_s"]-cost["photo"])}
            entry["plan"] = decision(case,adaptive,None,raw=3,pose=1,values=remaining)
        snapshots.append(entry)
    # Define sensitivity levels from the existing singleton calibration before reading the newer assay.
    scenarios = [None]+[cost["single"]*k for k in (1.,1.3,1.6,2.,2.5,4.)]
    reg = dict(clock_audit_sha256=digest(source), config_sha256=digest(CFG), cross_sensitivity_s=list(scenarios),
               input_visibility="Prior turn history only before task. Actual one-image responses enter after return. "
               "The later cross-camera calibration is a separate retrospective scenario, never a prior for the older task.")
    assert_snapshot(OUT/"planning_inputs.json",reg)
    if CROSS.is_file():
        additional = read(CROSS)
        scenarios += [c["observations"]["joint"]["service_s"] for c in additional["cases"]
                       if c["paired_answers_delivered"]]
    actual_values = {int(k[1:]):int(v=="positive") for c in original["calls"] for k,v in c["output"].items()}
    rows = []
    for name, model in models.items():
        defaults = decision(case,model,None)
        for number, cross in enumerate(scenarios):
            rows.append(dict(motion=name, cross_s=cross, cross_from_later_assay=number>=len(reg["cross_sensitivity_s"]),
                             policy=defaults if cross is None else decision(case,model,cross),
                             fixed_observed_branch=observed_schedule(case,model,cross,actual_values)))
    # A separate exact-input audit describes the tail after both arms ended.
    # These repetitions were not available to the original planner at dispatch.
    first_calls = [audit["actual"][a]["calls"][0] for a in ("joint","strong-prefetch")]
    extremes = sorted(c["service_s"] for c in first_calls)
    fingerprints = read(BASE/"causal_branch.json")["identical_request_replications"]
    if not any(b["case"]==CASE and b["count"]==2 and b["has_censored"]
               and [b["minimum_observed_s"],b["maximum_observed_s"]]==extremes for b in fingerprints):
        raise ValueError("No identical-input witness for the pending service calculation")
    tail = CausalMeasurements()
    for c in first_calls:
        if c["keys"]!=["q0"] or c["site"]!="front":
            raise ValueError("Not the matched first request")
        tail.record("matched",available_at=0,seconds=c["service_s"],delivered=c["output"] is not None)
    pending = [dict(elapsed_s=t,**tail.conditional_remaining("matched",0,t,120)) for t in (.2,4.,30.)]
    out = dict(new_model_requests=0, comparisons=len(rows), snapshots=snapshots, scenarios=rows,
               matched_request_tail=pending,
               later_assay_sha256=digest(CROSS) if CROSS.exists() else None,
               warning="Fixed clocks and declared independent .5 gate priors. This replay uses observed answers on one branch after the policy computation; it is neither a fresh model outcome nor causal evidence about unobserved service durations.")
    save(OUT/"summary.json",out)
    print([(r["motion"],r["cross_s"],r["policy"]["joint"]["expected_s"],
            r["policy"]["joint"]["first"],r["fixed_observed_branch"]["optimal"]["estimated_task_s"]) for r in rows])


if __name__ == "__main__":
    main()
