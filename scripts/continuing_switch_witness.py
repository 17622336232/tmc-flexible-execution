"""A three-question control separating the cost of physical preparation from question release."""
from dataclasses import asdict
from itertools import product
from pathlib import Path

from experiments.flexible_execution.revision_audit import digest,read,save
from research.flexible_scheduling.cross_source_workflow import CrossSourceWorkflow
from research.flexible_scheduling.event_workflow import Dispatch,Mission,Query
from research.flexible_scheduling.priced_cross_source import PricedCrossSourcePlanner
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_online_cost_replay import BASE,CROSS,OUT,ticks

DEST = OUT/"switch-witness"


def compute(trip,single,pair,cross,prior):
    # A and B are independent starting questions. Arrival is required for B.
    # C uses the same side photo as B and requires the true A result.
    m = Mission(("front","side"),(Query("A",(0,),prior=prior),Query("B",(1,)),
          Query("C",(1,),(((0,True),),))),((0,trip-1),(trip-1,0)),(1,1),
          (True,True),0,1,(0,single,pair),0,1,1)
    graph = CrossSourceWorkflow(m,cross)
    initial = graph.initial()
    joint = PricedCrossSourcePlanner(graph).solve(initial,budget=100000)
    generic = PricedCrossSourcePlanner(graph,quotient=False).solve(initial,budget=100000)
    if not (joint["proved"] and generic["proved"] and abs(joint["value_ticks"]-generic["value_ticks"])<1e-6):
        raise ValueError("This small task should allow an independent exact certificate")
    # Formula values for two specified plans; an exactly optimal policy can mix strategies.
    ask_first = max(single,trip)+(1-prior)*single+prior*pair
    collect_merge = trip+cross+prior*single if cross is not None else None
    layer = initial
    for stage in range(2):
        commands = graph.dispatches(layer)
        admitted = {mask for action in commands for _,mask in action.reads}
        if any(mask&4 for mask in admitted):
            raise ValueError("Premature evidence-use is possible in the benchmark")
        if stage==0:
            dt, branches = graph.successors(layer,Dispatch((1,True)))
            if dt!=trip or len(branches)!=1:
                raise ValueError("Taking a frame should reveal no permit-triggering judgment")
            layer = branches[0][1]
    opt = joint["value_ticks"]
    if min([ask_first]+([collect_merge] if cross is not None else []))+1e-6<opt:
        raise ValueError("Declared feasible plans cannot beat the exact continuation")
    return dict(prior=prior,turn_s=trip/100,cross_s=cross/100 if cross is not None else None,
                ask_first_s=ask_first/100,collect_merge_s=collect_merge/100 if collect_merge is not None else None,
                joint_s=opt/100,first=asdict(joint["action"]),strong_s=joint["strong_ticks"]/100)


def main():
    clock = read(OUT/"clock_audit.json")
    cal = read(BASE/"calibration/summary.json")
    cross = read(CROSS)
    prices = [None]+[ticks(c["observations"]["joint"]["service_s"]) for c in cross["cases"]]
    trips = [ticks(clock["branch"][i]["trip_s"]) for i in (0,1)]
    DEST.mkdir(exist_ok=True)
    assert_snapshot(DEST/"registration.json",dict(driver=digest(Path(__file__)),clocks=digest(OUT/"clock_audit.json"),
        prior=[0,.5,1],trips=trips,cross=prices,photos=digest(CROSS),
        scope="Known event-model outcomes at p=0, p=1, and midpoint convention. No observations are replaced in any archived episode."))
    rows = [compute(t,ticks(cal["single_s"]),ticks(cal["pair_s"]),x,p)
            for t,x,p in product(trips,prices,(0,.5,1))]
    result = dict(rows=rows,count=len(rows),new_model_requests=0,
                  cost_switches_at_half_prior=[r for r in rows if r["prior"]==.5 and r["cross_s"]==prices[1]/100],
                  status="development sensitivity, the expensive service profile is an observed sensitivity and not a predicted outcome")
    save(DEST/"summary.json",result)
    print(result["cost_switches_at_half_prior"])


if __name__=="__main__":
    main()
