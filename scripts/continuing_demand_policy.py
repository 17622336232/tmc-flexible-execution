"""Test whether learning camera demand changes an exact legal schedule on fixed observed offers."""
import csv
from itertools import combinations

from experiments.flexible_execution.revision_audit import read,save,write_table
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_critical_path import section
from scripts.continuing_exposures import before_restoration
from scripts.continuing_view_demand import OUT,VARIANTS,stated


def expected_section(menu,probabilities,wait):
    def value(option):
        travel=sum(menu[k] for k in option)
        future=sum(probabilities[k]*d for k,d in menu.items() if k not in option)
        return max(wait,travel)+future,travel,option
    return value


def evaluate():
    with (OUT/"cross_fitted.csv").open() as fp:
        probabilities={(r["turn"],r["candidate"],r["variant"]):float(r["p"]) for r in csv.DictReader(fp)}
    results=[]
    for path in before_restoration():
        rec=read(path);ep=path.with_name(rec["method"]);robot=ep/"private/robot"
        calls=read_trace_rows(robot/"model_calls.jsonl")[0]
        for ev in read_trace_rows(robot/"timeline.jsonl")[0]:
            if ev["event"]!="native_policy_checked" or not ev["options"]:
                continue
            frame="onboard_%04d"%ev["step"];turn=f"{ep.parent.name}/{ep.name}/{frame}"
            c=next(v for v in reversed(calls) if v["input"]["current_image_id"]==frame)
            menu={o["id"]:o["roundtrip_s"] for o in ev["options"]}
            cap=1 if ep.parents[1].name=="native-views-dev01" else 2
            wait=ev["request_window_s"];rem=c["input"]["execution_clock"]["remaining_seconds"]
            feasible=[s for n in range(min(cap,len(menu))+1) for s in combinations(sorted(menu),n) if sum(menu[k] for k in s)<=max(0.,rem-wait)]
            forecasts={}
            for variant in VARIANTS:
                p={k:probabilities[(turn,k,variant)] for k in menu}
                forecasts[variant]=min(feasible,key=expected_section(menu,p,wait))
            actual=c.get("output");shown={o["id"] for o in c["input"]["offered_views"]}
            expressed=stated(actual,shown)
            if expressed is None:
                continue
            recognized=stated({**actual,"action":{"type":actual["action"]["type"]}},shown)
            if recognized is None:
                continue
            service=c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"]
            for variant,s in forecasts.items():
                for label,d in (("expressed",expressed),("recognized-top-level",recognized)):
                    demand=set(menu)&d
                    optimal=min(section(menu,x,demand,service) for x in feasible)
                    value=section(menu,s,demand,service)
                    results.append(dict(layout=ep.parent.name,turn=turn,label_scope=label,variant=variant,
                         choices=list(s),demand=sorted(demand),matches_fixed_half=s==forecasts["fixed-half"],
                         unrequested_selected=len(set(s)-demand),wanted_selected=len(set(s)&demand),
                         nominal_oracle_gap_s=value-optimal,nominal_gain_from_serial_s=section(menu,(),demand,service)-value))
    write_table(OUT/"cross_fitted_policy.csv",results)
    stats={}
    for label in ("expressed","recognized-top-level"):
        stats[label]={}
        for variant in VARIANTS:
            rr=[r for r in results if r["label_scope"]==label and r["variant"]==variant]
            stats[label][variant]=dict(decisions=len(rr),same_set_as_fixed_half=sum(r["matches_fixed_half"] for r in rr),
                unrequested_selected=sum(r["unrequested_selected"] for r in rr),wanted_selected=sum(r["wanted_selected"] for r in rr),
                sum_oracle_gap_nominal_s=sum(r["nominal_oracle_gap_s"] for r in rr),
                sum_gain_from_serial_nominal_s=sum(r["nominal_gain_from_serial_s"] for r in rr))
    doc=dict(pre_recovery_cross_fitted=stats,training_independent_unit="full development layout",
        interpretation="Offline single frozen decision. Exact legal subsets and nominal round-trip estimates are identical for all four variants. Only probabilities differ. The later real response and observed transport window score choices after selection. Unseen travel times and later ordinary model replies are never synthesized. Section estimates do not establish a complete-task improvement.",
        online_deployed=False,new_model_requests=0)
    save(OUT/"policy_summary.json",doc)
    print(stats)


if __name__=="__main__":
    evaluate()
