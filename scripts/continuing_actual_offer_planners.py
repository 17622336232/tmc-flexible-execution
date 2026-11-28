"""Replay only real pre-dispatch camera offers under matched information, budgets and action capacity."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

from experiments.flexible_execution.revision_audit import read,save,write_table
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.workflow import View,Workflow,DecisionProblem
from scripts.continuing_critical_path import section
from scripts.continuing_exposures import before_restoration


ROOT=Path("outputs/continuing")
MODES=("factored","generic","strong-prefetch","async","serial")


class BoundedOffers(DecisionProblem):
    def __init__(self,task,capacity):
        super().__init__(task)
        self.capacity=capacity

    def actions(self):
        return ((b,o) for b,o in super().actions() if len(b)+len(o)<=self.capacity)


def replay():
    rows=[]
    frozen=before_restoration()
    for cohort in ("native-views-dev01","native-evidence-dev02"):
        root=ROOT/cohort
        for path in (p for p in frozen if p.parents[1]==root):
            receipt=read(path)
            if receipt["state"]!="scored":
                continue
            robot=path.with_name(receipt["method"])/"private/robot"
            log=read_trace_rows(robot/"model_calls.jsonl")[0]
            events=read_trace_rows(robot/"timeline.jsonl")[0]
            bystep={int(c["input"]["current_image_id"].rsplit("_",1)[1]):c for c in log}
            for e in events:
                if e["event"]!="native_policy_checked" or not e["options"]:
                    continue
                c=bystep[e["step"]];remaining=c["input"]["execution_clock"]["remaining_seconds"]
                options=e["options"];keys=[r["id"] for r in options]
                if len(keys)!=len(set(keys)) or len(keys)>2:
                    raise ValueError("Unexpected active study capacity")
                cap=1 if cohort=="native-views-dev01" else 2
                priors=[x.get("need_probability",1. if x.get("independent") else .5) for x in options]
                costs={v["id"]:v["roundtrip_s"] for v in options}
                window=e["request_window_s"]
                task=Workflow(tuple(((1<<i,1.),) if p==1 else ((0,1.-p),(1<<i,p)) for i,p in enumerate(priors)),
                              tuple(View(v["id"],1<<i,v["roundtrip_s"],True,False) for i,v in enumerate(options)),
                              0,(window,),(0.,0.),0.,max_early_motion=max(0.,remaining-window))
                forecasts={};times=defaultdict(list)
                # Cold instance each time, rotating the algorithm order. Repetition measures compute only.
                for rep in range(15):
                    for i in range(len(MODES)):
                        m=MODES[(i+rep)%len(MODES)]
                        sol=BoundedOffers(task,cap).solve(m)
                        choice=tuple(keys[i] for i in sol["overlap"])
                        if m in forecasts and forecasts[m][:2]!=(choice,sol["expected_seconds"]):
                            raise ValueError("Non-deterministic decision on fixed observations")
                        forecasts[m]=(choice,sol["expected_seconds"],sol["evaluated"])
                        times[m].append(sol["solve_seconds"]*1000)
                # Only after all choices are committed, inspect what the frozen real reply actually requested.
                decision=c.get("output") or {}
                uses=decision.get("use_views")
                if uses is None:
                    uses=[decision.get("use_view")]
                demand={v for v in uses if isinstance(v,str) and v in costs} if isinstance(uses,list) and len(uses)<=2 else set()
                if (decision.get("action") or {}).get("type") not in ("observe","record"):
                    demand=set()
                duration=c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"]
                feasible=[subset for k in range(cap+1) for subset in combinations(keys,k) if sum(costs[i] for i in subset)<=max(0.,remaining-window)]
                best=min(section(costs,s,demand,duration) for s in feasible)
                for m,(choice,val,count) in forecasts.items():
                    rows.append(dict(cohort=cohort,case=receipt["case"],method_trace=receipt["method"],step=e["step"],planner=m,
                        candidates=len(keys),capacity=cap,chosen=list(choice),expected_section_s=val,plans_evaluated=count,
                        equal_value_to_exact=abs(val-forecasts["generic"][1])<1e-7,equal_choice_to_exact=choice==forecasts["generic"][0],
                        posthoc_nominal_regret_s=section(costs,choice,demand,duration)-best,
                        reproduced_actual_v2_choice=(set(choice)==set(e["selected"])) if cohort=="native-evidence-dev02" and
                              m=={"evidence-deferred":"factored","evidence-off":"serial"}.get(receipt["method"]) else None,
                        solver_wall_median_ms=median(times[m]),actual_transport_s=duration))
    return rows


def main():
    dest=ROOT/"results";out=replay()
    write_table(dest/"actual_offer_planners.csv",out)
    exemplars=[r for r in out if r["planner"]=="generic"]
    stats={}
    for m in MODES:
        rr=[r for r in out if r["planner"]==m]
        stats[m]=dict(snapshots=len(rr),equal_value_to_exact=sum(r["equal_value_to_exact"] for r in rr),
               equal_choice_to_exact=sum(r["equal_choice_to_exact"] for r in rr),
               maximum_median_compute_ms=max(r["solver_wall_median_ms"] for r in rr),
               realized_demand_oracle_nominal_regret_s=sum(r["posthoc_nominal_regret_s"] for r in rr))
    matched=[r["reproduced_actual_v2_choice"] for r in out if r["reproduced_actual_v2_choice"] is not None]
    d=dict(real_decisions_with_eligible_options=len(exemplars),unique_layouts=len({r["case"] for r in exemplars}),
           options_histogram=dict(Counter(r["candidates"] for r in exemplars)),actual_v2_reproductions=[sum(matched),len(matched)],
           fixed_compute_repeats=15,planners=stats,new_model_requests=0,
           interpretation="Frozen real pre-request camera candidates and costs. The full factored and generic algorithms use identical legal subsets and priors. At this point an in-flight image request cannot accept additional photos, so strong physical prefetch can enumerate the same feasible acquisitions. The oracle regret uses nominal trip costs and the subsequently realized model choice, never task time.")
    save(dest/"actual_offer_planners.json",d)
    print(d)


if __name__=="__main__":
    main()
