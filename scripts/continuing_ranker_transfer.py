"""Identify the relative order actually implied by frozen learned weights on comparable camera offers."""
from hashlib import sha256
from itertools import product
from pathlib import Path

from experiments.flexible_execution.candidate_ranking import LinearRanker
from experiments.flexible_execution.ranker_train import SIZES,SEEDS
from experiments.flexible_execution.revision_audit import read,save,write_table,digest
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_exposures import before_restoration


ROOT=Path("outputs/continuing/results")
MODELS=Path("outputs/ranker-diagnostic-v1/models")


def weights():
    models=[]
    for size,seed,obj in product(SIZES,SEEDS,("ordinary","cost")):
        path=MODELS/(f"{size}-{seed}-{obj}.json")
        m=LinearRanker.load(path,obj)
        models.append(dict(size=size,seed=seed,objective=obj,checkpoint_sha256=m.digest,duration_slope=m.weights[1]/m.scale[1]))
    return models


def diagnose(models):
    rows=[]
    for path in before_restoration():
        state=read(path)
        ep=path.with_name(state["method"]);robot=ep/"private/robot"
        budget=read(ep/"result.json")["episode_timeout_seconds"]
        calls=read_trace_rows(robot/"model_calls.jsonl")[0]
        log=read_trace_rows(robot/"timeline.jsonl")[0]
        offers={e["id"]:e for e in log if e["event"]=="native_offer"}
        for event in log:
            opts=event.get("options") or []
            if event["event"]!="native_policy_checked" or len(opts)!=2:
                continue
            step=event["step"];call=next(c for c in calls if c["input"]["current_image_id"]=="onboard_%04d"%step)
            items=[offers[o["id"]] for o in opts]
            shared=len({(o["field"],o["independent"],o["permission"],o["source"]) for o in items})==1
            static=not (call["input"].get("scene_conditions") or {}).get("dynamic")
            if not shared or not static:
                continue
            cost={o["id"]:o["roundtrip_s"] for o in opts}
            ref=items[0]["id"]
            guesses=[]
            for m in models:
                # Each candidate here supplies one identity, visits one skill, returns to the
                # same anchor and has the same scope, duration of validity and permission.
                # For any shared state, all nine other linear-feature components cancel.
                differences={i:m["duration_slope"]*(seconds-cost[ref])/budget for i,seconds in cost.items()}
                guesses.append(min(differences,key=lambda i:(-differences[i],i)))
            # The retrospective label is consulted only after all 18 orderings are generated.
            out=call["output"]
            selected=out.get("use_views")
            if selected is None:selected=[out.get("use_view")]
            demand=[k for k in selected if k in cost] if isinstance(selected,list) and len(selected)<=2 else []
            if (out.get("action") or {}).get("type") not in ("observe","record"):demand=[]
            signature=sha256(str((items[0]["reference_sha256"],sorted(tuple(o["source_bbox"]) for o in items))).encode()).hexdigest()
            rows.append(dict(case=ep.parent.name,method=ep.name,step=step,source_log_sha256=digest(robot/"model_calls.jsonl"),
                      nomination_sha256=signature,choices=list(cost),trip_seconds=cost,models=len(models),
                      unique_ranker_top_choices=len(set(guesses)),top=guesses[0] if len(set(guesses))==1 else None,
                      tie_between_routes=len(set(cost.values()))==1,actual_model_demand=demand,
                      top_models_matching_demand=sum(i in demand for i in guesses) if demand else None))
    return rows


def main():
    models=weights()
    rows=diagnose(models)
    requested=[r for r in rows if len(r["actual_model_demand"])==1]
    data=dict(scope="retrospective relative ordering on capacity-one identity trips, including the first slot of two-capacity policies",
       frozen_models=len(models),all_duration_coefficients_negative=all(m["duration_slope"]<0 for m in models),
       comparable_two_view_decisions=len(rows),equal_estimated_cost_ties=sum(r["tie_between_routes"] for r in rows),
       actual_choices_of_one=len(requested),no_view_requested=sum(not r["actual_model_demand"] for r in rows),
       requested_decisions_with_any_top1_match=sum(bool(r["top_models_matching_demand"]) for r in requested),
       requested_decisions_with_all_models_matching=sum(r["top_models_matching_demand"]==len(models) for r in requested),
       independent_requested_layout_count=len({r["case"] for r in requested}),
       distinct_requested_nomination_sources=len({r["nomination_sha256"] for r in requested}),
       language_model_training=False,new_ranker_training=False,new_model_requests=0,models=models,
       interpretation="The old linear predictors were trained for synthetic branch success or deadline-restricted gain. On these comparable one-trip offers the other nine coordinates are common to the two candidates; their exact differences can be scored without inventing the unobserved open-row task size or a service deadline. Ranking can only pick by duration here. A raw GPT request for a photo is a demand label, not a verified advantage or final task benefit. Retrieval models and larger mixed-coverage prefixes are outside this result.")
    write_table(ROOT/"old_ranker_on_actual_offers.csv",rows)
    save(ROOT/"old_ranker_on_actual_offers.json",data)
    print({k:v for k,v in data.items() if k not in ("scope","models","interpretation")})


if __name__=="__main__":
    main()
