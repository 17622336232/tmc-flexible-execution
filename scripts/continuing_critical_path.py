"""Bound reorder-only action gains on fixed genuine model inputs, physical returns, and complete tasks."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from experiments.flexible_execution.revision_audit import digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_native_report import length
from scripts.continuing_exposures import before_restoration


ROOT=Path("outputs/continuing")
DEST=ROOT/"results"


def meter(c):
    return c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"]


def successful_trips(timeline,proofs,step):
    measured={}
    for p in proofs:
        shot=p.get("image")
        if not (shot and p.get("raw_bound") and p.get("actual_goal_reached") and p.get("returned_to_anchor")):
            continue
        key=p["candidate"]
        match=next((e for e in timeline if e["event"]=="native_raw_photo" and e["step"]==step and e["image_id"]==shot),None)
        if not match:
            continue
        depart=[e for e in timeline if e["event"]=="action_started" and e["step"]==str(step)+"-side"
                and e.get("candidate")==key and e.get("early")==p["early"] and e["monotonic_seconds"]<match["monotonic_seconds"]]
        back=[e for e in timeline if e["event"]=="native_anchor_checked" and e["step"]==step and e["candidate"]==key
               and e["restored"] and e["monotonic_seconds"]>=match["monotonic_seconds"]]
        if not depart or not back:
            raise ValueError("Cannot reconstruct an achieved excursion")
        d=back[0]["monotonic_seconds"]-depart[-1]["monotonic_seconds"]
        if key in measured:
            raise ValueError("Multiple successful trips for the same requested view; disambiguate before analysis")
        measured[key]=d
    return measured


def section(costs,selected,requested,window):
    return max(window,sum(costs[k] for k in selected))+sum(costs[k] for k in requested if k not in selected)


def compare_turns(cohort,ep,calls,events,proofs):
    results=[]
    for c in calls:
        current=c["input"].get("current_image_id","")
        options=c["input"].get("offered_views",[])
        if not options or not c.get("output"):
            continue
        step=int(current.rsplit("_",1)[1])
        option_ids={o["id"] for o in options}
        out=c["output"]
        raw=out.get("use_views")
        if raw is None:
            raw=[out.get("use_view")]
        requested={x for x in raw if isinstance(x,str) and x in option_ids} if isinstance(raw,list) and len(raw)<=2 else set()
        if (out.get("action") or {}).get("type") not in ("observe","record"):
            requested=set()
        check=next((e for e in events if e["event"]=="native_policy_checked" and e["step"]==step),{})
        nominal={v["id"]:v["roundtrip_s"] for v in check.get("options",[])}
        demand=requested.intersection(nominal)
        started={e["candidate"] for e in events if e["event"]=="native_early_start" and e["step"]==step}
        if not started<=set(nominal):
            raise ValueError("An early excursion lacked a legal nominal option")
        cap=1 if ep.name.startswith("native-") else 2
        possible=[s for n in range(min(len(nominal),cap)+1) for s in combinations(nominal,n)]
        w=meter(c)
        serial=section(nominal,(),demand,w)
        best=min(section(nominal,s,demand,w) for s in possible)
        incurred=section(nominal,started,demand,w)
        trips=successful_trips(events,proofs,step)
        known=sorted((trips[k] for k in demand if k in trips),reverse=True)
        mlo=min(w,sum(known[:cap]))
        mhi=w if any(k not in trips for k in demand) else mlo
        release=next((e for e in events if e["event"]=="native_material_release" and e["step"]==step),{})
        genuinely_moved=any(e["event"]=="action_started" and e["step"]==str(step)+"-side" for e in events)
        stall=max(0.,release["monotonic_seconds"]-c["return_monotonic_seconds"]) if genuinely_moved and release else None
        results.append(dict(cohort=cohort,case=ep.parent.name,method=ep.name,step=step,call=c["logical_job_attempt"],
           request_sha=digest(ep/"private/robot/model_calls.jsonl"),offered=len(option_ids),admissible=len(nominal),
           actually_requested=sorted(requested),admissible_demand=sorted(demand),early_started=sorted(started),
           requested_and_started=len(demand&started),unrequested_started=len(started-demand),
           measured_route_seconds=trips,service_s=w,forecast_s=check.get("request_window_s"),
           post_reply_release_delay_s=stall,
           nominal_oracle_vs_serial_s=serial-best,nominal_executed_vs_serial_s=serial-incurred,
           oracle_measured_trip_saving_lower_s=mlo,oracle_measured_trip_saving_upper_s=mhi))
    return results


def main():
    episodes=[];turns=[]
    frozen=before_restoration()
    cohorts=sorted({p.parents[1].name for p in frozen})
    for name in cohorts:
        root=ROOT/name
        proofs=read(root/"audit/photo_chains.json")
        for path in (p for p in frozen if p.parents[1]==root):
            receipt=read(path)
            if receipt["state"]!="scored":
                continue
            ep=path.with_name(receipt["method"])
            calls=read_trace_rows(ep/"private/robot/model_calls.jsonl")[0]
            events=read_trace_rows(ep/"private/robot/timeline.jsonl")[0]
            dur=length([(c["dispatched_monotonic_seconds"],c["return_monotonic_seconds"]) for c in calls])
            if abs(dur-sum(meter(c) for c in calls))>1.e-5:
                raise ValueError("Model requests overlap; sum is no longer a lower bound")
            photos=[p for p in proofs if (p["case"],p["method"])==(ep.parent.name,ep.name)]
            observed=compare_turns(name,ep,calls,events,photos)
            turns.extend(observed)
            episodes.append(dict(cohort=name,case=ep.parent.name,method=ep.name,strict=receipt["strict"],time_s=receipt["seconds"],
              actual_requests=len(calls),serial_transport_floor_s=dur,nontransport_wall_s=receipt["seconds"]-dur,
              offered_turns=len(observed),requested_turns=sum(bool(r["actually_requested"]) for r in observed),
              unrequested_excursions=sum(r["unrequested_started"] for r in observed),
              total_post_reply_delays_with_trips_s=sum(r["post_reply_release_delay_s"] or 0. for r in observed)))
    pairs=[]
    bycase=defaultdict(dict)
    for r in episodes:
        bycase[r["case"]][r["method"]]=r
    for case,methods in bycase.items():
        original=methods.get("original-gpt")
        proposed=next((r for k,r in methods.items() if k.endswith("-deferred")),None)
        if not (original and proposed and original["strict"] and proposed["strict"]):
            continue
        target=.8*original["time_s"]
        pairs.append(dict(case=case,original_time_s=original["time_s"],observed_proposed_time_s=proposed["time_s"],
          original_fixed_transport_s=original["serial_transport_floor_s"],proposed_fixed_transport_s=proposed["serial_transport_floor_s"],
          target_wall_s=target,minimum_proposed_transport_to_remove_s=max(0.,proposed["serial_transport_floor_s"]-target),
          minimum_original_transport_to_remove_s=max(0.,original["serial_transport_floor_s"]-target),
          proposed_transport_above_entire_original_s=proposed["serial_transport_floor_s"]-original["time_s"],
          proposed_post_reply_excursion_delay_s=proposed["total_post_reply_delays_with_trips_s"]))
    write_table(DEST/"observed_offer_opportunities.csv",turns)
    write_table(DEST/"fixed_transport_episodes.csv",episodes)
    summary=dict(scored=len(episodes),paired_strict_device_tasks=pairs,ordinary_turns_offering_views=len(turns),
        turns_requesting_a_view=sum(bool(r["actually_requested"]) for r in turns),
        early_excursions=sum(len(r["early_started"]) for r in turns),unrequested_early_excursions=sum(r["unrequested_started"] for r in turns),
        measured_demanded_route_turns=sum(bool(r["admissible_demand"]) and set(r["admissible_demand"])<=set(r["measured_route_seconds"]) for r in turns),
        comparison="Frozen realized prompts, replies and service times in the two scored cohorts. With serial sends, their union is a lower bound even if all motion and local work vanish. Removing or regrouping requests can change both outputs and service. Do not extrapolate the bound to a changed controller.",
        route_scope="One frozen ordinary call. For available options use the real returned demand after dispatch; this is an oracle diagnostic. Observed route time includes acquisition and verified return. An untraveled demanded alternative has unknown cost, at most one service window can be hidden. The separate nominal table uses pre-request trip estimates for unrealized routes. These are not whole-task effects.",
        new_model_requests=0)
    save(DEST/"critical_path.json",summary)
    print(summary)

if __name__=="__main__":
    main()
