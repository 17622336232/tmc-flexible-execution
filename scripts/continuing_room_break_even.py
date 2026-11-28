"""Exact-source comparison of delaying a read versus issuing it during transit; preserve censored calls."""
from collections import defaultdict
from pathlib import Path
import re
from statistics import median

from experiments.flexible_execution.revision_audit import digest,read,save,write_table
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_room_event_calibration import ROOT,CAL


def previous_turns():
    folder=Path("outputs/public-visual/validation-pilot-v1")
    rows=[]
    for p in sorted(folder.rglob("timeline.jsonl")):
        first=p.relative_to(folder).parts[0];site=re.search(r"house-(\d+)",first)
        if not site:continue
        starts={}
        for e in read_trace_rows(p)[0]:
            k=e["step"];kind=e["event"]
            if kind=="action_started":starts[k]=e["monotonic_seconds"]
            elif kind=="action_finished" and e.get("ok") and k in starts:
                rows.append(dict(house=int(site[1]),turn=k,s=e["monotonic_seconds"]-starts.pop(k),log=str(p),sha256=digest(p)))
    return rows


def main():
    original=read(ROOT/"live/summary.json");cal=read(CAL/"summary.json")
    rows=original["cases"]
    repeated=[]
    for b in original["identical_request_bindings"]:
        obs=b["observations"];times=[v["service_s"] for v in obs]
        repeated.append(dict(case=b["case"],input_sha256=b["input_sha256"],count=len(obs),
                             result_available=sum(v["output"] is not None for v in obs),minimum_observed_s=min(times),
                             maximum_observed_s=max(times),has_censored=any(v["output"] is None for v in obs)))
    case="crossed-screens-house-56"
    reports={r["method"]:r for r in rows if r["case"]==case}
    a,b=reports["joint"],reports["serial"]
    if not (a["strict"] and b["strict"] and a["report"]["source_digests"]==b["report"]["source_digests"] and
            a["expected_fields"]==b["expected_fields"]==a["report"]["fields"]==b["report"]["fields"]):
        raise ValueError("Two completed identical camera bindings are required")
    front={v["questions"][0]:v for v in a["bindings"]}
    bundled=next(v for v in b["bindings"] if v["questions"]==["q0","q3"])
    if (front["q0"]["output"]["q0"]!="negative" or front["q1"]["output"]["q1"]!="positive" or
            {k:front[k]["output"][k] for k in ("q0","q3")}!=bundled["output"]):
        raise ValueError("Conditional readings and exact source answers differ")
    saved=front["q0"]["service_s"]+front["q3"]["service_s"]-bundled["service_s"]
    historic=previous_turns()
    target=[r["s"] for r in historic if r["turn"]=="RotateLeft" and r["house"]==56]
    elsewhere=[r["s"] for r in historic if r["turn"]=="RotateLeft" and r["house"]!=56]
    s,g=cal["single_s"],cal["pair_s"]
    threshold=2*s-g
    cal_lift=lambda trip:max(s,trip)+2*s-(trip+s+g)
    motions=[]
    for name,row in reports.items():
        ep=ROOT/"live"/case/name
        e=read_trace_rows(ep/"timeline.jsonl")[0]
        begin=next(x["monotonic_seconds"] for x in e if x["event"]=="action_started")
        # Actual elapsed from starting the outward turn until the new side image exists, including any scheduling.
        end=next(x["monotonic_seconds"] for x in e if x["event"]=="camera_observation" and x["step"]=="side")
        motions.append(dict(method=name,measured_transit_and_capture_s=end-begin,
                            predicted_side_first_gain_on_observed_branch_s=cal_lift(end-begin)))
    cost=read(ROOT/"live/registration.json")["costs"][case]
    prior_trip=cost["outward"]+cost["photo"]
    out=dict(registered_live_sha256=original["source"],identical_request_replications=repeated,new_model_requests=0,
        calibration_single_s=s,calibration_double_s=g,paired_completed_case=case,identical_front_and_side_hashes=True,
        actual_front_two_single_service_s=front["q0"]["service_s"]+front["q3"]["service_s"],
        actual_joint_front_service_s=bundled["service_s"],same_observed_answers=True,saved_front_service_s=saved,
        actual_whole_side_first_minus_front_first_s=b["task_s"]-a["task_s"],
        old_planner_outward_and_shutter_s=prior_trip,
        predicted_side_first_gain_on_observed_branch_at_old_cost_s=cal_lift(prior_trip),
        positive_threshold_trip_s=threshold,observed_task_trips=motions,
        prior_same_house=dict(n=len(target),median_s=median(target) if target else None,below_threshold=sum(x<threshold for x in target)),
        prior_other_houses=dict(n=len(elsewhere),median_s=median(elsewhere) if elsewhere else None,below_threshold=sum(x<threshold for x in elsewhere)),
        formula="For the observed branch front-negative and side-positive, both views fixed and finish at side: front-first costs max(s,t)+2s; side-first costs t+s+b. Terms are equal singleton s, actual side transit with capture t, and paired request b. Numeric calibration implies b>s; side-first wins for t<2s-b. Field outcomes were unavailable to the controller; this is path-specific and excludes startup, controller overhead and timeout.",
        reliability="If independent caller-level failure rates are h1 for a single and h2 for a grouped call, this particular two-call plan is more likely to deliver than three singles when h2 < 1-(1-h1)^2. No failure probabilities are inferred from the two exact-input timeout pairs.")
    write_table(ROOT/"prior_turns.csv",historic);save(ROOT/"causal_branch.json",out)
    print(saved,threshold,out["prior_same_house"],out["prior_other_houses"],motions)


if __name__=="__main__":
    main()
