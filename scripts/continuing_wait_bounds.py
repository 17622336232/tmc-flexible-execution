"""Observed service censoring and potential trip overrun, without inventing missing answers or times."""
from argparse import ArgumentParser
from collections import Counter
from math import inf
from pathlib import Path
from statistics import mean, median

from experiments.flexible_execution.revision_audit import digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows


ROOT = Path("outputs/protocol-v2-optimization/custom-final")
DEST = Path("outputs/continuing/observed-queries")
NAV = Path("outputs/continuing/local-view-navigation01")


def query_rows(paths):
    for path in sorted(paths):
        checksum = digest(path)
        for row in read_trace_rows(path, allow_truncated=True)[0]:
            inp = row["input"]
            if "bbox_in_attached_photo" not in inp or len(row["image_sha256"]) != 1:
                continue
            if row["instructions_sha256"] != "156ec720e7f3a67903acc205c15d17e27e6779685d6afd00b69054c3b15d75f0":
                raise ValueError("Unexpected frozen visual-question prompt")
            outcome = (row.get("output") or {}).get("outcome")
            code = (row.get("error") or {}).get("code")
            if code == "timeout" and outcome is None:
                state = "right-censored"
            elif outcome in ("positive", "negative", "uncertain") and not code:
                state = "reply"
            else:
                raise ValueError("Unclassified query outcome: " + str((outcome, code)))
            duration = row["return_monotonic_seconds"] - row["dispatched_monotonic_seconds"]
            if duration < 0:
                raise ValueError("Reversed service clock")
            yield dict(log=str(path), log_sha256=checksum, attempt=row["logical_job_attempt"],
                       image_sha256=row["image_sha256"][0], explicit_bbox=inp["bbox_in_attached_photo"] is not None,
                       status=state, outcome=outcome,
                       service_s=duration, wall_s=row["wall_seconds"], effective_timeout_s=row["effective_timeout_seconds"])


def interval(rows, duration):
    """Complete replies give exact overrun. For a timeout the unobserved answer may arrive arbitrarily late."""
    if duration < 0 or not rows:
        raise ValueError("A nonnegative duration and service observations are required")
    exact = [max(0., duration-r["service_s"]) for r in rows if r["status"]=="reply"]
    upper = [max(0., duration-r["service_s"]) for r in rows if r["status"]=="right-censored"]
    lower_mean = sum(exact) / len(rows)
    upper_mean = (sum(exact)+sum(upper)) / len(rows)
    return dict(trip_motion_s=duration, mean_unhidden_motion_lower_s=lower_mean,
                mean_unhidden_motion_upper_s=upper_mean, mean_overlap_lower_s=duration-upper_mean,
                mean_overlap_upper_s=duration-lower_mean,
                demand_probability_break_even_lower=(lower_mean/duration if duration else None),
                demand_probability_break_even_upper=(upper_mean/duration if duration else None))


def analyze(rows):
    seen = [r["service_s"] for r in rows]
    upper = [r["service_s"] if r["status"]=="reply" else inf for r in rows]
    completed = [r for r in rows if r["status"]=="reply"]
    failed = [r for r in rows if r["status"]=="right-censored"]
    upper_median = median(upper)
    return dict(total=len(rows), replies=len(completed), right_censored=len(failed),
                outcome_counts=dict(Counter(r["outcome"] for r in completed)),
                previously_successful_only_median_s=median(r["service_s"] for r in completed),
                measured_timeout_median_s=median(r["service_s"] for r in failed),
                measured_timeout_wall_median_s=median(r["wall_s"] for r in failed),
                effective_deadline_min_s=min(r["effective_timeout_s"] for r in rows),
                effective_deadline_max_s=max(r["effective_timeout_s"] for r in rows),
                effective_deadline_median_s=median(r["effective_timeout_s"] for r in rows),
                underlying_median_lower_s=median(seen), underlying_median_upper_s=upper_median if upper_median<inf else None,
                mean_observed_service_s=mean(seen))


def action_slots(dest):
    rows=[]
    for base in sorted(Path("outputs/continuing").glob("*-dev*")):
        for path in sorted(base.glob("*/*/private/robot/model_calls.jsonl")):
            log=read_trace_rows(path,allow_truncated=True)[0]
            checksum=digest(path)
            for i,r in enumerate(log):
                offers={o["id"] for o in r["input"].get("offered_views",[])}
                if not offers:
                    continue
                decision=r.get("output") or {}
                action=(decision.get("action") or {}).get("type")
                picked=decision.get("use_views")
                if picked is None:
                    picked=[decision.get("use_view")]
                selected=sorted(offers.intersection([s for s in picked if isinstance(s,str)])) if isinstance(picked,list) and len(picked)<=2 else []
                after=log[i+1] if i+1<len(log) else {}
                following=((after.get("output") or {}).get("action") or {}).get("type")
                rows.append(dict(cohort=base.name, case=path.parents[3].name, method=path.parents[2].name,
                        log_sha256=checksum, attempt=r["logical_job_attempt"], offered=len(offers),
                        selected=selected, actual_action=action, following_actual_action=following,
                        supplied_next=[f["image_id"] for f in (after.get("input") or {}).get("recent_images",[]) if f["image_id"].startswith("onboard_side_")]))
    if rows:
        write_table(dest/"offered_action_slots.csv",rows)
    picked=[r for r in rows if r["selected"]]
    return dict(turns_with_offer=len(rows), turns_with_selected_offer=len(picked),
                actions_when_selected=dict(Counter(r["actual_action"] for r in picked)),
                following_ordinary_actions=dict(Counter(r["following_actual_action"] for r in picked)))


def main():
    parser=ArgumentParser()
    parser.add_argument("--output",type=Path,default=DEST)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    source=ROOT.glob("custom-*/*/attempt-*/episode/private/robot/model_calls.jsonl")
    rows=list(query_rows(source))
    write_table(args.output/"bound_query_clocks.csv",rows)
    trips=[]
    for path in sorted(NAV.glob("*/*/navigation.json")):
        n=read(path)
        if n["outward_arrived"] and n["return_arrived"]:
            d=n["outward_s"]+n["return_s"]
            trips.append(dict(case=n["case"],arm=n["arm"],navigation_sha256=digest(path),**interval(rows,d)))
    write_table(args.output/"trip_time_bounds.csv",trips)
    result=dict(service=analyze(rows),explicit_box_service=analyze([r for r in rows if r["explicit_bbox"]]),
        full_frame_service=analyze([r for r in rows if not r["explicit_bbox"]]),
        actual_offered_slots=action_slots(args.output),physical_durations=trips,
        definition="Each actually completed route provides a fixed motion duration evaluated against the archival query clocks. No simultaneous paired camera/service measurement is inferred. A cutoff supplies only a lower bound on time to a hypothetical valid reply, which can be infinite. No missing judgment is imputed.",
        break_even="Illustrative risk-neutral one-trip model: if a later on-demand trip costs the same duration d, and a useful early capture eliminates that trip, expected time advantage equals p*d minus mean uncovered motion. The probability interval is the range of thresholds allowed by censoring, not a measured activation rate. Excludes recognition, route changes and task success.",
        new_model_requests=0)
    save(args.output/"wait_bounds.json",result)
    print(result["service"],result["actual_offered_slots"])
    for t in trips:
        print(t["case"],t["arm"],*[round(t[k],3) for k in ("trip_motion_s","mean_unhidden_motion_lower_s","mean_unhidden_motion_upper_s")])

if __name__=="__main__":
    main()
