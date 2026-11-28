"""Audit all formal request logs, including attempts ending before a model reply."""
import csv
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from experiments.flexible_execution.revision_audit import BASE, OUT, digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows


def main():
    details=[]
    raw_dispatches=0
    for domain in ("custom","public"):
        report=BASE/"reports"/(domain+"-final")/"episodes.csv"
        with report.open() as f:
            episodes=list(csv.DictReader(f))
        for row in episodes:
            arm=BASE/(domain+"-final")/row["case"]/row["method"]
            ledger=BASE/"control/requests"/(row["logical_id"]+".jsonl")
            reservations=read_trace_rows(ledger,allow_truncated=True)[0] if ledger.exists() else []
            # Repetition requires a previous successful return in the same physical attempt.
            for attempt in sorted(arm.glob("attempt-*")):
                ep=attempt/"episode"
                root=ep/"private/robot" if domain=="custom" else ep
                path=root/"model_calls.jsonl"
                calls,cut=read_trace_rows(path,allow_truncated=True) if path.is_file() else ([],False)
                complete=set()
                exact_repeats=0
                repeated_judgments=0
                known_views=set()
                image_uses=0
                tokens=0
                roles=Counter()
                for call in calls:
                    payload=call.get("input",{})
                    photos=call.get("image_sha256") or []
                    key=sha256(json.dumps([call.get("instructions_sha256"),payload,photos,call.get("model"),
                                          call.get("reasoning_effort")],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                    role="compile" if "supported_fields" in payload else "controller" if "current_image_id" in payload else (
                         "judgment" if "bbox_in_attached_photo" in payload else "material" if "reference_frame_id" in payload else "other")
                    roles[role]+=1
                    if key in complete:
                        exact_repeats+=1
                        repeated_judgments+=role in ("judgment","material")
                    if isinstance(call.get("output"),dict):
                        complete.add(key)
                    known_views.update(photos)
                    image_uses+=len(photos)
                    tokens+=isinstance(call.get("usage"),dict)
                ended=sum(isinstance(c.get("return_monotonic_seconds"),(int,float)) for c in calls)
                timed=sum(c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"] for c in calls
                          if isinstance(c.get("return_monotonic_seconds"),(int,float)) and isinstance(c.get("dispatched_monotonic_seconds"),(int,float)))
                raw_dispatches+=len(calls)
                details.append(dict(domain=domain,case=row["case"],method=row["method"],attempt=attempt.name,
                     state_of_logical_job=row["state"],logical_job=row["logical_id"],logical_job_sends=row["actual_dispatches"],
                     logical_job_reservations=len(reservations),call_records=len(calls),records_with_return=ended,
                     returned_transport_s=timed,unique_image_hashes=len(known_views),image_inputs_including_reuse=image_uses,
                     exact_input_repeated_after_success=exact_repeats,repeated_bound_judgment_or_read=repeated_judgments,
                     tokens_with_usage_record=tokens,roles=dict(roles),partial_log_tail=cut,log_sha256=digest(path) if path.exists() else None))
    assert len(details)==457 and len({x["logical_job"] for x in details})==456
    write_table(OUT/"all_raw_attempts.csv",details)
    summary={k:sum(r[k] for r in details) for k in ("call_records","records_with_return","exact_input_repeated_after_success",
               "repeated_bound_judgment_or_read","tokens_with_usage_record")}
    summary.update(attempts=len(details),logical_jobs=456,recorded_calls=raw_dispatches,
                   note="Image reuse across different questions is not an avoidable request. A missing response is never a zero-duration inference.")
    save(OUT/"raw_cost_summary.json",summary)
    print(json.dumps(summary))


if __name__=="__main__":
    main()
