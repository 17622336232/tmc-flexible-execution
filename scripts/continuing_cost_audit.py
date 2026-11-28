"""Counterfactual bounds for a fixed original request sequence, and all 11 raw call chains."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from experiments.flexible_execution.revision_audit import OUT as PRIOR, digest, read, save, write_table

OUT = Path("outputs/continuing/audit")


def role(payload):
    if "supported_fields" in payload:
        return "compile"
    if "current_image_id" in payload:
        return "ordinary_control"
    if "object_scope" in payload:
        return "locate"
    if "bbox_in_attached_photo" in payload:
        return "bound_judgment"
    if "reference_frame_id" in payload:
        return "material_read"
    if "proposed_report" in payload:
        return "report_check"
    return "other"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, phases, bounds = [], [], []
    for root in (PRIOR, PRIOR / "round3"):
        for receipt in sorted(root.glob("dev-*/*.receipt.json")):
            data = read(receipt)
            if data["state"] != "scored":
                raise ValueError("Historical receipt is not closed")
            ep = receipt.with_name(receipt.name.replace(".receipt.json",""))
            where = dict(block="third" if root != PRIOR else "first", case=ep.parent.name, method=ep.name)
            path = ep / "private/robot/model_calls.jsonl"
            calls = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(calls) == data["sends"] and digest(ep / "result.json") == data["result_sha256"]
            times = defaultdict(list)
            for c in calls:
                name = role(c["input"])
                times[name].append(c["wall_seconds"])
                rows.append(dict(**where, role=name, attempt=c["logical_job_attempt"], seconds=c["wall_seconds"],
                                 images=len(c["image_sha256"]), source_log=digest(path), response="output" in c,
                                 error=(c.get("error") or {}).get("code")))
            phases.extend(dict(**where, role=k, sends=len(v), total_s=sum(v), median_s=median(v))
                          for k,v in times.items())
            if ep.name != "original-gpt":
                continue
            clock = read(ep / "job_clock.json")["elapsed_seconds"]
            transport = [(c["dispatched_monotonic_seconds"], c["return_monotonic_seconds"]) for c in calls]
            merged = []
            for start,end in sorted(transport):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1],end)
                else:
                    merged.append([start,end])
            in_request = sum(b-a for a,b in merged)
            action = sum(e["seconds"] for e in read(ep/"clean_timing.json")["intervals"].get("action",[]))
            # With one serial transport, every observed request still has to finish.
            ceiling = max(0.,clock-in_request)
            bounds.append(dict(**where, total_s=clock, request_union_s=in_request, logged_motion_s=action,
                               all_nontransport_upper_s=ceiling, fixed_sequence_ceiling_pct=100*ceiling/clock,
                               additional_request_time_to_eliminate_for_20pct_s=max(0.,.2*clock-ceiling)))
    assert len(rows) == 93 and len(bounds) == 2
    write_table(OUT/"request_stages.csv",rows)
    write_table(OUT/"stage_costs.csv",phases)
    write_table(OUT/"fixed_sequence_bounds.csv",bounds)
    save(OUT/"summary.json",dict(new_requests=0, historical_episodes=11, by_role=dict(Counter(r["role"] for r in rows)),
                                 original_fixed_sequence=bounds, scope="Fixed observed transport durations, serialized; no inference is parallelized or removed. Overlap of motion cannot exceed the total outside all call windows."))
    print([(r["case"],round(r["fixed_sequence_ceiling_pct"],2),round(r["additional_request_time_to_eliminate_for_20pct_s"],2)) for r in bounds])


if __name__ == "__main__":
    main()
