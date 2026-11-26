"""Rebuild a zero-request cost inventory from the retained development logs."""

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median, quantiles

from experiments.task_attack.records import facts, object_id
from experiments.task_attack.runtime import write_json

from .runner import _hash, read_json
from .timeline import read_trace_rows


def _read(path):
    return read_json(path) if path.is_file() else {}


def _rows(path):
    return read_trace_rows(path, allow_truncated=True)[0] if path.is_file() else []


def _union(spans):
    right, length = float("-inf"), 0.
    for start, end in sorted(spans):
        length += max(0., end - max(start, right))
        right = max(right, end)
    return length


def _purpose(call):
    payload = call.get("input", {})
    if not isinstance(payload, dict):
        return "other"
    if "current_image_id" in payload:
        return "action_and_reading"
    if set(payload) == {"authorized_task"} or "supported_fields" in payload:
        return "task_parse"
    if "onboard_image_id" in payload:
        return "localize"
    if "proposed_report" in payload or "supplied_images" in payload:
        return "acceptance"
    if "claims" in payload and "frame" not in payload:
        return "acceptance"
    if "question" in payload or payload.get("frame") in ("front", "side"):
        return "visual_reading"
    if "instruction" in payload and not call.get("image_sha256"):
        return "task_parse_or_report"
    return "other"


def _spans(events):
    starts, phases = {}, {}
    for event in events:
        name = event["event"]
        kind, _, phase = name.rpartition("_")
        if kind not in ("model_request", "action", "search", "observation"):
            continue
        key = kind, event["step"]
        if phase == "started":
            starts[key] = event["monotonic_seconds"]
        elif phase == "finished" and "ok" in event and key in starts:
            phases.setdefault(kind, []).append((starts.pop(key), event["monotonic_seconds"]))
    return phases, len(starts)


def _field_map(records):
    return {(str(object_id(record)).casefold(), field): value for record in (records or [])
            for field, value in facts(record).items() if object_id(record) and value is not None}


def _attempt(job, launch, root, terminal, failures):
    public = bool(job.get("public_runtime"))
    robot = root if public else root / "private/robot"
    events = _rows(robot / "timeline.jsonl")
    calls = [r for r in _rows(robot / "model_calls.jsonl") if "logical_job_attempt" in r]
    spans, unclosed = _spans(events)
    motion = spans.get("action", [])
    waiting = spans.get("model_request", [])
    phases = Counter()
    fingerprints = Counter()
    no_new_acceptance = 0
    reviewed = set()
    errors = Counter()
    for call in calls:
        phase = _purpose(call)
        images = call.get("image_sha256") or []
        phases[phase + "_requests"] += 1
        phases[phase + "_model_seconds"] += call.get("wall_seconds", 0)
        fingerprints[_hash([call.get("instructions_sha256"), call.get("input"), images,
                            call.get("model"), call.get("reasoning_effort")])] += 1
        if phase == "acceptance" and images:
            no_new_acceptance += set(images) <= reviewed
            reviewed.update(images)
        if call.get("error"):
            errors[call["error"].get("code") or call["error"].get("type") or "unknown"] += 1
    clock = _read(root / "job_clock.json")
    launched = launch.get("started_at")
    finished = terminal.get("finished_at")
    started = datetime.fromisoformat(launched).timestamp() if launched else None
    ended = datetime.fromisoformat(finished).timestamp() if finished else None
    task_start, task_seconds = clock.get("started_wall"), clock.get("elapsed_seconds")
    if task_seconds is None and terminal:
        task_seconds = (terminal.get("result") or {}).get("charged_seconds")
    coverage = _read(root / "clean_coverage.json")
    decisions = _rows(robot / "decisions.jsonl")
    recorded = [row for entry in decisions if not entry.get("late_not_executed")
                for decision in [entry.get("decision") or {}]
                for row in [*(decision.get("records") or []), *((decision.get("report") or {}).get("records") or [])]]
    mentions = Counter((str(object_id(row)).casefold(), field) for row in recorded for field in facts(row))
    case = job["case"]
    identity = dict(platform="ProcTHOR" if public else "Gazebo", case=case["id"], method=job["method"],
                    family=case["family"], job_id=job["id"], attempt=launch.get("attempt", job.get("attempt")),
                    cohort=job["stage"])
    controller = [call for call in calls if _purpose(call) == "action_and_reading" and isinstance(call.get("output"), dict)]
    dropped = 0
    for before, after in zip(controller, controller[1:]):
        previous = _field_map(before["input"].get("records"))
        added = _field_map(before["output"].get("records"))
        current = _field_map(after["input"].get("records"))
        updated = {obj for obj, _ in added}
        overwritten = {key for key in previous if key[0] in updated and key not in added and key not in current}
        dropped += len(overwritten)
        for obj, field in sorted(overwritten):
            failures.append({**identity, "object": obj, "field": field, "finding": "complementary_field_disappeared",
                             "raw_index": str(root)})
    for obj, fields in coverage.get("missing_fields_by_object", {}).items():
        for field in fields:
            count = mentions[obj.casefold(), field]
            failures.append({**identity, "object": obj, "field": field, "model_record_mentions": count,
                             "finding": "record_requires_source_or_value_review" if count else "no_field_record",
                             "raw_index": str(root)})
    if errors:
        for code, count in errors.items():
            failures.append({**identity, "finding": code, "count": count, "raw_index": str(root)})
    dispatched = [call for call in calls if isinstance(call.get("dispatched_monotonic_seconds"), (int, float))]
    prefix = next((e["monotonic_seconds"] for e in events if e["event"] == "full_inspection_started"), None)
    front = [v for group in spans.values() for start, end in group for v in (start, end)]
    front_span = prefix - min(front) if prefix is not None and front else None
    timings = {kind + "_interval_seconds": sum(b - a for a, b in group) for kind, group in spans.items()}
    cpu = _read(root.parent / "worker_cpu.json")
    lengths = {key + "_json_utf8_bytes": sum(len(json.dumps(call[key], ensure_ascii=False).encode("utf8"))
                                             for call in calls if key in call) for key in ("input", "output")}
    return {**identity, "model": job.get("model"), "effort": job.get("reasoning_effort"),
            "code_git_commit": launch.get("code_git_commit", job.get("code_git_commit")), "started_at": launched,
            "terminal_state": terminal.get("state"), "dispatches_with_log": len(calls),
            "usage_token_samples": sum(bool(c.get("usage")) for c in calls),
            "local_worker_cpu_seconds": cpu["user_seconds"] + cpu["system_seconds"] if cpu else None,
            "complementary_field_disappearances": dropped,
            "model_transport_seconds": sum(c.get("wall_seconds", 0) for c in calls),
            "repeat_exact_requests": sum(amount - 1 for amount in fingerprints.values()),
            "acceptance_without_new_image_hash": no_new_acceptance,
            "matched_frontend_cost_still_sent": sum(e.get("alignment") == "matched" for e in events),
            "model_errors": sum(errors.values()), "unclosed_intervals": unclosed,
            "model_action_overlap_seconds": _union(waiting) + _union(motion) - _union(waiting + motion),
            "observed_interval_union_seconds": _union([s for group in spans.values() for s in group]),
            "worker_wall_seconds": ended - started if ended is not None and started is not None else None,
            "task_seconds": task_seconds, "setup_seconds": task_start - started if task_start and started else None,
            "cleanup_seconds": max(0., ended - task_start - task_seconds)
            if ended and task_start and task_seconds is not None else None,
            "prefix_seconds": front_span, "prefix_fraction": front_span / task_seconds
            if front_span is not None and task_seconds else None,
            "prefix_query_submissions": sum(e["event"] == "query_submitted" for e in events),
            "controller_requests_after_handoff": sum(c["dispatched_monotonic_seconds"] >= prefix for c in dispatched)
            if prefix is not None and dispatched else None,
            "later_strategy_checks": sum(e["event"] == "pending_strategy" for e in events),
            "controller_report_claim": next((e.get("outcome") for e in reversed(events) if e["event"] ==
                                             "report_written"), None),
            "independent_coverage_pass": coverage.get("complete"), "raw_index": str(root), **phases, **timings, **lengths}


def _csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def audit(output_root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for cohort in (output_root / "natural-pilot", output_root / "public-visual/validation-pilot-v1"):
        for path in sorted(cohort.rglob("job.json")):
            job = read_json(path)
            if not job.get("artifact_dir") or job["state"] == "RUNNING":
                continue
            launches = sorted(path.parent.glob("attempt-*/launch.json"))
            if not launches:
                rows.append(_attempt(job, job, Path(job["artifact_dir"]), job, failures))
            for file in launches:
                launch = read_json(file)
                terminal = job if launch["attempt"] == job["attempt"] else {}
                rows.append(_attempt(job, launch, file.parent / "episode", terminal, failures))
    _csv(destination / "cost_breakdown.csv", rows)
    _csv(destination / "failure_groups.csv", failures)
    platforms = {}
    for platform, main in (("Gazebo", 360), ("ProcTHOR", 96)):
        sample = [r["worker_wall_seconds"] for r in rows if r["platform"] == platform and
                  r["terminal_state"] in {"FINISHED_VALID", "FINISHED_TASK_FAILURE", "TIMEOUT"} and
                  r["worker_wall_seconds"] is not None]
        q = quantiles(sample, n=4, method="inclusive")
        platforms[platform] = {"timed_worker_launches": len(sample), "median_seconds": median(sample),
                               "quartile_seconds": [q[0], q[-1]], "planned_main_arms": main,
                               "serial_hours_q1_median_q3": [main * t / 3600 for t in (q[0], median(sample), q[-1])]}
    old = {"calibration": 144, "online_selection": 768, "custom_final": 9600, "public_final": 800}
    new = {"first_development_max": 12, "specific_repair_rounds_max": 2, "repair_arms_per_round_max": 6,
           "custom_main": 360, "public_main": 96, "optional_ablations_max": 72}
    result = {"protocol_version": "protocol-v2", "evidence_level": "TRACE_SUPPORTED",
              "actual_new_model_requests": 0, "saved_worker_attempts": len(rows),
              "logged_dispatches": sum(r["dispatches_with_log"] for r in rows),
              "exact_duplicate_requests": sum(r["repeat_exact_requests"] for r in rows),
              "complementary_field_disappearances": sum(r["complementary_field_disappearances"] for r in rows),
              "acceptance_without_new_images": sum(r["acceptance_without_new_image_hash"] for r in rows),
              "old": old, "old_total": sum(old.values()), "new": new, "new_main": 456, "new_final_ceiling": 528,
              "new_final_request_ceiling": 12672, "initial_projections_by_worker": platforms,
              "primary": {"reference": "PLANNER", "metric": "deadline_restricted_complete_task_seconds", "alpha": .05}}
    write_json(destination / "old_new_job_counts.json", result)
    return result
