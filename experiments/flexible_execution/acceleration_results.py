"""Independent reports for the accelerated online development cohort."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median, quantiles

from experiments.task_attack.runtime import write_json

from .control import TASK_TERMINAL
from .live_audit import _camera_frames, _rows, _verified_judgment, audit_capture_permissions
from .runner import _hash, read_json


def _attempt_history(directory):
    ledger = directory.parent / "control/jobs.jsonl"
    closed = defaultdict(dict)
    unreadable = 0
    if ledger.is_file():
        with ledger.open("rb") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    unreadable += 1
                    continue
                if (row.get("stage") == "acceleration-v2-development" and row.get("attempt")
                        and row.get("started_at") and row.get("finished_at")):
                    # A later budget-only transition can close the logical job
                    # without launching another worker.
                    closed[row["id"]].setdefault(row["attempt"], row)
    return closed, unreadable


def _wall_seconds(attempt):
    return (datetime.fromisoformat(attempt["finished_at"]) -
            datetime.fromisoformat(attempt["started_at"])).total_seconds()


def development_summary(directory):
    catalog = read_json(directory / "pilot_manifest.json")
    protocol = read_json(directory / "run_protocol.json")
    if protocol["catalog_sha256"] != _hash(catalog) or catalog["protocol_version"] != "protocol-v2":
        raise ValueError("Development observations do not match the v2 schedule")
    history, unreadable = _attempt_history(directory)
    rows = []
    for path in sorted(directory.glob("*/*/job.json")):
        job = read_json(path)
        if job["state"] not in TASK_TERMINAL:
            continue
        root = Path(job["artifact_dir"])
        history[job["id"]].setdefault(job["attempt"], job)
        attempts = []
        previous_requests = 0
        for _, attempt in sorted(history[job["id"]].items()):
            cumulative_requests = attempt["request_budget"]["reserved"]
            attempts.append({"number": attempt["attempt"], "state": attempt["state"],
                             "worker_wall_seconds": _wall_seconds(attempt),
                             "observed_dispatches": cumulative_requests - previous_requests,
                             "reason": (attempt.get("result") or {}).get("reason"),
                             "source": attempt["artifact_dir"]})
            previous_requests = cumulative_requests
        robot = root / "private/robot"
        timeline, calls = robot / "timeline.jsonl", robot / "model_calls.jsonl"
        events = _rows(timeline, allow_truncated=job["state"] == "TIMEOUT") if timeline.is_file() else []
        logs = _rows(calls, allow_truncated=job["state"] == "TIMEOUT") if calls.is_file() else []
        frames = _camera_frames(robot) if robot.is_dir() else {}
        queries = []
        for row in events:
            if row["event"] != "pending_strategy":
                continue
            slice_for_step = [event for event in events if event["step"] == row["step"]]
            reply, request = _verified_judgment(slice_for_step, logs, frames)
            queries.append({"id": row["query"], "step": row["step"], "policy": row["policy"],
                            "selected": row["selection"], "binding_verified": bool(reply and request),
                            "outcome": reply.get("outcome"), "model_wait_seconds": request.get("wall_seconds"),
                            "decision_at": row["monotonic_seconds"], "applied_at": reply.get("monotonic_seconds")})
        permission = audit_capture_permissions(root, allow_truncated=job["state"] == "TIMEOUT")
        prior_violations = 0
        for attempt in attempts[:-1]:
            audit = audit_capture_permissions(Path(attempt["source"]), allow_truncated=True)
            prior_violations += (audit or {}).get("violations", 0)
        measured_all_attempts = len(attempts) == job["attempt"]
        rows.append({"job": job["id"], "case": job["case"]["id"], "method": job["method"], "state": job["state"],
                     "started_at": job["started_at"], "code": job["code_git_commit"],
                     "model": job["model"], "reasoning_effort": job["reasoning_effort"],
                     "dispatches": job["request_budget"]["reserved"], "seconds": job["charged_seconds"],
                     "worker_wall_seconds": sum(a["worker_wall_seconds"] for a in attempts),
                     "all_attempt_timings_available": measured_all_attempts, "attempts": attempts,
                     "cached": sum(row.get("cache_hits", 0) + (row["event"] == "acceptance_cache_hit") for row in events),
                     "source": str(root), "queries": queries, "permit_audit": permission,
                     "prior_attempt_permission_violations": prior_violations,
                     "controller_status": next((r.get("outcome") for r in reversed(events)
                                                if r["event"] == "report_written"), None)})
    worker = [row["worker_wall_seconds"] for row in rows if row["all_attempt_timings_available"]]
    quantile = quantiles(worker, n=4, method="inclusive") if len(worker) >= 2 else []
    return {"protocol_version": "protocol-v2", "evidence_level": "NATURAL_ONLINE",
            "registered_episodes": sum(len(row["order"]) for row in catalog["cases"]), "recorded_episodes": len(rows),
            "actual_requests": sum(row["dispatches"] for row in rows),
            "all_closed_attempts_requests": sum(attempts[max(attempts)]["request_budget"]["reserved"]
                                                for attempts in history.values()),
            "all_closed_attempts_worker_seconds": sum(_wall_seconds(a) for runs in history.values()
                                                       for a in runs.values()),
            "worker_timing_episode_count": len(worker), "unreadable_control_rows": unreadable,
            "worker_median_seconds": median(worker) if worker else None,
            "worker_quartile_seconds": [quantile[0], quantile[-1]] if quantile else None,
            "ours_complete_with_multiple_verified_queries": [
                row["case"] for row in rows if row["method"] == "conditional-a" and row["state"] == "FINISHED_VALID"
                and sum(query["binding_verified"] for query in row["queries"]) >= 2
                and row["permit_audit"] is not None and not row["permit_audit"]["violations"]
                and not row["prior_attempt_permission_violations"]],
            "rows": rows}


def update(output_root, development):
    result = development_summary(development)
    root = output_root / "acceleration-v2"
    write_json(root / "development_results.json", result)
    initial = read_json(root / "old_new_job_counts.json")["initial_projections_by_worker"]
    native, public = initial["Gazebo"], initial["ProcTHOR"]
    use_v2 = result["worker_timing_episode_count"] >= 4
    median_native = result["worker_median_seconds"] if use_v2 else native["median_seconds"]
    native_range = result["worker_quartile_seconds"] if use_v2 else native["quartile_seconds"]
    times = [360 * gazebo + 96 * procthor for gazebo, procthor in zip(
        [native_range[0], median_native, native_range[1]],
        [public["quartile_seconds"][0], public["median_seconds"], public["quartile_seconds"][1]])]
    write_json(root / "measured_estimate.json", {
        "model": "one_worker_measured_start_to_finish_by_platform", "sampled_v2_episodes": result["recorded_episodes"],
        "native_sample_uses_v2": use_v2, "fully_timed_v2_episodes": result["worker_timing_episode_count"],
        "public_measurement_source": "16 matched validation arms", "main_hours_q1_median_q3": [t / 3600 for t in times],
        "optional_additional_gazebo_hours": 72 * median_native / 3600,
        "validated_parallelism": None, "applied_speedup_assumption": 1})
    return result
