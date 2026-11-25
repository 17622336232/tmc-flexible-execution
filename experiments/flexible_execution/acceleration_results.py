"""Independent reports for the accelerated online development cohort."""

from datetime import datetime
from pathlib import Path
from statistics import median, quantiles

from experiments.task_attack.runtime import write_json

from .control import TASK_TERMINAL
from .live_audit import _camera_frames, _rows, _verified_judgment, audit_capture_permissions
from .runner import _hash, read_json


def development_summary(directory):
    catalog = read_json(directory / "pilot_manifest.json")
    protocol = read_json(directory / "run_protocol.json")
    if protocol["catalog_sha256"] != _hash(catalog) or catalog["protocol_version"] != "protocol-v2":
        raise ValueError("Development observations do not match the v2 schedule")
    rows = []
    for path in sorted(directory.glob("*/*/job.json")):
        job = read_json(path)
        if job["state"] not in TASK_TERMINAL:
            continue
        root = Path(job["artifact_dir"])
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
        rows.append({"job": job["id"], "case": job["case"]["id"], "method": job["method"], "state": job["state"],
                     "started_at": job["started_at"], "code": job["code_git_commit"],
                     "model": job["model"], "reasoning_effort": job["reasoning_effort"],
                     "dispatches": job["request_budget"]["reserved"], "seconds": job["charged_seconds"],
                     "worker_wall_seconds": (datetime.fromisoformat(job["finished_at"]) -
                                             datetime.fromisoformat(job["started_at"])).total_seconds(),
                     "cached": sum(row.get("cache_hits", 0) + (row["event"] == "acceptance_cache_hit") for row in events),
                     "source": str(root), "queries": queries, "permit_audit": permission,
                     "controller_status": next((r.get("outcome") for r in reversed(events)
                                                if r["event"] == "report_written"), None)})
    worker = [row["worker_wall_seconds"] for row in rows]
    quantile = quantiles(worker, n=4, method="inclusive") if len(worker) >= 2 else []
    return {"protocol_version": "protocol-v2", "evidence_level": "NATURAL_ONLINE",
            "registered_episodes": sum(len(row["order"]) for row in catalog["cases"]), "recorded_episodes": len(rows),
            "actual_requests": sum(row["dispatches"] for row in rows),
            "worker_median_seconds": median(worker) if worker else None,
            "worker_quartile_seconds": [quantile[0], quantile[-1]] if quantile else None,
            "ours_complete_with_multiple_verified_queries": [
                row["case"] for row in rows if row["method"] == "conditional-a" and row["state"] == "FINISHED_VALID"
                and sum(query["binding_verified"] for query in row["queries"]) >= 2
                and not (row["permit_audit"] or {}).get("violations")],
            "rows": rows}


def update(output_root, development):
    result = development_summary(development)
    root = output_root / "acceleration-v2"
    write_json(root / "development_results.json", result)
    initial = read_json(root / "old_new_job_counts.json")["initial_projections_by_worker"]
    native, public = initial["Gazebo"], initial["ProcTHOR"]
    median_native = result["worker_median_seconds"] if result["recorded_episodes"] >= 4 else native["median_seconds"]
    native_range = result["worker_quartile_seconds"] if result["recorded_episodes"] >= 4 else native["quartile_seconds"]
    times = [360 * gpu + 96 * cpu for gpu, cpu in zip(
        [native_range[0], median_native, native_range[1]],
        [public["quartile_seconds"][0], public["median_seconds"], public["quartile_seconds"][1]])]
    write_json(root / "measured_estimate.json", {
        "model": "one_worker_measured_start_to_finish_by_platform", "sampled_v2_episodes": result["recorded_episodes"],
        "native_sample_uses_v2": result["recorded_episodes"] >= 4,
        "public_measurement_source": "16 matched validation arms", "main_hours_q1_median_q3": [t / 3600 for t in times],
        "optional_additional_gazebo_hours": 72 * median_native / 3600,
        "validated_parallelism": None, "applied_speedup_assumption": 1})
    return result
