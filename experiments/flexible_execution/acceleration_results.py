"""Independent reports for the accelerated online development cohort."""

import csv
import json
from collections import defaultdict
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import median, quantiles, stdev

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


def _new_dispatches(job):
    parent = job.get("budget_parent") or {}
    inherited = parent.get("prior_requests", 0) if job.get("budget_class") == "original_development" else 0
    return job["request_budget"]["reserved"] - inherited


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
        parent = job.get("budget_parent") if job.get("budget_class") == "original_development" else None
        inherited_requests = (parent or {}).get("prior_requests", 0)
        previous_requests = inherited_requests
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
                            "decision_at": row["monotonic_seconds"], "applied_at": reply.get("monotonic_seconds"),
                            "sent_at": request.get("dispatched_monotonic_seconds"),
                            "returned_at": request.get("return_monotonic_seconds")})
        permission = audit_capture_permissions(root, allow_truncated=job["state"] == "TIMEOUT")
        prior_violations = 0
        for attempt in attempts[:-1]:
            audit = audit_capture_permissions(Path(attempt["source"]), allow_truncated=True)
            prior_violations += (audit or {}).get("violations", 0)
        measured_all_attempts = len(attempts) == job["attempt"] - (parent or {}).get("prior_attempts", 0)
        handoff = next((r for r in events if r["event"] == "full_inspection_started"), {})
        first_controller = next((c["input"] for c in logs if "current_image_id" in c.get("input", {})), {})
        permission_steps = {row["step"] for row in (permission or {}).get("actions", []) if row["permitted"]}
        early_actions = [event for event in events if event["event"] == "action_started"
                         and event["step"] in permission_steps and any(
                             q["sent_at"] is not None and q["returned_at"] is not None
                             and q["sent_at"] <= event["monotonic_seconds"] < q["returned_at"] for q in queries)]
        quiet = [q for q in queries if q["binding_verified"] and not q["selected"]]
        rows.append({"job": job["id"], "case": job["case"]["id"], "method": job["method"], "state": job["state"],
                     "started_at": job["started_at"], "code": job["code_git_commit"],
                     "model": job["model"], "reasoning_effort": job["reasoning_effort"],
                     "dispatches": job["request_budget"]["reserved"] - inherited_requests, "seconds": job["charged_seconds"],
                     "execution_revision": job.get("execution_revision", "original"),
                     "budget_class": job.get("budget_class"), "inherited_dispatches": inherited_requests,
                     "inherited_task_seconds": (parent or {}).get("prior_charged_seconds", 0.),
                     "worker_wall_seconds": sum(a["worker_wall_seconds"] for a in attempts),
                     "all_attempt_timings_available": measured_all_attempts, "attempts": attempts,
                     "cached": sum(row.get("cache_hits", 0) + (row["event"] == "acceptance_cache_hit") for row in events),
                     "source": str(root), "queries": queries, "permit_audit": permission,
                     "prior_attempt_permission_violations": prior_violations,
                     "seeded_records": handoff.get("seeded_records"), "first_controller_records":
                     len(first_controller.get("records", [])) if first_controller else None,
                     "permitted_actions_started_before_return": len(early_actions),
                     "audited_late_permission": any(e["event"] == "permission_checked" and e.get("early") is False
                                                    and e.get("after") == ["positive"] and e.get("source") for e in events),
                     "ordinary_wait_or_no_gain": bool(quiet),
                     "uncertain_or_evidence_rejected": any(q["outcome"] == "uncertain" for q in queries)
                     or any(e["event"] in ("evidence_rejected", "query_rejected", "pending_query_deferred") for e in events),
                     "controller_status": next((r.get("outcome") for r in reversed(events)
                                                if r["event"] == "report_written"), None)})
    worker = [row["worker_wall_seconds"] for row in rows if row["all_attempt_timings_available"]]
    quantile = quantiles(worker, n=4, method="inclusive") if len(worker) >= 2 else []
    return {"protocol_version": "protocol-v2", "evidence_level": "NATURAL_ONLINE",
            "registered_episodes": sum(len(row["order"]) for row in catalog["cases"]), "recorded_episodes": len(rows),
            "actual_requests": sum(row["dispatches"] for row in rows),
            "all_closed_attempts_requests": sum(_new_dispatches(attempts[max(attempts)]) for attempts in history.values()),
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
    root = development.parent
    write_json(root / "development_results.json", result)
    initial = read_json(output_root / "acceleration-v2/old_new_job_counts.json")["initial_projections_by_worker"]
    native, public = initial["Gazebo"], initial["ProcTHOR"]
    use_v2 = result["worker_timing_episode_count"] >= 4
    revised = root.name == "protocol-v2-optimization"
    native_source = "revised_development" if revised and use_v2 else "original_v2_development" if use_v2 else "legacy"
    if use_v2:
        median_native, native_range = result["worker_median_seconds"], result["worker_quartile_seconds"]
    elif revised:
        historical = development_summary(output_root / "acceleration-v2/development")
        median_native, native_range = historical["worker_median_seconds"], historical["worker_quartile_seconds"]
        native_source = "original_v2_development"
    else:
        median_native, native_range = native["median_seconds"], native["quartile_seconds"]
    times = [360 * gazebo + 96 * procthor for gazebo, procthor in zip(
        [native_range[0], median_native, native_range[1]],
        [public["quartile_seconds"][0], public["median_seconds"], public["quartile_seconds"][1]])]
    estimate = {
        "model": "one_worker_measured_start_to_finish_by_platform", "sampled_v2_episodes": result["recorded_episodes"],
        "native_sample_uses_v2": use_v2, "fully_timed_v2_episodes": result["worker_timing_episode_count"],
        "native_measurement_source": native_source,
        "public_measurement_source": "16 matched validation arms", "main_hours_q1_median_q3": [t / 3600 for t in times],
        "optional_additional_gazebo_hours": 72 * median_native / 3600,
        "validated_parallelism": None, "applied_speedup_assumption": 1}
    write_json(root / "measured_estimate.json", estimate)
    if revised:
        write_json(root / "cost_projection.json", _cost_projection(output_root, result, estimate))
        write_json(root / "development_power.json", _power(development, result))
    return result


def _method_times(rows):
    methods = defaultdict(list)
    for row in rows:
        methods[row["method"]].append(row)
    return {method: {"recorded_episodes": len(group), "source_cases": len({row["case"] for row in group}),
                     "median_worker_seconds": median(row["worker_wall_seconds"] for row in group),
                     "median_task_seconds": median(row["seconds"] for row in group),
                     "median_dispatches": median(row["dispatches"] for row in group)}
            for method, group in methods.items()}


def _cost_projection(output_root, development, estimate):
    original = output_root / "acceleration-v2"
    public = output_root / "public-visual/validation-pilot-v1"
    old = development_summary(original / "development")
    public_rows = []
    for path in public.glob("*/*/job.json"):
        job = read_json(path)
        if job["state"] in TASK_TERMINAL:
            public_rows.append({"method": job["method"], "case": job["case"]["group_id"],
                                "worker_wall_seconds": _wall_seconds(job), "seconds": job["charged_seconds"],
                                "dispatches": job["request_budget"]["reserved"]})
    worker = estimate["main_hours_q1_median_q3"]
    audit = read_json(output_root / "protocol-v2-optimization/audit/current_v2_timeline.json")
    with (output_root / "protocol-v2-optimization/audit/current_v2_attempts.csv").open(
            encoding="utf8", newline="") as handle:
        original_timing = list(csv.DictReader(handle))
    original_attempts, _ = _attempt_history(original / "development")
    calendar_gaps = []
    for attempts in original_attempts.values():
        ordered = [attempts[n] for n in sorted(attempts)]
        for before, after in zip(ordered, ordered[1:]):
            if before["state"] == "BLOCKED_EXTERNAL":
                gap = (datetime.fromisoformat(after["started_at"]) -
                       datetime.fromisoformat(before["finished_at"])).total_seconds()
                calendar_gaps.append({"condition": "external_limit_then_later_restart", "seconds": gap,
                                      "source": before["artifact_dir"]})
    return {"execution_revision": "evidence-handoff-r1", "one_worker": True,
            "natural_main_jobs": 456, "natural_main_dispatch_ceiling": 10944, "optional_dispatches_registered": 0,
            "original_gazebo_by_method": _method_times(old["rows"]),
            "revised_gazebo_by_method": _method_times(development["rows"]),
            "public_validation_by_method": _method_times(public_rows),
            "revised_timed_case_count": len({row["case"] for row in development["rows"]
                                             if row["all_attempt_timings_available"]}),
            "worker_active_hours_quartile_scenarios": worker,
            "active_projection_sources": {"gazebo": estimate["native_measurement_source"], "public_validation": 16},
            "original_setup_seconds_observed": sum(float(row["initialization_seconds"] or 0) for row in original_timing),
            "original_setup_samples": sum(bool(row["initialization_seconds"]) for row in original_timing),
            "original_post_task_seconds_including_score_and_teardown": sum(
                float(row["post_task_worker_seconds"] or 0) for row in original_timing),
            "original_active_worker_dispatches_per_hour": 3600 * audit["dispatches"] / audit["elapsed_worker_seconds"],
            "revised_development_active_worker_seconds": development["all_closed_attempts_worker_seconds"],
            "revised_development_actual_dispatches": development["all_closed_attempts_requests"],
            "observed_external_restart_gaps": calendar_gaps, "verified_quota_reset_schedule": None,
            "available_fraction_scenarios": [{"assumed_active_fraction": value,
                                              "illustrative_median_calendar_hours": worker[1] / value}
                                             for value in (1.0, .75, .5)],
            "calendar_range_available": False, "scope": "Service availability scenarios are assumptions and not forecasts"}


def _power(directory, result):
    catalog = read_json(directory / "pilot_manifest.json")
    deadline = {row["id"]: row["episode_timeout_seconds"] for row in catalog["cases"]}
    groups = {row["id"]: row.get("group_id", row["id"]) for row in catalog["cases"]}
    recorded = {(row["case"], row["method"]): row for row in result["rows"]}
    differences = defaultdict(list)
    for case, limit in deadline.items():
        a = recorded.get((case, "conditional-a"))
        planner = recorded.get((case, "conditional-planner"))
        if a and planner:
            a_cost = a["seconds"] if a["state"] == "FINISHED_VALID" else limit
            planner_cost = planner["seconds"] if planner["state"] == "FINISHED_VALID" else limit
            differences[groups[case]].append(a_cost - planner_cost)
    paired = {group: sum(values) / len(values) for group, values in differences.items()}
    observed_sd = stdev(paired.values()) if len(paired) > 1 else None
    return {"primary_domain": "custom", "primary_reference": "conditional-planner", "alpha_two_sided": .05,
            "design_power_target": .8, "planned_independent_groups": 72,
            "matched_development_groups": len(paired), "paired_development_seconds": paired,
            "paired_sd_seconds": observed_sd, "planning_mde_normal_approx_seconds":
            2.8 * observed_sd / sqrt(72) if observed_sd is not None else None,
            "small_development_sample": len(paired) < 10,
            "assumed_sd_sensitivity": [{"sd_seconds": sd, "planning_mde_seconds": 2.8 * sd / sqrt(72)}
                                       for sd in (60, 120, 180)],
            "scope": "Pre-holdout planning, with the normal approximation and assumptions reported separately"}
