"""Rebuild paired development metrics from saved arm outcomes and event ledgers."""

import csv
import json
from collections import Counter
from hashlib import sha256
from math import isfinite
from pathlib import Path
from random import Random
from statistics import fmean, mean

from experiments.task_attack.records import object_id
from experiments.task_attack.runtime import write_json

from .candidate_ranking import RANKED_METHODS, RETRIEVAL_METHOD
from .clean_coverage import audit_run
from .custom_schedule import FINAL_ROLE as CUSTOM_FINAL_ROLE
from .live_audit import _camera_frames, _rows, _verified_judgment, audit_capture_permissions
from .pilot_schedule import _read, _valid_preflight
from .public_schedule import FINAL_ROLE as PUBLIC_FINAL_ROLE
from .runner import _hash
from .timeline import read_trace_rows, summarize


RECORDED = {"FINISHED_VALID", "FINISHED_TASK_FAILURE", "TIMEOUT"}
MECHANISM_COUNTS = ("conditional_moves_started_before_bound_reply", "conditional_moves_finished_before_bound_reply",
                    "conditional_actual_photos_from_early_moves", "conditional_photos_completed_before_bound_reply",
                    "conditional_early_materials_used_in_report", "conditional_early_materials_without_report_use")


def _attempt_roots(arm, final):
    return sorted({path.resolve() for path in arm.glob("attempt-*/episode")} | {final.resolve()})


def _add_permission_counts(row, audits):
    audited = [audit for audit in audits if audit is not None]
    if audited:
        row.update(attempts_with_permission_audit=len(audited),
                   audited_conditional_actions=sum(audit["audited"] for audit in audited),
                   conditional_permission_violations=sum(audit["violations"] for audit in audited))


def _attempt_costs(arm, final, public, *, allow_final_truncated=False):
    final = final.resolve()
    costs = {"total_model_transport_wall_seconds": 0., "total_action_seconds": 0.,
             "total_observation_seconds": 0., "total_search_seconds": 0., "unclosed_timing_intervals": 0,
             "attempts_with_raw_timing": 0, "transport_calls_with_duration": 0,
             "interrupted_raw_log_tails": 0}
    dispatch_times = []
    cpu_times = []
    task_windows = []
    roots = _attempt_roots(arm, final)
    for root in roots:
        allow_truncated = root != final or allow_final_truncated
        clock = _read(root / "job_clock.json") or {}
        started, elapsed = clock.get("started_wall"), clock.get("elapsed_seconds")
        if all(isinstance(value, (int, float)) and isfinite(value) and value >= 0 for value in (started, elapsed)):
            task_windows.append((started, started + elapsed))
        cpu = _read(root.parent / "worker_cpu.json")
        if cpu is not None:
            seconds = (cpu["user_seconds"], cpu["system_seconds"])
            if cpu["scope"] != "reaped_local_worker_and_descendants" or any(not isfinite(value) or value < 0 for value in seconds):
                raise ValueError("Invalid recorded worker CPU usage for " + str(root))
            cpu_times.append(sum(seconds))
        logs = root if public else root / "private/robot"
        calls, timeline = logs / "model_calls.jsonl", logs / "timeline.jsonl"
        if calls.is_file():
            records, truncated = read_trace_rows(calls, allow_truncated=allow_truncated)
            costs["interrupted_raw_log_tails"] += truncated
            for call in records:
                if "logical_job_attempt" in call:
                    costs["transport_calls_with_duration"] += 1
                    costs["total_model_transport_wall_seconds"] += call["wall_seconds"]
                    dispatch_times.append(call["started_at"])
        if timeline.is_file():
            clock = summarize(timeline, allow_truncated=allow_truncated)
            costs["attempts_with_raw_timing"] += 1
            costs["interrupted_raw_log_tails"] += clock["truncated_tail"]
            costs["unclosed_timing_intervals"] += len(clock["unfinished_intervals"])
            for kind in ("action", "observation", "search"):
                costs[f"total_{kind}_seconds"] += sum(item["seconds"] for item in clock["intervals"].get(kind, []))
    gaps = [after[0] - before[1] for before, after in zip(task_windows, task_windows[1:])]
    if len(task_windows) == len(roots) and all(gap >= 0 for gap in gaps):
        costs.update(task_elapsed_including_recovery_seconds=task_windows[-1][1] - task_windows[0][0],
                     task_recovery_gap_seconds=sum(gaps))
    return {**costs, "attempts_with_worker_cpu": len(cpu_times),
            "total_local_worker_cpu_seconds": sum(cpu_times) if cpu_times else None,
            "first_request_wall_time": min(dispatch_times, default=None),
            "last_request_wall_time": max(dispatch_times, default=None)}


def _first_action_seconds(events):
    starts = [row["monotonic_seconds"] for row in events if row["event"] == "action_started"]
    return min(starts) - events[0]["monotonic_seconds"] if starts else None


def _mechanism(root, events, report, allow_truncated):
    robot = root / "private/robot"
    frames = _camera_frames(robot)
    calls = _rows(robot / "model_calls.jsonl", allow_truncated=allow_truncated)
    evidence = _rows(robot / "evidence_ledger.jsonl", allow_truncated=allow_truncated)
    reviewed = audit_capture_permissions(root, allow_truncated=allow_truncated) or {}
    permitted = {(row["step"], row["candidate"]) for row in reviewed.get("actions", []) if row["permitted"]}
    accepted = {event["frame_id"] for event in events if event["event"] == "evidence_accepted"}
    reported = {row["evidence_image_id"] for row in report.get("records", ()) if row.get("evidence_image_id")}
    reported_objects = {object_id(row) for row in report.get("records", ())}
    reported.update(proof["identity_proof"]["image_id"] for proof in evidence
                    if proof["status"] in ("frontend_bound_observation", "frontend_material_verified")
                    and proof["photo"] in reported and proof["object"] in reported_objects)
    completed = {(row["step"], row.get("candidate")): row["monotonic_seconds"] for row in events
                 if row["event"] == "action_finished" and row.get("ok") is True}
    taken = []
    early_branches, moves_before, captures_before, used = 0, 0, 0, 0
    conditions = Counter({key: 0 for key in ("verified_judgments", "early_allowed_predicted_gain", "early_allowed_no_witnessed_gain",
                                             "confirmation_required", "reply_already_arrived",
                                             "revealed_collection_after_early_motion")})
    query_steps = sorted({row["step"] for row in events if row["event"] == "query_submitted"})
    for index, step in enumerate(query_steps):
        matching = [row for row in events if row["step"] == step]
        _, request = _verified_judgment(matching, calls, frames)
        sent, returned = request.get("dispatched_monotonic_seconds"), request.get("return_monotonic_seconds")
        if sent is None or returned is None:
            continue
        conditions["verified_judgments"] += 1
        grants = [row for row in matching if row["event"] == "permission_checked" and row.get("source")]
        if any(row["early"] for row in grants):
            baseline = next((row.get("baseline_finish") for row in matching
                             if row["event"] == "candidate_search_summary"), None)
            conditions["early_allowed_predicted_gain"] += baseline is not None and any(
                row["event"] == "candidate_selected" and row["expected_finish"] < baseline for row in matching)
            conditions["early_allowed_no_witnessed_gain"] += any(
                row["event"] == "candidate_rejected" and row.get("reason") == "no witnessed cost improvement"
                for row in matching)
        conditions["confirmation_required"] += any(not row["early"] and row["after"] == ["positive"] for row in grants)
        conditions["reply_already_arrived"] += any(row["event"] == "pending_strategy" and
                                                   row.get("reason") == "reply_already_arrived" for row in matching)
        upper = query_steps[index + 1] if index + 1 < len(query_steps) else float("inf")
        while_pending = [row for row in events if row["event"] == "action_started"
                         and step <= row["step"] < upper and (row["step"], row.get("candidate")) in permitted
                         and sent <= row["monotonic_seconds"] < returned]
        early_branches += len(while_pending)
        conditions["revealed_collection_after_early_motion"] += bool(while_pending) and any(
            row["event"] == "revealed_branch_collection" and row["monotonic_seconds"] >= returned for row in matching)
        for action in while_pending:
            finished = completed.get((action["step"], action["candidate"]))
            moves_before += finished is not None and action["monotonic_seconds"] <= finished < returned
            picture = next((row for row in matching if row["event"] == "candidate_executed"
                            and row.get("candidate") == action["candidate"]
                            and row["monotonic_seconds"] >= action["monotonic_seconds"]), None)
            if picture and picture.get("frame_id") in frames:
                taken.append(picture)
                captures_before += picture["monotonic_seconds"] < returned
                used += picture["frame_id"] in accepted and picture["image_id"] in reported
    scored = _read(root / "independent_scoring_time.json") or {}
    return {"conditional_moves_started_before_bound_reply": early_branches,
            "conditional_moves_finished_before_bound_reply": moves_before,
            "conditional_actual_photos_from_early_moves": len(taken),
            "conditional_photos_completed_before_bound_reply": captures_before,
            "conditional_early_materials_used_in_report": used,
            "conditional_early_materials_without_report_use": len(taken) - used,
            **{f"judgment_condition_{name}": count for name, count in conditions.items()},
            "first_action_from_first_logged_event_seconds": _first_action_seconds(events),
            "independent_scoring_seconds": scored["end_monotonic"] - scored["start_monotonic"] if scored else None}


def _arm_row(directory, case, method, binding=None):
    arm = directory / case["id"] / method
    job = _read(arm / "job.json")
    binding = binding or {}
    row = {"case": case["id"], "group": case.get("group_id", case["id"]), "family": case["family"],
           "domain": case["domain"], "stratum": case.get("stratum"), "seed": case["seed"], "method": method,
           "execution_method": binding.get("method", method), "ranker_check_cap": binding.get("check_cap"),
           "training_seed": binding.get("seed"),
           "deadline_seconds": case["episode_timeout_seconds"], "state": "PENDING"}
    if not job:
        return row
    row.update(state=job["state"], logical_id=job["id"], infrastructure_attempts=job["attempt"],
               actual_dispatches=job.get("request_budget", {}).get("reserved"),
               cumulative_task_seconds=job.get("charged_seconds"))
    detail = job.get("result") or {}
    if job.get("execution_revision"):
        row.update(execution_revision=job["execution_revision"],
                   external_interruption=detail.get("external_interruption"))
    if job.get("natural_continuous"):
        row.update(continuous_task_seconds=detail.get("charged_seconds"),
                   task_interruption=detail.get("task_interruption"),
                   continuous_source_attempt=detail.get("continuous_source_attempt"))
    if job["state"] not in RECORDED:
        return row
    output = job.get("artifact_dir")
    row.update(artifact_dir=output, success=False, restricted_seconds=case["episode_timeout_seconds"])
    if not output:
        return row
    root = Path(output).resolve()
    allow_truncated = job["state"] == "TIMEOUT" or bool(row.get("external_interruption") or row.get("task_interruption"))
    row.update(_attempt_costs(arm, root, bool(job.get("public_runtime")), allow_final_truncated=allow_truncated))
    if job.get("public_runtime"):
        return _public_arm(row, arm, root, job)
    _add_permission_counts(row, (audit_capture_permissions(attempt, allow_truncated=attempt != root or allow_truncated)
                                 for attempt in _attempt_roots(arm, root)))
    preflight, result = _read(root / "clean_preflight.json"), _read(root / "result.json")
    if preflight:
        _valid_preflight(preflight, case, job.get("execution_method", method), root)
    for earned, name in (("probe_completed", "base_frontend_completed"),
                         ("conditional_query_bound", "conditional_query_bound"),
                         ("conditional_materials", "active_conditional_materials"),
                         ("typed_task_program_compiled", "typed_program_compiled")):
        row[name] = (result or {}).get("robot_metrics", {}).get(earned)
    original_score = (result or {}).get("score") or {}
    row["legacy_outcome"] = original_score.get("outcome")
    row["failure_reason"] = (job.get("result") or {}).get("reason")
    row["legacy_wrong_accepted"] = original_score.get("wrong_accepted")
    timing = root / "private/robot/timeline.jsonl"
    if timing.is_file():
        clock = summarize(timing, allow_truncated=allow_truncated)
        events, _ = read_trace_rows(timing, allow_truncated=allow_truncated)
        event_counts = Counter(event["event"] for event in events)
        search = next((event for event in events if event["event"] == "candidate_search_summary"), None)
        selected = next((event for event in events if event["event"] == "candidate_selected"), None)
        aligned = next((event for event in events if event["event"] == "frontend_alignment"), None)
        typed_guard = [event["accepted"] for event in events if event["event"] == "typed_task_acceptance"]
        row.update(model_request_seconds=clock["model_request_seconds"],
                   model_action_overlap_seconds=clock["model_action_overlap_seconds"],
                   reported_outcome=clock["report_outcome"],
                   unfinished_intervals=len(clock["unfinished_intervals"]),
                   rejected_frozen_judgments=event_counts["query_rejected"],
                   conditional_frontend_fallbacks=event_counts["conditional_frontend_fallback"],
                   rejected_materials=event_counts["evidence_rejected"],
                   rejected_live_submissions=sum(event.get("accepted") is False for event in events
                                                 if event["event"] == "full_task_acceptance"),
                   action_seconds=sum(interval["seconds"] for interval in clock["intervals"].get("action", [])),
                   observation_seconds=sum(interval["seconds"] for interval in clock["intervals"].get("observation", [])),
                   search_seconds=sum(interval["seconds"] for interval in clock["intervals"].get("search", [])))
        if search and "ranker" in search:
            row.update(ranked_candidates=search["generated"], ranked_candidates_verified=search["checked"],
                       ranking_stop_reason=search["stop_reason"], checkpoint_sha256=search["checkpoint"])
            if search.get("candidate_prefixes_sha256"):
                row["candidate_prefixes_sha256"] = search["candidate_prefixes_sha256"]
        if search and selected:
            row.update(selected_candidate=selected["candidate"], selected_forecast_seconds=selected["expected_finish"],
                       baseline_forecast_seconds=search["baseline_finish"])
        if aligned:
            row.update(frontend_alignment=aligned["alignment"],
                       frontend_reference_sha256=aligned["reference_sha256"],
                       frontend_fresh_targets=aligned["fresh_targets"], frontend_used_targets=aligned["used_targets"])
        if typed_guard:
            row.update(typed_submission_checks=len(typed_guard), typed_submission_accepted=typed_guard[-1])
        if job.get("execution_revision"):
            accepted_report = (result or {}).get("report") if job["state"] == "FINISHED_VALID" else {}
            row.update(_mechanism(root, events, accepted_report or {}, allow_truncated))
    if result and result.get("execution_completed") and (root / "clean_coverage.json").is_file():
        coverage = audit_run(root)
        saved = _read(root / "clean_coverage.json")
        if saved != coverage:
            raise ValueError("Stored coverage does not agree with raw evidence for " + row["logical_id"])
        row["complete_coverage"] = coverage["complete"]
        row["missing_objects"] = len(coverage["missing_fields_by_object"])
        row["required_objects"] = coverage["audited_objects"]
        if original_score.get("score_version") == 5 and row["legacy_outcome"] in ("correct", "wrong_accepted"):
            recheck = audit_run(root, recheck_legacy_fragments=True)
            bound = recheck["report_matches_bound_submission"]
            row.update(legacy_fragment_recheck_complete=recheck["complete"], legacy_fragment_submission_bound=bound,
                       legacy_fragment_recheck_verified_change=not coverage["complete"] and recheck["complete"] and bound)
        if not coverage["complete"] and row["failure_reason"] is None:
            if row["typed_program_compiled"] is False:
                row["failure_reason"] = "task_program_parse"
            elif row["legacy_wrong_accepted"]:
                row["failure_reason"] = "incorrect_report"
            elif row["legacy_outcome"] in ("refused", "incomplete"):
                row["failure_reason"] = row["legacy_outcome"]
            else:
                row["failure_reason"] = "full_task_coverage"
    if job["state"] == "FINISHED_VALID":
        seconds = row["cumulative_task_seconds"]
        if (not row.get("complete_coverage") or row["legacy_outcome"] != "correct"
                or not isinstance(seconds, (int, float)) or not isfinite(seconds)
                or not 0 < seconds <= row["deadline_seconds"]):
            raise ValueError("A saved success failed independent validation for " + row["logical_id"])
        row.update(success=True, restricted_seconds=seconds)
    if row.get("reported_outcome") is not None:
        row["wrong_completion_claim"] = row["reported_outcome"] == "accepted" and not row["success"]
    if method in ("conditional-stream", "conditional-stream-buffered"):
        from .stream_baseline import audit

        packets = audit(root)
        row.update(segmented_navigation_packets=packets["navigation_packets"],
                   verified_segmented_moves=packets["moves_during_model_continuation"],
                   verified_segmented_overlap_seconds=packets["overlap_seconds"])
    return row


def _public_mechanism(root, job, clock, permission, allow_truncated):
    events = _rows(root / "timeline.jsonl", allow_truncated=allow_truncated)
    observations = {event["step"]: event for event in events if event["event"] == "camera_observation"}
    front = observations.get("front")
    submitted = next((event["binding"] for event in events if event["event"] == "query_submitted"), {})
    applied = next((event for event in events if event["event"] == "query_applied"), None)
    if not (front and submitted and permission is not None and (root / "front.jpg").is_file()):
        return {}
    front_hash = sha256((root / "front.jpg").read_bytes()).hexdigest()
    if front["image_sha256"] != front_hash or submitted["frame_id"] != "front":
        return {}

    calls = _rows(root / "model_calls.jsonl", allow_truncated=allow_truncated)
    ledger_path = root / "requests.jsonl"
    if not ledger_path.is_file():
        ledger_path = Path(job["request_ledger"])
    ledger = {entry["reservation"]: entry for entry in _rows(ledger_path, allow_truncated=allow_truncated)}

    def dispatched(call, hashes):
        send = ledger.get(call.get("logical_job_attempt"), {})
        return (call.get("image_sha256") == hashes and send.get("image_sha256") == hashes
                and call.get("model") == send.get("model") == job["model"]
                and call.get("reasoning_effort") == send.get("reasoning_effort") == job["reasoning_effort"])

    matches = []
    for call in calls:
        payload, output = call.get("input", {}), call.get("output")
        if not (payload.get("frame") == "front" and payload.get("sha256") == front_hash
                and dispatched(call, [front_hash]) and isinstance(output, dict)
                and set(output) == {"outcome"} and output["outcome"] in submitted["outcomes"]):
            continue
        prompt = json.dumps(payload, sort_keys=True)
        if sha256((front_hash + "\n" + prompt).encode("utf8")).hexdigest() == submitted["input_hash"]:
            matches.append(call)
    if len(matches) != 1:
        return {}
    reply = matches[0]
    if applied and (applied["binding"] != submitted or reply["output"] != {"outcome": applied["outcome"]}):
        return {}
    sent, returned = reply.get("dispatched_monotonic_seconds"), reply.get("return_monotonic_seconds")
    if sent is None or returned is None or not front["monotonic_seconds"] <= sent <= returned:
        return {}

    motion = next((event for event in events if event["event"] == "action_started"), None)
    early = (permission["violations"] == 0 and motion is not None
             and sent <= motion["monotonic_seconds"] < returned)
    finished = early and any(interval["step"] == motion["step"] and interval["start"] == motion["monotonic_seconds"]
                             and interval["ok"] and interval["finish"] < returned
                             for interval in clock["intervals"].get("action", ()))
    side = observations.get("side") if early else None
    path = root / "side.jpg"
    if early and path.is_file() != (side is not None):
        return {}
    used = False
    if side:
        if sha256(path.read_bytes()).hexdigest() != side["image_sha256"]:
            return {}
        report = _read(root / "report.json")
        used = bool(isinstance(report, dict) and report.get("side_sha256") == side["image_sha256"] and
                    any(call.get("output") == report and dispatched(call, [front_hash, side["image_sha256"]])
                        for call in calls))
    return {"conditional_moves_started_before_bound_reply": int(early),
            "conditional_moves_finished_before_bound_reply": int(finished),
            "conditional_actual_photos_from_early_moves": int(side is not None),
            "conditional_photos_completed_before_bound_reply": int(side is not None and side["monotonic_seconds"] < returned),
            "conditional_early_materials_used_in_report": int(used),
            "conditional_early_materials_without_report_use": int(side is not None and not used)}


def _public_arm(row, arm, root, job):
    from .public_audit import audit, audit_camera_permissions, registered_outcome

    detail = job.get("result") or {}
    row["failure_reason"] = detail.get("reason")
    direction = job["case"]["expected_language_contract"]["direction"]
    allow_truncated = job["state"] == "TIMEOUT" or bool(row.get("external_interruption") or row.get("task_interruption"))
    permissions = {attempt: audit_camera_permissions(attempt, direction, allow_truncated=attempt != root or allow_truncated)
                   for attempt in _attempt_roots(arm, root)}
    _add_permission_counts(row, permissions.values())
    if job["state"] == "FINISHED_VALID":
        state, _ = registered_outcome(root, job, detail["charged_seconds"], job["returncode"], None)
        seconds = row["cumulative_task_seconds"]
        if state != job["state"] or not 0 < seconds <= row["deadline_seconds"]:
            raise ValueError("Saved public success could not be independently verified: " + job["id"])
        row.update(success=True, restricted_seconds=seconds, complete_coverage=True)
    if not row["success"] and job.get("execution_revision"):
        row["public_failed_mechanism_measured"] = False
    if (root / "timeline.jsonl").is_file():
        clock = summarize(root / "timeline.jsonl", allow_truncated=allow_truncated)
        events, _ = read_trace_rows(root / "timeline.jsonl", allow_truncated=allow_truncated)
        row["first_action_from_first_logged_event_seconds"] = _first_action_seconds(events)
        row.update(model_request_seconds=clock["model_request_seconds"],
                   model_action_overlap_seconds=clock["model_action_overlap_seconds"],
                   action_seconds=sum(interval["seconds"] for interval in clock["intervals"].get("action", [])),
                   observation_seconds=sum(interval["seconds"] for interval in clock["intervals"].get("observation", [])),
                   search_seconds=sum(interval["seconds"] for interval in clock["intervals"].get("search", [])))
        if row["success"] and job.get("execution_revision"):
            camera = audit(root)
            from_dispatch = camera["first_motion_seconds_after_dispatch"]
            before_reply = camera["first_motion_seconds_before_reply"]
            early = (camera["early_camera_motion_had_permission"] is True and from_dispatch is not None
                     and from_dispatch >= 0 and before_reply is not None and before_reply > 0)
            motions = clock["intervals"].get("action", [])
            finished = early and bool(motions) and motions[0]["ok"] and motions[0]["seconds"] < before_reply
            side_after = camera["side_capture_seconds_after_reply"]
            captured = early and side_after is not None
            used = captured and bool(_read(root / "report.json").get("side_sha256"))
            row.update(conditional_moves_started_before_bound_reply=int(early),
                       conditional_moves_finished_before_bound_reply=int(finished),
                       conditional_actual_photos_from_early_moves=int(captured),
                       conditional_photos_completed_before_bound_reply=int(captured and side_after < 0),
                       conditional_early_materials_used_in_report=int(used),
                       conditional_early_materials_without_report_use=int(captured and not used))
        elif "public_failed_mechanism_measured" in row:
            stats = _public_mechanism(root, job, clock, permissions[root.resolve()], allow_truncated)
            row["public_failed_mechanism_measured"] = bool(stats)
            row.update({"public_failed_" + field: value for field, value in stats.items()})
    if (root / "report.json").is_file():
        row["wrong_completion_claim"] = not row["success"]
    return row


def _interval(values, seed, replicates):
    if len(values) < 2:
        return None
    generator = Random(seed)
    draws = sorted(fmean(generator.choices(values, k=len(values))) for _ in range(replicates))
    return [draws[int(.025 * (replicates - 1))], draws[int(.975 * (replicates - 1))]]


def _paired_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    values = {key: [mean(row[key] for row in group) for group in groups.values()]
              for key in ("success_difference", "restricted_seconds_difference")}
    return {"matched_cases": len(rows), "independent_groups": len(groups),
            **{key: {"mean": mean(numbers) if numbers else None,
                     "cluster_bootstrap_95": _interval(numbers, 17, 2000)} for key, numbers in values.items()}}


def _pair_differences(rows, reference):
    cases = {}
    for row in rows:
        if row["state"] in RECORDED:
            cases.setdefault(row["case"], {})[row["method"]] = row
    reports = {}
    for arms in cases.values():
        if reference not in arms:
            continue
        target = arms[reference]
        for method, row in arms.items():
            if method != reference:
                difference = {"case": row["case"], "group": row["group"],
                              "success_difference": int(row["success"]) - int(target["success"]),
                              "restricted_seconds_difference": row["restricted_seconds"] - target["restricted_seconds"]}
                reports.setdefault(method, []).append(difference)
    return reports


def _pair_rows(rows, reference):
    return {method: _paired_summary(differences) for method, differences in _pair_differences(rows, reference).items()}


def _registered_primary(rows, ours, reference):
    matches = {}
    for row in rows:
        if row["state"] in RECORDED and row["method"] in (ours, reference):
            matches.setdefault(row["case"], {})[row["method"]] = row
    groups = {}
    for cases in matches.values():
        if {ours, reference} <= cases.keys():
            candidate, baseline = cases[ours], cases[reference]
            groups.setdefault(candidate["group"], []).append(candidate["restricted_seconds"] - baseline["restricted_seconds"])
    differences = [mean(group) for group in groups.values()]
    p_value = None
    if len(differences) >= 2:
        rng = Random(1729)
        observed = abs(sum(differences))
        trials = 20000
        exceed = sum(abs(sum(value if rng.getrandbits(1) else -value for value in differences)) >= observed
                     for _ in range(trials))
        p_value = (exceed + 1) / (trials + 1)
    return {"method": ours, "reference": reference, "metric": "deadline_restricted_complete_task_seconds",
            "independent_groups": len(differences), "p_two_sided_sign_flip": p_value, "alpha": .05}


def _service_block_sensitivity(rows, ours, reference, span_seconds=3600):
    blocks = {}
    for row in rows:
        if row["state"] in RECORDED:
            blocks.setdefault(row["case"], []).append(row)
    affected = {}
    for case, group in blocks.items():
        starts = [row["first_request_wall_time"] for row in group if row.get("first_request_wall_time") is not None]
        ends = [row["last_request_wall_time"] for row in group if row.get("last_request_wall_time") is not None]
        span = max(ends) - min(starts) if starts and ends else None
        interrupted = [row["method"] for row in group if row.get("external_interruption") or row.get("task_interruption")]
        if interrupted or (span is not None and span > span_seconds):
            affected[case] = {"recorded_service_block_seconds": span, "interrupted_methods": interrupted}
    remaining = [row for row in rows if row["case"] not in affected]
    return {"block_threshold_seconds": span_seconds, "affected_cases": affected,
            "matched_differences": _pair_rows(remaining, reference).get(ours),
            "scope": "descriptive complete-block sensitivity"}


def _paired_ranking(rows):
    by_case, planned = {}, {}
    for row in rows:
        policy = row["ranker_check_cap"], row["execution_method"]
        seed = row["training_seed"]
        planned.setdefault(policy, set()).add(seed)
        by_case.setdefault(row["case"], {})[(*policy, seed)] = row
    heuristic, ordinary, b1, adaptive = RANKED_METHODS
    contrasts = [(b1, ordinary), (b1, heuristic), (b1, RETRIEVAL_METHOD), (adaptive, b1)]
    output = []
    for cap, method in planned:
        for target, reference in contrasts:
            if method != target or (cap, reference) not in planned:
                continue
            targets, references = planned[cap, target], planned[cap, reference]
            seeds = (targets | references) - {None} or {None}
            if any(group != {None} and group != seeds for group in (targets, references)):
                raise ValueError("Ranking comparison does not have matching declared training seeds")
            pairs = []
            observed_pools = shared_pools = 0
            for arms in by_case.values():
                matches = [(arms.get((cap, target, seed if seed in targets else None)),
                            arms.get((cap, reference, seed if seed in references else None))) for seed in seeds]
                if any(row is None or row["state"] not in RECORDED for pair in matches for row in pair):
                    continue
                if all(row.get("candidate_prefixes_sha256") for pair in matches for row in pair):
                    observed_pools += 1
                    shared_pools += all(a["candidate_prefixes_sha256"] == b["candidate_prefixes_sha256"]
                                        for a, b in matches)
                pairs.append({"group": matches[0][0]["group"],
                    "success_difference": mean(int(a["success"]) - int(b["success"]) for a, b in matches),
                    "restricted_seconds_difference": mean(a["restricted_seconds"] - b["restricted_seconds"]
                                                          for a, b in matches)})
            output.append({"target": target, "reference": reference, "check_cap": cap, "seed_pairs_per_case": len(seeds),
                           "planned_cases": len(by_case), "cases_with_comparable_pool_fingerprints": observed_pools,
                           "cases_with_identical_ranker_inputs": shared_pools, **_paired_summary(pairs)})
    return output


def _native_forecast_check(rows):
    baseline = {row["case"]: row["restricted_seconds"] for row in rows
                if row["method"] == "conditional-async" and row["state"] in RECORDED}
    checked = []
    for row in rows:
        if row["case"] not in baseline or "selected_forecast_seconds" not in row:
            continue
        forecast = row["baseline_forecast_seconds"] - row["selected_forecast_seconds"]
        observed = baseline[row["case"]] - row["restricted_seconds"]
        checked.append({"case": row["case"], "method": row["method"], "selected": row["selected_candidate"],
                        "forecast_gain_seconds": forecast, "matched_observed_gain_seconds": observed,
                        "absolute_gap_seconds": abs(forecast - observed)})
    return checked


def _failure_category(row):
    reason = row.get("task_interruption") or row.get("external_interruption") or row.get("failure_reason")
    if isinstance(reason, dict):
        return reason.get("code") or reason.get("category") or reason.get("type") or row["state"]
    return reason or row["state"]


def _method_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    methods = {}
    for method, cases in groups.items():
        valid = [row for row in cases if row["state"] in RECORDED]
        methods[method] = {"planned": len(cases), "recorded": len(valid),
                           "successes": sum(row["success"] for row in valid),
                           "success_rate": mean(row["success"] for row in valid) if valid else None,
                           "scored_completion_claims": sum("wrong_completion_claim" in row for row in valid),
                           "wrong_completion_claims": sum(row.get("wrong_completion_claim", False) for row in valid),
                           "permission_scored_episodes": sum("conditional_permission_violations" in row for row in valid),
                           "audited_conditional_actions": sum(row.get("audited_conditional_actions", 0) for row in valid),
                           "conditional_permission_violations": sum(row.get("conditional_permission_violations", 0) for row in valid),
                           "interrupted_raw_log_tails": sum(row.get("interrupted_raw_log_tails", 0) for row in valid),
                           "worker_cpu_measured_episodes": sum(row.get("total_local_worker_cpu_seconds") is not None
                                                               for row in valid),
                           "task_clock_measured_episodes": sum("task_elapsed_including_recovery_seconds" in row
                                                               for row in valid),
                           "first_action_measured_episodes": sum(
                               row.get("first_action_from_first_logged_event_seconds") is not None for row in valid),
                           "failure_types": dict(Counter(_failure_category(row) for row in valid if not row["success"])),
                           "frontend_alignment_counts": dict(Counter(row["frontend_alignment"] for row in valid
                                                                     if "frontend_alignment" in row)),
                           "statuses": {state: sum(row["state"] == state for row in cases)
                                        for state in sorted({row["state"] for row in cases})},
                           "mean_restricted_seconds": mean(row["restricted_seconds"] for row in valid) if valid else None}
        for field in ("actual_dispatches", "total_model_transport_wall_seconds", "total_action_seconds",
                      "total_observation_seconds", "total_search_seconds", "total_local_worker_cpu_seconds",
                      "task_elapsed_including_recovery_seconds", "task_recovery_gap_seconds",
                      "continuous_task_seconds", "first_action_from_first_logged_event_seconds",
                      "independent_scoring_seconds"):
            measures = [row[field] for row in valid if row.get(field) is not None]
            methods[method]["mean_" + field] = mean(measures) if measures else None
        mechanism = [row for row in valid if "conditional_moves_started_before_bound_reply" in row]
        if mechanism:
            methods[method]["conditional_mechanism_episodes"] = len(mechanism)
            for field in MECHANISM_COUNTS:
                methods[method][field] = sum(row[field] for row in mechanism)
        public_failures = [row for row in valid if "public_failed_mechanism_measured" in row]
        if public_failures:
            measured = [row for row in public_failures if row["public_failed_mechanism_measured"]]
            supplement = {"recorded_nonpassing_episodes": len(public_failures), "measured_episodes": len(measured)}
            if measured:
                supplement.update({field: sum(row["public_failed_" + field] for row in measured)
                                   for field in MECHANISM_COUNTS})
            methods[method]["public_failed_mechanism"] = supplement
        condition_counts = Counter()
        condition_prefix = "judgment_condition_"
        for row in mechanism:
            condition_counts.update({key[len(condition_prefix):]: value for key, value in row.items()
                                     if key.startswith(condition_prefix)})
        if condition_counts:
            methods[method]["native_judgment_conditions"] = dict(condition_counts)
        successes = [row["restricted_seconds"] for row in valid if row["success"]]
        methods[method]["mean_seconds_given_success"] = mean(successes) if successes else None
    return methods


def _family_summary(rows):
    families = {}
    for row in rows:
        families.setdefault(row["family"], []).append(row)
    summary = {}
    for family, cases in families.items():
        recorded = [row for row in cases if row["state"] in RECORDED]
        planned_counts = Counter(row["case"] for row in cases)
        recorded_counts = Counter(row["case"] for row in recorded)
        summary[family] = {"planned_cases": len(planned_counts),
                           "recorded_cases": sum(recorded_counts[case] == count for case, count in planned_counts.items()),
                           "planned_episodes": len(cases), "recorded_episodes": len(recorded),
                           "successes": sum(row["success"] for row in recorded),
                           "failure_types": dict(Counter(_failure_category(row) for row in recorded if not row["success"]))}
    return summary


def _object_coverage(rows, target, domain):
    label = target.strip().casefold()
    if not label:
        raise ValueError("Provide a nonempty object ID for the coverage audit")
    selected = [row for row in rows if domain is None or row["domain"] == domain]
    recorded = [row for row in selected if row["state"] in RECORDED]
    audited = 0
    relevant = []
    for row in recorded:
        if not row.get("artifact_dir"):
            continue
        root = Path(row["artifact_dir"])
        coverage = _read(root / "clean_coverage.json")
        if not coverage:
            continue
        audited += 1
        name = next((key for key in coverage["supported_by_frame"] if key.casefold() == label), None)
        if name is None:
            continue
        decisions = root / "private/robot/decisions.jsonl"
        count = None
        if decisions.is_file():
            trace, _ = read_trace_rows(decisions, allow_truncated=row["state"] == "TIMEOUT")
            count = 0
            for step in trace:
                if step.get("late_not_executed"):
                    continue
                decision = step.get("decision") or {}
                report = decision.get("report") or {}
                records = list(decision.get("records", [])) + list(report.get("records", []))
                count += sum(isinstance(key, str) and key.strip().casefold() == label
                             for key in map(object_id, records))
        relevant.append({"case": row["case"], "method": row["method"], "missing_fields":
                         coverage["missing_fields_by_object"].get(name, []), "model_records": count,
                         "records_without_bound_frame": sum(item["object"] == name
                                                            for item in coverage["records_without_supplied_frame"])})
    missing = [item for item in relevant if item["missing_fields"]]
    return {"object": target.strip(), "domain": domain, "planned_episodes": len(selected),
            "recorded_episodes": len(recorded), "audited_episodes": audited, "episodes_requiring_object": len(relevant),
            "episodes_missing_evidence": len(missing),
            "missing_without_model_record": sum(item["model_records"] == 0 for item in missing),
            "missing_without_decision_log": sum(item["model_records"] is None for item in missing),
            "missing_with_unbound_record": sum(item["records_without_bound_frame"] > 0 for item in missing),
            "episodes": relevant}


def rebuild(directory, output, *, coverage_object=None, coverage_domain=None):
    catalog = _read(directory / "pilot_manifest.json")
    if not catalog or not catalog["cases"]:
        raise ValueError("A frozen nonempty pilot manifest is required")
    bindings = catalog.get("ranker_bindings", {})
    rows = [_arm_row(directory, case, method, bindings.get(method))
            for case in catalog["cases"] for method in case["order"]]
    methods = _method_summary(rows)
    fallback_references = ("conditional-async", "paired-async", "conditional-rank-heuristic")
    if catalog.get("execution_revision"):
        fallback_references = ("conditional-planner", *fallback_references)
    reference = catalog.get("primary_reference") or next((name for name in fallback_references if name in methods), None)
    formal = catalog["role"] in (PUBLIC_FINAL_ROLE, CUSTOM_FINAL_ROLE)
    mode = "ACTUAL_LANGUAGE_AND_VISION_CLOSED_LOOP" if formal else "NATURAL_LATENCY_DEVELOPMENT"
    permission_rows = [row for row in rows if "conditional_permission_violations" in row]
    permission = (None if not permission_rows else {"scope": "registered conditional camera permissions",
        "scored_episodes": len(permission_rows),
        "audited_attempts": sum(row["attempts_with_permission_audit"] for row in permission_rows),
        "audited_actions": sum(row["audited_conditional_actions"] for row in permission_rows),
        "violations": sum(row["conditional_permission_violations"] for row in permission_rows)})
    result = {"mode": mode, "role": catalog["role"], "reference": reference,
              "planned_episodes": len(rows), "recorded_episodes": sum(r["state"] in RECORDED for r in rows),
              "methods": methods, "families": _family_summary(rows), "matched_differences": _pair_rows(rows, reference),
              "paired_ranking_comparisons": _paired_ranking(rows),
              "energy_measurement": None, "independently_scored_constraint_violations": permission,
              "scientific_conclusion": "INCONCLUSIVE"}
    if catalog.get("execution_revision"):
        protocol = _read(directory / "run_protocol.json")
        result.update(execution_revision=catalog["execution_revision"], evidence_level=catalog["evidence_level"],
                      catalog_sha256=_hash(catalog), worker_protocol_sha256=_hash(protocol) if protocol else None,
                      analysis_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    if coverage_object is not None:
        result["object_coverage_audit"] = _object_coverage(rows, coverage_object, coverage_domain)
    if formal and catalog.get("protocol_version") == "protocol-v2":
        result["protocol_version"] = "protocol-v2"
        if catalog.get("execution_revision"):
            result["execution_revision"] = catalog["execution_revision"]
            result["analysis_role"] = "confirmatory" if catalog["role"] == CUSTOM_FINAL_ROLE else "external_validation"
            result["service_block_sensitivity"] = _service_block_sensitivity(rows, catalog["ours_method"], reference)
            result["exploratory_baseline_minus_a"] = _pair_rows(rows, catalog["ours_method"])
        if result.get("analysis_role") != "external_validation":
            result["registered_primary_inference"] = _registered_primary(rows, catalog["ours_method"], reference)
    if catalog.get("data_role") == "INDEPENDENT_NATIVE_TRAINING_CHECK":
        result["real_trajectory_forecast_checks"] = _native_forecast_check(rows)
    fragment_rechecks = [row for row in rows if "legacy_fragment_recheck_complete" in row]
    if fragment_rechecks:
        result["legacy_fragment_recheck"] = {
            "scope": "completed reports scored with legacy version 5",
            "episodes": len(fragment_rechecks),
            "original_complete": sum(row["complete_coverage"] for row in fragment_rechecks),
            "complete_with_fragment_merging": sum(row["legacy_fragment_recheck_complete"] for row in fragment_rechecks),
            "verified_changes": [{"case": row["case"], "method": row["method"], "artifact_dir": row["artifact_dir"]}
                                 for row in fragment_rechecks if row["legacy_fragment_recheck_verified_change"]],
        }
    strata = {}
    for row in rows:
        if row["stratum"]:
            strata.setdefault(row["stratum"], []).append(row)
    if strata:
        result["strata"] = {label: {"cases": len({row["case"] for row in cohort}),
            "methods": _method_summary(cohort), "matched_differences": _pair_rows(cohort, reference),
            "paired_ranking_comparisons": _paired_ranking(cohort)} for label, cohort in strata.items()}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf8") as handle:
        header = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    if catalog.get("execution_revision"):
        fields = ("method", "reference", "case", "group", "success_difference", "restricted_seconds_difference")
        with (output / "paired_effects.csv").open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method, pairs in _pair_differences(rows, reference).items():
                writer.writerows({**pair, "method": method, "reference": reference} for pair in pairs)
    write_json(output / "summary.json", result)
    return result
