"""Evaluate only actions and releases supported by registered observations."""

from collections import defaultdict
from datetime import datetime, timezone
from heapq import heappop, heappush
from pathlib import Path
from statistics import mean, median
from time import process_time

from experiments.task_attack.runtime import write_json

from .acceleration_audit import _purpose
from .learning_schedule import build_config
from .live_audit import _camera_frames, _rows, _verified_judgment, audit_capture_permissions
from .ranker_replay import replay
from .runner import read_json
from .timeline import read_trace_rows


def _trace(directory):
    """Reveal recorded query outcomes only at their actual application event."""
    catalog = read_json(directory / "pilot_manifest.json")
    observations = {}
    attempts, supported, unbound, exact_captures = 0, 0, 0, 0
    for status in sorted(directory.glob("*/*/job.json")):
        job = read_json(status)
        if not job.get("artifact_dir") or not job.get("finished_at"):
            continue
        episode = Path(job["artifact_dir"])
        robot = episode / "private/robot"
        path = robot / "timeline.jsonl"
        if not path.is_file():
            continue
        events, truncated = read_trace_rows(path, allow_truncated=True)
        if not events:
            continue
        frames = _camera_frames(robot)
        calls = _rows(robot / "model_calls.jsonl", allow_truncated=True)
        permissions = audit_capture_permissions(episode, allow_truncated=True) or {}
        permitted = {(row["step"], row["candidate"]): row["permitted"] for row in permissions.get("actions", [])}
        queue, by_step = [], defaultdict(list)
        usable = {"query_submitted", "query_applied", "query_rejected", "candidate_executed",
                  "action_started", "action_finished", "evidence_accepted"}
        origin = events[0]["monotonic_seconds"]
        for number, event in enumerate(events):
            if event["event"] in usable:
                heappush(queue, (event["monotonic_seconds"] - origin, number, event))
                by_step[event["step"]].append(event)
        pending, results, rejected, running, moves, captures, acceptances = {}, [], [], {}, {}, [], []
        while queue:
            now, _, event = heappop(queue)
            kind, step, binding = event["event"], event["step"], event.get("binding") or {}
            key = binding.get("id")
            if kind == "query_submitted" and key:
                pending[step] = (now, binding)
                attempts += 1
            elif kind == "query_applied":
                reply, transport = _verified_judgment(by_step[step], calls, frames)
                if step not in pending or pending[step][1] != binding or reply != event:
                    unbound += 1
                    continue
                started, _ = pending.pop(step)
                sent, ready = transport.get("dispatched_monotonic_seconds"), transport.get("return_monotonic_seconds")
                has_transport_interval = (isinstance(sent, (int, float)) and isinstance(ready, (int, float))
                                          and origin <= sent <= ready <= origin + now)
                results.append({"query": key, "submitted_at": started, "released_at": now,
                                "elapsed_seconds": now - started, "outcome": event["outcome"],
                                "dispatched_at": sent - origin if has_transport_interval else None,
                                "transport_ready_at": ready - origin if has_transport_interval else None,
                                "input_sha256": binding["input_hash"], "photo_sha256": transport["image_sha256"][0],
                                "prompt_sha256": transport.get("instructions_sha256"),
                                "source": "EXACT_RECORDED_QUERY"})
                supported += 1
            elif kind == "query_rejected" and step in pending:
                started, frozen = pending.pop(step)
                rejected.append({"query": frozen["id"], "ended_at": now, "elapsed_seconds": now - started,
                                 "reason": event.get("reason"), "source": "EXACT_RECORDED_NO_REPLY_APPLIED"})
            elif kind == "action_started" and event.get("candidate"):
                running[(step, event["candidate"])] = now
            elif kind == "action_finished" and (step, event.get("candidate")) in running:
                candidate = event["candidate"]
                started = running.pop((step, candidate))
                if event["ok"]:
                    moves[candidate] = (step, started, now)
            elif kind == "candidate_executed":
                candidate = event["candidate"]
                move = moves.pop(candidate, None)
                frame = frames.get(event["frame_id"], {})
                if move and frame.get("sha256") == event["frame_sha256"]:
                    captures.append({"candidate": candidate, "start": move[1], "end": move[2],
                                     "photo_available_at": now, "photo_sha256": frame["sha256"],
                                     "permission_verified": permitted.get((move[0], candidate)),
                                     "source": "EXACT_RECORDED_ACTION_AND_PHOTO"})
                    exact_captures += 1
            elif kind == "evidence_accepted" and event.get("frame_id") in frames:
                frame = frames[event["frame_id"]]
                if any(frame["sha256"] in call.get("image_sha256", []) for call in calls
                       if isinstance(call.get("return_monotonic_seconds"), (int, float))
                       and call["return_monotonic_seconds"] <= origin + now):
                    acceptances.append({"requirement": event["requirement"], "logged_at": now,
                                        "photo_sha256": frame["sha256"],
                                        "source": "RECORDED_CONTROLLER_ACCEPTANCE"})
        observations[job["id"]] = {"source": str(path), "case": job["case"]["id"], "method": job["method"],
                                   "model": job["model"], "reasoning_effort": job["reasoning_effort"],
                                   "code_git_commit": job["code_git_commit"],
                                   "source_protocol": catalog.get("protocol_version", "legacy"),
                                   "released": results, "rejected": rejected, "captured": captures,
                                   "accepted": acceptances, "unresolved_queries": len(pending),
                                   "truncated_trace_tail": truncated, "unrecorded_action_or_view": "OUT_OF_SUPPORT"}
    return {"protocol_version": "protocol-v2", "evidence_level": "TRACE_SUPPORTED",
            "virtual_clock": "recorded_monotonic_event_order",
            "attempted_queries": attempts, "exact_bound_releases": supported, "unbound_releases": unbound,
            "exact_action_photo_transitions": exact_captures, "sampled_transitions": 0,
            "symbolic_transitions": 0, "counterfactual_full_episodes": 0, "new_requests": 0,
            "source_episodes": observations}


def _calibration_waits(output_root):
    cohorts = ("natural-pilot/native-training-check-v2", "natural-pilot/visual-calibration-v3",
               "public-visual/validation-pilot-v1")
    groups = defaultdict(list)
    for cohort in cohorts:
        for path in sorted((output_root / cohort).glob("*/*/job.json")):
            job = read_json(path)
            if not job.get("finished_at") or not job.get("artifact_dir"):
                continue
            suffix = Path("model_calls.jsonl") if job.get("public_runtime") else Path("private/robot/model_calls.jsonl")
            roots = set(path.parent.glob("attempt-*/episode")) | {Path(job["artifact_dir"])}
            platform = "ProcTHOR" if job.get("public_runtime") else "Gazebo"
            for root in sorted(roots):
                for number, call in enumerate(_rows(root / suffix, allow_truncated=True), 1):
                    if ("logical_job_attempt" not in call or not isinstance(call.get("wall_seconds"), (int, float))
                            or not call.get("started_at")):
                        continue
                    stamp = call["started_at"]
                    started = (datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
                               if isinstance(stamp, str) else datetime.fromtimestamp(stamp, timezone.utc))
                    hour = started.hour
                    error = call.get("error") or {}
                    cause = str(error.get("code") or error.get("type") or ("unknown" if error else ""))
                    timed_out = ("timeout" in cause.lower() or "deadline" in cause.lower())
                    purpose, images = _purpose(call), len(call.get("image_sha256") or ())
                    key = (platform, purpose, images, hour, call.get("model") or job["model"],
                           call.get("reasoning_effort") or job["reasoning_effort"])
                    groups[key].append({"case": job["case"]["id"], "method": job["method"], "cohort": cohort,
                                        "log": str(root / suffix), "log_row": number,
                                        "seconds": call["wall_seconds"], "right_censored": timed_out,
                                        "terminal_error": cause or None})
    strata = []
    for (platform, purpose, images, hour, model, effort), samples in sorted(groups.items()):
        replies = [row["seconds"] for row in samples if row["terminal_error"] is None]
        strata.append({"platform": platform, "purpose": purpose, "image_count": images, "service_hour_utc": hour,
                       "model": model, "reasoning_effort": effort, "attempts": len(samples),
                       "right_censored": sum(row["right_censored"] for row in samples),
                       "other_errors": sum(bool(row["terminal_error"]) and not row["right_censored"]
                                           for row in samples),
                       "successful_return_median_seconds": median(replies) if replies else None,
                       "samples": samples})
    return {"protocol_version": "protocol-v2", "evidence_level": "TRACE_SUPPORTED",
            "scope": "observed development and training calibration attempts",
            "new_requests": 0, "strata": strata, "sampled_counterfactual_waits": 0}


def offline(output_root):
    root = output_root / "acceleration-v2"
    root.mkdir(parents=True, exist_ok=True)
    diagnostic = output_root / "ranker-diagnostic-v1"
    native = output_root / "natural-pilot/native-training-check-v2"
    configurations = {f"{size}-seed-{seed}-cap-{cap}": build_config(diagnostic, size, seed, cap, include_retrieval=True)
                      for size in (1000, 3000, 10000) for seed in (17, 29, 43) for cap in (4, 8)}
    cpu_start = process_time()
    fixed = replay(native, root / "offline-configuration-replay.json", configurations=configurations)
    grid = {}
    for key in fixed["snapshots"][0]["methods"]:
        rows = [row["methods"][key] for row in fixed["snapshots"]]
        grid[key] = {"checked": sum(row["checked"] for row in rows),
                     "selected": sum(bool(row["selected"]) for row in rows),
                     "mean_forecast_seconds": mean(row["predicted_finish_seconds"] for row in rows),
                     "selected_infeasible": sum(not row["selected_prefix_feasible"] for row in rows),
                     "unexamined_feasible_prefixes": sum(row["unexamined_feasible_prefixes"] for row in rows),
                     "unexamined_improving_prefixes": sum(row["unexamined_improving_prefixes"] for row in rows),
                     "missed_best_forecast_pools": sum(row["missed_best_forecast"] for row in rows),
                     "worst_forecast_gap_seconds": max(row["forecast_gap_to_best_seconds"] for row in rows),
                     "cpu_seconds": sum(row["process_cpu_seconds"] for row in rows)}
    traces = _trace(native)
    write_json(root / "trace-replay.json", traces)
    write_json(root / "calibration-waits.json", _calibration_waits(output_root))
    symbol = read_json(diagnostic / "latest_summary.json")
    scale = {}
    for row in symbol["run_rows"]:
        if row["objective"] == "cost":
            cost = next(item["mean_restricted_seconds"] for item in row["fixed"] if item["cap"] == 4)
            scale.setdefault(row["training_states"], []).append(cost)
    symbolic = {size: mean(scores) for size, scores in scale.items() if len(scores) == 3}
    size = min(symbolic, key=lambda item: (symbolic[item], item))
    # Seed is fixed in advance. Compare the deployable B1 check cap on the same pools.
    cap = min((4, 8), key=lambda item: (
        grid[f"{size}-seed-17-cap-{item}/conditional-rank-b1"]["mean_forecast_seconds"], item))
    result = {"protocol_version": "protocol-v2", "evidence_level": "DECISION_REPLAY", "actual_new_model_requests": 0,
              "fixed_pools": fixed["source_snapshots"], "independent_sources": fixed["independent_case_groups"],
              "settings": len(configurations), "paired_policy_settings": len(grid), "grid": grid,
              "exhaustive_same_pool": {key: sum(row["exhaustive_same_pool"][key] for row in fixed["snapshots"])
                                       for key in ("checked", "feasible", "improving", "cpu_seconds")},
              "training_size": size, "training_seed": 17, "check_cap": cap,
              "symbolic_restricted_seconds": symbolic, "trace_bound_releases": traces["exact_bound_releases"],
              "cpu_seconds": process_time() - cpu_start, "full_closed_loop_claim_supported": False}
    write_json(root / "offline_selection.json", result)
    return result
