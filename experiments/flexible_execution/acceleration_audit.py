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


def _indexed_rows(path, gaps):
    if not path.is_file():
        return []
    rows = []
    with path.open("rb") as handle:
        for line, raw in enumerate(handle, 1):
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
            except (ValueError, UnicodeDecodeError):
                gaps.append({"file": str(path), "line": line, "recoverable_boundary": not raw.endswith(b"\n")})
                continue
            rows.append((line, row))
    return rows


_V2_ROLES = {
    "compile_joint_task": "task_compile", "locate_visible_targets": "target_localization",
    "judge_frozen_photo": "frozen_judgment", "verify_conditional_material": "material_confirmation",
    "accept_complete_task": "whole_task_acceptance",
    "verify_evidence_group": "evidence_group_acceptance",
    "segmented_action": "controller_first_segment", "segmented_continuation": "controller_next_segment",
}


def _request_windows(events):
    starts, windows = {}, []
    for line, event in events:
        if event["event"] == "model_request_started":
            starts[event["step"]] = (line, event)
        elif event["event"] == "model_request_finished":
            prior = starts.pop(event["step"], None)
            if prior:
                windows.append((prior[0], prior[1], line, event))
    return windows, list(starts.values())


def _v2_request_rows(key, physical, events, raw_path, ledger, ledger_path, gaps):
    windows, open_windows = _request_windows(events)
    handoff = next((e["monotonic_seconds"] for _, e in events if e["event"] == "full_inspection_started"), None)
    first_reject = next((e["monotonic_seconds"] for _, e in events
                         if e["event"] in ("typed_task_acceptance", "full_task_acceptance")
                         and e.get("accepted") is False), None)
    seen, exact, window_hashes, claimed = set(), set(), set(), set()
    records = []
    for line, call in physical:
        number = call.get("logical_job_attempt")
        if number is None:
            gaps.append({"file": str(raw_path), "line": line, "reason": "no_dispatch_ordinal"})
            continue
        if number in seen:
            gaps.append({"file": str(raw_path), "line": line, "reason": "duplicate_dispatch_ordinal"})
            continue
        seen.add(number)
        start, finish = call.get("dispatched_monotonic_seconds"), call.get("return_monotonic_seconds")
        matched = [(a, st, b, en) for a, st, b, en in windows
                   if start is not None and finish is not None
                   and st["monotonic_seconds"] <= start <= finish <= en["monotonic_seconds"]]
        role, evidence, step = "unknown", "no_unique_request_event", None
        payload = call.get("input") or {}
        images = call.get("image_sha256") or []
        fingerprint = _hash([call.get("model"), call.get("reasoning_effort"),
                             call.get("instructions_sha256"), payload, images])
        within_retry = False
        if len(matched) == 1:
            opener, event, _, _ = matched[0]
            purpose = event.get("purpose")
            step, evidence = event["step"], "unique_enclosing_timeline_interval"
            role = _V2_ROLES.get(purpose, "unknown")
            if purpose == "accept_complete_task" and "verified_observations" in payload and not images:
                role = "acceptance_merge"
            elif purpose is None and handoff is not None and start >= handoff and "current_image_id" in payload:
                role = "controller_decision"
            within_retry = (opener, fingerprint) in window_hashes
            window_hashes.add((opener, fingerprint))
        err = call.get("error") or {}
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        ledger_match = number in ledger and ledger[number][1].get("image_sha256", []) == images
        if ledger_match:
            claimed.add(number)
        records.append({**key, "dispatch_ordinal": number, "role": role, "role_evidence": evidence, "step": step,
                        "model": call.get("model"), "reasoning_effort": call.get("reasoning_effort"),
                        "prompt_sha256": call.get("instructions_sha256"), "input_sha256": fingerprint,
                        "start_wall_epoch": call.get("started_at"), "start_monotonic": start, "finish_monotonic": finish,
                        "service_seconds": call.get("wall_seconds"), "images": len(images),
                        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                        "total_tokens": usage.get("total_tokens"), "token_usage_reported": bool(usage),
                        "exact_input_repeat": fingerprint in exact, "same_request_retry": within_retry,
                        "error": err.get("code") or err.get("type"), "post_handoff": handoff is not None and
                        start is not None and start >= handoff,
                        "after_report_rejection": first_reject is not None and start is not None and start >= first_reject,
                        "ledger_match": ledger_match, "ledger_line": ledger[number][0] if number in ledger else None,
                        "ledger_path": str(ledger_path), "model_log_path": str(raw_path), "model_log_line": line})
        exact.add(fingerprint)
    for number in sorted(ledger.keys() - seen):
        line, entry = ledger[number]
        records.append({**key, "dispatch_ordinal": number, "role": "unknown", "role_evidence": "ledger_only",
                        "ledger_match": False, "ledger_line": line, "ledger_path": str(ledger_path),
                        "start_wall_epoch": entry.get("wall"), "images": len(entry.get("image_sha256") or ())})
    return records, windows, open_windows, len(claimed)


def audit_current_v2(output_root, destination):
    from .control import plans

    source = output_root / "acceleration-v2"
    development = source / "development"
    catalog, protocol = (_read(development / name) for name in ("pilot_manifest.json", "run_protocol.json"))
    if not catalog or protocol.get("catalog_sha256") != _hash(catalog):
        raise ValueError("The audited v2 development directory does not match its frozen catalog")
    gaps, attempts, request_rows, timelines = [], [], [], []
    journal = source / "control/jobs.jsonl"
    transitions = {}
    for _, row in _indexed_rows(journal, gaps):
        if row.get("stage") == protocol["stage"] and row.get("attempt") and row.get("finished_at"):
            transitions.setdefault((row["id"], row["attempt"]), row)
    pending_jobs = []
    for template in plans(development, protocol, catalog):
        path = Path(template["arm"]) / "job.json"
        job = _read(path) or template
        raw_ledger = Path(job["request_ledger"]) if job.get("request_ledger") else (
            source / "control/requests" / (job["id"] + ".jsonl"))
        ledger = {row["reservation"]: (line, row) for line, row in _indexed_rows(raw_ledger, gaps)}
        launches = sorted(path.parent.glob("attempt-*/launch.json"))
        if not launches:
            pending_jobs.append({"job": job["id"], "case": job["case"]["id"], "method": job["method"]})
        for launch_path in launches:
            launch = _read(launch_path)
            number = launch["attempt"]
            terminal = transitions.get((job["id"], number), job if job["attempt"] == number else {})
            started_at, finished_at = launch.get("started_at"), terminal.get("finished_at")
            worker_start = datetime.fromisoformat(started_at).timestamp() if started_at else None
            worker_end = datetime.fromisoformat(finished_at).timestamp() if finished_at else None
            lower = launch.get("request_budget", {}).get("reserved", 0)
            upper = terminal.get("request_budget", {}).get("reserved")
            attempt_ledger = {n: row for n, row in ledger.items() if n > lower and
                              (n <= upper if upper is not None else worker_start is None or row[1]["wall"] >= worker_start)}
            key = {"protocol_version": "protocol-v2", "execution_revision": "original",
                   "case": job["case"]["id"], "method": job["method"], "job_id": job["id"], "attempt": number,
                   "code_git_commit": launch.get("code_git_commit", protocol["code_git_commit"])}
            root = launch_path.parent / "episode"
            robot = root / "private/robot"
            trace_path, raw_path = robot / "timeline.jsonl", robot / "model_calls.jsonl"
            events = _indexed_rows(trace_path, gaps)
            physical = _indexed_rows(raw_path, gaps)
            rows, windows, open_windows, matched = _v2_request_rows(key, physical, events, raw_path, attempt_ledger,
                                                                     raw_ledger, gaps)
            request_rows.extend(rows)
            spans, unclosed = _spans([event for _, event in events])
            model = [(r["start_monotonic"], r["finish_monotonic"]) for r in rows
                     if r.get("start_monotonic") is not None and r.get("finish_monotonic") is not None]
            actions = spans.get("action", [])
            observation = spans.get("observation", [])
            explicit = model + actions + observation + spans.get("search", [])
            task = _read(root / "job_clock.json")
            task_start = task.get("started_wall")
            task_elapsed = task.get("elapsed_seconds", (terminal.get("result") or {}).get("charged_seconds"))
            coverage = _read(root / "clean_coverage.json")
            decisions = _indexed_rows(robot / "decisions.jsonl", gaps)
            before = next((call["input"] for _, call in physical if "current_image_id" in call.get("input", {})), {})
            handoff = next((r for _, r in events if r["event"] == "full_inspection_started"), {})
            reject = [(line, r) for line, r in events if r["event"] in ("typed_task_acceptance", "full_task_acceptance")
                      and r.get("accepted") is False]
            submits = Counter(_hash(row["decision"]["report"]) for _, row in decisions
                              if (row.get("decision") or {}).get("report") and not row.get("late_not_executed"))
            report_repeats = sum(n - 1 for n in submits.values())
            row = {**key, "state": terminal.get("state", "UNRESOLVED"), "worker_started_at": started_at,
                   "worker_finished_at": finished_at, "worker_seconds": worker_end-worker_start
                   if worker_start is not None and worker_end is not None else None,
                   "initialization_seconds": task_start-worker_start
                   if task_start is not None and worker_start is not None else None,
                   "post_task_worker_seconds": max(0., worker_end-task_start-task_elapsed)
                   if all(v is not None for v in (worker_end, task_start, task_elapsed)) else None,
                   "independent_scoring_seconds": None, "task_seconds": task_elapsed,
                   "task_complete": terminal.get("state") == "FINISHED_VALID" and coverage.get("complete") is True,
                   "physical_dispatches": len(rows), "raw_dispatch_ledger_matches": matched,
                   "reported_service_seconds_sum": sum(r.get("service_seconds") or 0. for r in rows),
                   "transport_union_seconds": _union(model), "action_union_seconds": _union(actions),
                   "transport_action_overlap_seconds": max(0., _union(model)+_union(actions)-_union(model+actions)),
                   "observed_intervals_union_seconds": _union(explicit),
                   "task_seconds_outside_logged_intervals": max(0., task_elapsed-_union(explicit))
                   if task_elapsed is not None else None,
                   "post_handoff_calls": sum(bool(r.get("post_handoff")) for r in rows),
                   "first_controller_structured_records": len(before.get("records") or []) if before else None,
                   "verified_frontend_materials": handoff.get("accepted_active_materials"),
                   "report_rejections": len(reject), "requests_after_first_rejection": sum(
                       bool(r.get("after_report_rejection")) for r in rows),
                   "exact_repeated_proposed_reports": report_repeats, "unclosed_intervals": unclosed,
                   "terminal_reason": (terminal.get("result") or {}).get("reason"), "raw_episode": str(root),
                   "launch": str(launch_path), "timeline": str(trace_path), "coverage": str(root / "clean_coverage.json")}
            attempts.append(row)
            dependencies = [{"kind": "frozen_outcome_applied", "step": r["step"], "record_line": line,
                             "at": r["monotonic_seconds"], "query": r.get("binding", {}).get("id")}
                            for line, r in events if r["event"] == "query_applied"]
            dependencies.extend({"kind": "report_rejection_before_followup", "step": r["step"], "record_line": line,
                                 "at": r["monotonic_seconds"], "reason": r.get("issue")}
                                for line, r in reject)
            timelines.append({**key, "events": str(trace_path), "raw_requests": str(raw_path),
                              "model_intervals": model, "action_intervals": actions,
                              "request_event_pairs": [[a, b, st.get("purpose")] for a, st, b, _ in windows],
                              "unclosed_requests": [line for line, _ in open_windows], "observed_dependencies": dependencies,
                              "independent_score_index": row["coverage"], "missing_fields": coverage.get("missing_fields_by_object"),
                              "exact_critical_path_reconstructed": False})
    role_names = sorted({r["role"] for r in request_rows} |
                        {"evidence_group_acceptance", "acceptance_merge", "unknown"})
    roles = {}
    for name in role_names:
        group = [r for r in request_rows if r["role"] == name]
        roles[name] = {"dispatches": len(group), "reported_service_seconds_sum": sum(
            r.get("service_seconds") or 0. for r in group), "sent_images": sum(r.get("images") or 0 for r in group),
            "same_request_retries": sum(bool(r.get("same_request_retry")) for r in group),
            "token_usage_rows": sum(bool(r.get("token_usage_reported")) for r in group),
            "errors": dict(Counter(r["error"] for r in group if r.get("error")))}
    totals = {"actual_new_model_requests": 0, "actual_new_flights": 0,
              "source_cohort": str(development), "source_code_git_commit": protocol["code_git_commit"],
              "source_fingerprint": _hash(protocol["source_sha256"]), "role_totals": roles,
              "recorded_episodes": sum(r["state"] in ("FINISHED_VALID", "FINISHED_TASK_FAILURE", "TIMEOUT")
                                       for r in attempts), "saved_attempts": len(attempts),
              "dispatches": len(request_rows), "raw_dispatch_ledger_matches": sum(
                  r["raw_dispatch_ledger_matches"] for r in attempts),
              "elapsed_worker_seconds": sum(r["worker_seconds"] or 0. for r in attempts),
              "post_handoff_calls": sum(r["post_handoff_calls"] for r in attempts),
              "post_rejection_calls": sum(r["requests_after_first_rejection"] for r in attempts),
              "exact_repeated_proposed_reports": sum(r["exact_repeated_proposed_reports"] for r in attempts),
              "exact_input_repeats": sum(bool(r.get("exact_input_repeat")) for r in request_rows),
              "gap_records": gaps, "pending_jobs_without_attempt": pending_jobs}
    destination.mkdir(parents=True, exist_ok=True)
    _csv(destination / "current_v2_attempts.csv", attempts)
    _csv(destination / "current_v2_request_roles.csv", request_rows)
    write_json(destination / "current_v2_timeline.json", {**totals, "attempt_timelines": timelines})
    _write_v2_bottlenecks(destination, totals, attempts)
    return totals


def _write_v2_bottlenecks(destination, totals, attempts):
    empty_handoffs = sum(bool(row["verified_frontend_materials"]) and row["first_controller_structured_records"] == 0
                         for row in attempts)
    active_episodes = sum(row["task_seconds"] > 0 for row in attempts if row["task_seconds"] is not None)
    parts = [
        "# v2 开发回合热点审计", "",
        f"原执行提交 `{totals['source_code_git_commit']}`。"
        f"共 {totals['saved_attempts']} 次启动，{totals['recorded_episodes']} 个终态回合。"
        f"{totals['dispatches']} 笔发送中 {totals['raw_dispatch_ledger_matches']} 笔对上账本。"
        f"worker 活跃墙钟 {totals['elapsed_worker_seconds']:.0f} 秒。"
        "重算未发送模型请求或启动仿真。", "",
        "## 已确认热点", "",
        "| 用途 | 发送 | 模型请求耗时合计（秒） | 图片发送数 | 同一请求重试 | 令牌可见条目 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(totals["role_totals"].items(),
                            key=lambda item: (-item[1]["reported_service_seconds_sum"], item[0])):
        parts.append(f"| {name} | {row['dispatches']} | {row['reported_service_seconds_sum']:.2f} | "
                     f"{row['sent_images']} | {row['same_request_retries']} | {row['token_usage_rows']} |")
    parts += [
        "", "服务时段按发送相加。另有模型和动作区间并集、重叠、任务时间及区间外时间，见逐尝试表。",
        f"共 {totals['post_handoff_calls']} 笔在整项控制器接管后发生，"
        f"其中 {totals['post_rejection_calls']} 笔位于该次尝试第一次报告拒绝之后，含合法补做和后续验收。"
        f"在 {empty_handoffs}/{active_episodes} 次启动到任务阶段的尝试中，"
        "前置阶段已经核验材料，整项控制器首个请求的结构化记录仍为空。",
        f"完整输入重复 {totals['exact_input_repeats']} 笔，重提完全相同的报告"
        f" {totals['exact_repeated_proposed_reports']} 次。已知外部错误列于逐请求表。",
        "", "## 仅由源码提出的怀疑", "",
        "全图分组在插入照片时可能失配，超过四图才走的分组与合并终审在这份样本中均未发生。"
        "当前没有这一分支的热点证据。已核验前置观察的交接可能减少后续查询，实际可避免的调用"
        "及整项时差需在同版本真实开发中核验。",
        "", "## 无法确认项", "",
        "服务未返回令牌计数时保留缺失。既有事件能确认冻结结果应用和报告拒绝后的顺序，"
        "不足以复原完整关键路径。独立评分与退出的各自时长缺少起止事件，worker 退出前的合并时段仍保留。",
        f"原始日志解析缺口 {len(totals['gap_records'])} 条，日志索引及不可分类调用见时间线与逐请求表。"
        "所有有模型请求的尝试来自首个物理场景。跨作业的额度等待可由原始日历时间重算。",
        "", "重算命令见本目录父项目的实验协议。", "",
    ]
    (destination / "BOTTLENECK_REPORT.md").write_text("\n".join(parts), encoding="utf8")


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
