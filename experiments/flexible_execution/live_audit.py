"""Recheck conditional dispatch and evidence against the raw model and camera records."""

import hashlib
import json

from .live_planning import needs_body_identity
from .runner import read_json


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()] if path.is_file() else []


def audit_capture_permissions(episode):
    robot = episode / "private/robot"
    contract = robot / "conditional_contract.json"
    trace = robot / "timeline.jsonl"
    if not contract.is_file() or not trace.is_file():
        return None
    task = read_json(contract)["task"]
    events = _rows(trace)
    skills = {skill["id"]: skill for skill in task["skills"]}
    fields = {r["id"]: r["field"] for r in task["requirements"]}
    grants = {row["field"]: row["source"] for row in events if row["event"] == "permission_checked" and row["early"]}
    applied = next((row for row in events if row["event"] == "query_applied"), {})
    instruction = task["instruction"].splitlines()
    checked = []
    for row in events:
        if row["event"] != "action_started" or row.get("candidate") not in skills:
            continue
        skill = skills[row["candidate"]]
        gate = skill["permission"]
        if gate is None:
            permitted = all(grants.get(fields[identifier]) in instruction for identifier in skill["covers"])
        else:
            permitted = (applied.get("binding", {}).get("query_id") == gate["query_id"]
                         and applied.get("outcome") in gate["outcomes"]
                         and applied["monotonic_seconds"] <= row["monotonic_seconds"])
        checked.append({"step": row["step"], "candidate": row["candidate"], "permitted": permitted})
    return {"scope": "conditional camera actions", "audited": len(checked),
            "violations": sum(not row["permitted"] for row in checked), "actions": checked}


def audit_live_connection(episode):
    robot = episode / "private" / "robot"
    rows = _rows(robot / "timeline.jsonl")
    calls = _rows(robot / "model_calls.jsonl")
    lookup = {}
    for path in robot.glob("probe_[0-9][0-9][0-9][0-9].json"):
        item = read_json(path)
        image = path.with_suffix(".jpg")
        if image.is_file() and path.stem == item["image_id"]:
            if hashlib.sha256(image.read_bytes()).hexdigest() == item["sha256"]:
                lookup[item["frame_id"]] = item
    submitted = next((row for row in rows if row["event"] == "query_submitted"), {})
    applied = next((row for row in rows if row["event"] == "query_applied"), {})
    binding = submitted.get("binding") or {}
    photo = lookup.get(binding.get("frame_id")) or {}
    frozen_call = next((call for call in calls if call["input"].get("frame_id") == binding.get("frame_id")
                        and "outcome" in call.get("output", {}) and call["input"].get("frame_id")), {})
    frozen_bound = bool(binding and binding == applied.get("binding") and photo
                        and frozen_call.get("image_sha256") == [photo["sha256"]])
    interval = {row["event"]: row["monotonic_seconds"] for row in rows
                if row.get("purpose") == "judge_frozen_photo" and row["event"].startswith("model_request_")}
    precise_interval = {"model_request_started": frozen_call.get("dispatched_monotonic_seconds"),
                        "model_request_finished": frozen_call.get("return_monotonic_seconds")}
    timing_source = "no_complete_model_interval"
    if all(isinstance(value, (float, int)) for value in precise_interval.values()):
        interval = precise_interval
        timing_source = "recorded_dispatch_and_model_return"
    elif {"model_request_started", "model_request_finished"} <= interval.keys():
        timing_source = "outer_request_timeline"
    else:
        interval = {}
    witnesses = [row for row in rows if row["event"] == "candidate_checked" and row.get("common")]
    grant = {row["field"] for row in rows if row["event"] == "permission_checked" and row["early"]}
    accepted = []
    for item in rows:
        if item["event"] != "evidence_accepted":
            continue
        frame = lookup.get(item.get("frame_id"))
        if not frame:
            continue
        verification = next((call for call in calls if call["input"].get("current_frame_id") == item["frame_id"]
                             and call["image_sha256"][-1:] == [frame["sha256"]]
                             and call.get("output", {}).get("readable") is True
                             and call.get("output", {}).get("same_object") is True), {})
        if not verification:
            continue
        captured = next((row for row in rows if row["event"] == "candidate_executed"
                         and row["frame_id"] == item["frame_id"]), {})
        started = next((row for row in rows if row["event"] == "action_started"
                        and row["step"] == int(frame["image_id"].split("_", 1)[1])
                        and row.get("candidate") == captured.get("candidate") and row.get("candidate")), {})
        sent_to_controller = any(frame["image_id"] in {row["image_id"] for row in
                                 call["input"].get("recent_images", [])}
                                 and frame["sha256"] in call["image_sha256"] for call in calls)
        witness = any(captured.get("candidate") in row["candidate"].split("|") and
                      {value for branch in row["branches"] for value in branch["outcomes"].values()}
                      >= {"positive", "negative", "uncertain"} for row in witnesses)
        early = bool(interval and started and interval["model_request_started"] <= started["monotonic_seconds"]
                     < interval["model_request_finished"] and witness and item["requirement"].split(":")[-1] in grant)
        source_role = verification.get("output", {}).get("source_role")
        requires_body = (item["requirement"].split(":")[-1] == "identity"
                         and needs_body_identity(verification["input"].get("authorized_task", "")))
        source_valid = source_role == "body_identity" if requires_body else source_role in ("body_identity", "other")
        accepted.append({"requirement": item["requirement"], "image_id": frame["image_id"],
                         "image_sha256": frame["sha256"], "shared_branch_witness": witness,
                         "source_role": source_role, "required_source_verified": source_valid,
                         "motion_started_while_pending": early,
                         "photo_captured_before_reply": early and
                         captured["monotonic_seconds"] < interval["model_request_finished"],
                         "used_by_full_controller": sent_to_controller})
    report = episode / "clean_coverage.json"
    return {"frozen_photo_bound": frozen_bound, "model_timing_source": timing_source, "accepted_materials": accepted,
            "full_report_coverage": bool(report.is_file() and read_json(report)["complete"]),
            "live_report_verifier_accepted": any(row["event"] == "full_task_acceptance" and row["accepted"]
                                                 for row in rows),
            "early_motion_dispatched": frozen_bound and any(
                item["motion_started_while_pending"] for item in accepted),
            "early_motion_with_usable_material": frozen_bound and any(
                item["motion_started_while_pending"] and item["used_by_full_controller"]
                and item["required_source_verified"] for item in accepted),
            "material_completed_before_reply": frozen_bound and any(
                item["photo_captured_before_reply"] and item["used_by_full_controller"]
                and item["required_source_verified"] for item in accepted)}
