"""Recheck conditional dispatch and evidence against the raw model and camera records."""

import hashlib
import json

from .gpt_vision import Material
from .live_planning import needs_body_identity, permission_for
from .material_schema import INSPECTION_FIELDS, value_matches
from .runner import read_json


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()] if path.is_file() else []


def _camera_frames(robot):
    lookup = {}
    for path in robot.glob("probe_[0-9][0-9][0-9][0-9].json"):
        item = read_json(path)
        image = path.with_suffix(".jpg")
        if image.is_file() and path.stem == item["image_id"]:
            if hashlib.sha256(image.read_bytes()).hexdigest() == item["sha256"]:
                lookup[item["frame_id"]] = item
    return lookup


def _verified_judgment(events, calls, frames):
    submitted = next((row for row in events if row["event"] == "query_submitted"), {})
    applied = next((row for row in events if row["event"] == "query_applied"), {})
    binding = submitted.get("binding") or {}
    photo = frames.get(binding.get("frame_id"))
    if not photo or binding != applied.get("binding") or applied.get("outcome") not in binding.get("outcomes", ()):
        return {}, {}
    matches = []
    for call in calls:
        payload, outcome = call.get("input", {}), call.get("output")
        if (payload.get("frame_id") != photo["frame_id"] or payload.get("target_id") != binding["object_id"]
                or outcome != {"outcome": applied["outcome"]} or call.get("image_sha256") != [photo["sha256"]]):
            continue
        prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256((photo["sha256"] + "\n" + prompt).encode("utf8")).hexdigest()
        if digest == binding["input_hash"]:
            matches.append(call)
    return (applied, matches[0]) if len(matches) == 1 else ({}, {})


def audit_capture_permissions(episode):
    robot = episode / "private/robot"
    contract = robot / "conditional_contract.json"
    trace = robot / "timeline.jsonl"
    if not contract.is_file() or not trace.is_file():
        return None
    task = read_json(contract)["task"]
    events = _rows(trace)
    skills = {skill["id"]: skill for skill in task["skills"]}
    requirements = {r["id"]: r for r in task["requirements"]}
    grants = {row["field"]: row["source"] for row in events if row["event"] == "permission_checked" and row["early"]}
    applied, _ = _verified_judgment(events, _rows(robot / "model_calls.jsonl"), _camera_frames(robot))
    checked = []
    for row in events:
        if row["event"] != "action_started" or row.get("candidate") not in skills:
            continue
        skill = skills[row["candidate"]]
        gate = skill["permission"]
        allowed = []
        for identifier in skill["covers"]:
            requirement = requirements[identifier]
            condition = requirement["when"]
            if condition is None:
                allowed.append(False)
                continue
            field = requirement["field"]
            permission = permission_for(task["instruction"], Material(field, tuple(condition["outcomes"]), "", None))
            if gate is None:
                allowed.append(permission.early and permission.source == grants.get(field))
            else:
                allowed.append(not permission.early and tuple(gate["outcomes"]) == permission.after
                               and gate["query_id"] == condition["query_id"] == applied.get("binding", {}).get("id")
                               and applied.get("outcome") in permission.after
                               and applied["monotonic_seconds"] <= row["monotonic_seconds"])
        permitted = bool(allowed) and all(allowed)
        checked.append({"step": row["step"], "candidate": row["candidate"], "permitted": permitted})
    return {"scope": "conditional camera actions", "audited": len(checked),
            "violations": sum(not row["permitted"] for row in checked), "actions": checked}


def audit_live_connection(episode):
    robot = episode / "private" / "robot"
    rows = _rows(robot / "timeline.jsonl")
    calls = _rows(robot / "model_calls.jsonl")
    lookup = _camera_frames(robot)
    _, frozen_call = _verified_judgment(rows, calls, lookup)
    frozen_bound = bool(frozen_call)
    contract = robot / "conditional_contract.json"
    task = read_json(contract)["task"]["instruction"] if contract.is_file() else None
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
        field = item["requirement"].rsplit(":", 1)[-1]
        kind = "label" if field == "identity" else INSPECTION_FIELDS.get(field)
        frame = lookup.get(item.get("frame_id"))
        if not frame:
            continue
        details = item.get("details", {})
        verification = {}
        for call in calls:
            payload, answer = call.get("input", {}), call.get("output", {})
            if not isinstance(answer, dict):
                continue
            reference = lookup.get(payload.get("reference_frame_id"), {})
            if (payload.get("field") == field and answer.get("field") == field
                    and payload.get("authorized_task") == task and payload.get("current_frame_id") == frame["frame_id"]
                    and answer.get("readable") is True and answer.get("same_object") is True
                    and value_matches(kind, answer.get("value")) and reference
                    and call.get("image_sha256") == [reference["sha256"], frame["sha256"]]
                    and details.get("reference_sha256") == reference["sha256"]
                    and details.get("image_sha256") == frame["sha256"]
                    and details.get("source_role") == answer.get("source_role")
                    and details.get("object_region") == answer.get("object_bbox")
                    and details.get("material_region") == answer.get("material_bbox")):
                verification = call
                break
        if not verification:
            continue
        captured = next((row for row in rows if row["event"] == "candidate_executed"
                         and row["frame_id"] == frame["frame_id"] and row["image_id"] == frame["image_id"]
                         and row["frame_sha256"] == frame["sha256"]), {})
        started = next((row for row in rows if row["event"] == "action_started"
                        and row["step"] == int(frame["image_id"].split("_", 1)[1])
                        and row.get("candidate") == captured.get("candidate") and row.get("candidate")), {})
        sent_to_controller = any(
            payload.get("authorized_task") == task
            and any(name["image_id"] == frame["image_id"] and digest == frame["sha256"]
                    for name, digest in zip(payload.get("recent_images", []), call.get("image_sha256", [])))
            for call in calls for payload in [call.get("input", {})])
        witness = any(captured.get("candidate") in row["candidate"].split("|") and
                      {value for branch in row["branches"] for value in branch["outcomes"].values()}
                      >= {"positive", "negative", "uncertain"} for row in witnesses)
        early = bool(interval and started and interval["model_request_started"] <= started["monotonic_seconds"]
                     < interval["model_request_finished"] and witness and item["requirement"].split(":")[-1] in grant)
        source_role = verification.get("output", {}).get("source_role")
        requires_body = field == "identity" and needs_body_identity(task or "")
        source_valid = source_role == "body_identity" if requires_body else source_role in ("body_identity", "other")
        accepted.append({"requirement": item["requirement"], "image_id": frame["image_id"], "field": field,
                         "value": verification["output"]["value"],
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
