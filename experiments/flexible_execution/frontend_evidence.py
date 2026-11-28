"""Project current-episode camera evidence into the shared controller's records."""

import re
from collections import Counter, defaultdict
from hashlib import sha256

from .gpt_vision import JUDGE
from .live_audit import _camera_frames, _rows, _verified_judgment
from .material_schema import value_matches
from .runner import read_json


_DIRECT_CRACK = re.compile(
    r"(?:(?:(?:图像|照片|画面)中(?:的)?)?(?:该|这(?:台|个)?)?设备(?:的)?(?:正面)?(?:外壳)?是否(?:有|存在)裂纹[?？]?"
    r"|does (?:this|the) (?:device|equipment)(?:'s front (?:housing|surface))? "
    r"(?:have|show) (?:a )?(?:visible )?cracks?[?]?)", re.IGNORECASE)
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
# Static in the clean simulation fixtures. Presence, obstruction and counts
# from changeable views stay with the controller's current observation.
_STATIC_FIELDS = {"cracked", "temperature", "color", "order_count", "restricted"}


def atomic_facts(question, outcome, fields):
    if (outcome in ("positive", "negative") and fields.get("cracked") == "flag"
            and isinstance(question, str) and _DIRECT_CRACK.fullmatch(question.strip())):
        return {"cracked": outcome == "positive"}
    return {}


def _bound_transport(call, reservations, model, effort, available):
    reservation = reservations.get(call.get("logical_job_attempt"), {})
    dispatched, returned = call.get("dispatched_monotonic_seconds"), call.get("return_monotonic_seconds")
    return (isinstance(dispatched, (int, float)) and isinstance(returned, (int, float))
            and dispatched <= returned <= available
            and (call.get("model"), call.get("reasoning_effort")) == (model, effort)
            and (reservation.get("model"), reservation.get("reasoning_effort")) == (model, effort)
            and reservation.get("image_sha256") == call.get("image_sha256"))


def _material_confirmation(part, query, events, calls, frames, reservations, model, effort, available):
    from .adapters.live_inspection import READ_MATERIAL

    photo = frames.get(part.get("frame_id"), {})
    ref = frames.get(query["frame_id"], {})
    if (part.get("provenance") != "simulated_onboard_rgb" or not photo or not ref or
            part["frame_hash"] != photo["sha256"] or part["image_id"] != photo["image_id"]):
        return None
    matches = []
    digest = sha256(READ_MATERIAL.encode("utf8")).hexdigest()
    for row in events:
        if (row["event"] != "evidence_accepted" or row.get("requirement") != part["requirement_id"]
                or row.get("frame_id") != part["frame_id"] or row["monotonic_seconds"] > available):
            continue
        detail = row.get("details") or {}
        if detail.get("reference_sha256") != ref["sha256"] or detail.get("image_sha256") != photo["sha256"]:
            continue
        accepted = row["monotonic_seconds"]
        for call in calls:
            payload, output = call.get("input") or {}, call.get("output")
            if (not isinstance(output, dict) or call.get("instructions_sha256") != digest
                    or not _bound_transport(call, reservations, model, effort, accepted)
                    or call["image_sha256"] != [ref["sha256"], photo["sha256"]]
                    or payload.get("reference_frame_id") != query["frame_id"]
                    or not query.get("bbox_in_attached_photo")
                    or payload.get("reference_bbox") != query["bbox_in_attached_photo"]
                    or payload.get("reference_target") != query["target_description"]
                    or payload.get("current_frame_id") != part["frame_id"] or payload.get("field") != part["field"]
                    or output.get("same_object") is not True or output.get("readable") is not True
                    or output.get("field") != part["field"] or output.get("source_role") != detail.get("source_role")):
                continue
            raw = output.get("value")
            if (str(raw) != part["value"] and not (type(raw) is bool and str(raw).lower() == part["value"])):
                continue
            matches.append((call, photo, accepted))
    return matches[0] if len(matches) == 1 else None


def _source(call, photo, used_at):
    return {"image_id": photo["image_id"], "image_sha256": photo["sha256"],
            "dispatch": call["logical_job_attempt"], "prompt_sha256": call["instructions_sha256"],
            "model": call["model"], "reasoning_effort": call["reasoning_effort"], "ready_monotonic": used_at}


def project(directory, program, task, ledger, *, model, effort, cutoff, expires):
    """Use the original request and response chain, with no evaluator inputs."""
    if ledger is None or cutoff >= expires or not (directory / "conditional_result.json").is_file():
        return [], []
    reservations = {row["reservation"]: row for row in _rows(ledger)}
    calls = _rows(directory / "model_calls.jsonl")
    events = [row for row in _rows(directory / "timeline.jsonl") if row["monotonic_seconds"] <= cutoff]
    frames = _camera_frames(directory)
    summary = read_json(directory / "conditional_result.json")
    materials = defaultdict(list)
    for part in summary.get("accepted_materials", {}).values():
        materials[part["query_id"]].append(part)
    by_step = defaultdict(list)
    for row in events:
        if row["event"] in ("query_submitted", "query_applied"):
            by_step[row["step"]].append(row)
    candidates = []
    judge_prompt = sha256(JUDGE.encode("utf8")).hexdigest()
    for phase in by_step.values():
        applied, call = _verified_judgment(phase, calls, frames)
        if not call or call.get("instructions_sha256") != judge_prompt:
            continue
        judged_at = applied["monotonic_seconds"]
        if not _bound_transport(call, reservations, model, effort, judged_at):
            continue
        binding, payload = applied["binding"], call["input"]
        accepted = []
        for part in materials[binding["id"]]:
            if part["object_id"] == binding["object_id"] and payload.get("target_id") == part["object_id"]:
                match = _material_confirmation(part, payload, events, calls, frames, reservations, model, effort, cutoff)
                if match is not None and match[0]["input"].get("authorized_task") == task:
                    accepted.append((part["field"], *match))
        identities = [(part_call, photo, at) for field, part_call, photo, at in accepted if field == "identity"
                      and part_call["output"]["source_role"] == "body_identity"
                      and isinstance(part_call["output"]["value"], str) and _IDENTITY.fullmatch(part_call["output"]["value"])]
        if len(identities) != 1:
            continue
        identity_call, identity_photo, identity_at = identities[0]
        identity = identity_call["output"]["value"]
        explicit = [(field, part_call["output"]["value"], _source(part_call, photo, at), "frontend_material_verified")
                    for field, part_call, photo, at in accepted if field in _STATIC_FIELDS
                    and value_matches(program.fields.get(field), part_call["output"]["value"])]
        observed = [(field, value, _source(call, frames[binding["frame_id"]], judged_at), "frontend_bound_observation")
                    for field, value in atomic_facts(payload.get("question"), applied["outcome"], program.fields).items()]
        candidates.append((identity, binding, _source(identity_call, identity_photo, identity_at), observed + explicit))
    counts = Counter(identity.casefold() for identity, *_ in candidates)
    records, evidence = [], []
    for identity, binding, association, claims in candidates:
        if counts[identity.casefold()] != 1:
            continue
        for field, value, source, status in claims:
            if any(other == field and other_value != value for other, other_value, _, _ in claims):
                continue
            record = {"object_id": identity, "facts": {field: value}, "evidence_image_id": source["image_id"]}
            records.append(record)
            evidence.append({"object": identity, "field": field, "value": value, "photo": source["image_id"],
                             "image_sha256": source["image_sha256"], "query": binding["id"],
                             "query_input_sha256": binding["input_hash"], "source": source, "identity_proof": association,
                             "available_monotonic": max(source["ready_monotonic"], association["ready_monotonic"]),
                             "valid_until_monotonic": expires, "scope": "current_static_simulation_episode",
                             "status": status})
    return records, evidence
