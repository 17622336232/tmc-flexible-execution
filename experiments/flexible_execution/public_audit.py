"""Preserve and independently recheck a public-camera inspection's evidence."""

from hashlib import sha256
import json
from pathlib import Path

from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES
from experiments.task_attack.runtime import write_json

from .runner import read_json


def save_segmentation(root, name, event, category, minimum_pixels):
    from PIL import Image

    scoring = root / "scoring"
    scoring.mkdir(exist_ok=True)
    target_ids = {item["objectId"] for item in event.metadata["objects"] if item["objectType"] == category}
    colors = [[int(channel) for channel in color] for color, identity in event.color_to_object_id.items()
              if identity in target_ids]
    path = scoring / (name + ".png")
    Image.fromarray(event.instance_segmentation_frame).save(path)
    write_json(scoring / (name + ".json"), {"category": category, "minimum_instance_pixels": minimum_pixels,
               "colors": colors, "segmentation_sha256": sha256(path.read_bytes()).hexdigest()})


def _grade_segmentation(scoring, name):
    import numpy as np
    from PIL import Image

    data = read_json(scoring / (name + ".json"))
    path = scoring / (name + ".png")
    if sha256(path.read_bytes()).hexdigest() != data["segmentation_sha256"]:
        raise ValueError("The saved ground-truth image does not match its checksum")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    present = any(int(np.count_nonzero(np.all(rgb == color, axis=-1))) >= data["minimum_instance_pixels"]
                  for color in data["colors"])
    return "positive" if present else "negative"


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def audit(root):
    result = read_json(root / "result.json")
    report = read_json(root / "report.json")
    calls, timeline = _rows(root / "model_calls.jsonl"), _rows(root / "timeline.jsonl")
    request_meta = result["requests"]
    local_ledger = root / "requests.jsonl"
    ledger_path = local_ledger if local_ledger.is_file() else Path(request_meta["ledger"])
    ledger_rows = _rows(ledger_path)
    ledger = {row["reservation"]: row for row in ledger_rows}
    expected_reservations = list(range(request_meta.get("reserved_before_attempt", 0) + 1,
                                       request_meta["reserved"] + 1))
    hashes = {name: sha256((root / (name + ".jpg")).read_bytes()).hexdigest()
              for name in ("front", "side") if (root / (name + ".jpg")).is_file()}
    front = next(call for call in calls if call["input"].get("frame") == "front" and "output" in call)
    side = next((call for call in calls if call["input"].get("frame") == "side" and "output" in call), None)
    handoff = next(call for call in calls if call.get("output") == report and "front_sha256" in call["input"])
    submitted = next(event["binding"] for event in timeline if event["event"] == "query_submitted")
    applied = next(event for event in timeline if event["event"] == "query_applied")
    prompt = json.dumps(front["input"], sort_keys=True)
    binding_hash = sha256((hashes["front"] + "\n" + prompt).encode("utf8")).hexdigest()
    uses_side = report.get("front") == "positive"
    allowed = [hashes["front"], hashes.get("side")] if uses_side else [hashes["front"]]
    photos_bound = (front["image_sha256"] == [hashes["front"]] and report["front_sha256"] == hashes["front"]
                    and handoff["image_sha256"] == allowed and (not uses_side or
                        (side and side["image_sha256"] == [hashes["side"]] and report["side_sha256"] == hashes["side"])))
    budget_verified = (request_meta["reserved"] <= len(ledger_rows) <= request_meta["limit"]
                       and [row["reservation"] for row in ledger_rows] == list(range(1, len(ledger_rows) + 1))
                       and all(row["limit"] == request_meta["limit"] for row in ledger_rows)
                       and sorted(call["logical_job_attempt"] for call in calls if "logical_job_attempt" in call)
                       == expected_reservations
                       and all(ledger[call["logical_job_attempt"]]["image_sha256"] == call["image_sha256"]
                               for call in calls if "logical_job_attempt" in call))
    motion = next((event["monotonic_seconds"] for event in timeline if event["event"] == "action_started"), None)
    shutter = next((event["monotonic_seconds"] for event in timeline if event["event"] == "camera_observation"
                    and event["step"] == "side"), None)
    sent, returned = front["dispatched_monotonic_seconds"], front["return_monotonic_seconds"]
    terminal_pose = next((event["pose"] for event in timeline if event["event"] == "terminal_pose"), None)
    pose_verified = permission_verified = None
    if terminal_pose is not None:
        from .public_task import TURNS, _same_pose

        observations = {event["step"]: event for event in timeline if event["event"] == "camera_observation"}
        start_pose = observations["front"]["pose"]
        turn, restore, angle = TURNS[result["parsed"]["direction"]]
        acts = [event["step"] for event in timeline if event["event"] == "action_started"]
        pose_verified = (_same_pose(start_pose, terminal_pose) and (acts == [turn, restore]
                         and _same_pose(start_pose, observations["side"]["pose"], angle)
                         if "side" in observations else not acts))
        grants = [event for event in timeline if event["event"] == "approved_preparation"]
        source = read_json(root / "source.json")
        permission_verified = (len(grants) == 1 and grants[0]["permission"]
                               == "采集规则：允许在判读结果返回前采集以下材料：side_photo。"
                               and grants[0]["permission"] in source["instruction"].splitlines()
                               if motion is not None and motion < returned else True)
    raw_truth = None
    scoring = root / "scoring"
    if (scoring / "front.png").is_file():
        raw_truth = {"front": _grade_segmentation(scoring, "front"),
                     "side": _grade_segmentation(scoring, "side") if uses_side else None}
    return {"reported_complete": result["complete"], "camera_photos_match_model_and_report": bool(photos_bound),
            "send_ledger_matches_all_requests": budget_verified,
            "frozen_reply_matches_request": submitted == applied["binding"] and binding_hash == submitted["input_hash"]
            and front["output"]["outcome"] == applied["outcome"],
            "first_motion_seconds_after_dispatch": None if motion is None else motion - sent,
            "first_motion_seconds_before_reply": None if motion is None else returned - motion,
            "side_capture_seconds_after_reply": None if shutter is None else shutter - returned,
            "terminal_heading_verified_by_runner": result["restored_heading"],
            "poses_match_the_contracted_turn_and_return": pose_verified,
            "early_camera_motion_had_permission": permission_verified,
            "report_matches_saved_score": result["expected"] == report, "raw_segmentation_scores": raw_truth,
            "recomputed_segmentation_agrees": None if raw_truth is None else
            all(report[field] == value for field, value in raw_truth.items())}


def registered_outcome(root, job, charged, returncode, stop):
    detail = {"output": str(root), "charged_seconds": charged, "returncode": returncode}
    if stop in {"TIMEOUT", "CANCELLED_BY_USER"}:
        return stop, dict(detail, reason="supervisor")
    try:
        path = root / "result.json"
        if not path.is_file():
            return "INFRA_ERROR", dict(detail, reason="public worker exited without a result")
        result, source = read_json(path), read_json(root / "source.json")
        case = job["case"]
        expected = {**job["public_runtime"]["source"], "case_id": case["id"], "instruction": case["instruction"],
                    "method": job["method"], "model": job["model"], "reasoning_effort": job["reasoning_effort"],
                    "house_index": case["official_house_index"], "request_limit": job["request_limit"],
                    "ground_truth_input_to_model": False}
        if any(source.get(name) != value for name, value in expected.items()):
            raise ValueError("The room result does not match its frozen case and model")
        if result["requests"]["ledger"] != job["request_ledger"]:
            raise ValueError("The worker used a different request ledger")
        if result.get("cleanup_error"):
            return "INFRA_ERROR", dict(detail, reason="simulator cleanup", error=result["cleanup_error"])
        error = result.get("error", {})
        sent = result["requests"]["reserved"] - result["requests"]["reserved_before_attempt"]
        detail.update(score="correct" if result["complete"] else "incomplete", coverage_complete=result["complete"],
                      logged_dispatches=sent, episode_seconds=result["episode_seconds"],
                      model_action_overlap_seconds=result["model_action_overlap_seconds"])
        if error.get("code") in EXTERNAL_RESPONSE_CODES or error.get("status") in (401, 403, 429):
            return "BLOCKED_EXTERNAL", dict(detail, reason=error)
        if (result["episode_seconds"] > source["episode_timeout"] or
                error.get("category") in {"episode_deadline", "request_budget_exhausted"}):
            return "TIMEOUT", dict(detail, reason=error.get("category") or "episode deadline")
        if result["complete"]:
            check = audit(root)
            required = ("camera_photos_match_model_and_report", "send_ledger_matches_all_requests",
                        "frozen_reply_matches_request", "terminal_heading_verified_by_runner",
                        "report_matches_saved_score", "recomputed_segmentation_agrees",
                        "poses_match_the_contracted_turn_and_return", "early_camera_motion_had_permission")
            if not all(check.get(key) is True for key in required):
                raise ValueError("The claimed complete room report failed independent evidence verification")
            if result["parsed"] != case["expected_language_contract"]:
                raise ValueError("A completed report has a different interpreted task")
            return "FINISHED_VALID", detail
        invalid_output = {"invalid_json", "invalid_output", "unsupported_output", "inconsistent_output", "incomplete_output"}
        if not error or error.get("category") == "task_contract" or error.get("code") in invalid_output:
            return "FINISHED_TASK_FAILURE", dict(detail, reason=error or "report failed independent scoring")
        return "INFRA_ERROR", dict(detail, reason=error)
    except (KeyError, OSError, ValueError, TypeError, StopIteration) as error:
        return "INFRA_ERROR", dict(detail, reason="public artifact verification: " + str(error))
