"""Preserve and independently recheck a public-camera inspection's evidence."""

from hashlib import sha256
import json

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
    ledger = {row["reservation"]: row for row in _rows(root / "requests.jsonl")}
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
    budget_verified = (len(ledger) == result["requests"]["reserved"]
                       and sorted(ledger) == list(range(1, len(ledger) + 1))
                       and sorted(call["logical_job_attempt"] for call in calls if "logical_job_attempt" in call) == sorted(ledger)
                       and all(ledger[call["logical_job_attempt"]]["image_sha256"] == call["image_sha256"]
                               for call in calls if "logical_job_attempt" in call))
    motion = next((event["monotonic_seconds"] for event in timeline if event["event"] == "action_started"), None)
    shutter = next((event["monotonic_seconds"] for event in timeline if event["event"] == "camera_observation"
                    and event["step"] == "side"), None)
    sent, returned = front["dispatched_monotonic_seconds"], front["return_monotonic_seconds"]
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
            "report_matches_saved_score": result["expected"] == report, "raw_segmentation_scores": raw_truth,
            "recomputed_segmentation_agrees": None if raw_truth is None else
            all(report[field] == value for field, value in raw_truth.items())}
