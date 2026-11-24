"""Derive initial live scheduling costs from recorded development episodes."""

from dataclasses import asdict
import json
from math import dist
from pathlib import Path
from statistics import median

from .live_planning import Costs
from .runner import _hash, read_json


def calibrate(warmup, completed):
    binding_audit = read_json(warmup / "warmup_verification.json")
    if not binding_audit["observed_safe_action_entirely_while_pending"]:
        raise ValueError("Calibration requires a witnessed live motion and photo")
    timing = read_json(warmup / "clean_timing.json")["intervals"]
    motion = {item["step"]: item["seconds"] for item in timing["action"] if item["ok"]}
    robot = warmup / "private" / "robot"
    poses = [read_json(robot / ("probe_%04d.json" % index))["position"] for index in (1, 2, 3)]
    points = [tuple(position[key] for key in ("x", "y", "z")) for position in poses]
    first, second = dist(*points[:2]), dist(*points[1:])
    if first <= second:
        raise ValueError("Calibration requires two different leg lengths")
    per_meter = (motion[1] - motion[2]) / (first - second)
    start = motion[2] - second * per_meter
    task_times = []
    for path in completed:
        record = read_json(path / "result.json")
        coverage = read_json(path / "clean_coverage.json")
        if not (record["execution_completed"] and coverage["complete"]):
            raise ValueError("Continuation calibration must use completed development tasks")
        task_times.append(record["robot_metrics"]["episode_wall_seconds"])
    if not task_times:
        raise ValueError("A measured complete task is required for the continuation reserve")
    judge = [item["seconds"] for item in timing["model_request"] if item["ok"] and item["step"] == 3]
    # Use the slower observed setup/model call as the small-sample bound until
    # enough first-judge samples exist. The estimate does not use the next reply.
    longest = max(item["seconds"] for item in timing["model_request"] if item["ok"])
    evidence = [item["seconds"] for item in timing["observation"] if item["ok"]]
    samples = {"warmup": str(warmup), "completed": [str(path) for path in completed],
               "move_seconds": [motion[1], motion[2]], "travel_m": [first, second],
               "judge_seconds": judge, "complete_task_seconds": task_times, "max_model_step_seconds": longest,
               "observation_seconds": evidence}
    cost = Costs(median(judge), longest, start, per_meter, max(evidence), max(task_times), _hash(samples))
    return {"costs": asdict(cost), "samples": samples, "purpose": "SMALL_DEVELOPMENT_CALIBRATION"}


def calibrate_visual(directory, previous):
    """Fit live timing estimates only from the matched calibration cohort."""
    manifest = read_json(directory / "pilot_manifest.json")
    if manifest["role"] != "VISUAL_METHOD_CALIBRATION":
        raise ValueError("Live cost fitting requires the designated calibration split")
    samples = {"observed_runs": [], "completed_runs": [], "frozen_seconds": [], "photo_seconds": [], "report_seconds": []}
    for case in manifest["cases"]:
        for method in case["order"]:
            job = read_json(directory / case["id"] / method / "job.json")
            if not job.get("artifact_dir"):
                continue
            root = Path(job["artifact_dir"])
            trace = root / "private/robot/timeline.jsonl"
            if not trace.is_file():
                continue
            completed = job["state"] == "FINISHED_VALID"
            with trace.open(encoding="utf8") as handle:
                events = [json.loads(line) for line in handle]
            start = {}
            for event in events:
                name, at = event["event"], event["monotonic_seconds"]
                key = event["step"]
                if name == "model_request_started" and event.get("purpose") == "judge_frozen_photo":
                    start[("model", key)] = at
                elif name == "model_request_finished" and event.get("ok") and ("model", key) in start:
                    samples["frozen_seconds"].append(at - start[("model", key)])
                elif name == "observation_started":
                    start[("photo", key)] = at
                elif name == "observation_finished" and event.get("ok") and ("photo", key) in start:
                    samples["photo_seconds"].append(at - start[("photo", key)])
                elif name == "full_inspection_started":
                    start["report"] = at
                elif name == "report_written" and "report" in start and completed:
                    samples["report_seconds"].append(at - start["report"])
            samples["observed_runs"].append(job["id"])
            if completed:
                samples["completed_runs"].append(job["id"])

    def empirical(values, fraction):
        return sorted(values)[int((len(values) - 1) * fraction)]

    costs = dict(previous["costs"])
    if samples["frozen_seconds"]:
        costs["frozen_query_seconds"] = median(samples["frozen_seconds"])
        costs["frozen_query_ceiling_seconds"] = max(costs["frozen_query_seconds"], empirical(samples["frozen_seconds"], .95))
    if samples["photo_seconds"]:
        costs["observation_seconds"] = empirical(samples["photo_seconds"], .9)
    if samples["report_seconds"]:
        costs["remaining_controller_seconds"] = empirical(samples["report_seconds"], .75)
    samples["inherited_motion_calibration"] = previous["costs"]["calibration_id"]
    costs["calibration_id"] = _hash(samples)
    Costs(**costs)
    return {"costs": costs, "samples": samples, "purpose": "VISUAL_CALIBRATION_ONLY"}
