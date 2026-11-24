"""Derive initial live scheduling costs from recorded development episodes."""

from dataclasses import asdict
from math import dist
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
