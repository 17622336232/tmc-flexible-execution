"""Derive initial live scheduling costs from recorded development episodes."""

from dataclasses import asdict
from hashlib import sha256
import json
from itertools import combinations
from math import ceil, dist, isfinite
from pathlib import Path
from statistics import median, quantiles

from .live_planning import Costs
from .runner import _hash, read_json
from .timeline import summarize


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


def _camera_motion_samples(robot, events):
    prior = active = finished = None
    moves = {"move", "approach", "capture_view", "capture_initial_judgment_view",
             "verified_common_handoff", "safe_recovery_return"}
    samples = []
    for event in events:
        name, at, step = event["event"], event["monotonic_seconds"], event["step"]
        if name == "action_started":
            active = (step, at, prior) if event.get("action") in moves else None
            prior = finished = None
        elif name == "action_finished":
            if active and active[0] == step and active[2] and event["ok"]:
                finished = (step, at - active[1], active[2])
            active = None
        elif name == "observation_finished" and event["ok"]:
            image_id = event.get("image_id") or f"probe_{step:04d}"
            source = robot / (image_id + ".json")
            meta = read_json(source) if source.is_file() else {}
            image = source.with_suffix(".jpg")
            position = meta.get("position", {})
            point = tuple(position.get(axis) for axis in ("x", "y", "z"))
            if (meta.get("image_id") != image_id or not image.is_file()
                    or sha256(image.read_bytes()).hexdigest() != meta.get("sha256")
                    or any(not isinstance(v, (int, float)) or not isfinite(v) for v in point)):
                prior = finished = None
                continue
            if finished and finished[0] <= step and finished[1] > 0:
                origin = finished[2]
                meters = dist(origin["point"], point)
                if meters >= 0.1:
                    samples.append({"meters": meters, "seconds": finished[1], "action_step": finished[0],
                                    "from_frame": origin["frame"], "to_frame": image_id})
            prior, finished = {"point": point, "frame": image_id}, None
    return samples


def _motion_fit(samples):
    data = sorted((row["meters"], row["seconds"]) for row in samples)
    if len(data) < 8 or data[-1][0] - data[0][0] < 1.0:
        return None
    slopes = [(second_time - first_time) / (second_length - first_length)
              for (first_length, first_time), (second_length, second_time) in combinations(data, 2)
              if second_length - first_length >= 0.5]
    slope = median(slopes)
    start = median(seconds - meters * slope for meters, seconds in data)
    return (start, slope) if start > 0 and slope > 0 else None


def calibrate_visual(directory, previous):
    """Fit live timing estimates only from the matched calibration cohort."""
    manifest = read_json(directory / "pilot_manifest.json")
    if manifest["role"] != "VISUAL_METHOD_CALIBRATION":
        raise ValueError("Live cost fitting requires the designated calibration split")
    samples = {"observed_runs": [], "completed_runs": [], "frozen_seconds": [], "censored_query_seconds": [],
               "photo_seconds": [], "report_seconds": [], "measured_motion": []}
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
            samples["measured_motion"].extend({**sample, "job": job["id"]} for sample in
                                               _camera_motion_samples(trace.parent, events))
            start = {}
            for event in events:
                name, at = event["event"], event["monotonic_seconds"]
                key = event["step"]
                if name == "model_request_started" and event.get("purpose") == "judge_frozen_photo":
                    start[("model", key)] = at
                elif name == "model_request_finished" and event.get("purpose") == "judge_frozen_photo":
                    began = start.pop(("model", key), None)
                    if began is not None and event.get("ok"):
                        samples["frozen_seconds"].append(at - began)
                    elif began is not None and event.get("error_type") in ("ModelDeadlineExceeded", "TimeoutError"):
                        samples["censored_query_seconds"].append(at - began)
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
        return sorted(values)[ceil(len(values) * fraction) - 1]

    costs = dict(previous["costs"])
    if samples["frozen_seconds"]:
        costs["frozen_query_seconds"] = median(samples["frozen_seconds"])
        costs["frozen_query_ceiling_seconds"] = max(costs["frozen_query_seconds"],
                                                   empirical(samples["frozen_seconds"], .95))
    if samples["censored_query_seconds"]:
        costs["frozen_query_ceiling_seconds"] = max(costs["frozen_query_ceiling_seconds"],
            previous["costs"]["frozen_query_ceiling_seconds"], max(samples["censored_query_seconds"]))
    if samples["photo_seconds"]:
        costs["observation_seconds"] = empirical(samples["photo_seconds"], .9)
    if samples["report_seconds"]:
        costs["remaining_controller_seconds"] = empirical(samples["report_seconds"], .75)
    motion = _motion_fit(samples["measured_motion"])
    if motion is not None:
        costs["move_start_seconds"], costs["move_seconds_per_meter"] = motion
        samples["motion_fit"] = "median_pairwise_slopes_and_median_intercept"
    else:
        samples["inherited_motion_calibration"] = previous["costs"]["calibration_id"]
    costs["calibration_id"] = _hash(samples)
    Costs(**costs)
    return {"costs": costs, "samples": samples, "purpose": "VISUAL_CALIBRATION_ONLY"}


def calibrate_public_visual(directory, catalog):
    """Fit the public comparison's planning estimates from its registered validation houses."""
    from .public_schedule import VALIDATION_ROLE
    from .public_task import MODEL_TIMES, PLATFORM_TIMES

    if catalog.get("role") != VALIDATION_ROLE:
        raise ValueError("Public timing calibration requires the registered validation cohort")
    by_house = {}
    skipped_arms = []
    for case in catalog["cases"]:
        sample = by_house.setdefault(case["group_id"], {key: [] for key in ("front", "side", "report", "turn", "photo")})
        for method in case["order"]:
            job = read_json(directory / case["id"] / method / "job.json")
            if job["state"] not in ("FINISHED_VALID", "FINISHED_TASK_FAILURE", "TIMEOUT"):
                raise ValueError("An incomplete public validation job cannot calibrate the final comparison")
            trace = Path(job["artifact_dir"]) / "timeline.jsonl" if job.get("artifact_dir") else None
            if trace is None or not trace.is_file():
                skipped_arms.append(job["id"])
                continue
            intervals = summarize(trace, allow_truncated=job["state"] == "TIMEOUT")["intervals"]
            for kind, key in (("model_request", None), ("action", "turn"), ("observation", "photo")):
                for interval in intervals.get(kind, []):
                    field = key or interval["step"]
                    seconds = interval["seconds"]
                    if field in sample and interval["ok"] and isfinite(seconds) and seconds > 0:
                        sample[field].append(seconds)

    def by_source(field):
        return [median(samples[field]) for samples in by_house.values() if samples[field]]

    def upper(values):
        return quantiles(values, n=4, method="inclusive")[2] if len(values) > 1 else values[0]

    front, side, report = (by_source(field) for field in ("front", "side", "report"))
    inference = dict(MODEL_TIMES)
    inherited = []
    if front:
        inference["initial_prediction"] = min(median(front), inference["initial_ceiling"])
    else:
        inherited.append("initial_prediction")
    if (front or side) and report:
        inference["remaining_judgments"] = max(upper(values) for values in (front, side) if values) + upper(report)
    else:
        inherited.append("remaining_judgments")
    motion = dict(PLATFORM_TIMES)
    for field in motion:
        values = by_source(field)
        if values:
            motion[field] = upper(values)
        else:
            inherited.append(field)
    return {"schema": 1, "motion": motion, "inference": inference, "samples_by_house": by_house,
            "arms_without_trace": skipped_arms, "inherited_estimates": inherited,
            "method": "equal_house_medians_and_upper_quartiles_of_completed_development_intervals"}
