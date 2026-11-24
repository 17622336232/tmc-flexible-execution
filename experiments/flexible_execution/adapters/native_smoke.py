"""Exercise real CPU-simulator sensing and navigation with a controlled query barrier."""

import argparse
import asyncio
import math
import os
import threading
from hashlib import sha256
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from experiments.task_attack.runtime import write_json
from experiments.task_attack_integrity.run import main as legacy_run

from ..evidence import Frame
from ..queries import AsyncQueryWorker, QueryBook
from ..task_ir import QuerySpec


async def _exercise(backend, observation, directory, log):
    if backend.config["mode"] != "gazebo":
        raise RuntimeError("This probe requires the simulated control backend")
    begun = threading.Event()
    allow_reply = threading.Event()
    measured = {}

    def inspect_frame(payload, prompt):
        started = monotonic()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        measured["decode_end"] = monotonic()
        measured["decode_seconds"] = measured["decode_end"] - started
        outcome = "decodable" if image is not None and image.size else "unreadable"
        begun.set()
        if not allow_reply.wait(timeout=45):
            raise RuntimeError("The controlled query moved past its release deadline")
        return outcome

    received = monotonic()
    spec = QuerySpec("rgb-diagnostic", "initial-station", ("decodable", "unreadable"), 0, 45)
    frame = Frame(observation["observation_id"], spec.object_id, 0, received,
                  observation["data"], "onboard_cpu_simulator_rgb")
    book = QueryBook()
    worker = AsyncQueryWorker(inspect_frame)
    future = worker.submit(spec, frame, "Decode this actual camera frame", monotonic(), book)
    loop = asyncio.get_running_loop()
    try:
        if not await loop.run_in_executor(None, begun.wait, 5):
            raise RuntimeError("The image worker failed to process the observation")
        if future.done() or spec.id not in book.pending:
            raise RuntimeError("The controlled query was not pending before movement")
        start = observation["position"]
        yaw = observation["yaw"]
        goal = {"x": start["x"] + 0.7 * math.cos(yaw),
                "y": start["y"] + 0.7 * math.sin(yaw), "z": start["z"], "yaw": yaw}
        backend.execution_deadline_monotonic = monotonic() + 30
        moving_at = monotonic()
        log("async_probe_independent_motion_started", time_monotonic=moving_at, query_pending=True)
        arrived = await loop.run_in_executor(None, backend.move, goal)
        arrived_at = monotonic()
        if future.done():
            raise RuntimeError("The probe lost its query before the independent move completed")
        after, _ = await loop.run_in_executor(None, backend.observe)
        observed_at = monotonic()
        (directory / "onboard_after_async_move.jpg").write_bytes(after["data"])
        allow_reply.set()
        reply = await asyncio.wait_for(future, timeout=5)
        returned = monotonic()
        if not book.accept(reply, returned) or reply.outcome != "decodable":
            raise RuntimeError("The original onboard frame or its query did not verify")
        distance = math.hypot(arrived["position"]["x"] - start["x"], arrived["position"]["y"] - start["y"])
        image_hash = sha256(after["data"]).hexdigest()
        if distance < 0.25 or image_hash == frame.digest:
            raise RuntimeError("The independent move did not produce a new measured camera viewpoint")
        write_json(directory / "async_navigation_probe.json", {
            "mode": "CPU_SIM_ACTUAL_MOTION_AND_RGB_CONTROLLED_QUERY",
            "first_rgb_sha256": frame.digest,
            "next_rgb_sha256": image_hash,
            "first_rgb_ros_capture": observation["ros_stamp"],
            "next_rgb_ros_capture": after["ros_stamp"],
            "first_rgb_local_receipt": received,
            "actual_decoding_seconds": measured["decode_seconds"],
            "decoding_complete_monotonic": measured["decode_end"],
            "query_submission_monotonic": reply.binding.submitted_at,
            "independent_motion_started_monotonic": moving_at,
            "independent_motion_arrived_monotonic": arrived_at,
            "next_observation_monotonic": observed_at,
            "query_returned_monotonic": returned,
            "distance_m": distance,
            "query_pending_during_independent_motion": moving_at < arrived_at < returned,
            "query_result_matches_original_frame": True,
            "release": "Test barrier after real motion and next frame, not a measured VLM delay",
        })
        log("async_probe_complete", distance_m=distance)
    finally:
        allow_reply.set()
        if not future.done():
            future.cancel()
        worker.close()


def probe(backend, observation, directory, log):
    asyncio.run(_exercise(backend, observation, directory, log))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ros-port", default=11672, type=int)
    parser.add_argument("--gazebo-port", default=11673, type=int)
    parser.add_argument("--onboard-evidence-only", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("ROS_HOME", str(args.output.resolve() / "ros_home"))
    code = legacy_run([
        "--backend", "cpu", "--headless", "--no-rviz", "--sensor-smoke",
        "--family", "cracks", "--policy", "clean", "--source-profile", "single",
        "--equipment-count", "2", "--seed", "20261124", "--output", str(args.output),
        "--ros-port", str(args.ros_port), "--gazebo-port", str(args.gazebo_port),
    ], sensor_exercise=probe, archive_external_rgb=not args.onboard_evidence_only)
    if args.onboard_evidence_only:
        from ..storage import archive_sensor_logs

        archive_sensor_logs(args.output.resolve())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
