#!/usr/bin/env python3
"""Read-only acceptance of real rays, hit-only registration, and map evidence.

Connects to ROS_MASTER_URI; never launches processes, changes the world, or sends
flight commands. The only publication is a PointStamped map-query request. No
world/landmark/map files are read. Run soon after a world starts, with the drone
stationary, so a surface can still have unobserved space behind it.

Example:
  ROS_MASTER_URI=http://127.0.0.1:11342 python3 sensor_evidence_test.py \
      --output /tmp/sensor-evidence.json

Exit codes: 0 = all required evidence observed; 1 = violated data contract;
2 = inconclusive (missing stream, moving observer, or no suitable occlusion).
"""
import argparse
from collections import Counter, OrderedDict, deque
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


RAY_FIELDS = ("x", "y", "z", "hit", "origin_x", "origin_y", "origin_z",
              "body_x", "body_y", "body_z")


class Failed(Exception):
    pass


class Inconclusive(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Failed(message)


def xyz(message):
    values = list(point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=False))
    return np.asarray(values, dtype=np.float64).reshape((-1, 3))


class ObservationClient:
    def __init__(self, args):
        self.args = args
        self.changed = threading.Condition()
        self.rays, self.clouds = OrderedDict(), OrderedDict()
        self.responses = deque(maxlen=512)
        self.latest_rays = None
        self.protocol_errors = []
        self.subscribers = [
            rospy.Subscriber(args.rays_topic, PointCloud2, self.on_rays, queue_size=3, buff_size=8*1024*1024),
            rospy.Subscriber(args.registered_topic, PointCloud2, self.on_cloud, queue_size=3, buff_size=8*1024*1024),
            rospy.Subscriber(args.result_topic, String, self.on_result, queue_size=30),
        ]
        self.query_pub = rospy.Publisher(args.query_topic, PointStamped, queue_size=10)

    def store(self, target, message):
        with self.changed:
            target[message.header.stamp.to_nsec()] = message
            while len(target) > 24:
                target.popitem(last=False)
            self.changed.notify_all()

    def on_rays(self, message):
        self.latest_rays = message
        self.store(self.rays, message)

    def on_cloud(self, message):
        self.store(self.clouds, message)

    def on_result(self, message):
        try:
            response = json.loads(message.data)
            required = ("seq", "x", "y", "z", "state", "inflated_blocked", "scan_stale")
            if not isinstance(response, dict) or any(key not in response for key in required):
                raise ValueError("Missing required query-result fields")
            if response["state"] not in ("unknown", "free", "occupied", "outside"):
                raise ValueError("Invalid evidence state")
            if not isinstance(response["inflated_blocked"], bool) or not isinstance(response["scan_stale"], bool):
                raise ValueError("Query safety flags must be JSON booleans")
            response["_received_wall"] = time.monotonic()
            with self.changed:
                self.responses.append(response)
                self.changed.notify_all()
        except (ValueError, TypeError) as exc:
            with self.changed:
                self.protocol_errors.append(str(exc))
                self.changed.notify_all()

    def collect(self):
        deadline = time.monotonic() + self.args.stream_timeout
        pairs = []
        seen = set()
        with self.changed:
            while len(pairs) < self.args.frames and not rospy.is_shutdown():
                for stamp in sorted(set(self.rays).intersection(self.clouds) - seen):
                    seen.add(stamp)
                    pairs.append((self.rays[stamp], self.clouds[stamp]))
                    if len(pairs) == self.args.frames:
                        break
                if len(pairs) == self.args.frames:
                    return pairs
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.changed.wait(min(.25, remaining))
        raise Inconclusive("Received only %s/%s exact-timestamp ray/registered pairs within %.1f wall seconds" %
                           (len(pairs), self.args.frames, self.args.stream_timeout))

    def query(self, point):
        deadline = time.monotonic() + self.args.query_timeout
        while not self.query_pub.get_num_connections() and time.monotonic() < deadline and not rospy.is_shutdown():
            time.sleep(.02)
        if not self.query_pub.get_num_connections():
            raise Inconclusive("Map-query subscriber is unavailable")
        message = PointStamped()
        message.header.frame_id = "world"
        message.header.stamp = rospy.Time.now()
        message.point.x, message.point.y, message.point.z = map(float, point)
        sent = time.monotonic()
        self.query_pub.publish(message)
        # rospy serializes synchronously and replaces Header.seq with its own
        # publication sequence. Read it AFTER publish, rather than guessing.
        seq = int(message.header.seq)
        with self.changed:
            while time.monotonic() < deadline and not rospy.is_shutdown():
                if self.protocol_errors:
                    raise Failed("Malformed map query response: " + self.protocol_errors[-1])
                for response in self.responses:
                    if response["_received_wall"] < sent or response["seq"] != seq:
                        continue
                    returned = np.array([response["x"], response["y"], response["z"]], dtype=float)
                    if not np.allclose(returned, point, atol=1e-3, rtol=0):
                        continue  # Another diagnostic publisher may share a sequence number.
                    result = {key: value for key, value in response.items() if not key.startswith("_")}
                    result["round_trip_wall_seconds"] = time.monotonic() - sent
                    require(not result["scan_stale"], "Map reports stale scan during query")
                    return result
                self.changed.wait(min(.05, max(0, deadline-time.monotonic())))
        raise Inconclusive("Map query seq=%s timed out after %.1f wall seconds" % (seq, self.args.query_timeout))

    def close(self):
        for subscriber in self.subscribers:
            subscriber.unregister()
        self.query_pub.unregister()


def verify_streams(pairs, args, report):
    arrays = []
    origins, bodies, ranges = [], [], []
    hits_total = misses_total = registered_total = 0
    frame_stats = []
    for rays_message, cloud_message in pairs:
        require(rays_message.header.frame_id == "world" and cloud_message.header.frame_id == "world",
                "Ray and registered clouds must both be in world coordinates")
        fields = {field.name: field for field in rays_message.fields}
        require(all(name in fields for name in RAY_FIELDS), "Ray cloud lacks hit/origin/body metadata")
        require(fields["hit"].datatype == PointField.UINT8, "hit field must be UINT8")
        ray = np.asarray(list(point_cloud2.read_points(rays_message, field_names=RAY_FIELDS,
                                                      skip_nans=False)), dtype=np.float64).reshape((-1, 10))
        require(len(ray) > 0 and np.isfinite(ray).all(), "Ray cloud is empty or contains non-finite values")
        require(np.isin(ray[:, 3], [0, 1]).all(), "Ray hit flags must be 0 or 1")
        require(np.max(np.linalg.norm(ray[:, 4:7]-ray[0, 4:7], axis=1)) < 1e-4,
                "A ray frame contains inconsistent sensor origins")
        require(np.max(np.linalg.norm(ray[:, 7:10]-ray[0, 7:10], axis=1)) < 1e-4,
                "A ray frame contains inconsistent body positions")
        distance = np.linalg.norm(ray[:, :3]-ray[:, 4:7], axis=1)
        require(np.max(distance) <= args.range_max + .05,
                "A ray endpoint exceeds sensor range_max + 0.05 m, measured from the sensor origin")
        offset = float(np.linalg.norm(ray[0, 4:7]-ray[0, 7:10]))
        require(abs(offset-args.expected_extrinsic) <= args.extrinsic_tolerance,
                "Observed sensor-to-body offset %.6f does not match %.6f m" % (offset, args.expected_extrinsic))
        hit = ray[:, 3] == 1
        miss = ~hit
        if miss.any():
            require(np.max(np.abs(distance[miss]-args.range_max)) <= args.shell_margin+2e-6,
                    "A no-return ray does not terminate in the maximum-range shell")
        # Genuine near-limit hits are retained. Their range must not be
        # rejected using the obsolete 8 cm shell heuristic. Exact hit/miss
        # subset checks below establish that limit rays do not enter the cloud.
        registered = xyz(cloud_message)
        require(np.isfinite(registered).all(), "Registered cloud contains non-finite points")
        require(len(registered) == int(hit.sum()), "Registered point count differs from the real-hit count")
        require(np.allclose(registered, ray[hit, :3], atol=1e-5, rtol=0),
                "Registered cloud is not exactly the ordered true-hit subset of the synchronized rays")
        hits_total += int(hit.sum())
        misses_total += int(miss.sum())
        registered_total += len(registered)
        origins.append(ray[0, 4:7])
        bodies.append(ray[0, 7:10])
        ranges.append(distance)
        arrays.append(ray)
        frame_stats.append({"stamp": rays_message.header.stamp.to_sec(), "rays": len(ray),
                            "hits": int(hit.sum()), "no_return": int(miss.sum()),
                            "registered": len(registered), "maximum_sensor_range": float(distance.max()),
                            "sensor_to_body_offset": offset})
    origins, bodies = np.asarray(origins), np.asarray(bodies)
    body_motion = float(np.max(np.linalg.norm(bodies-bodies[0], axis=1)))
    origin_motion = float(np.max(np.linalg.norm(origins-origins[0], axis=1)))
    report["sensor_statistics"] = {
        "matched_frames": len(pairs), "rays": sum(len(a) for a in arrays), "hits": hits_total,
        "no_return_rays": misses_total, "registered_points": registered_total,
        "maximum_range_from_sensor_origin": float(np.max(np.concatenate(ranges))),
        "body_motion_m": body_motion, "sensor_origin_motion_m": origin_motion,
        "registered_is_exact_true_hit_subset": True, "registered_has_max_range_shell": False,
        "frames": frame_stats,
    }
    if max(body_motion, origin_motion) > args.stationary_tolerance:
        raise Inconclusive("Observer moved during collection; run just after startup while stationary")
    if not hits_total or not misses_total:
        raise Inconclusive("Both real hits and no-return rays are required to exercise shell filtering")
    return arrays[-1]


def occlusion_candidates(ray, args):
    """Rank actual near-horizontal returns; prefer dense planar support.

    World axes only index observed points. This does not read scene geometry or
    assume that any location is a wall. Candidate responses remain authoritative.
    """
    delta = ray[:, :3]-ray[:, 4:7]
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(distance[:, None], 1e-9)
    valid = ((ray[:, 3] == 1) & (distance >= 1.0) & (distance <= min(6, args.range_max-.5)) &
             (np.abs(direction[:, 2]) < .3) & (ray[:, 2] > .20))
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    xy_bins = np.rint(ray[indices, :2]/.05).astype(int)
    x_counts = Counter(xy_bins[:, 0])
    y_counts = Counter(xy_bins[:, 1])
    rank = []
    for row, idx in enumerate(indices):
        support = max(x_counts[xy_bins[row, 0]], y_counts[xy_bins[row, 1]])
        rank.append((-support, distance[idx], abs(direction[idx, 2]), int(idx)))
    rank.sort()
    chosen, cells = [], set()
    for _, _, _, idx in rank:
        # Space samples apart so all queries are not inside one tiny voxel.
        cell = tuple(np.rint(ray[idx, :3]/.3).astype(int))
        if cell in cells:
            continue
        cells.add(cell)
        chosen.append((ray[idx], direction[idx], float(distance[idx])))
        if len(chosen) >= args.candidates:
            break
    return chosen


def verify_queries(client, ray, args, report):
    origin = ray[0, 4:7]
    outside_results = []
    for vector in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
        point = origin + np.asarray(vector)*(args.range_max*4)
        result = client.query(point)
        outside_results.append(result)
        require(result["state"] in ("outside", "unknown"),
                "Far beyond current sensor range was classified as " + result["state"])
        require(result["inflated_blocked"], "Unobserved/outside space must remain blocked for planning")
    report["out_of_range_queries"] = outside_results
    attempts = []
    report["occlusion_queries"] = attempts
    report["occlusion_note"] = (
        "Samples come exclusively from real ray returns. A behind-surface point can legitimately be free "
        "if observed from another direction earlier; those candidates are recorded but do not establish "
        "the required unknown-space evidence. No universal behind-surface-unknown assumption is made.")
    for observed, direction, distance in occlusion_candidates(ray, args):
        endpoint = observed[:3]
        before = endpoint - direction*args.before_distance
        behind = endpoint + direction*args.behind_distance
        attempt = {"observed_hit": endpoint.tolist(), "sensor_origin": observed[4:7].tolist(),
                   "hit_distance": distance, "before_distance": args.before_distance,
                   "behind_distance": args.behind_distance}
        attempt["before"] = client.query(before)
        attempt["hit"] = client.query(endpoint)
        attempt["behind"] = client.query(behind)
        attempts.append(attempt)
        if (attempt["before"]["state"] == "free" and attempt["hit"]["state"] == "occupied" and
                attempt["behind"]["state"] == "unknown"):
            require(attempt["behind"]["inflated_blocked"], "Unknown behind-surface voxel was not blocked")
            report["verified_free_occupied_unknown_triplet"] = attempt
            break
    else:
        states = Counter(a["behind"]["state"] for a in attempts)
        report["behind_state_counts"] = dict(states)
        raise Inconclusive("No free→occupied→unknown triplet in %s observed candidates; "
                           "restart with a stationary near-wall observation before historical viewpoints accumulate" % len(attempts))
    report["behind_state_counts"] = dict(Counter(a["behind"]["state"] for a in attempts))
    latest = client.latest_rays
    if latest is not None:
        first = next(point_cloud2.read_points(latest, field_names=("body_x", "body_y", "body_z"), skip_nans=False), None)
        if first is not None and np.linalg.norm(np.asarray(first)-ray[0, 7:10]) > args.stationary_tolerance:
            raise Inconclusive("Observer moved while map evidence was being queried")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("/tmp/vla-sensor-evidence.json"))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--stream-timeout", type=float, default=30)
    parser.add_argument("--query-timeout", type=float, default=2.5)
    parser.add_argument("--range-max", type=float, default=8.0, help="Expected physical range; fast=8, dense=12")
    parser.add_argument("--shell-margin", type=float, default=2e-6)
    parser.add_argument("--expected-extrinsic", type=float, default=.155)
    parser.add_argument("--extrinsic-tolerance", type=float, default=.005)
    parser.add_argument("--stationary-tolerance", type=float, default=.05)
    parser.add_argument("--before-distance", type=float, default=.6)
    parser.add_argument("--behind-distance", type=float, default=.6)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--rays-topic", default="/vla/lidar/rays_world")
    parser.add_argument("--registered-topic", default="/cloud_registered")
    parser.add_argument("--query-topic", default="/vla_planner/grid_map/query")
    parser.add_argument("--result-topic", default="/vla_planner/grid_map/query_result")
    args = parser.parse_args(rospy.myargv()[1:])
    positive = [args.stream_timeout, args.query_timeout, args.range_max, args.extrinsic_tolerance,
                args.stationary_tolerance, args.before_distance, args.behind_distance]
    if any(not math.isfinite(v) or v <= 0 for v in positive) or args.frames < 2 or args.candidates < 1:
        parser.error("Timeouts/distances must be finite and positive; frames>=2 and candidates>=1")
    if not 0 <= args.shell_margin < args.range_max:
        parser.error("shell-margin must be between zero and range-max")
    return args


def main():
    args = arguments()
    report = {"success": False, "status": "inconclusive", "read_only": True,
              "ros_master": os.environ.get("ROS_MASTER_URI", ""),
              "range_max": args.range_max, "range_tolerance": .05,
              "expected_sensor_to_body_offset": args.expected_extrinsic,
              "query_timeout_wall_seconds": args.query_timeout,
              "sources": {"rays": args.rays_topic, "registered": args.registered_topic,
                          "map_query": args.query_topic, "map_result": args.result_topic}}
    started = time.monotonic()
    client = None
    code = 2
    try:
        rospy.init_node("vla_sensor_evidence_acceptance", anonymous=True)
        client = ObservationClient(args)
        pairs = client.collect()
        ray = verify_streams(pairs, args, report)
        verify_queries(client, ray, args, report)
        report.update(success=True, status="passed")
        code = 0
    except Failed as exc:
        report.update(status="failed", reason=str(exc))
        code = 1
    except Inconclusive as exc:
        report.update(status="inconclusive", reason=str(exc))
    except Exception as exc:
        report.update(status="error", reason="%s: %s" % (type(exc).__name__, exc))
        code = 1
    finally:
        if client is not None:
            client.close()
        report["wall_seconds"] = time.monotonic()-started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
