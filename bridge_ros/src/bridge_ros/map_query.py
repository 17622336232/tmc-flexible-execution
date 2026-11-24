"""Bounded, correlated read-only admission query for the Gazebo mapper."""
import json
import math
import re
import secrets
import threading
import time

from .visual_geometry import VisualGeometryError


def _unobserved(reason):
    return VisualGeometryError("goal_unobserved", "接近点的机体空间尚未充分观测或地图不可用，请重新观察", {"map_query_reason": reason})


class KnownFreeQuery:
    """The caller waits on an Event; ROS's independent subscriber resolves it.

    No flight lock, ROS service call, flight topic, or target adjustment occurs.
    send(request_id, xyz) must publish only the mapper's read-only PointStamped
    query. request_id is an opaque decimal nanosecond nonce in header.stamp;
    header.seq is NOT usable because rospy overwrites it while serializing.
    """
    def __init__(self, send, timeout=0.25):
        if not 0.2 <= timeout <= 0.3:
            raise ValueError("map query timeout must be 200–300 ms")
        self.send, self.timeout = send, timeout
        self.lock = threading.Lock()
        self.pending = {}

    def ensure(self, goal):
        try:
            xyz = tuple(float(goal[axis]) for axis in ("x", "y", "z"))
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError()
        except (TypeError, ValueError, KeyError):
            raise _unobserved("invalid_goal") from None
        started = time.monotonic()
        entry = dict(xyz=xyz, event=threading.Event(), response=None, deadline=started+self.timeout)
        with self.lock:
            if len(self.pending) >= 8:
                raise _unobserved("busy")
            limit = (2**32-1)*1_000_000_000-1
            request_id = str(secrets.randbelow(limit)+1)
            while request_id in self.pending:
                request_id = str(secrets.randbelow(limit)+1)
            self.pending[request_id] = entry
        try:
            try:
                self.send(request_id, xyz)
            except Exception:
                raise _unobserved("unavailable") from None
            entry["event"].wait(max(0., entry["deadline"]-time.monotonic()))
            response = entry["response"]
            if response is None:
                raise _unobserved("timeout")
            if response["scan_stale"]:
                raise _unobserved("scan_stale")
            if response["state"] != "free" or response["inflated_blocked"]:
                raise _unobserved("not_known_free")
            return dict(source="mapper_grid_query", state="free", inflated_blocked=False, scan_stale=False,
                        scope=response.get('observation_scope', "full_vehicle_envelope"), request_id=request_id,
                        point=dict(zip(("x", "y", "z"), xyz)), stamp=time.time(),
                        query_ms=1000*(time.monotonic()-started))
        finally:
            with self.lock:
                self.pending.pop(request_id, None)

    def receive(self, raw):
        if not isinstance(raw, str) or len(raw) > 4096:
            return
        try:
            response = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if (not isinstance(response, dict) or not isinstance(response.get("request_stamp_ns"), str)
                    or not re.fullmatch(r"[0-9]{1,19}", response["request_stamp_ns"])
                    or response.get("state") not in ("outside", "unknown", "free", "occupied")
                    or type(response.get("inflated_blocked")) is not bool
                    or type(response.get("scan_stale")) is not bool
                    or response.get('observation_scope', 'full_vehicle_envelope') not in
                       ('full_vehicle_envelope', 'observed_navigation_layer_and_vehicle_obstacles')):
                return
            xyz = tuple(response[axis] for axis in ("x", "y", "z"))
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in xyz):
                return
        except (ValueError, TypeError, KeyError):
            return
        with self.lock:
            entry = self.pending.get(response["request_stamp_ns"])
            if (entry is None or entry["response"] is not None or time.monotonic() > entry["deadline"]
                    or any(abs(a-b) > 1e-5 for a, b in zip(xyz, entry["xyz"]))):
                return
            entry["response"] = response
            entry["event"].set()
