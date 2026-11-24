"""Explicit demonstration adapter, with smooth simulated telemetry.

This module never imports ROS and never sends commands to a vehicle. Its status
always says 'mock'. It exists for APK/API testing without installed robotics tools.
"""

import math
import threading
import time


class MockAdapter:
    def __init__(self, config):
        home = config.get("home") or config.get("mock_origin", {"x": -6.0, "y": -3.0, "z": 0.18})
        self._position = {"x": float(home["x"]), "y": float(home["y"]), "z": 0.18}
        self._target = dict(self._position)
        self._velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._yaw = 0.0
        self._armed = False
        self._landed = True
        self._action = "LANDED"
        self._last = time.monotonic()
        self._lock = threading.RLock()
        self._speed = 0.8

    def _advance(self):
        now = time.monotonic()
        dt, self._last = min(now - self._last, 0.25), now
        delta = {k: self._target[k] - self._position[k] for k in ("x", "y", "z")}
        distance = math.sqrt(sum(v * v for v in delta.values()))
        travel = min(distance, self._speed * dt)
        ratio = travel / distance if distance > 1e-8 else 0.0
        self._velocity = {k: delta[k] * ratio / dt if dt > 0 else 0.0 for k in delta}
        self._position = {k: self._position[k] + delta[k] * ratio for k in delta}
        if distance <= 1e-6:
            self._velocity = {k: 0.0 for k in delta}
            if self._action == "LANDING":
                self._landed, self._armed, self._action = True, False, "LANDED"
            elif not self._landed:
                self._action = "HOLD"

    def snapshot(self):
        with self._lock:
            self._advance()
            return {"mode": "mock", "connected": True, "armed": self._armed,
                    "landed": self._landed, "position": dict(self._position),
                    "velocity": dict(self._velocity), "yaw": self._yaw,
                    "state_stamp": time.time(), "position_stamp": time.time(),
                    "flight_mode": self._action, "battery": 0.98}

    def takeoff(self, height):
        with self._lock:
            self._advance()
            if not self._landed:
                raise RuntimeError("Vehicle is already airborne")
            self._armed, self._landed, self._action = True, False, "TAKING_OFF"
            self._target = dict(self._position, z=float(height))

    def goto(self, x, y, z, yaw=None):
        with self._lock:
            self._advance()
            if not self._armed or self._landed:
                raise RuntimeError("Take off before requesting a destination")
            self._target = {"x": float(x), "y": float(y), "z": float(z)}
            if yaw is not None:
                self._yaw = float(yaw)
            self._action = "FLYING"

    def land(self):
        with self._lock:
            self._advance()
            self._target = dict(self._position, z=0.18)
            self._action = "LANDING"

    def hold(self):
        with self._lock:
            self._advance()
            self._target = dict(self._position)
            self._velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
            self._action = "LANDED" if self._landed else "HOLD"

    def close(self):
        pass
