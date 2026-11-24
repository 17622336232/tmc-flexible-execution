"""A stationary polynomial for admission/acknowledgement, not control output.

All actual position sampling and yaw control run in the original C++ traj_server.
"""
import math
import json
from pathlib import Path


def rc_command_allowed(channels):
    """Mirror the existing PX4CTRL CH5/CH6 switches; never command the radio.

    PWM > 1750 is the vendor's normalized > 0.75 threshold. Physical switch
    direction is transmitter-specific. An absent RC message is not a switch.
    """
    if not isinstance(channels, (list, tuple)) or len(channels) < 6:
        return None
    values = channels[4:6]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return None
    return all((v-1000.)/1000. > .75 for v in values)



class RCFeedback:
    """Original RC receipt timeout, with optional source-stamp replay protection.

    MAVROS variants/legacy producers may use a zero or absent header stamp;
    those retain the original receipt-only timeout rather than being rejected.
    """
    def __init__(self, required, timeout):
        if type(required) is not bool or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Invalid original PX4CTRL RC feedback configuration')
        self.required, self.timeout = required, float(timeout)
        self.received = self.stamp = None

    def fresh(self, now, ros_now):
        if not self.required:
            return True
        if self.received is None or not 0 <= now-self.received <= self.timeout:
            return False
        return self.stamp is None or (-.1 <= ros_now-self.stamp <= self.timeout)

    def observe(self, message, now, ros_now):
        stamp = getattr(getattr(message, 'header', None), 'stamp', None)
        try:
            stamp = stamp.to_sec() if stamp is not None else 0.
        except (AttributeError, TypeError, ValueError):
            return False
        if not math.isfinite(stamp) or stamp < 0:
            return False
        if stamp > 0 and not -.1 <= ros_now-stamp <= self.timeout:
            return False
        self.received, self.stamp = now, stamp if stamp > 0 else None
        return True


def stationary_coefficients(position):
    if len(position) != 3 or not all(math.isfinite(v) for v in position):
        raise ValueError('invalid stationary position')
    return [[0., 0., 0., 0., 0., float(v)] for v in position]


def load_flight_bounds(path):
    """None selects the original planner; legacy explicit task fences still work."""
    config = json.loads(Path(path).read_text())
    if config.get('mode') != 'real':
        raise ValueError('Executor requires a real task configuration')
    if config.get('photo', {}).get('navigation_scope') == 'planner':
        if config.get('ros', {}).get('execution_backend') != 'diff_vla_real_v1' or 'bounds' in config:
            raise ValueError('Planner scope requires the real backend and no duplicate bounds')
        return None
    bounds = config.get('bounds', {})
    for axis in 'xyz':
        values = bounds.get(axis)
        if (not isinstance(values, list) or len(values) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values) or
                not 0 < values[1]-values[0] <= 40):
            raise ValueError('Invalid flight bounds: '+axis)
    return {axis: list(bounds[axis]) for axis in 'xyz'}


def trajectory_current(start, duration, now):
    # Same future tolerance as the existing bridge. Normal delayed replans
    # remain valid for their original time interval; never shift the curve.
    return (all(math.isfinite(v) for v in (start, duration, now)) and start > 0 and
            duration > 0 and start <= now+1. and start+duration > now)
