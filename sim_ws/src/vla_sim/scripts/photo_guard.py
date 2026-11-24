#!/usr/bin/env python3
"""Conservative, ROS-independent person exclusion guard for photo missions.

The radius includes the measured person envelope, rotor radius and requested
clearance.  The cylinder is deliberately infinite vertically: flying over a
person is not an escape from the exclusion boundary.  This is a simulation
control guard, not a substitute for a real flight controller's braking model.
"""
import math
import re
import time
from collections import deque


class GuardError(ValueError):
    pass


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def outside_segment(a, b, center, radius):
    """True only when the complete straight XY segment clears the cylinder."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx+dy*dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0,
        ((center[0]-a[0])*dx+(center[1]-a[1])*dy)/length2))
    x, y = a[0]+t*dx-center[0], a[1]+t*dy-center[1]
    return x*x+y*y > (radius+1e-7)**2


def power_to_bernstein(descending, duration):
    """Convert p(t), t in [0,T], to its six degree-5 Bezier control values."""
    a = list(reversed(descending))
    return [sum(math.comb(k, i)/math.comb(5, i)*a[i]*duration**i
                for i in range(k+1)) for k in range(6)]


def split_controls(points):
    levels = [points]
    while len(levels[-1]) > 1:
        level = levels[-1]
        levels.append([((a[0]+b[0])*.5, (a[1]+b[1])*.5)
                       for a, b in zip(level, level[1:])])
    return [row[0] for row in levels], [row[-1] for row in reversed(levels)]


def prove_piece_clear(x, y, duration, center, radius, max_depth=20, max_nodes=4096, deadline=None):
    """Prove every point is outside via recursively bounded Bezier hull boxes.

    An inconclusive box is subdivided.  Touching, numerical uncertainty, or
    exhausted proof budget is rejected; sparse samples never certify a path.
    """
    controls = list(zip(power_to_bernstein(x, duration), power_to_bernstein(y, duration)))
    if not all(math.isfinite(v) for p in controls for v in p):
        return False
    stack, visited = [(controls, 0)], 0
    limit2 = (radius+1e-7)**2
    while stack:
        points, depth = stack.pop()
        visited += 1
        if visited > max_nodes or (deadline is not None and visited % 16 == 0 and time.monotonic() >= deadline):
            return False
        for p in (points[0], points[-1]):
            if (p[0]-center[0])**2+(p[1]-center[1])**2 <= limit2:
                return False
        lo = [min(p[i] for p in points) for i in range(2)]
        hi = [max(p[i] for p in points) for i in range(2)]
        # Inflate the box by a scale-aware rounding cushion before proving it.
        eps = 1e-10*(1.0+max(abs(v) for p in points for v in p))
        closest = [max(lo[i]-eps, min(center[i], hi[i]+eps)) for i in range(2)]
        if sum((closest[i]-center[i])**2 for i in range(2)) > limit2:
            continue
        if depth >= max_depth:
            return False
        left, right = split_controls(points)
        stack.extend(((right, depth+1), (left, depth+1)))
    return True


class ExecutionTrajectory:
    def __init__(self, execution_id, start, durations, coefficients, order=5):
        if (isinstance(execution_id, bool) or not isinstance(execution_id, int)
                or execution_id <= 0 or not finite_number(start) or start <= 0
                or order != 5 or not 1 <= len(durations) <= 128):
            raise GuardError('invalid_execution_trajectory')
        if (any(not finite_number(d) or not 0 < d <= 120 for d in durations)
                or sum(durations) > 300 or len(coefficients) != 3
                or any(len(c) != 6*len(durations) for c in coefficients)
                or any(not finite_number(v) or abs(v) > 1e9 for c in coefficients for v in c)):
            raise GuardError('invalid_execution_coefficients')
        self.id, self.start = execution_id, start
        self.durations = list(durations)
        self.coefficients = [list(c) for c in coefficients]
        self.duration = sum(durations)
        for i, d in enumerate(self.durations[:-1]):
            for axis in self.coefficients:
                previous = list(reversed(axis[6*i:6*i+6]))
                following = list(reversed(axis[6*(i+1):6*(i+1)+6]))
                for deriv, tolerance in enumerate((.002, .01, .05)):
                    end = sum(previous[k]*math.factorial(k)/math.factorial(k-deriv)*d**(k-deriv)
                              for k in range(deriv, 6))
                    begin = following[deriv]*math.factorial(deriv)
                    if not math.isfinite(end) or abs(end-begin) > tolerance:
                        raise GuardError('discontinuous_execution_trajectory')

    def prove_clear(self, center, radius, deadline=None):
        deadline = time.monotonic()+.04 if deadline is None else deadline
        return all(prove_piece_clear(self.coefficients[0][6*i:6*i+6],
                                    self.coefficients[1][6*i:6*i+6], d, center, radius,
                                    deadline=deadline)
                   for i, d in enumerate(self.durations))

    def evaluate(self, stamp):
        t = max(0.0, min(stamp-self.start, self.duration))
        i = 0
        while i+1 < len(self.durations) and t > self.durations[i]:
            t -= self.durations[i]
            i += 1
        at_end = stamp-self.start >= self.duration
        out = []
        for deriv in range(3):
            xyz = []
            for c in self.coefficients:
                asc = list(reversed(c[6*i:6*i+6]))
                xyz.append(0.0 if deriv and at_end else sum(
                    asc[k]*math.factorial(k)/math.factorial(k-deriv)*t**(k-deriv)
                    for k in range(deriv, 6)))
            out.append(xyz)
        return out

    def matches(self, stamp, position, velocity, acceleration):
        if not finite_number(stamp) or stamp < self.start:
            return False
        expected = self.evaluate(stamp)
        return all(all(finite_number(v) and abs(v-e) <= tol for v, e in zip(actual, wanted))
                   and len(actual) == 3
                   for actual, wanted, tol in zip((position, velocity, acceleration), expected,
                                                 (0.002, 0.003, 0.01)))


class PhotoGuard:
    """Fresh, nonce-bound lease; renewal never resurrects an expired guard."""
    def __init__(self):
        self.id = self.nonce = ''
        self.center = None
        self.radius = 0.0
        self.zones = []
        self.active = False
        self.expires = 0.0
        self.epoch = 0.0
        self.last_stamp = 0.0
        self.error = ''
        self.seen_nonces = deque(maxlen=512)
        self.seen_ids = deque(maxlen=128)
        self.validated_trajectory_id = 0

    def expire(self, now):
        if self.active and now >= self.expires:
            self.deactivate('photo_guard_expired', now)
            return True
        return False

    def deactivate(self, reason, now):
        self.active = False
        self.expires = 0.0
        self.epoch = max(self.epoch, now)
        self.validated_trajectory_id = 0
        self.error = reason

    def apply(self, message, now):
        if (not isinstance(message, dict) or type(message.get('version')) is not int
                or message.get('version') != 1):
            raise GuardError('invalid_photo_guard_version')
        op, guard_id, nonce = message.get('op'), message.get('id'), message.get('nonce')
        token = re.compile(r'^[A-Za-z0-9_-]{16,96}$')
        if (op not in ('enable', 'refresh', 'clear')
                or not isinstance(guard_id, str) or not token.fullmatch(guard_id)
                or not isinstance(nonce, str) or not token.fullmatch(nonce)
                or nonce in self.seen_nonces):
            raise GuardError('invalid_or_replayed_photo_guard')
        stamp, ttl = message.get('stamp'), message.get('ttl')
        if (not finite_number(stamp) or stamp <= 0 or not 0 <= now-stamp <= .6
                or stamp < self.last_stamp or not finite_number(ttl) or not .2 <= ttl <= 3.0
                or stamp+ttl <= now):
            raise GuardError('stale_photo_guard_message')
        if op != 'enable' and guard_id != self.id:
            raise GuardError('photo_guard_id_mismatch')
        if op == 'clear':
            self.deactivate('', now)
        else:
            center, radius = message.get('center'), message.get('radius')
            if (not isinstance(center, (tuple, list)) or len(center) != 2
                    or any(not finite_number(v) or abs(v) > 100 for v in center)
                    or not finite_number(radius) or not 2.05 <= radius <= 10.0):
                raise GuardError('invalid_photo_guard_geometry')
            zones = message.get('zones', [dict(center=list(center), radius=radius)])
            if not isinstance(zones, list) or not 1 <= len(zones) <= 8:
                raise GuardError('invalid_photo_guard_zones')
            clean_zones = []
            for zone in zones:
                if not isinstance(zone, dict):
                    raise GuardError('invalid_photo_guard_zone')
                zc, zr = zone.get('center'), zone.get('radius')
                if (not isinstance(zc, (list, tuple)) or len(zc) != 2
                        or any(not finite_number(v) or abs(v) > 100 for v in zc)
                        or not finite_number(zr) or not 2.05 <= zr <= 10):
                    raise GuardError('invalid_photo_guard_zone')
                clean_zones.append(dict(center=list(zc), radius=float(zr)))
            if clean_zones[0] != dict(center=list(center), radius=float(radius)):
                raise GuardError('selected_person_missing_from_guard_zones')
            if op == 'refresh':
                if not self.active or now >= self.expires:
                    raise GuardError('photo_guard_renewal_after_expiry')
                if list(center) != self.center or radius != self.radius or clean_zones != self.zones:
                    raise GuardError('photo_guard_geometry_changed')
            else:
                if guard_id in self.seen_ids:
                    raise GuardError('reused_photo_guard_id')
                self.id, self.center, self.radius = guard_id, list(center), float(radius)
                self.zones = clean_zones
                self.epoch = now
                self.active = True
                self.seen_ids.append(guard_id)
                self.validated_trajectory_id = 0
            self.expires = stamp+ttl
            self.error = ''
        self.nonce, self.last_stamp = nonce, stamp
        self.seen_nonces.append(nonce)
        return op

    def clears(self, position, candidate=None):
        return not self.active or all(outside_segment(position, position if candidate is None else candidate,
                                      zone['center'], zone['radius']) for zone in self.zones)

    def proves(self, trajectory):
        deadline = time.monotonic()+.04
        return all(trajectory.prove_clear(zone['center'], zone['radius'], deadline=deadline) for zone in self.zones)

    def status(self):
        return dict(supported=True, id=self.id, nonce=self.nonce, active=self.active,
                    center=self.center, radius=self.radius, expires_ros_stamp=self.expires,
                    zones=[dict(center=list(z['center']), radius=z['radius']) for z in self.zones],
                    error=self.error, validated_trajectory_id=self.validated_trajectory_id)
