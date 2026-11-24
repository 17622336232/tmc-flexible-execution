"""ROS-independent fail-closed policy for the real-flight command gate."""
import math


class GatePolicy:
    def __init__(self, lease_seconds=1.0):
        self.lease_seconds = lease_seconds
        self.close()

    def close(self):
        self.enabled = False
        self.lease_deadline = 0.0
        self.request_stamp = float("inf")
        self.epoch_start = None
        self.current_id = None
        self.current_start = None

    def open(self, now, ros_stamp, latest_id=None):
        # The upstream planner resets its local trajectory id to 1 for EVERY
        # goal. latest_id is diagnostic only, never a cross-goal lower bound.
        self.close()
        self.enabled = True
        self.lease_deadline = now + self.lease_seconds
        self.request_stamp = ros_stamp

    def renew(self, now):
        if self.enabled and now <= self.lease_deadline:
            self.lease_deadline = now + self.lease_seconds
        else:
            self.close()

    def trajectory(self, trajectory_id, start_stamp):
        if not (self.enabled and math.isfinite(start_stamp) and start_stamp >= self.request_stamp):
            return False
        if self.epoch_start is None:
            # An old goal can still be replanning after the gate closes. Its
            # fresh id 4 is not a new-goal acknowledgement. Wait for reset id 1.
            if trajectory_id != 1:
                return False
            self.epoch_start = start_stamp
        elif trajectory_id == self.current_id and start_stamp == self.current_start:
            return True  # Duplicate transport delivery of the same polynomial.
        elif trajectory_id <= self.current_id or start_stamp <= self.current_start:
            return False
        self.current_id, self.current_start = trajectory_id, start_stamp
        return True

    def allows(self, now, trajectory_id):
        if now > self.lease_deadline:
            self.close()
        return self.enabled and self.current_id == trajectory_id


class PolynomialTrajectory:
    """Decode PolyTraj quintics to reject commands from a reused old ROS id.

    This verifies correspondence with the selected polynomial, not ownership by
    a particular task: the upstream planner has no task acknowledgement field.
    """
    def __init__(self, start_stamp, durations, coefficients):
        self.start_stamp = float(start_stamp)
        self.durations = tuple(float(value) for value in durations)
        self.coefficients = tuple(tuple(float(value) for value in axis) for axis in coefficients)
        values = [self.start_stamp] + list(self.durations)
        values.extend(value for axis in self.coefficients for value in axis)
        if (not self.durations or len(self.coefficients) != 3
                or any(value <= 0 for value in self.durations)
                or not math.isfinite(sum(self.durations))
                or any(len(axis) != 6*len(self.durations) for axis in self.coefficients)
                or not all(math.isfinite(value) for value in values)):
            raise ValueError("Invalid quintic trajectory")

    def sample(self, stamp):
        relative = stamp - self.start_stamp
        duration = sum(self.durations)
        if not math.isfinite(relative) or relative < -0.02 or relative > duration + 0.02:
            return None
        relative = max(0.0, min(duration, relative))
        index = 0
        while index + 1 < len(self.durations) and relative > self.durations[index]:
            relative -= self.durations[index]
            index += 1
        result = []
        for derivative in range(3):
            vector = []
            for axis in self.coefficients:
                coeff = list(axis[6*index:6*index+6])
                for _ in range(derivative):
                    coeff = [value*(len(coeff)-1-i) for i, value in enumerate(coeff[:-1])]
                value = 0.0
                for term in coeff:
                    value = value*relative + term
                vector.append(value)
            result.append(tuple(vector))
        return tuple(result)

    def matches(self, stamp, position, velocity, acceleration):
        expected = self.sample(stamp)
        if expected is None:
            return False
        # ROS sets the command header slightly after trajectory evaluation;
        # tolerate that scheduling jitter without accepting an unrelated path.
        for actual, target, tolerance in zip((position, velocity, acceleration), expected, (0.03, 0.15, 0.5)):
            if any(not math.isfinite(value) for value in actual):
                return False
            if math.sqrt(sum((left-right)**2 for left, right in zip(actual, target))) > tolerance:
                return False
        return True
