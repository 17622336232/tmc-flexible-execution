#!/usr/bin/env python3
"""Real-flight fail-closed PositionCommand gate, never an arming controller."""
import json
import math
import threading
import time

import rosgraph
import rospy
from quadrotor_msgs.msg import PositionCommand
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, String
from std_srvs.srv import SetBool, SetBoolResponse
from traj_utils.msg import PolyTraj

try:
    from bridge_ros.gate_policy import GatePolicy, PolynomialTrajectory
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from bridge_ros.gate_policy import GatePolicy, PolynomialTrajectory


class CommandGate:
    def __init__(self):
        self.lock = threading.RLock()
        self.policy = GatePolicy(lease_seconds=1.0)
        self.latest_id = 0
        self.epoch = 0
        self.selected = None
        self.odom = None
        self.odom_seen = 0.0
        self.epoch_origin = None
        self.frame_id = rospy.get_param("~frame_id", "world").lstrip("/")
        self.status = "CLOSED"
        self.output = rospy.Publisher("/setpoints_cmd", PositionCommand, queue_size=1, latch=False)
        self.status_pub = rospy.Publisher("~state", String, queue_size=1, latch=True)
        rospy.Subscriber("/diff_vla/planner_cmd", PositionCommand, self.on_command, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber("/diff_vla/gate_lease", Empty, self.on_lease, queue_size=1)
        rospy.Subscriber("/ekf/ekf_odom", Odometry, self.on_odom, queue_size=1)
        trajectory_topic = rospy.get_param("~trajectory_topic", "/drone_0_planning/trajectory")
        rospy.Subscriber(trajectory_topic, PolyTraj, self.on_trajectory, queue_size=10)
        rospy.Service("~set_enabled", SetBool, self.set_enabled)
        self.monitor = threading.Thread(target=self.watchdog, daemon=True)
        self.monitor.start()

    def set_enabled(self, request):
        with self.lock:
            self.policy.close()
            self.selected = None
            self.status = "CLOSED"
            try:
                publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
                names = dict(publishers).get("/setpoints_cmd", [])
                if any(name != rospy.get_name() for name in names):
                    return SetBoolResponse(False, "Another publisher bypasses the gate on /setpoints_cmd")
            except Exception as exc:
                return SetBoolResponse(False, "Cannot verify command publisher exclusivity: " + str(exc))
            if not request.data:
                return SetBoolResponse(True, "Planner output closed; PX4CTRL command timeout will enter hover")
            if self.output.get_num_connections() < 1:
                return SetBoolResponse(False, "No controller subscribes to /setpoints_cmd")
            now_ros = rospy.Time.now().to_sec()
            if now_ros <= 0:
                return SetBoolResponse(False, "ROS clock unavailable")
            if self.odom is None or time.monotonic() - self.odom_seen > 1.0:
                return SetBoolResponse(False, "Fresh odometry is required to capture the new task origin")
            self.policy.open(time.monotonic(), now_ros, self.latest_id)
            self.epoch += 1
            self.epoch_origin = self.odom
            self.status = "WAITING_FOR_NEW_GOAL_TRAJECTORY_ID_1"
            return SetBoolResponse(True, "Waiting for a fresh reset-id-1 trajectory for task epoch %d" % self.epoch)

    def on_odom(self, message):
        p = message.pose.pose.position
        values = (p.x, p.y, p.z)
        age = (rospy.Time.now() - message.header.stamp).to_sec()
        if (message.header.frame_id.lstrip("/") == self.frame_id
                and all(math.isfinite(value) for value in values) and -0.1 <= age <= 1.0):
            with self.lock:
                self.odom, self.odom_seen = values, time.monotonic()

    def on_lease(self, _message):
        with self.lock:
            self.policy.renew(time.monotonic())

    def on_trajectory(self, message):
        with self.lock:
            self.latest_id = message.traj_id
            if message.order != 5:
                return
            try:
                candidate = PolynomialTrajectory(message.start_time.to_sec(), message.duration,
                                                 (message.coef_x, message.coef_y, message.coef_z))
            except (ValueError, TypeError):
                return
            if self.policy.epoch_start is None and self.epoch_origin is not None:
                start_position = candidate.sample(candidate.start_stamp)[0]
                if math.sqrt(sum((a-b)**2 for a, b in zip(start_position, self.epoch_origin))) > 0.75:
                    return
            if self.policy.trajectory(message.traj_id, candidate.start_stamp):
                self.selected = candidate

    def on_command(self, message):
        with self.lock:
            if not self.policy.allows(time.monotonic(), message.trajectory_id):
                return
            age = (rospy.Time.now() - message.header.stamp).to_sec()
            vectors = [message.position, message.velocity, message.acceleration, message.jerk]
            values = [number for vector in vectors for number in (vector.x, vector.y, vector.z)]
            values.extend((message.yaw, message.yaw_dot))
            if (not all(math.isfinite(number) for number in values) or not -0.1 <= age <= 0.25
                    or message.trajectory_flag != PositionCommand.TRAJECTORY_STATUS_READY):
                self.policy.close()
                self.status = "INVALID_COMMAND_CLOSED"
                return
            vectors = [tuple(getattr(vector, key) for key in ("x", "y", "z")) for vector in vectors[:3]]
            if self.selected is None or not self.selected.matches(message.header.stamp.to_sec(), *vectors):
                # A newly received id-1 PolyTraj can race with one last command
                # from an older id-1 path. Drop it, never authorize by id alone.
                self.status = "WAITING_FOR_MATCHING_COMMAND"
                return
            self.status = "FORWARDING"
            self.output.publish(message)

    def watchdog(self):
        while not rospy.is_shutdown():
            with self.lock:
                if self.policy.enabled and time.monotonic() > self.policy.lease_deadline:
                    self.policy.close()
                    self.status = "LEASE_EXPIRED_CLOSED"
                message = json.dumps({"state": self.status, "enabled": self.policy.enabled,
                                      "task_epoch": self.epoch, "selected_trajectory_id": self.policy.current_id,
                                      "latest_trajectory_id": self.latest_id})
            self.status_pub.publish(String(data=message))
            # Wall clock is deliberate: a paused ROS clock must not preserve a lease.
            time.sleep(0.1)


if __name__ == "__main__":
    rospy.init_node("command_gate")
    CommandGate()
    rospy.spin()
