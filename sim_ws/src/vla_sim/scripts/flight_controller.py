#!/usr/bin/env python3
"""Interface-level flight simulation, NOT a PX4/MAVLink or motor dynamics SITL.

Acceleration-limited position tracking, ground-start/takeoff/landing state machine,
explicit trajectory gate and timeout hold. Gazebo provides physical ray sensing.
"""
import json
import copy
import math
import threading
import time
import secrets
import rospy
import tf
from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
from quadrotor_msgs.msg import PositionCommand, TakeoffLand
from std_msgs.msg import Empty, String
from sensor_msgs.msg import PointCloud2
from traj_utils.msg import PolyTraj
from photo_guard import PhotoGuard, GuardError, ExecutionTrajectory, finite_number
from photo_search import PhotoSearch, PhotoSearchError


def stamp_fresh(stamp, now, timeout):
    return stamp > 0 and 0 <= now-stamp <= timeout


def clip(v, limit):
    return max(-limit, min(limit, v))


def norm(v):
    return math.sqrt(sum(a*a for a in v))


class FlightController:
    def __init__(self):
        self.lock = threading.RLock()
        self.ground = float(rospy.get_param('~ground_height', 0.18))
        self.height = float(rospy.get_param('~takeoff_height', 1.2))
        self.p = [float(rospy.get_param('~start_x', -6)), float(rospy.get_param('~start_y', -3)), self.ground]
        self.v, self.a = [0.0]*3, [0.0]*3
        self.yaw = float(rospy.get_param('~start_yaw', 0.0))
        if not math.isfinite(self.yaw):
            raise ValueError('start_yaw must be finite')
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        self.hold = self.p[:]
        self.state = 'GROUNDED'
        self.model = rospy.get_param('~model_name', 'vla_uav')
        self.model_ready = False
        self.model_seen = 0.0
        self.cmd = None
        self.accept_commands = False
        self.minimum_id = 0
        self.latest_id = 0
        self.goal_at = 0.0
        self.command_at = 0.0
        self.command_timeout = float(rospy.get_param('~command_timeout', 0.6))
        self.contact_count = 0
        self.lidar_at = 0.0
        self.lidar_timeout = 0.6
        self.photo_guard = PhotoGuard()
        self.pending_guard_message = None
        self.guard_message_timing = None
        self.guard_message_rejection = None
        self.guard_clock_wait_count = 0
        self.photo_search = PhotoSearch()
        self.search_query = None
        self.search_envelope_at = 0.0
        self.search_reject_before = 0.0
        self.photo_trajectories = {}
        self.execution_start_times = {}
        self.photo_execution_cache = {}
        self.pending_photo_command = None
        self.photo_pending_since = None
        self.photo_last_command_id = 0
        self.photo_last_command_stamp = 0.0
        self.photo_reject_detail = None
        self.photo_reject_before = 0.0
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.model_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.state_pub = rospy.Publisher('/vla/flight_state', String, queue_size=1, latch=True)
        self.detail_pub = rospy.Publisher('/diff_vla/sim/state', String, queue_size=1, latch=True)
        self.search_query_pub = rospy.Publisher('/vla_planner/grid_map/query', PointStamped, queue_size=2)
        self.tf = tf.TransformBroadcaster()
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.models, queue_size=1)
        rospy.Subscriber('/position_cmd', PositionCommand, self.command, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/px4ctrl/takeoff_land', TakeoffLand, self.takeoff_land, queue_size=5)
        rospy.Subscriber('/goal', PoseStamped, self.goal, queue_size=1)
        rospy.Subscriber('/vla/hold', Empty, self.pause, queue_size=1)
        rospy.Subscriber('/vla/uav_contacts', ContactsState, self.contacts, queue_size=1)
        rospy.Subscriber('/vla/lidar/rays_world', PointCloud2, self.lidar, queue_size=1)
        rospy.Subscriber('/vla/photo_guard', String, self.guard_message, queue_size=10)
        rospy.Subscriber('/vla/photo_search', String, self.search_message, queue_size=10)
        rospy.Subscriber('/vla_planner/grid_map/query_result', String, self.search_map_result, queue_size=10)
        rospy.Subscriber('/vla/execution_trajectory', PolyTraj, self.execution_trajectory, queue_size=4)
        self.timer = rospy.Timer(rospy.Duration(0.02), self.update)
        self.status_timer = rospy.Timer(rospy.Duration(0.1), self.publish_status)

    def lidar(self, msg):
        if msg.width*msg.height:
            with self.lock:
                stamp = msg.header.stamp.to_sec()
                if stamp < self.lidar_at:
                    self.clear_gate()
                    if self.state not in ('GROUNDED', 'LANDING'):
                        self.transition('HOVERING')
                self.lidar_at = stamp

    def models(self, msg):
        with self.lock:
            self.model_ready = self.model in msg.name
            self.model_seen = rospy.get_time()

    def transition(self, state):
        if state != self.state:
            rospy.loginfo('Flight: %s -> %s', self.state, state)
        self.state = state

    def _discard_pending_guard_message(self, tombstone_enable=True):
        pending = getattr(self, 'pending_guard_message', None)
        if pending is not None and tombstone_enable and pending['payload'].get('op') == 'enable':
            # A queued enable cancelled by hold/land must not be replayed once
            # its original timestamp matures on this process's /clock.
            self.photo_guard.seen_ids.append(pending['payload']['id'])
        if pending is not None and pending.get('timer') is not None:
            pending['timer'].cancel()
        self.pending_guard_message = None

    def clear_gate(self):
        self._discard_pending_guard_message()
        self.accept_commands = False
        self.cmd = None
        self.minimum_id = self.latest_id
        self.hold = self.p[:]
        self.photo_trajectories.clear()
        self.photo_execution_cache.clear()
        self.pending_photo_command = None
        self.photo_pending_since = None
        self.photo_last_command_id = self.latest_id
        self.photo_last_command_stamp = 0.0
        self.photo_guard.validated_trajectory_id = 0

    def stop_photo(self, reason, deactivate=False):
        """A rejected photo motion cannot coast into the person's cylinder."""
        self.clear_gate()
        self.v, self.a = [0.0]*3, [0.0]*3
        now = rospy.get_time()
        self.photo_reject_before = max(self.photo_reject_before, now)
        if deactivate:
            self.photo_guard.deactivate(reason, now)
        else:
            self.photo_guard.error = reason
        if self.state not in ('GROUNDED', 'LANDING', 'FAULT'):
            self.transition('HOVERING')

    def stop_search(self, reason=''):
        if not (self.photo_search.active or self.photo_search.pending):
            return
        self.photo_search.cancel(reason, rospy.get_time())
        self.search_query = None
        self.search_reject_before = max(self.search_reject_before, rospy.get_time())
        self.photo_reject_before = max(self.photo_reject_before, self.search_reject_before)
        self.clear_gate()
        self.v, self.a = [0.0]*3, [0.0]*3
        if self.state not in ('GROUNDED', 'LANDING', 'FAULT'):
            self.transition('HOVERING')

    def request_search_envelope(self):
        if self.search_query is not None:
            return
        request_id = secrets.randbelow((2**32-1)*1000000000-1)+1
        self.search_query = dict(id=str(request_id), position=self.p[:], wall=time.monotonic())
        query = PointStamped()
        query.header.frame_id = 'world'
        query.header.stamp = rospy.Time(request_id//1000000000, request_id%1000000000)
        query.point.x, query.point.y, query.point.z = self.p
        self.search_query_pub.publish(query)

    def search_message(self, msg):
        with self.lock:
            try:
                payload = json.loads(msg.data)
                now = rospy.get_time()
                if isinstance(payload, dict) and payload.get('op') == 'start' and payload.get('stamp', 0) <= self.search_reject_before:
                    raise PhotoSearchError('photo_search_start_before_cancel')
                owned_running = (isinstance(payload, dict) and payload.get('id') == self.photo_search.id
                                 and (self.photo_search.active or self.photo_search.pending))
                operation = self.photo_search.apply(payload, now, self.p, self.yaw)
                if operation == 'start':
                    if (self.state != 'HOVERING' or self.p[2] < self.ground+.3
                            or norm(self.v) > .12 or norm(self.a) > .4
                            or not self.model_ready or now-self.model_seen > 1.0
                            or not stamp_fresh(self.lidar_at, now, self.lidar_timeout)
                            or self.photo_guard.active):
                        # Rejecting an inadmissible search must not interrupt
                        # an unrelated active flight or photo guard.
                        self.photo_search.cancel('photo_search_requires_fresh_stable_airborne_hold', now)
                        return
                    self.clear_gate()
                    self.search_reject_before = max(self.search_reject_before, now)
                    self.photo_reject_before = max(self.photo_reject_before, now)
                    self.v, self.a = [0.0]*3, [0.0]*3
                    self.hold = self.p[:]
                    self.search_query = None
                    self.search_envelope_at = 0.0
                    self.request_search_envelope()
                elif operation == 'stop':
                    # apply already deactivated the lease. Do not change a
                    # descent target if the owner's async stop follows LAND.
                    self.search_query = None
                    self.search_reject_before = max(self.search_reject_before, now)
                    if owned_running and self.state != 'LANDING':
                        self.photo_reject_before = max(self.photo_reject_before, now)
                        self.clear_gate()
                        self.v, self.a = [0.0]*3, [0.0]*3
                        if self.state not in ('GROUNDED', 'FAULT'):
                            self.transition('HOVERING')
            except (ValueError, TypeError, OverflowError) as exc:
                # A stale worker's stop/refresh must never stop a newer owner.
                rospy.logwarn_throttle(1, 'Photo search request rejected: %s', str(exc))

    def search_map_result(self, msg):
        with self.lock:
            pending = self.search_query
            if pending is None or not (self.photo_search.pending or self.photo_search.active):
                return
            try:
                result = json.loads(msg.data)
                if not isinstance(result, dict):
                    raise ValueError('photo_search_envelope_response_invalid')
                if result.get('request_stamp_ns') != pending['id']:
                    return
                xyz = [result[a] for a in ('x', 'y', 'z')]
                if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in xyz)
                        or any(abs(a-b) > 1e-5 for a, b in zip(xyz, pending['position']))):
                    raise ValueError('photo_search_envelope_response_invalid')
                if (time.monotonic()-pending['wall'] > .3 or result.get('state') != 'free'
                        or result.get('inflated_blocked') is not False or result.get('scan_stale') is not False
                        or norm([self.p[i]-pending['position'][i] for i in range(3)]) > .02):
                    raise ValueError('photo_search_current_envelope_not_known_free')
                self.search_query = None
                self.search_envelope_at = rospy.get_time()
                if self.photo_search.pending:
                    self.photo_search.activate(rospy.get_time())
            except (ValueError, TypeError, KeyError) as exc:
                self.stop_search(str(exc))

    def update_search(self, now, dt):
        if not (self.photo_search.active or self.photo_search.pending):
            return 0.0
        if (not self.model_ready or now-self.model_seen > 1.0
                or not stamp_fresh(self.lidar_at, now, self.lidar_timeout)
                or self.photo_guard.active or self.state != 'HOVERING'
                or norm([self.p[i]-self.photo_search.position[i] for i in range(3)]) > .02):
            self.stop_search('photo_search_pose_or_sensor_not_ready')
            return 0.0
        if self.search_query is not None and time.monotonic()-self.search_query['wall'] >= .3:
            self.stop_search('photo_search_envelope_timeout')
            return 0.0
        if self.photo_search.active and now-self.search_envelope_at >= .25:
            self.request_search_envelope()
        delta = self.photo_search.step(now, dt)
        if not (self.photo_search.active or self.photo_search.pending):
            self.search_query = None
            self.search_reject_before = max(self.search_reject_before, now)
            self.photo_reject_before = max(self.photo_reject_before, now)
            self.clear_gate()
            self.v, self.a = [0.0]*3, [0.0]*3
        return delta

    def _guard_timing(self, payload, now, outcome):
        stamp = payload.get('stamp') if isinstance(payload, dict) else None
        ttl = payload.get('ttl') if isinstance(payload, dict) else None
        return dict(outcome=outcome, op=payload.get('op') if isinstance(payload, dict) else None,
                    message_stamp=stamp if finite_number(stamp) else None,
                    received_ros_stamp=now, age_seconds=now-stamp if finite_number(stamp) else None,
                    previous_message_stamp=self.photo_guard.last_stamp,
                    previous_expires_ros_stamp=self.photo_guard.expires,
                    ttl_seconds=ttl if finite_number(ttl) else None)

    def _reject_guard_message(self, payload, now, error):
        detail = self._guard_timing(payload, now, 'rejected')
        detail['error'] = str(error)
        self.guard_message_timing = self.guard_message_rejection = detail
        self._discard_pending_guard_message()
        if self.photo_guard.active:
            self.stop_photo(str(error))
        rospy.logwarn_throttle(1, 'Photo guard rejected: %s; timing=%s', str(error), json.dumps(detail))

    def _apply_guard_message(self, payload, now):
        if (isinstance(payload, dict) and payload.get('op') == 'enable'
                and payload.get('stamp', 0) <= self.search_reject_before):
            raise GuardError('photo_guard_enable_before_search_boundary')
        if isinstance(payload, dict) and payload.get('op') == 'enable' and self.state not in ('HOVERING', 'FLYING'):
            raise GuardError('photo_guard_requires_airborne_hold')
        detail = self._guard_timing(payload, now, 'applied')
        op = self.photo_guard.apply(payload, now)
        self.guard_message_timing = detail
        if op == 'enable':
            self.stop_search('new_photo_guard')
            self.photo_reject_detail = None
        if op in ('enable', 'clear'):
            if op == 'clear' and self.state == 'LANDING':
                # The adapter's async clear may follow LAND. Never replace
                # its ground target or erase descent motion.
                self.photo_trajectories.clear()
                self.photo_reject_before = max(self.photo_reject_before, rospy.get_time())
            else:
                self.stop_photo('')
        if self.photo_guard.active and not self.photo_guard.clears(self.p):
            self.stop_photo('photo_guard_current_position_inside', deactivate=True)

    def _guard_clock_wait_timeout(self, pending):
        # rospy.Timer itself stops if /clock stops. This wall timer cannot
        # renew the lease; it can only discard and reject this exact packet.
        with self.lock:
            if (getattr(self, 'pending_guard_message', None) is pending and
                    time.monotonic()-pending['received_wall'] >= .2):
                self._reject_guard_message(pending['payload'], rospy.get_time(),
                                           GuardError('photo_guard_clock_wait_timeout'))

    def consume_pending_guard_message(self):
        pending = getattr(self, 'pending_guard_message', None)
        if pending is None:
            return
        now = rospy.get_time()
        if self.photo_guard.expire(now):
            self.stop_photo('photo_guard_expired')
            return
        try:
            if time.monotonic()-pending['received_wall'] > .2:
                raise GuardError('photo_guard_clock_wait_timeout')
            if now < pending['payload']['stamp']:
                return  # No ACK, nonce consumption, expiry extension, or authorization.
            self._discard_pending_guard_message(tombstone_enable=False)
            self._apply_guard_message(pending['payload'], now)
            self.guard_message_timing.update(clock_wait_seconds=time.monotonic()-pending['received_wall'],
                                             original_received_ros_stamp=pending['received_ros'])
        except (ValueError, TypeError, OverflowError) as exc:
            self._reject_guard_message(pending['payload'], now, exc)

    def guard_message(self, msg):
        with self.lock:
            payload, now = None, rospy.get_time()
            try:
                payload = json.loads(msg.data)
                self.consume_pending_guard_message()
                pending = getattr(self, 'pending_guard_message', None)
                if pending is not None:
                    # A valid current clear may cancel a pending renewal. No
                    # other packet can replace it and renew its wall deadline.
                    if not isinstance(payload, dict) or payload.get('op') != 'clear':
                        raise GuardError('photo_guard_clock_wait_overlap')
                    self._discard_pending_guard_message()
                stamp = payload.get('stamp') if isinstance(payload, dict) else None
                if finite_number(stamp) and 0 < stamp-now <= .1:
                    # Validate schema/owner/nonce/geometry without touching the
                    # live lease. Apply remains strict after local clock catches up.
                    copy.deepcopy(self.photo_guard).apply(payload, stamp)
                    if (payload.get('op') == 'enable' and
                            (stamp <= self.search_reject_before or self.state not in ('HOVERING', 'FLYING'))):
                        raise GuardError('photo_guard_enable_boundary_or_state_invalid')
                    self.pending_guard_message = dict(payload=payload, received_ros=now, received_wall=time.monotonic())
                    timer = threading.Timer(.2, self._guard_clock_wait_timeout, args=(self.pending_guard_message,))
                    timer.daemon = True
                    self.pending_guard_message['timer'] = timer
                    timer.start()
                    self.guard_clock_wait_count = getattr(self, 'guard_clock_wait_count', 0)+1
                    self.guard_message_timing = self._guard_timing(payload, now, 'waiting_for_clock')
                    return
                self._apply_guard_message(payload, now)
            except (ValueError, TypeError, OverflowError) as exc:
                self._reject_guard_message(payload, now, exc)

    def execution_trajectory(self, msg):
        with self.lock:
            start = msg.start_time.to_sec()
            if msg.traj_id <= 0 or not math.isfinite(start) or start <= 0:
                return
            self.latest_id = max(self.latest_id, msg.traj_id)
            self.execution_start_times[msg.traj_id] = start
            for old_id in sorted(self.execution_start_times)[:-32]:
                del self.execution_start_times[old_id]
            if not self.photo_guard.active:
                return
            now = rospy.get_time()
            if (msg.traj_id <= self.minimum_id or msg.traj_id < self.photo_last_command_id
                    or start <= self.photo_reject_before or not -.2 <= now-start <= .6):
                return
            try:
                trajectory = ExecutionTrajectory(msg.traj_id, start, list(msg.duration),
                    [list(msg.coef_x), list(msg.coef_y), list(msg.coef_z)], msg.order)
            except (ValueError, TypeError, OverflowError) as exc:
                self.stop_photo(str(exc))
                return
            # A planner can process the published goal before this subscriber.
            # Cache a bounded number of full executions; never infer authority
            # from their arrival order or move before the goal itself arrives.
            self.photo_execution_cache[trajectory.id] = trajectory
            for old_id in sorted(self.photo_execution_cache)[:-8]:
                del self.photo_execution_cache[old_id]
            if self.accept_commands and start >= self.goal_at:
                self.validate_photo_execution(trajectory)
            self.consume_pending_photo_command()

    def validate_photo_execution(self, trajectory):
        if not self.photo_guard.active or not self.accept_commands:
            return False
        if (trajectory.id <= self.minimum_id or trajectory.start < self.goal_at
                or trajectory.start <= self.photo_reject_before):
            return False
        if not self.photo_guard.proves(trajectory):
            self.stop_photo('photo_trajectory_enters_person_guard')
            return False
        if self.photo_guard.expire(rospy.get_time()):
            self.stop_photo('photo_guard_expired')
            return False
        self.photo_trajectories[trajectory.id] = trajectory
        for old_id in sorted(self.photo_trajectories)[:-8]:
            del self.photo_trajectories[old_id]
        self.photo_guard.validated_trajectory_id = trajectory.id
        return True

    def queue_photo_command(self, msg):
        # Let the previous future sample mature before replacing it. Otherwise
        # a consistent 5 ms /clock lag can starve a valid 100 Hz stream forever.
        self.consume_pending_photo_command()
        if (not self.photo_guard.active or self.photo_guard.error
                or msg.trajectory_id < self.photo_last_command_id
                or (msg.trajectory_id == self.photo_last_command_id
                    and msg.header.stamp.to_sec() <= self.photo_last_command_stamp)):
            return
        pending = self.pending_photo_command
        if pending is not None and (msg.trajectory_id, msg.header.stamp.to_sec()) <= (
                pending.trajectory_id, pending.header.stamp.to_sec()):
            return
        if self.photo_pending_since is None:
            self.photo_pending_since = (rospy.get_time(), time.monotonic())
        proven_continuation = (self.accept_commands and self.cmd is not None
            and self.cmd.trajectory_id == msg.trajectory_id
            and msg.trajectory_id in self.photo_trajectories
            and stamp_fresh(self.cmd.header.stamp.to_sec(), rospy.get_time(), self.command_timeout))
        if not proven_continuation:
            self.hold = self.p[:]
            self.v, self.a = [0.0]*3, [0.0]*3
            self.cmd = None
            self.transition('HOVERING')
        # One newest command only. A new ID does not extend the wait deadline.
        # A same-execution clock-only wait retains its already validated sample;
        # the future sample is never executed until its original stamp matures.
        self.pending_photo_command = msg
        self.consume_pending_photo_command()

    def consume_pending_photo_command(self):
        if self.photo_pending_since is None:
            return
        now = rospy.get_time()
        since_ros, since_wall = self.photo_pending_since
        if now-since_ros >= .4 or time.monotonic()-since_wall >= 1.0 or now < since_ros:
            self.stop_photo('photo_trajectory_wait_timeout')
            return
        if self.photo_guard.expire(now):
            self.stop_photo('photo_guard_expired')
            return
        msg = self.pending_photo_command
        if msg is None or not self.accept_commands or not self.photo_guard.active:
            return
        stamp = msg.header.stamp.to_sec()
        start = self.execution_start_times.get(msg.trajectory_id)
        if (msg.trajectory_id <= self.minimum_id or stamp < self.goal_at
                or (start is not None and (start < self.goal_at or start <= self.photo_reject_before))):
            self.pending_photo_command = None
            return
        trajectory = self.photo_trajectories.get(msg.trajectory_id)
        if trajectory is None or stamp > now:
            return
        vals = [msg.position.x, msg.position.y, msg.position.z,
                msg.velocity.x, msg.velocity.y, msg.velocity.z,
                msg.acceleration.x, msg.acceleration.y, msg.acceleration.z]
        fresh = stamp_fresh(stamp, now, self.command_timeout)
        matches = trajectory.matches(stamp, vals[0:3], vals[3:6], vals[6:9])
        clear = self.photo_guard.clears(self.p, vals[0:3])
        if not (fresh and matches and clear):
            expected = trajectory.evaluate(stamp)
            self.photo_reject_detail = dict(execution_id=msg.trajectory_id,
                command_stamp=stamp, execution_start=trajectory.start, controller_now=now,
                fresh=fresh, trajectory_matches=matches, person_segment_clear=clear,
                maximum_errors=[max(abs(vals[3*i+j]-expected[i][j]) for j in range(3)) for i in range(3)])
            self.stop_photo('photo_command_not_on_validated_safe_trajectory')
            return
        if norm([vals[i]-self.p[i] for i in range(3)]) > .8:
            self.stop_photo('photo_trajectory_tracking_error')
            self.transition('FAULT')
            return
        self.cmd = msg
        self.command_at = now
        self.photo_last_command_id = msg.trajectory_id
        self.photo_last_command_stamp = stamp
        self.pending_photo_command = None
        self.photo_pending_since = None

    def takeoff_land(self, msg):
        with self.lock:
            if msg.takeoff_land_cmd == TakeoffLand.TAKEOFF and self.state == 'GROUNDED' and self.model_ready:
                if not stamp_fresh(self.lidar_at, rospy.get_time(), self.lidar_timeout):
                    rospy.logwarn('Takeoff rejected: LiDAR is stale')
                    return
                self.clear_gate()
                self.hold[2] = self.height
                self.transition('TAKING_OFF')
            elif msg.takeoff_land_cmd == TakeoffLand.LAND and self.state not in ('GROUNDED', 'LANDING'):
                self.stop_search('land')
                if self.photo_guard.active:
                    self.stop_photo('', deactivate=True)
                self.clear_gate()
                self.hold[2] = self.ground
                self.transition('LANDING')

    def goal(self, msg):
        with self.lock:
            vals = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            if self.state not in ('HOVERING', 'FLYING') or not all(math.isfinite(x) for x in vals):
                return
            if msg.header.frame_id not in ('world', ''):
                return
            # A new task cannot let an earlier queued guard enable mature
            # after it. Existing active-guard trajectory checks still apply.
            self._discard_pending_guard_message()
            if not (-8 <= vals[0] <= 8 and -5 <= vals[1] <= 5 and 0.6 <= vals[2] <= 2.5):
                return
            now = rospy.get_time()
            stamp = msg.header.stamp.to_sec()
            if self.photo_reject_before and stamp <= self.photo_reject_before:
                return
            if self.photo_guard.active:
                if (self.photo_guard.expire(now) or not stamp_fresh(stamp, now, .6)
                        or stamp < self.photo_guard.epoch or not self.photo_guard.clears(vals)):
                    self.stop_photo('photo_goal_rejected')
                    return
            self.stop_search('new_goal')
            # Goal source time is shared with the planner; callback receipt
            # time is not a valid ordering barrier across independent topics.
            if self.photo_guard.active and stamp <= self.goal_at:
                return
            pending = self.pending_photo_command
            pending_since = self.photo_pending_since
            self.hold = self.p[:]
            self.minimum_id = self.minimum_id if self.photo_guard.active else self.latest_id
            self.goal_at = stamp if self.photo_guard.active else now
            self.photo_reject_detail = None
            self.cmd = None
            self.pending_photo_command = None
            self.photo_pending_since = None
            self.photo_trajectories.clear()
            self.photo_guard.validated_trajectory_id = 0
            self.accept_commands = True
            if self.photo_guard.active:
                self.v, self.a = [0.0]*3, [0.0]*3
                # Preserve a pre-goal command only as untrusted pending data.
                if pending is not None and pending.header.stamp.to_sec() >= stamp:
                    self.pending_photo_command = pending
                    self.photo_pending_since = pending_since
                for trajectory in list(self.photo_execution_cache.values()):
                    if trajectory.start >= stamp and not self.validate_photo_execution(trajectory):
                        break
                self.consume_pending_photo_command()

    def pause(self, _):
        with self.lock:
            if self.state in ('HOVERING', 'FLYING', 'TAKING_OFF'):
                self.stop_search('hold')
                if self.photo_guard.active:
                    self.stop_photo('', deactivate=True)
                self.clear_gate()
                self.transition('HOVERING')

    def contacts(self, msg):
        with self.lock:
            if msg.states and self.p[2] > self.ground + 0.2:
                self.contact_count += 1
                self.stop_search('collision')
                self.clear_gate()
                if self.photo_guard.active:
                    self.stop_photo('photo_guard_collision', deactivate=True)
                self.transition('FAULT')

    def command(self, msg):
        with self.lock:
            self.latest_id = max(self.latest_id, msg.trajectory_id)
            vals = [msg.position.x, msg.position.y, msg.position.z,
                    msg.velocity.x, msg.velocity.y, msg.velocity.z,
                    msg.acceleration.x, msg.acceleration.y, msg.acceleration.z, msg.yaw]
            stamp = msg.header.stamp.to_sec()
            if (self.state not in ('HOVERING', 'FLYING') or msg.trajectory_id <= self.minimum_id
                    or msg.trajectory_flag != PositionCommand.TRAJECTORY_STATUS_READY
                    or not all(math.isfinite(x) for x in vals) or not math.isfinite(stamp)):
                return
            if self.photo_search.active or self.photo_search.pending:
                return
            if self.photo_guard.active:
                now = rospy.get_time()
                if self.photo_guard.expire(now):
                    self.stop_photo('photo_guard_expired')
                    return
                if (stamp <= self.photo_reject_before or stamp < self.goal_at
                        or msg.trajectory_id < self.photo_last_command_id
                        or (msg.trajectory_id == self.photo_last_command_id and stamp <= self.photo_last_command_stamp)):
                    return
                if (msg.header.frame_id not in ('world', '') or stamp > now+.2
                        or not (-8 <= vals[0] <= 8 and -5 <= vals[1] <= 5 and .4 <= vals[2] <= 2.7)):
                    self.stop_photo('photo_command_frame_time_or_fence_invalid')
                    return
                execution_start = self.execution_start_times.get(msg.trajectory_id)
                if execution_start is not None and (execution_start <= self.photo_reject_before or execution_start < self.goal_at):
                    return
                # This same path handles command-before-proof, execution-before-
                # goal, and command timestamps slightly ahead of local /clock.
                # Ready commands are consumed immediately without a motion stop.
                trajectory = self.photo_trajectories.get(msg.trajectory_id)
                if (self.photo_pending_since is None and self.accept_commands
                        and trajectory is not None and stamp <= now):
                    self.pending_photo_command = msg
                    self.photo_pending_since = (now, time.monotonic())
                    self.consume_pending_photo_command()
                else:
                    self.queue_photo_command(msg)
                return
            if not self.accept_commands or stamp < self.goal_at:
                return
            execution_start = self.execution_start_times.get(msg.trajectory_id, 0)
            if self.photo_reject_before and (execution_start <= self.photo_reject_before or execution_start < self.goal_at):
                return
            if not (-8 <= vals[0] <= 8 and -5 <= vals[1] <= 5 and .4 <= vals[2] <= 2.7):
                self.clear_gate()
                rospy.logwarn_throttle(1, 'Out-of-fence command rejected; holding')
                return
            if norm([vals[i]-self.p[i] for i in range(3)]) > .8:
                self.clear_gate()
                self.transition('FAULT')
                rospy.logerr('Trajectory tracking error exceeds 0.8 m; refusing an unsafe catch-up')
                return
            self.cmd = msg
            self.command_at = rospy.get_time()

    def update(self, event):
        with self.lock:
            now = rospy.Time.now()
            dt = 0.02 if event.last_real is None else min(0.05, max(0.001, (event.current_real-event.last_real).to_sec()))
            connected = self.model_ready and now.to_sec()-self.model_seen < 1.0
            if self.photo_guard.expire(now.to_sec()):
                self.stop_photo('photo_guard_expired')
            self.consume_pending_guard_message()
            self.consume_pending_photo_command()
            if not connected and self.state != 'GROUNDED':
                self.stop_search('model_disconnected')
                self.clear_gate()
                self.v, self.a = [0.0]*3, [0.0]*3
                self.transition('FAULT')
                # No task progress against a missing Gazebo vehicle.
                return
            if (self.state in ('FLYING', 'TAKING_OFF') or self.accept_commands) and not stamp_fresh(self.lidar_at, now.to_sec(), self.lidar_timeout):
                self.clear_gate()
                self.transition('HOVERING')
                rospy.logwarn_throttle(1, 'LiDAR is stale; gate closed and holding')
            target, velocity, feedforward = self.hold[:], [0.0]*3, [0.0]*3
            wanted_yaw = self.yaw + self.update_search(now.to_sec(), dt)
            if self.cmd is not None:
                if now.to_sec() - self.command_at > self.command_timeout:
                    self.clear_gate()
                    self.transition('HOVERING')
                    target = self.hold[:]
                else:
                    target = [self.cmd.position.x, self.cmd.position.y, self.cmd.position.z]
                    velocity = [self.cmd.velocity.x, self.cmd.velocity.y, self.cmd.velocity.z]
                    feedforward = [self.cmd.acceleration.x, self.cmd.acceleration.y, self.cmd.acceleration.z]
                    wanted_yaw = self.cmd.yaw
                    self.transition('FLYING' if norm(velocity) > 0.08 or norm([target[i]-self.p[i] for i in range(3)]) > 0.08 else 'HOVERING')
            if self.state == 'GROUNDED':
                self.v, self.a = [0.0]*3, [0.0]*3
                self.p[2] = self.ground
            else:
                # PD + trajectory acceleration, with limited actuator slew.
                old_position = self.p[:]
                for i in range(3):
                    desired = clip(6.0*(target[i]-self.p[i]) + 4.5*(velocity[i]-self.v[i]) + feedforward[i], 3.0)
                    self.a[i] += clip(desired-self.a[i], 10.0*dt)
                    limit = 0.5 if self.state in ('TAKING_OFF', 'LANDING') else 1.7
                    self.v[i] = clip(self.v[i]+self.a[i]*dt, limit)
                    self.p[i] += self.v[i]*dt
                if self.photo_guard.active and not self.photo_guard.clears(old_position, self.p):
                    self.p = old_position
                    self.stop_photo('photo_tracking_would_enter_person_guard')
                if self.p[2] < self.ground:
                    self.p[2], self.v[2], self.a[2] = self.ground, 0.0, 0.0
                if self.state == 'TAKING_OFF' and abs(self.p[2]-self.height) < 0.035 and abs(self.v[2]) < 0.08:
                    self.transition('HOVERING')
                if self.state == 'LANDING' and self.p[2] <= self.ground+0.02 and abs(self.v[2]) < 0.10:
                    self.p[2] = self.ground
                    self.v, self.a = [0.0]*3, [0.0]*3
                    self.hold = self.p[:]
                    self.transition('GROUNDED')
            delta_yaw = math.atan2(math.sin(wanted_yaw-self.yaw), math.cos(wanted_yaw-self.yaw))
            # Keep photo rotation within the RGB/LiDAR registration budget.
            yaw_rate = .25 if self.photo_guard.active else 1.2
            self.yaw += clip(delta_yaw, yaw_rate*dt)
            # Mild inertial tilt appears in BOTH Gazebo and odometry, therefore
            # lidar registration uses the same orientation as physical rays.
            ax = math.cos(self.yaw)*self.a[0] + math.sin(self.yaw)*self.a[1]
            ay = -math.sin(self.yaw)*self.a[0] + math.cos(self.yaw)*self.a[1]
            quat = tf.transformations.quaternion_from_euler(clip(-ay/9.81, 0.25), clip(ax/9.81, 0.25), self.yaw)
            odom = Odometry()
            odom.header.stamp, odom.header.frame_id, odom.child_frame_id = now, 'world', 'base_link'
            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = self.p
            odom.pose.pose.orientation = Quaternion(*quat)
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = self.v
            self.odom_pub.publish(odom)
            self.tf.sendTransform(self.p, quat, now, 'base_link', 'world')
            self.tf.sendTransform((0,0,0.155), (0,0,0,1), now, 'mid360_link', 'base_link')
            # Gazebo camera looks along body +X. ROS optical frame uses +Z
            # forward, +X right, +Y down (REP-103 camera convention).
            self.tf.sendTransform((0.17,0,0.02), (0,0,0,1), now, 'camera_link', 'base_link')
            self.tf.sendTransform((0,0,0), (-0.5,0.5,-0.5,0.5), now, 'camera_optical', 'camera_link')
            if self.model_ready:
                model = ModelState()
                model.model_name, model.reference_frame, model.pose = self.model, 'world', odom.pose.pose
                self.model_pub.publish(model)

    def publish_status(self, _):
        with self.lock:
            self.state_pub.publish(String(data=self.state))
            self.detail_pub.publish(String(data=json.dumps({
                'connected': self.model_ready and rospy.get_time()-self.model_seen < 1.0,
                'armed': self.state != 'GROUNDED', 'landed': self.state == 'GROUNDED',
                'flight_mode': self.state, 'simulation': True,
                'physics': 'acceleration_limited_tracking', 'contacts': self.contact_count,
                'photo_guard': dict(self.photo_guard.status(),
                    message_timing=getattr(self, 'guard_message_timing', None),
                    last_message_rejection=getattr(self, 'guard_message_rejection', None),
                    clock_wait_count=getattr(self, 'guard_clock_wait_count', 0),
                    clock_wait_pending=getattr(self, 'pending_guard_message', None) is not None),
                'photo_search': self.photo_search.status(),
                'photo_guard_error': self.photo_guard.error,
                'photo_command_rejection': self.photo_reject_detail,
                'photo_command_pending': self.pending_photo_command is not None})))


if __name__ == '__main__':
    rospy.init_node('vla_flight_controller')
    FlightController()
    rospy.spin()
