"""Fixed ROS messages only: no user-provided shell, ROS topic or service names.

The HTTP service supplies authorization, geofence and mission sequencing. This
adapter independently requires live feedback and preserves PX4CTRL preflight
checks. A successful method call means a ROS request was sent, never that flight
has succeeded; completion is determined from snapshot feedback by the service.
"""
import json
import logging
import math
import os
import re
import secrets
import threading
import time

from .gate_policy import PolynomialTrajectory
from .camera import CameraBuffer
from .bounded_call import BoundedCall
from .real_motion import rc_command_allowed, RCFeedback
from uav_bridge.flight_limits import apply_planner_limits, planner_scope, within_bounds


def create_adapter(config):
    return ROSAdapter(config)


class ROSAdapter:
    def __init__(self, config):
        try:
            import rospy
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import PointCloud2
            from quadrotor_msgs.msg import PositionCommand, TakeoffLand
            from std_msgs.msg import Empty, String
            from std_srvs.srv import SetBool
            from traj_utils.msg import PolyTraj
        except ImportError as exc:
            raise RuntimeError("ROS 1 is unavailable: source ROS Noetic and the built Diff-VLA workspace first") from exc
        self.ros = rospy
        self.msg = {"goal": PoseStamped, "yaw": PositionCommand,
                    "takeoff_land": TakeoffLand, "empty": Empty}
        self.mode = os.environ.get("VLA_MODE", config.get("mode", "gazebo"))
        if self.mode not in ("gazebo", "real"):
            raise ValueError("ROS adapter mode must be explicitly gazebo or real")
        self.config = config
        self._real_backend = (self.mode == 'real' and config.get('ros', {}).get('execution_backend') == 'diff_vla_real_v1')
        self._execution_stamp = 0.
        self._execution_status = {}
        self.frame_id = str(config.get("ros", {}).get("frame_id", "world")).lstrip("/")
        if not self.frame_id:
            raise ValueError("ros.frame_id must name the shared odometry/goal coordinate frame")
        self.lock = threading.RLock()
        self.command_lock = threading.RLock()
        self.timeout = float(config.get("ros", {}).get("telemetry_timeout", 2.0))
        self.trajectory_timeout = float(config.get("ros", {}).get("trajectory_start_timeout", 8.0))
        if not math.isfinite(self.trajectory_timeout) or not 1 <= self.trajectory_timeout <= 60:
            raise ValueError("ros.trajectory_start_timeout must be between 1 and 60 seconds")
        self.planner_startup_timeout = float(config.get("ros", {}).get("planner_startup_timeout", 180.0))
        if not math.isfinite(self.planner_startup_timeout) or self.planner_startup_timeout <= 0:
            raise ValueError("ros.planner_startup_timeout must be positive and finite")
        self.takeoff_height = float(config.get("takeoff_height", 1.2))
        self._position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._yaw = 0.0
        self._quaternion = dict(x=0., y=0., z=0., w=1.)
        self._odom_child_frame = ""
        self._position_stamp = self._state_stamp = self._landed_stamp = 0.0
        self._cloud_stamp = self._planner_stamp = 0.0
        self._armed = self._landed = self._connected = False
        self._flight_mode = "UNKNOWN"
        self._rc_command_allowed = None
        self._rc_feedback = RCFeedback(False, .5)
        self._rc_feedback_failed = False
        self._control_epoch = 0
        self._battery = None
        self._battery_stamp = 0.0
        self._target_yaw = None
        self._yaw_epoch = 0
        self._takeoff_target_z = None
        self._mission_active = False
        self._trajectory_pending = None
        self._navigation_request = None
        self._command_error = None
        self._gate_active = False
        self._visual = None
        self._goal_query = None
        self._photo_guard = None
        self._photo_guard_status = {"supported": False}
        self._photo_search = None
        self._photo_search_status = {"supported": False}
        self._photo_search_cancelled = set()
        self._photo_search_used = set()
        self._photo_search_exhausted = False
        if not rospy.core.is_initialized():
            rospy.init_node("diff_vla_bridge", anonymous=False, disable_signals=True)
        if planner_scope(config):
            self._wait_for_planner_limits()
        self.goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=1, latch=False)
        self.yaw_pub = rospy.Publisher("/planning/yaw", PositionCommand, queue_size=1, latch=False)
        self.takeoff_pub = rospy.Publisher("/px4ctrl/takeoff_land", TakeoffLand, queue_size=1, latch=False)
        self.hold_pub = rospy.Publisher("/vla/hold", Empty, queue_size=1, latch=False)
        self.photo_guard_pub = rospy.Publisher("/vla/photo_guard", String, queue_size=3, latch=False)
        self.photo_search_pub = rospy.Publisher("/vla/photo_search", String, queue_size=3, latch=False)
        self.msg["string"] = String
        self.lease_pub = rospy.Publisher("/diff_vla/gate_lease", Empty, queue_size=1, latch=False)
        self.gate = rospy.ServiceProxy("/diff_vla/command_gate/set_enabled", SetBool)
        odom_topic = config.get('ros', {}).get('odom_topic', "/odom" if self.mode == "gazebo" else "/ekf/ekf_odom")
        self.subs = [rospy.Subscriber(odom_topic, Odometry, self._odom, queue_size=1)]
        cloud_topic = config.get('ros', {}).get('cloud_topic', "/cloud_registered" if self.mode == "gazebo" else "/laserMapping/cloud_registered")
        self._cloud_topic = cloud_topic
        heartbeat_topic = ("/vla_traj_server/heartbeat" if self.mode == "gazebo"
                           else "/drone_%d_traj_server/heartbeat" % int(config.get("ros", {}).get("drone_id", 0)))
        trajectory_topic = ("/planning/trajectory" if self.mode == "gazebo"
                            else "/drone_%d_planning/trajectory" % int(config.get("ros", {}).get("drone_id", 0)))
        trajectory_topic = config.get('ros', {}).get('trajectory_topic', trajectory_topic)
        self.subs.extend([rospy.Subscriber(cloud_topic, PointCloud2, self._cloud, queue_size=1),
                          rospy.Subscriber(heartbeat_topic, Empty, self._planner_heartbeat, queue_size=1),
                          rospy.Subscriber(trajectory_topic, PolyTraj, self._trajectory, queue_size=20)])
        if self.mode == "gazebo":
            self.subs.append(rospy.Subscriber("/diff_vla/sim/state", String, self._sim_state, queue_size=1))
        else:
            try:
                from mavros_msgs.msg import ExtendedState, State, RCIn
                from sensor_msgs.msg import BatteryState
            except ImportError as exc:
                raise RuntimeError("Real mode requires mavros_msgs and sensor_msgs") from exc
            self.subs.extend([
                rospy.Subscriber("/mavros/state", State, self._real_state, queue_size=1),
                rospy.Subscriber("/mavros/extended_state", ExtendedState, self._real_extended, queue_size=1),
                rospy.Subscriber("/mavros/battery", BatteryState, self._real_battery, queue_size=1),
            ])
            self._rc_feedback = RCFeedback(not rospy.get_param('/px4ctrl/auto_takeoff_land/no_RC', False),
                                           float(rospy.get_param('/px4ctrl/msg_timeout/rc', .5)))
            if self._rc_feedback.required:
                self.subs.append(rospy.Subscriber('/mavros/rc/in', RCIn, self._real_rc, queue_size=20))
            if self._real_backend:
                self.subs.append(rospy.Subscriber('/diff_vla/executor/state', String, self._real_execution_state, queue_size=1))
        self._init_camera()
        self._init_visual_grounding()
        self._init_goal_query()
        self.timer = rospy.Timer(rospy.Duration(0.1), self._tick)

    def _wait_for_planner_limits(self):
        """Construction-only wait; runtime trajectory and service budgets stay unchanged."""
        started = time.monotonic()
        deadline = started + self.planner_startup_timeout
        next_log = started + 5.0
        # Reuse one bounded call slot: a stalled master cannot accumulate
        # abandoned workers or defeat the overall initialization deadline.
        read_limits = BoundedCall('原规划器启动参数')
        launcher_pid = os.environ.get('VLA_REAL_LAUNCH_PID')
        if launcher_pid is not None:
            if not launcher_pid.isdecimal() or int(launcher_pid) <= 0:
                raise ValueError('VLA_REAL_LAUNCH_PID must name the current ROS launcher process')
            launcher_pid = int(launcher_pid)

        def current_limits():
            if launcher_pid is not None:
                sequence = self.ros.get_param('/diff_vla/startup_sequence', None)
                # The launcher publishes this only after this parent's
                # clear_params has removed any previous C++ report. Merely
                # finding limits on a reused master is not sufficient.
                if (not isinstance(sequence, dict) or sequence.get('launcher_pid') != launcher_pid
                        or sequence.get('planner_limits_reset') is not True):
                    return None
            return self.ros.get_param('/diff_vla/planner/loaded_limits', None)
        logging.info('正在等待原规划器按顺序启动并提供实际范围（启动等待最多 %.0f 秒，任务执行超时不变）。',
                     self.planner_startup_timeout)
        while True:
            if self.ros.is_shutdown():
                raise RuntimeError('原规划器初始化等待已中止')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('原规划器在启动等待期限内尚未提供实际加载的范围配置（%.0f 秒）；'
                                   '请查看分阶段启动日志，手机服务尚未开放。' % self.planner_startup_timeout)
            try:
                limits = read_limits(current_limits, timeout=min(.25, remaining))
            except Exception:
                limits = None  # Missing/unreachable master is not proof of usable bounds.
            if self.ros.is_shutdown():
                raise RuntimeError('原规划器初始化等待已中止')
            if limits is not None and time.monotonic() <= deadline:
                # Use only the C++ planner's actual loaded values. This retains
                # their existing schema validation and does not invent defaults.
                apply_planner_limits(self.config, limits)
                logging.info('原规划器实际范围已收到，启动等待用时 %.1f 秒；继续初始化手机服务。',
                             time.monotonic() - started)
                return
            now = time.monotonic()
            if now >= next_log:
                logging.info('原规划器仍在初始化，已等待 %.0f 秒；尚未开放手机任务服务。', now - started)
                next_log = now + 5.0
            if now < deadline:
                time.sleep(min(.1, deadline - now))

    def _photo_execution_available(self):
        if self.mode == 'gazebo':
            return True
        return bool(getattr(self, '_real_backend', False) and
                    time.monotonic()-getattr(self, '_execution_stamp', 0.) <= 1. and
                    getattr(self, '_execution_status', {}).get('ready') is True)

    def _real_execution_state(self, message):
        try:
            if len(message.data) > 16384:
                return
            status = json.loads(message.data)
            if (status.get('protocol') != 1 or status.get('backend') != 'diff_vla_real'
                    or type(status.get('ready')) is not bool
                    or not isinstance(status.get('photo_guard'), dict)
                    or not isinstance(status.get('photo_search'), dict)):
                return
        except (ValueError, TypeError):
            return
        with self.lock:
            self._execution_stamp, self._execution_status = time.monotonic(), status
            self._photo_guard_status = dict(status['photo_guard'])
            self._photo_search_status = dict(status['photo_search'])
            if (self._mission_active and status.get('enabled') is False
                    and isinstance(status.get('error'), str) and status['error']):
                self._command_error = '实机执行停止：'+status['error'][:180]

    def _lookup_visual_transform(self, stamp, source_body, camera_frame):
        """Allow TF callbacks to catch up, keeping the exact RGB timestamp.

        RGB can arrive a few milliseconds before the 50 Hz transform. Wait in
        the observation worker only, bounded by wall time even if ROS pauses.
        Never replace this request with Time(0) or extrapolate the calibration.
        """
        requested = self.ros.Time.from_sec(stamp)
        deadline = time.monotonic() + .12
        while True:
            try:
                return self._visual_tf.lookupTransform(source_body, camera_frame, requested)
            except Exception as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("RGB 时刻相机光学外参不可用：" + str(exc)[:160]) from exc
                time.sleep(min(.01, remaining))

    def _init_visual_grounding(self):
        from itertools import islice
        import numpy as np
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs import point_cloud2
        from .visual_grounding import VisualGrounding, quaternion_matrix
        options = self.config.get("ros", {})
        body = options.get("body_frame", "base_link" if self.mode == "gazebo" else None)
        optical = options.get("camera_optical_frame", "camera_optical" if self.mode == "gazebo" else None)
        info_topic = options.get("camera_info_topic", "/vla/camera/camera_info" if self.mode == "gazebo" else None)
        configured = (self.mode == "gazebo" or
                      (options.get("camera_calibration_confirmed") is True and options.get("camera_topic")
                       and body and optical and info_topic))
        reason = None if configured else "实机视觉定位未配置：需要明确相机话题、CameraInfo、机体/光学坐标及已标定外参确认"
        self._visual_tf = None
        if configured:
            try:
                import tf
                self._visual_tf = tf.TransformListener()
            except ImportError:
                configured, reason = False, "缺少 ROS tf，相机外参不可用"

        def transform(stamp, source_body, camera_frame):
            translation, rotation = self._lookup_visual_transform(stamp, source_body, camera_frame)
            value = np.eye(4)
            value[:3, :3] = quaternion_matrix(rotation)
            value[:3, 3] = translation
            return value

        def decode_cloud(message):
            # Registered clouds contain physical ray hits only. Limit decoding
            # work without growing a queue; no map/scene geometry is consulted.
            step = max(1, (int(message.width)*int(message.height)+29999)//30000)
            return list(islice(point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True),
                               0, None, step))

        self._visual = VisualGrounding(self.camera_frame, self.snapshot, transform, decode_cloud,
                                       world_frame=self.frame_id, body_frame=body or "",
                                       optical_frame=optical or "", configured=bool(configured), disabled_reason=reason,
                                       sensor_skew=float(options.get("visual_sensor_skew", 0.15)),
                                       ros_clock=lambda: self.ros.Time.now().to_sec(),
                                       cloud_topic=getattr(self, '_cloud_topic', "/cloud_registered" if self.mode == "gazebo" else "/laserMapping/cloud_registered"),
                                       camera_info_topic=info_topic,
                                       demo_target_options=self.config.get('photo', {}),
                                       allow_pinhole_approximation=(self.mode == 'real' and
                                           os.environ.get('VLA_CAMERA_MODE', 'calibrated')
                                           in ('auto', 'elastic_pinhole')))
        if configured:
            self.subs.append(self.ros.Subscriber(info_topic, CameraInfo,
                                                lambda message: self._visual.add_info(message, self._receipt_stamp(message)),
                                                queue_size=10))

    def visual_observation(self):
        """Freeze a locally authenticated RGB/real-LiDAR observation for a VLM."""
        return self._visual.freeze()

    def resolve_visual_target(self, observation, bbox, standoff_m=1.2):
        result = self._visual.resolve(observation, bbox, standoff_m)
        if self._goal_query is not None:
            result["evidence"]["goal_map_check"] = self._goal_query.ensure(result["goal"])
        else:
            result["evidence"]["goal_map_check"] = {"supported": False, "reason": "real_mapper_query_not_configured"}
        return result

    def measure_photo_target(self, observation, bbox):
        return self._visual.measure_photo(observation, bbox, getattr(self, "config", {}).get("photo", {}))

    def photo_bearing(self, observation, bbox):
        return self._visual.photo_bearing(observation, bbox)

    @staticmethod
    def _search_identity(identity):
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,96}", identity):
            raise ValueError("搜索必须使用预先分配的唯一任务标识")
        return identity

    def _search_bookkeeping(self):
        if not hasattr(self, "_photo_search_cancelled"):
            self._photo_search_cancelled, self._photo_search_used = set(), set()
            self._photo_search_exhausted = False

    def _publish_photo_search(self, owner, operation):
        nonce = secrets.token_urlsafe(18)
        message = dict(version=1, op=operation, id=owner["id"], nonce=nonce,
                       stamp=self.ros.Time.now().to_sec(), ttl=2.0,
                       rate_rad_s=owner["rate_rad_s"], max_sweep_rad=owner["max_sweep_rad"], timeout_s=owner["timeout_s"])
        if owner.get('guard_id'):
            message['guard_id'] = owner['guard_id']
        self._ensure_subscriber(self.photo_search_pub, "/vla/photo_search")
        self.photo_search_pub.publish(self.msg["string"](data=json.dumps(message, allow_nan=False)))
        owner["last_sent"], owner["nonce"] = time.monotonic(), nonce
        return nonce

    def _wait_photo_search_ack(self, identity, nonce, active):
        deadline = time.monotonic()+1.5
        while time.monotonic() < deadline:
            with self.lock:
                status = dict(getattr(self, "_photo_search_status", {}))
                if active and identity in self._photo_search_cancelled:
                    raise RuntimeError("搜索在启动确认前已取消")
            if status.get("id") == identity and status.get("nonce") == nonce:
                if status.get("pending") is True and not status.get("error"):
                    time.sleep(.02)
                    continue
                if status.get("supported") is True and status.get("active") is active and not status.get("error"):
                    return status
                raise RuntimeError("搜索控制器拒绝请求："+str(status.get("error") or status.get("reason") or "invalid_ack"))
            time.sleep(.02)
        raise RuntimeError("原地转向搜索确认超时，未授权搜索动作")

    def start_photo_search(self, search_id, max_sweep_rad=2*math.pi, timeout=30., rate=.35,
                           guard_token=None, control_token=None, before_dispatch=None):
        if before_dispatch is not None and not callable(before_dispatch):
            raise TypeError('before_dispatch must be callable')
        identity = self._search_identity(search_id)
        values = (max_sweep_rad, timeout, rate)
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                or not 0 < max_sweep_rad <= 2*math.pi or not 1 <= timeout <= 35 or not .10 <= rate <= .35):
            raise ValueError("搜索转角、时限或角速度超出已验证范围")
        if not self._photo_execution_available():
            raise RuntimeError("实机任务执行后端尚未就绪，请检查 diff_vla_real 状态")
        if guard_token is not None and (self.mode != 'real' or
                getattr(self, '_execution_status', {}).get('guarded_photo_search') is not True):
            raise RuntimeError('执行服务缺少保留人物保护的搜索能力，请更新并重启本版执行服务。')
        control_epoch = self.capture_control_token() if control_token is None else control_token
        with self.command_lock:
            with self.lock:
                if self._control_changed(control_epoch):
                    raise RuntimeError('搜索排队期间发生遥控接管，旧搜索已取消。')
                self._search_bookkeeping()
                if (self._photo_search_exhausted or identity in self._photo_search_cancelled
                        or identity in self._photo_search_used):
                    raise RuntimeError("搜索标识已取消或使用，不得重新激活")
                if len(self._photo_search_used)+len(self._photo_search_cancelled) >= 4096:
                    self._photo_search_exhausted = True
                    raise RuntimeError("搜索标识保护容量已满，请重启桥接服务")
                guard = getattr(self, '_photo_guard', None)
                if getattr(self, "_photo_search", None) is not None:
                    raise RuntimeError("请先结束已有搜索或摄影任务")
                if guard_token is not None:
                    if (guard is None or guard['id'] != guard_token or
                            not guard['ready'] or guard['failed']):
                        raise RuntimeError('搜索必须沿用当前有效的人物保护任务。')
                elif guard is not None:
                    raise RuntimeError("请先结束已有搜索或摄影任务")
                if self._mission_active and guard_token is None:
                    raise RuntimeError("正在执行飞行任务，不能同时开始原地搜人")
            state = self._require_live(airborne=True)
            if (self.mode == 'gazebo' and state["flight_mode"] != "HOVERING") or not state["planning_ready"]:
                raise RuntimeError("原地搜人需要悬停和新鲜雷达、规划器反馈")
            # Guarded sub-search keeps the photo runner's existing hover
            # tolerance; the original pre-selection search remains unchanged.
            final_hover_admission = guard_token is not None and before_dispatch is not None
            if (not final_hover_admission and
                    math.sqrt(sum(state["velocity"][axis]**2 for axis in ("x", "y", "z"))) > (.15 if guard_token is not None else .12)):
                raise RuntimeError("请先稳定悬停后再原地搜人")
            if self._goal_query is None:
                raise RuntimeError("原地搜人需要当前机体完整包络的地图证据")
            if guard is not None and any(math.hypot(state['position']['x']-z['center'][0],
                    state['position']['y']-z['center'][1]) <= z['radius'] for z in guard['zones']):
                raise RuntimeError('当前机位进入原人物保护范围，不能开始搜索。')
            self._goal_query.ensure(state["position"])
            owner = dict(id=identity, rate_rad_s=float(rate), max_sweep_rad=float(max_sweep_rad),
                         timeout_s=float(timeout), ready=False, failed=False, last_sent=0.,
                         guard_id=guard_token or '')
            with self.lock:
                if identity in self._photo_search_cancelled or self._control_changed(control_epoch):
                    self._photo_search_cancelled.add(identity)
                    raise RuntimeError("搜索已取消或发生遥控接管，不得迟到启动，请重新确认。")
                if guard is not None and (self._photo_guard is not guard or
                        guard['id'] != guard_token or not guard['ready'] or guard['failed']):
                    raise RuntimeError('搜索准备期间人物保护已改变。')
                self._photo_search_used.add(identity)
                self._photo_search = owner
            self._reset_command_tracking()
            callback_rejected = start_attempted = False
            try:
                with self.lock:
                    if (identity in self._photo_search_cancelled or owner['failed'] or
                            self._control_changed(control_epoch) or
                            (guard is not None and (self._photo_guard is not guard or
                             not guard['ready'] or guard['failed']))):
                        raise RuntimeError("搜索已取消或发生遥控接管，不得迟到启动，请重新确认。")
                    if before_dispatch is not None:
                        current = self.snapshot()
                        try:
                            # Local, nonblocking admission only. Never acquire
                            # runner.lock or call adapter/ROS methods here.
                            before_dispatch(current)
                        except Exception:
                            callback_rejected = True
                            self._reset_command_tracking()
                            raise
                        # The guarded runner may retain its decision while a
                        # transient speed change settles. Give its final hook
                        # that opportunity, then retain the original speed
                        # ceiling even if another caller supplies a weaker hook.
                        if final_hover_admission and math.sqrt(sum(
                                current['velocity'][axis]**2 for axis in ('x', 'y', 'z'))) > .15:
                            raise RuntimeError('请先稳定悬停后再原地搜人')
                    # Publication errors cannot prove that START was unsent.
                    start_attempted = True
                    nonce = self._publish_photo_search(owner, "start")
                acknowledgement = self._wait_photo_search_ack(identity, nonce, True)
                if any(acknowledgement.get(key) != owner[key] for key in ("rate_rad_s", "max_sweep_rad", "timeout_s")):
                    raise RuntimeError("搜索控制器确认的参数与请求不一致")
                if guard_token is not None and acknowledgement.get('guard_id') != guard_token:
                    raise RuntimeError('搜索确认未保留原人物保护任务。')
                with self.lock:
                    if (identity in self._photo_search_cancelled or owner['failed'] or
                            self._control_changed(control_epoch) or
                            (guard is not None and (self._photo_guard is not guard or
                             not guard['ready'] or guard['failed']))):
                        raise RuntimeError("搜索已取消，不得迟到启动")
                    owner["ready"] = True
            except Exception as error:
                with self.lock:
                    self._photo_search_cancelled.add(identity)
                    owner["ready"], owner["failed"] = False, True
                try:
                    if callback_rejected and not start_attempted:
                        self._cancel_unpublished_motion(control_epoch, guard)
                    else:
                        nonce = self._publish_photo_search(owner, "stop")
                        self._wait_photo_search_ack(identity, nonce, False)
                except Exception as cleanup_error:
                    if before_dispatch is not None:
                        raise RuntimeError('搜索停止未得到确认，不能按未发送动作重试。') from cleanup_error
                    raise
                finally:
                    with self.lock:
                        if self._photo_search is owner:
                            self._photo_search = None
                if before_dispatch is not None and start_attempted:
                    raise RuntimeError('搜索已进入发送流程，不能按未发送动作重试：'+str(error)) from error
                raise
            return identity

    def stop_photo_search(self, search_id):
        identity = self._search_identity(search_id)
        # Record cancellation before waiting for command_lock. A delayed HTTP
        # worker can never start this owner after a stop-before-start race.
        with self.lock:
            self._search_bookkeeping()
            if len(self._photo_search_cancelled) < 4096:
                self._photo_search_cancelled.add(identity)
            else:
                self._photo_search_exhausted = True
        with self.command_lock:
            with self.lock:
                owner = getattr(self, "_photo_search", None)
                if owner is None or owner["id"] != identity:
                    return dict(stopped=False, cancelled=True, reason="absent_or_superseded")
                owner["ready"] = False
            nonce = self._publish_photo_search(owner, "stop")
            status = self._wait_photo_search_ack(identity, nonce, False)
            with self.lock:
                if self._photo_search is owner:
                    self._photo_search = None
            return dict(stopped=True, cancelled=True, id=identity,
                        angle_swept_rad=status.get('angle_swept_rad', 0.), photo_search=status)

    def measure_photo_candidates(self, observation, detections):
        if not isinstance(detections, list) or not 1 <= len(detections) <= 8:
            raise ValueError("候选测距需要同一帧中的 1 到 8 个人物框")
        measured = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, dict) or detection.get("label", "person") != "person":
                raise ValueError("候选只能来自同一帧人物检测")
            bbox = detection.get("bbox")
            result = self.measure_photo_target(observation, bbox)
            measured.append(dict(index=index, bbox=list(bbox),
                                 person_surface_position=result["person_surface_position"], evidence=result["evidence"],
                                 person_cluster=result["evidence"].get("person_cluster")))
        return measured

    @staticmethod
    def _photo_zones(evidence):
        source = evidence["person_surface_position"]
        primary = dict(center=[float(source[axis]) for axis in ("x", "y")],
                       radius=float(evidence["photo_guard_radius_m"]))
        zones = evidence.get("zones", [primary])
        if not isinstance(zones, list) or not 1 <= len(zones) <= 8:
            raise ValueError("人物保护需要 1 到 8 个有效范围")
        checked = []
        for zone in zones:
            if not isinstance(zone, dict) or not isinstance(zone.get("center"), (list, tuple)) or len(zone["center"]) != 2:
                raise ValueError("人物保护范围格式无效")
            values = list(zone["center"])+[zone.get("radius")]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("人物保护范围必须为有限数值")
            center, radius = list(map(float, zone["center"])), float(zone["radius"])
            if any(abs(v) > 100 for v in center) or not 2.05 <= radius <= 10:
                raise ValueError("人物保护范围超出已验证限制")
            checked.append(dict(center=center, radius=radius))
        if checked[0] != primary:
            raise ValueError("第一个人物保护范围必须对应已选人物")
        return checked

    def resolve_selected_photo(self, observation, selected_bbox, other_bboxes):
        if not self._photo_execution_available():
            raise RuntimeError("实机任务执行后端尚未就绪，请检查 diff_vla_real 状态")
        if not isinstance(other_bboxes, list) or len(other_bboxes) > 7:
            raise ValueError("同一帧最多支持 8 个可测人物")
        boxes = [selected_bbox]+other_bboxes
        people = self.measure_photo_candidates(observation, [dict(label="person", bbox=box) for box in boxes])
        result = self._visual.resolve_photo(observation, selected_bbox, getattr(self, "config", {}).get("photo", {}))
        zones = [dict(center=[person["person_surface_position"][axis] for axis in ("x", "y")],
                      radius=person["evidence"]["photo_guard_radius_m"]) for person in people]
        result["evidence"].update(zones=zones, people=people)
        result["evidence"]["candidate_map_check"] = self.validate_photo_goal(result["goal"], result["evidence"])
        return result

    def resolve_photo_target(self, observation, bbox):
        if not self._photo_execution_available():
            raise RuntimeError("实机任务执行后端尚未就绪，请检查 diff_vla_real 状态")
        result = self._visual.resolve_photo(observation, bbox, getattr(self, "config", {}).get("photo", {}))
        result["evidence"]["candidate_map_check"] = self.validate_photo_goal(result["goal"], result["evidence"])
        return result

    def photo_camera_context(self, observation, include_demo_targets=True):
        return self._visual.photo_camera_context(observation, include_demo_targets=include_demo_targets)

    def validate_static_photo_goal(self, observation, goal, evidence):
        if not self._photo_execution_available() or self.config.get('photo',{}).get('subject_motion_mode')!='stationary':
            raise RuntimeError('当前接口只用于明确配置的静态人物模式')
        self._visual.photo_camera_context(observation, include_demo_targets=False)
        if not -.1<=time.time()-observation['stamp']<=2.5:
            raise RuntimeError('最终机位下发前需要新鲜图像与遥测')
        validation=self.validate_photo_goal(goal,evidence)
        return dict(goal=dict(goal),evidence=dict(position_source='confirmed_stationary_person',
                    original_measurement_ros_stamp=evidence.get('cloud_stamp'),
                    candidate_map_check=validation,goal_replaced=False))

    def validate_proposed_photo_goal(self, observation, selected_bbox, other_bboxes, goal):
        """Read-only validation of the model's exact proposed world x/y/z/yaw.

        Same-frame remeasurement and map admission do not start flight or
        replace an existing guard. The caller must uniquely bind all detected
        subjects, validate against its fixed guard and only then execute.
        """
        if not self._photo_execution_available():
            raise RuntimeError("实机任务执行后端尚未就绪，请检查 diff_vla_real 状态")
        stamp = observation.get("stamp") if isinstance(observation, dict) else None
        max_age = getattr(self, "config", {}).get("camera_timeout_seconds", 2.5)
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                or not -1 <= time.time()-stamp <= max_age):
            raise RuntimeError("模型返回后必须用新的机载图像和点云复核，旧观察不能授权飞行")
        if not isinstance(other_bboxes, list) or len(other_bboxes) > 7:
            raise ValueError("模型摄影机位需要同一帧全部人物框，最多 8 人")
        people = self.measure_photo_candidates(observation,
                    [dict(label="person", bbox=box) for box in [selected_bbox]+other_bboxes])
        result = self._visual.validate_proposed_photo(observation, selected_bbox, goal,
                    getattr(self, "config", {}).get("photo", {}))
        zones = [dict(center=[person["person_surface_position"][axis] for axis in ("x", "y")],
                      radius=person["evidence"]["photo_guard_radius_m"]) for person in people]
        result["evidence"].update(zones=zones, people=people)
        result["evidence"]["candidate_map_check"] = self.validate_photo_goal(result["goal"], result["evidence"])
        return result

    def validate_photo_goal(self, goal, evidence):
        """Read-only admission; the controller additionally guards the curve."""
        if not self._photo_execution_available() or self._goal_query is None:
            raise RuntimeError("拍照需要支持完整机体包络的地图查询和轨迹人物保护")
        state = self._require_live(airborne=True)
        zones = self._photo_zones(evidence)
        origin = [float(state["position"][axis]) for axis in ("x", "y", "z")]
        target = [float(goal[axis]) for axis in ("x", "y", "z")]
        if not all(math.isfinite(v) for v in origin+target):
            raise ValueError("人物保护几何无效")
        distance = math.sqrt(sum((a-b)**2 for a, b in zip(origin, target)))
        if getattr(self, 'config', {}).get('photo', {}).get('navigation_scope') in ('geofence', 'planner'):
            bounds = self.config['bounds']
            if not within_bounds(goal, bounds, planner_scope(self.config)):
                raise RuntimeError('最终摄影机位超出场地边界')
            if any(math.hypot(target[0]-z['center'][0], target[1]-z['center'][1]) < z['radius']+.02 for z in zones):
                raise RuntimeError('最终摄影机位进入人物保护范围')
            start_check = self._goal_query.ensure(dict(zip(('x','y','z'), origin)))
            return dict(source='mapper_grid_query', scope='local_route_intention',
                        start=start_check, destination_known_free_claimed=False,
                        route_planner_required=True, unknown_space_traversal_allowed=False,
                        complete_trajectory_person_guard_required=True)
        from .photo_geometry import photo_limits
        max_move = photo_limits(getattr(self, "config", {}).get("photo", {}))["max_move_m"]
        if distance > max_move+1e-6 or abs(origin[2]-target[2]) > .05:
            raise RuntimeError("拍照目标超出单次短程平飞范围")
        count = max(1, int(math.ceil(distance/.075)))
        checks = []
        for index in range(count+1):
            ratio = index/count
            point = [a+ratio*(b-a) for a, b in zip(origin, target)]
            if any(math.hypot(point[0]-zone["center"][0], point[1]-zone["center"][1]) < zone["radius"]+.02
                   for zone in zones):
                raise RuntimeError("拍照候选航段距人物不足，拒绝执行")
            checks.append(self._goal_query.ensure(dict(zip(("x", "y", "z"), point))))
        return dict(source="mapper_grid_query", scope="sampled_corridor_full_vehicle_envelopes",
                    sample_spacing_m=.075, sample_count=len(checks),
                    endpoint=checks[-1], all_known_free=True,
                    person_zones_checked=len(zones),
                    complete_trajectory_person_guard_required=True)

    def _publish_photo_guard(self, guard, operation):
        nonce = secrets.token_urlsafe(18)
        message = dict(version=1, op=operation, id=guard["id"], nonce=nonce,
                       stamp=self.ros.Time.now().to_sec(), ttl=2.0,
                       center=list(guard["center"]), radius=guard["radius"])
        if "zones" in guard:
            message["zones"] = guard["zones"]
        self._ensure_subscriber(self.photo_guard_pub, "/vla/photo_guard")
        self.photo_guard_pub.publish(self.msg["string"](data=json.dumps(message, allow_nan=False)))
        guard["last_sent"] = time.monotonic()
        guard["nonce"] = nonce
        return nonce

    def _wait_photo_guard_ack(self, identity, nonce, active):
        deadline = time.monotonic()+1.5
        while time.monotonic() < deadline:
            with self.lock:
                status = dict(self._photo_guard_status)
            if status.get("id") == identity and status.get("nonce") == nonce:
                if status.get("supported") is True and status.get("active") is active and not status.get("error"):
                    return status
                raise RuntimeError("人物轨迹保护未接受请求："+str(status.get("error") or "invalid_ack"))
            time.sleep(.02)
        raise RuntimeError("人物轨迹保护确认超时，未授权拍摄飞行")

    def set_photo_guard(self, evidence):
        """Activate before any goal; return an ownership token for cancellation."""
        if not self._photo_execution_available():
            raise RuntimeError("实机任务执行后端尚未就绪，请检查 diff_vla_real 状态")
        zones = self._photo_zones(evidence)
        center, radius = zones[0]["center"], zones[0]["radius"]
        with self.lock:
            control_epoch = getattr(self, '_control_epoch', 0)
        with self.command_lock:
            self._require_live(airborne=True)
            with self.lock:
                if getattr(self, "_photo_guard", None) is not None:
                    raise RuntimeError("已有拍照保护任务，必须先结束原任务")
                if getattr(self, "_photo_search", None) is not None:
                    raise RuntimeError("请先停止原地搜索并稳定悬停后再摄影")
            # enable atomically stops the controller itself. A separate hold
            # topic can arrive later and cancel a just-acknowledged guard.
            self._reset_command_tracking()
            guard = dict(id=secrets.token_urlsafe(18), center=center, radius=radius,
                         ready=False, failed=False, last_sent=0.)
            guard["zones"] = zones
            with self.lock:
                if self._control_changed(control_epoch):
                    raise RuntimeError('人物保护建立期间发生遥控接管，请重新确认。')
                self._photo_guard = guard
            try:
                with self.lock:
                    if guard['failed'] or self._control_changed(control_epoch):
                        raise RuntimeError('人物保护建立期间发生遥控接管，请重新确认。')
                    nonce = self._publish_photo_guard(guard, "enable")
                status = self._wait_photo_guard_ack(guard["id"], nonce, True)
                if (status.get("center") != center or abs(status.get("radius", -1)-radius) > 1e-8):
                    raise RuntimeError("人物保护确认的范围与请求不一致")
                if status.get("zones", [dict(center=center, radius=radius)]) != zones:
                    raise RuntimeError("人物保护确认未覆盖全部可见人物")
                with self.lock:
                    if guard['failed']:
                        raise RuntimeError('人物保护建立期间已被遥控接管，请重新确认。')
                    guard["ready"] = True
            except Exception:
                with self.lock:
                    guard["failed"] = True
                try:
                    self._publish_photo_guard(guard, "clear")
                finally:
                    with self.lock:
                        if self._photo_guard is guard:
                            self._photo_guard = None
                raise
            return guard["id"]

    def clear_photo_guard(self, token=None):
        """A stale task token cannot stop or clear a newer task's protection."""
        with self.command_lock:
            with self.lock:
                guard = getattr(self, "_photo_guard", None)
                if guard is None or (token is not None and token != guard["id"]):
                    return dict(cleared=False, reason="absent_or_superseded")
                guard["ready"] = False
            # clear atomically stops the controller; avoid cross-topic races.
            self._reset_command_tracking()
            nonce = self._publish_photo_guard(guard, "clear")
            self._wait_photo_guard_ack(guard["id"], nonce, False)
            with self.lock:
                if self._photo_guard is guard:
                    self._photo_guard = None
            return dict(cleared=True, id=guard["id"])

    def _init_goal_query(self):
        if self.mode != "gazebo" and not getattr(self, "_real_backend", False):
            return
        from geometry_msgs.msg import PointStamped
        from std_msgs.msg import String
        from .map_query import KnownFreeQuery
        options = self.config.get("ros", {})
        query_topic = options.get("goal_query_topic", "/vla_planner/grid_map/query")
        result_topic = options.get("goal_query_result_topic", "/vla_planner/grid_map/query_result")
        publisher = self.ros.Publisher(query_topic, PointStamped, queue_size=8, latch=False)

        def send(request_id, xyz):
            if publisher.get_num_connections() < 1:
                raise RuntimeError("Mapper query unavailable")
            message = PointStamped()
            message.header.frame_id = self.frame_id
            # rospy overwrites Header.seq during serialization. The mapper
            # echoes this exact integer nanosecond nonce as a JSON string;
            # it is request identity, not a sensor capture timestamp.
            nanoseconds = int(request_id)
            message.header.stamp = self.ros.Time(nanoseconds//1_000_000_000, nanoseconds%1_000_000_000)
            message.point.x, message.point.y, message.point.z = xyz
            publisher.publish(message)

        self._goal_query = KnownFreeQuery(send, timeout=0.25)
        self.subs.append(self.ros.Subscriber(result_topic, String,
                                            lambda message: self._goal_query.receive(message.data), queue_size=20))

    def _init_camera(self):
        camera_config = self.config.get("ros", {})
        topic = camera_config.get("camera_topic", "/vla/camera/image_raw" if self.mode == "gazebo" else None)
        fps = float(camera_config.get("camera_fps", 3.0))
        timeout = float(camera_config.get("camera_timeout", 2.5))
        if not topic:
            self._camera = CameraBuffer(reason="未配置机载相机话题", fps=fps, timeout=timeout)
            return
        if not isinstance(topic, str) or not topic.startswith("/") or any(c.isspace() for c in topic):
            raise ValueError("ros.camera_topic must be an absolute ROS topic or null")
        try:
            import cv2
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image
        except ImportError:
            self._camera = CameraBuffer(reason="缺少 cv_bridge/OpenCV，相机暂不可用", topic=topic,
                                        fps=fps, timeout=timeout)
            return
        conversion = CvBridge()

        def encode(message):
            bgr = conversion.imgmsg_to_cv2(message, desired_encoding="bgr8")
            height, width = bgr.shape[:2]
            scale = min(1.0, 640.0/width, 360.0/height)
            if scale < 1.0:
                bgr = cv2.resize(bgr, (max(1, round(width*scale)), max(1, round(height*scale))),
                                 interpolation=cv2.INTER_AREA)
            height, width = bgr.shape[:2]
            success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not success:
                raise RuntimeError("JPEG 编码失败")
            return encoded.tobytes(), width, height

        self._camera = CameraBuffer(encode, fps=fps, timeout=timeout, topic=topic)
        self.subs.append(self.ros.Subscriber(topic, Image, self._camera_image, queue_size=1,
                                             buff_size=8*1024*1024, tcp_nodelay=True))

    def _camera_image(self, message):
        # ROS image callback never acquires flight locks and never encodes.
        if message.width <= 0 or message.height <= 0:
            return
        self._camera.push(message, self._receipt_stamp(message), message.header.stamp.to_sec())

    def camera_frame(self):
        """Return JPEG bytes and capture timestamps; never serve stale pixels."""
        return self._camera.frame(include_data=True)

    def _receipt_stamp(self, message):
        stamp = message.header.stamp.to_sec()
        ros_now = self.ros.Time.now().to_sec()
        if not math.isfinite(stamp) or not math.isfinite(ros_now) or stamp <= 0 or stamp > ros_now + 0.25:
            return 0.0
        # Old stamped messages cannot become fresh merely by being replayed.
        return time.time() - max(0.0, ros_now - stamp)

    def _odom(self, message):
        p, v, q = message.pose.pose.position, message.twist.twist.linear, message.pose.pose.orientation
        values = [p.x, p.y, p.z, v.x, v.y, v.z, q.x, q.y, q.z, q.w]
        if not all(math.isfinite(value) for value in values):
            return
        if message.header.frame_id.lstrip("/") != self.frame_id:
            return
        norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if abs(norm - 1.0) > 0.1:
            return
        qx, qy, qz, qw = (value/math.sqrt(norm) for value in (q.x,q.y,q.z,q.w))
        with self.lock:
            self._position = dict(x=p.x, y=p.y, z=p.z)
            self._velocity = dict(x=v.x, y=v.y, z=v.z)
            self._yaw = math.atan2(2*(qw*qz + qx*qy), 1-2*(qy*qy+qz*qz))
            self._quaternion = dict(zip(("x", "y", "z", "w"), (qx,qy,qz,qw)))
            self._odom_child_frame = message.child_frame_id.lstrip("/")
            self._position_stamp = self._receipt_stamp(message)
        visual = self._visual
        if visual is not None:
            reset = visual.add_odom(message.header.stamp.to_sec(), (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                                    message.child_frame_id, self._receipt_stamp(message))
            if reset:
                self._camera.reset()

    def _sim_state(self, message):
        try:
            payload = json.loads(message.data)
            state = payload["flight_mode"]
            if any(type(payload[key]) is not bool for key in ("connected", "armed", "landed")):
                return
        except (ValueError, TypeError, KeyError):
            return
        if state not in ("GROUNDED", "TAKING_OFF", "HOVERING", "FLYING", "LANDING", "FAULT"):
            return
        with self.lock:
            contacts = payload.get('contacts')
            self._sim_contacts = contacts if type(contacts) is int and contacts >= 0 else None
            self._flight_mode = state
            self._state_stamp = self._landed_stamp = time.time()
            self._connected = payload["connected"] and state != "FAULT"
            # FAULT preserves the last observed armed/landed state.
            if state != "FAULT":
                self._landed = payload["landed"]
                self._armed = payload["armed"]
            photo_guard = payload.get("photo_guard")
            if isinstance(photo_guard, dict):
                self._photo_guard_status = dict(photo_guard)
            photo_search = payload.get("photo_search")
            if isinstance(photo_search, dict):
                self._photo_search_status = dict(photo_search)

    def _cloud(self, message):
        if message.header.frame_id.lstrip("/") == self.frame_id and message.width * message.height > 0:
            with self.lock:
                self._cloud_stamp = self._receipt_stamp(message)
        visual = self._visual
        if visual is not None:
            visual.add_cloud(message, self._receipt_stamp(message))

    def _planner_heartbeat(self, _message):
        with self.lock:
            self._planner_stamp = time.time()

    def _trajectory(self, message):
        if message.order != 5:
            return
        try:
            trajectory = PolynomialTrajectory(message.start_time.to_sec(), message.duration,
                                               (message.coef_x, message.coef_y, message.coef_z))
        except (ValueError, TypeError):
            return
        now_ros = self.ros.Time.now().to_sec()
        if (trajectory.start_stamp > now_ros + 1.0
                or trajectory.start_stamp + sum(trajectory.durations) < now_ros):
            return
        with self.lock:
            pending = self._trajectory_pending
            if pending is None or trajectory.start_stamp < pending["ros_stamp"]:
                return
            self._expire_trajectory_request()
            if self._command_error is None:
                # Monitor only the first usable path for this request. Normal
                # endpoint hover or the end of position output is not a fault.
                self._trajectory_pending = None

    def _expire_trajectory_request(self):
        # Caller holds self.lock. This reports request delivery failure without
        # inventing a flight mode or changing the controller's feedback state.
        if (self._trajectory_pending is not None and self._command_error is None
                and time.monotonic() - self._trajectory_pending["started_at"] >= self.trajectory_timeout):
            self._command_error = ("规划器在 %.0f 秒内未返回可用新轨迹；目标可能位于障碍物、不可达区域，或规划失败。请调整目标点后重试。"
                                   % self.trajectory_timeout)
            reason = getattr(self, '_execution_status', {}).get('error')
            if reason:
                self._command_error += ' 实机反馈：'+str(reason)[:180]

    def _real_state(self, message):
        with self.lock:
            if self._flight_mode == 'OFFBOARD' and message.mode != 'OFFBOARD':
                self._control_epoch = getattr(self, '_control_epoch', 0)+1
                self._yaw_epoch = getattr(self, '_yaw_epoch', 0)+1
            owned = (self._mission_active or getattr(self, '_photo_guard', None) is not None or
                     getattr(self, '_photo_search', None) is not None)
            if owned and self._flight_mode == 'OFFBOARD' and message.mode != 'OFFBOARD':
                # Latch the event: a quick mode return between task ticks must
                # not resurrect an old model action or a pending guard setup.
                self._command_error = '实机已退出 OFFBOARD；原任务已取消，请重新确认。'
                self._mission_active = self._gate_active = False
                self._target_yaw = None
                for name in ('_photo_guard', '_photo_search'):
                    owner = getattr(self, name, None)
                    if owner is not None:
                        owner['ready'], owner['failed'] = False, True
            self._connected = bool(message.connected)
            self._armed = bool(message.armed)
            self._flight_mode = message.mode or "UNKNOWN"
            self._state_stamp = self._receipt_stamp(message)

    def _real_rc(self, message):
        allowed = rc_command_allowed(message.channels)
        if allowed is None:
            return
        with self.lock:
            # Observe expiry before accepting a recovered packet: a gap that
            # occurred during a delayed callback must still cancel the old task.
            self._check_rc_feedback()
            accepted = self._rc_feedback.observe(message, time.monotonic(), self.ros.Time.now().to_sec())
            if accepted:
                self._rc_feedback_failed = False
            elif allowed:
                return  # A delayed allow packet cannot refresh/reopen control.
            if not allowed and getattr(self, '_rc_command_allowed', None) is not False:
                self._control_epoch = getattr(self, '_control_epoch', 0)+1
                self._yaw_epoch = getattr(self, '_yaw_epoch', 0)+1
            self._rc_command_allowed = allowed
            owned = (self._mission_active or getattr(self, '_photo_guard', None) is not None or
                     getattr(self, '_photo_search', None) is not None)
            if not allowed and owned:
                self._command_error = '遥控器已退出指令控制；原任务已取消，请重新确认。'
                self._mission_active = self._gate_active = False
                self._target_yaw = None
                for name in ('_photo_guard', '_photo_search'):
                    owner = getattr(self, name, None)
                    if owner is not None:
                        owner['ready'], owner['failed'] = False, True

    def _check_rc_feedback(self):
        # Caller holds the state lock. No service/master calls on this path.
        fresh = self._rc_feedback.fresh(time.monotonic(), self.ros.Time.now().to_sec())
        if not fresh and not self._rc_feedback_failed:
            self._rc_feedback_failed = True
            if self._rc_feedback.received is not None:
                self._control_epoch = getattr(self, '_control_epoch', 0)+1
                self._yaw_epoch = getattr(self, '_yaw_epoch', 0)+1
                owned = (self._mission_active or self._gate_active or
                         self._photo_guard is not None or self._photo_search is not None)
                if owned:
                    self._command_error = '遥控反馈已过期；原任务已取消，请恢复反馈后重新确认。'
                self._mission_active = self._gate_active = False
                self._target_yaw = None
                for name in ('_photo_guard', '_photo_search'):
                    owner = getattr(self, name, None)
                    if owner is not None:
                        owner['ready'], owner['failed'] = False, True
        return fresh

    def _real_extended(self, message):
        with self.lock:
            # MAVROS undefined landed_state=0 is not evidence of airborne/grounded.
            if message.landed_state not in (1, 2, 3, 4):
                self._landed_stamp = 0.0
                return
            self._landed = message.landed_state == 1
            self._landed_stamp = self._receipt_stamp(message)

    def _real_battery(self, message):
        percentage = message.percentage
        voltage = message.voltage
        with self.lock:
            self._battery = {
                "percentage": percentage * 100 if math.isfinite(percentage) and 0 <= percentage <= 1 else None,
                "voltage": voltage if math.isfinite(voltage) and voltage > 0 else None,
            }
            self._battery_stamp = self._receipt_stamp(message)

    def snapshot(self):
        now = time.time()
        with self.lock:
            self._expire_trajectory_request()
            state_stamp = min(self._state_stamp, self._landed_stamp)
            fresh = now - min(self._position_stamp, state_stamp) <= self.timeout
            result = dict(mode=self.mode, connected=self._connected and fresh,
                          armed=self._armed, landed=self._landed,
                          position=dict(self._position), velocity=dict(self._velocity),
                          yaw=self._yaw, position_stamp=self._position_stamp,
                          orientation=dict(self._quaternion), body_frame=self._odom_child_frame,
                          state_stamp=state_stamp, flight_mode=self._flight_mode,
                          control_state="UNKNOWN" if self.mode == "real" else self._flight_mode,
                          battery=None, takeoff_height=self.takeoff_height, frame_id=self.frame_id,
                          planning_ready=now-min(self._cloud_stamp, self._planner_stamp) <= self.timeout,
                          capabilities={"yaw_target": self._photo_execution_available(), "fixed_takeoff_height": True})
            result["command_error"] = self._command_error
            result["planning_request_pending"] = self._trajectory_pending is not None
            result["camera_status"] = self._camera.frame(include_data=False)
            result["visual_grounding"] = self._visual.status() if self._visual is not None else {"configured": False}
            result["visual_grounding"]["goal_query_supported"] = self.mode == "gazebo" or getattr(self, "_real_backend", False)
            if getattr(self, "_real_backend", False):
                result["real_execution"] = dict(getattr(self, "_execution_status", {}))
                # A fresh status may still describe the preceding goal. Bind
                # completion to the request actually sent, not to proximity or
                # a one-time cache reset that a delayed status could undo.
                request = getattr(self, '_navigation_request', None)
                result["navigation_completed"] = bool(
                    self._photo_execution_available() and self._mission_active and request is not None and
                    self._trajectory_pending is None and
                    result['real_execution'].get('goal_request') == request and
                    result['real_execution'].get('navigation_completed') is True)
            result["photo_guard"] = dict(getattr(self, "_photo_guard_status", {"supported": False}))
            result["photo_search"] = dict(getattr(self, "_photo_search_status", {"supported": False}))
            if self.mode == 'gazebo':
                result['contacts'] = getattr(self, '_sim_contacts', None)
            if self._takeoff_target_z is not None:
                result["takeoff_target_z"] = self._takeoff_target_z
            if self._battery and now - self._battery_stamp <= self.timeout:
                result["battery"] = dict(self._battery)
            return result

    def _require_live(self, airborne=False):
        state = self.snapshot()
        if not state["connected"]:
            raise RuntimeError("Fresh odometry and controller state are required")
        if airborne and (not state["armed"] or state["landed"]):
            raise RuntimeError("Controller has not reported an armed airborne vehicle")
        if airborne and self.mode == "real" and state["flight_mode"] != "OFFBOARD":
            raise RuntimeError("Real flight requires reported OFFBOARD mode; use existing RC/PX4CTRL preflight procedure")
        return state

    def _ensure_subscriber(self, publisher, label):
        if publisher.get_num_connections() < 1:
            raise RuntimeError("No ROS subscriber for " + label)

    def _control_changed(self, epoch):
        # Call under self.lock, including the final publish: callbacks must not
        # cancel an admission between its last check and the outgoing command.
        fresh = self._check_rc_feedback()
        return (not fresh or epoch != getattr(self, '_control_epoch', 0) or
                getattr(self, '_rc_command_allowed', None) is False)

    def capture_control_token(self):
        """Capture before an outer worker queue/command lock can delay action."""
        with self.lock:
            self._check_rc_feedback()
            return getattr(self, '_control_epoch', 0)

    def _set_gate(self, enabled):
        if self.mode != "real":
            return
        with self.lock:
            if enabled and (not self._check_rc_feedback() or
                            getattr(self, '_rc_command_allowed', None) is False):
                raise RuntimeError('遥控器反馈尚未就绪或未允许指令控制。')
            control_epoch = getattr(self, '_control_epoch', 0)
        self.ros.wait_for_service("/diff_vla/command_gate/set_enabled", timeout=1.0)
        if not hasattr(self, '_gate_call'):
            self._gate_call = BoundedCall('real command gate')
        response = self._gate_call(self.gate, enabled, timeout=1.75)
        if not response.success:
            raise RuntimeError("Command gate refused request: " + response.message)
        with self.lock:
            if enabled and self._control_changed(control_epoch):
                raise RuntimeError('授权等待期间发生遥控接管或反馈过期，旧指令已取消，请重新确认。')
            self._gate_active = enabled

    def _reset_command_tracking(self):
        with self.lock:
            self._yaw_epoch = getattr(self, '_yaw_epoch', 0)+1
            self._target_yaw = None
            self._gate_active = False
            self._mission_active = False
            self._trajectory_pending = None
            self._navigation_request = None
            self._command_error = None

    def _cancel_unpublished_motion(self, control_epoch, guard):
        """Revoke an unsent action without clearing its existing photo guard.

        The caller still owns command_lock, but must not hold the state lock
        while waiting for the gate service. A failed close is an unknown outcome,
        never the caller's recoverable pre-dispatch rejection.
        """
        with self.lock:
            self._reset_command_tracking()
            self._photo_navigation_goal = None
        try:
            self._set_gate(False)
        except Exception as error:
            with self.lock:
                self._command_error = '未发送动作的执行授权未确认撤销，请停止任务。'
            raise RuntimeError('未确认撤销本次执行授权，不能按未发送动作重试。') from error
        with self.lock:
            if self._control_changed(control_epoch):
                raise RuntimeError('撤销未发送动作期间发生遥控接管，旧动作已取消。')
            if guard is not None and (self._photo_guard is not guard or not guard['ready'] or guard['failed']):
                raise RuntimeError('撤销未发送动作期间人物保护已失效。')

    def _stop_commands(self):
        self._reset_command_tracking()
        with self.lock:
            guard = getattr(self, "_photo_guard", None)
            if guard is not None:
                guard["ready"] = False
            search = getattr(self, "_photo_search", None)
            if search is not None:
                search["ready"] = False
                self._search_bookkeeping()
                self._photo_search_cancelled.add(search["id"])
        if self.mode == "real":
            self._set_gate(False)
        else:
            self._ensure_subscriber(self.hold_pub, "/vla/hold")
            self.hold_pub.publish(self.msg["empty"]())

    def _wait_control_timeout(self):
        if self.mode == "real":
            timeout = float(self.ros.get_param("/px4ctrl/msg_timeout/cmd", 0.5))
            if not 0.1 <= timeout <= 3.0:
                raise RuntimeError("Unsupported PX4CTRL command timeout")
            # Wall time avoids blocking indefinitely if the ROS clock stops.
            time.sleep(timeout + 0.15)

    def takeoff(self, height):
        with self.command_lock:
            state = self._require_live()
            if state["armed"] or not state["landed"]:
                raise RuntimeError("Takeoff requires reported GROUNDED/disarmed state")
            parameter = ("/vla_flight_controller/takeoff_height" if self.mode == "gazebo"
                         else "/px4ctrl/auto_takeoff_land/takeoff_height")
            if not self.ros.has_param(parameter):
                raise RuntimeError("Controller takeoff height is unknown: " + parameter)
            configured = float(self.ros.get_param(parameter))
            if not math.isfinite(height) or abs(height - configured) > 0.05:
                raise ValueError("This controller has fixed takeoff height %.2f m; restart with matching configuration" % configured)
            if self.mode == "gazebo":
                # The controller clears its trajectory gate atomically when it
                # accepts takeoff. A separate /vla/hold can arrive AFTER the
                # takeoff topic and incorrectly cancel ascent; never send it
                # while the reported controller is GROUNDED/disarmed.
                with self.lock:
                    self._target_yaw = None
                    self._gate_active = False
                    self._mission_active = False
                    self._trajectory_pending = None
                    self._command_error = None
            else:
                self._stop_commands()
            self._wait_control_timeout()
            self._require_live()
            self._ensure_subscriber(self.takeoff_pub, "/px4ctrl/takeoff_land")
            with self.lock:
                self._takeoff_target_z = (self._position["z"] + configured
                                          if self.mode == "real" else configured)
            message = self.msg["takeoff_land"]()
            message.takeoff_land_cmd = 1
            self.takeoff_pub.publish(message)

    def land(self, control_token=None, before_dispatch=None):
        control_epoch = self.capture_control_token() if control_token is None else control_token
        if before_dispatch is not None and not callable(before_dispatch):
            raise TypeError('before_dispatch must be callable')
        with self.command_lock:
            state = self._require_live()
            if state["landed"] and not state["armed"]:
                self._stop_commands()
                return
            with self.lock:
                if self._control_changed(control_epoch):
                    raise RuntimeError('降落等待期间发生遥控接管，旧指令已取消，请重新确认。')
            self._require_live(airborne=True)
            self._stop_commands()
            self._wait_control_timeout()
            self._require_live(airborne=True)
            self._ensure_subscriber(self.takeoff_pub, "/px4ctrl/takeoff_land")
            message = self.msg["takeoff_land"]()
            message.takeoff_land_cmd = 2
            with self.lock:
                if self._control_changed(control_epoch):
                    raise RuntimeError('降落等待期间发生遥控接管，旧指令已取消，请重新确认。')
                if before_dispatch is not None:
                    # This is the send-commit boundary. The callback must not
                    # acquire a runner/state lock or call back into the adapter.
                    # It may atomically reject cancellation and mark dispatch.
                    before_dispatch()
                self.takeoff_pub.publish(message)

    def hold(self):
        # Closing the gate remains permitted even with stale flight feedback.
        with self.command_lock:
            self._stop_commands()

    def goto(self, x, y, z, yaw, control_token=None, before_dispatch=None):
        if before_dispatch is not None and not callable(before_dispatch):
            raise TypeError('before_dispatch must be callable')
        if not all(math.isfinite(value) for value in (x, y, z)) or (yaw is not None and not math.isfinite(yaw)):
            raise ValueError("Goal values must be finite")
        if self.mode == "real" and yaw is not None and not self._photo_execution_available():
            raise RuntimeError("Explicit real-flight yaw requires the fresh diff_vla_real executor")
        if (self.mode == 'real' and getattr(self, '_real_backend', False) and
                getattr(self, '_execution_status', {}).get('goal_request_correlation') is not True):
            raise RuntimeError('实机执行反馈版本缺少本次目标关联；请更新并重启本版执行服务。')
        control_epoch = self.capture_control_token() if control_token is None else control_token
        with self.command_lock:
            with self.lock:
                if self._control_changed(control_epoch):
                    raise RuntimeError('发送目标期间发生遥控接管，旧目标已取消，请重新确认。')
                self._navigation_request = None
            state = self._require_live(airborne=True)
            with self.lock:
                guard = getattr(self, "_photo_guard", None)
                if guard is not None and (not guard["ready"] or guard["failed"]):
                    raise RuntimeError("拍照人物保护尚未就绪或已失效，禁止发送新目标")
                if getattr(self, "_photo_search", None) is not None:
                    raise RuntimeError("原地搜索尚未结束，禁止同时发送平移目标")
            if not state["planning_ready"]:
                raise RuntimeError("Fresh registered point cloud and planner heartbeat are required")
            if self.mode == "gazebo" and state["flight_mode"] not in ("HOVERING", "FLYING"):
                raise RuntimeError("Wait for takeoff/landing to finish before sending a goal")
            self._ensure_subscriber(self.goal_pub, "/goal")
            self._ensure_subscriber(self.yaw_pub, "/planning/yaw")
            if self.mode == 'real':
                # Retire the previous heading before waiting for this new
                # command's gate; a delayed tick cannot keep publishing it.
                with self.lock:
                    self._yaw_epoch = getattr(self, '_yaw_epoch', 0)+1
                    self._target_yaw = None
            self._set_gate(False)
            self._set_gate(True)
            orientation_yaw = state["yaw"] if yaw is None else yaw
            goal = self.msg["goal"]()
            goal.header.stamp = self.ros.Time.now()
            goal.header.frame_id = self.frame_id
            goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = x, y, z
            goal.pose.orientation.z = math.sin(orientation_yaw/2)
            goal.pose.orientation.w = math.cos(orientation_yaw/2)
            callback_rejected = publish_attempted = False
            try:
                with self.lock:
                    if (self._control_changed(control_epoch) or
                            (guard is not None and (self._photo_guard is not guard or
                             not guard['ready'] or guard['failed']))):
                        self._mission_active = self._gate_active = False
                        self._target_yaw = self._trajectory_pending = None
                        self.hold_pub.publish(self.msg['empty']())
                        raise RuntimeError('发送目标期间发生遥控接管，旧目标已取消，请重新确认。')
                    if before_dispatch is not None:
                        current = self.snapshot()
                        try:
                            # Same state-lock boundary as the final goal send.
                            # The callback may only inspect this detached state;
                            # it must not acquire runner.lock or perform I/O.
                            before_dispatch(current)
                        except Exception:
                            callback_rejected = True
                            self._reset_command_tracking()
                            raise
                    self._yaw_epoch = getattr(self, '_yaw_epoch', 0)+1
                    self._target_yaw = None if yaw is None else math.atan2(math.sin(yaw), math.cos(yaw))
                    self._mission_active = True
                    # Real motion retains the fixed endpoint heading. Keep the
                    # existing simulator bearing policy after admission too.
                    self._photo_navigation_goal = (dict(x=x,y=y,z=z,yaw=yaw) if self.mode == 'gazebo' and
                        guard is not None and self.config.get('photo', {}).get('navigation_scope')
                        in ('geofence', 'planner') else None)
                    self._trajectory_pending = {"ros_stamp": goal.header.stamp.to_sec(),
                                                "started_at": time.monotonic()}
                    self._command_error = None
                    publish_attempted = True
                    self.goal_pub.publish(goal)
                    if self.mode == 'real':
                        # rospy owns Header.seq; read the serialized identity
                        # under the same lock as the send and state callbacks.
                        self._navigation_request = dict(seq=getattr(goal.header, 'seq', 0),
                                                        stamp=goal.header.stamp.to_sec())
            except Exception as error:
                if callback_rejected and not publish_attempted:
                    self._cancel_unpublished_motion(control_epoch, guard)
                elif before_dispatch is not None and publish_attempted:
                    raise RuntimeError('目标已进入发送流程，不能按未发送动作重试：'+str(error)) from error
                raise

    def _photo_guard_snapshot(self):
        with self.lock:
            guard = getattr(self, "_photo_guard", None)
            status = dict(getattr(self, "_photo_guard_status", {}))
            ready = bool(guard is not None and guard["ready"] and not guard["failed"])
            return guard, status, ready

    def _tick(self, _event):
        with self.lock:
            if not self._check_rc_feedback():
                return
            search = getattr(self, "_photo_search", None)
            search_status = dict(getattr(self, "_photo_search_status", {}))
            search_ready = bool(search is not None and search["ready"] and not search["failed"])
        if search_ready:
            with self.lock:
                if self._photo_search is search and search["ready"] and not search["failed"]:
                    if search_status.get("id") != search["id"] or search_status.get("error"):
                        search["ready"], search["failed"] = False, True
                        self._command_error = "原地搜索失效："+str(search_status.get("error") or "owner_mismatch")
                        try:
                            self._publish_photo_search(search, "stop")
                        except Exception:
                            pass
                    elif search_status.get("active") is not True:
                        search["ready"] = False  # Sweep completed; controller already holds.
                    elif time.monotonic()-search["last_sent"] >= .5:
                        try:
                            self._publish_photo_search(search, "refresh")
                        except Exception as exc:
                            search["ready"], search["failed"] = False, True
                            self._command_error = "原地搜索续期失败："+str(exc)[:100]
        photo_guard, status, guard_ready = self._photo_guard_snapshot()
        if guard_ready:
            if (status.get("id") != photo_guard["id"] or status.get("active") is not True
                    or status.get("error")):
                with self.lock:
                    photo_guard["failed"] = True
                    self._command_error = "人物轨迹保护失效："+str(status.get("error") or "missing_active_guard")
                    self._mission_active = False
                    self._target_yaw = None
                self.hold_pub.publish(self.msg["empty"]())
            elif time.monotonic()-photo_guard["last_sent"] >= .5:
                try:
                    with self.lock:
                        # clear() marks ready false under this same lock, so
                        # an in-flight tick cannot refresh a cleared owner.
                        if self._photo_guard is photo_guard and photo_guard["ready"] and not photo_guard["failed"]:
                            self._publish_photo_guard(photo_guard, "refresh")
                except Exception as exc:
                    with self.lock:
                        photo_guard["failed"] = True
                        self._command_error = "人物轨迹保护续期失败："+str(exc)[:100]
                    self.hold_pub.publish(self.msg["empty"]())
        state = self.snapshot()
        if search_ready and (not state["connected"] or not state["planning_ready"]):
            with self.lock:
                if self._photo_search is search and search["ready"]:
                    search["ready"], search["failed"] = False, True
                    self._command_error = "原地搜索停止：位姿或雷达规划反馈过期"
                    try:
                        self._publish_photo_search(search, "stop")
                    except Exception:
                        pass
        with self.lock:
            yaw, active, gate_active = self._target_yaw, self._mission_active, self._gate_active
            yaw_epoch = getattr(self, '_yaw_epoch', 0)
            navigation_goal = getattr(self, '_photo_navigation_goal', None)
        if self.mode == 'gazebo' and guard_ready and active and navigation_goal is not None and yaw is not None:
            center = photo_guard['zones'][0]['center']
            final_bearing = math.atan2(center[1]-navigation_goal['y'], center[0]-navigation_goal['x'])
            bearing = math.atan2(center[1]-state['position']['y'], center[0]-state['position']['x'])
            yaw = math.atan2(math.sin(bearing+navigation_goal['yaw']-final_bearing),
                             math.cos(bearing+navigation_goal['yaw']-final_bearing))
        if not state["connected"] or (active and not state["planning_ready"]):
            # The real gate expires independently if this node or ROS callbacks die.
            with self.lock:
                self._target_yaw = None
                self._gate_active = False
                self._mission_active = False
            if active and self.mode == "gazebo":
                self.hold_pub.publish(self.msg["empty"]())
            return
        if gate_active:
            self.lease_pub.publish(self.msg["empty"]())
        if yaw is not None and state["armed"] and not state["landed"]:
            message = self.msg["yaw"]()
            if self.mode == 'real':
                # Use the same state lock as goal installation/cancellation.
                # A new ROS stamp must never make an older tick's yaw current.
                with self.lock:
                    if (yaw_epoch != getattr(self, '_yaw_epoch', 0) or
                            self._target_yaw is None or not self._mission_active):
                        return
                    message.header.stamp = self.ros.Time.now()
                    message.yaw = yaw
                    self.yaw_pub.publish(message)
            else:
                message.header.stamp = self.ros.Time.now()
                message.yaw = yaw
                self.yaw_pub.publish(message)

    def close(self):
        try:
            self.hold()
        finally:
            self.timer.shutdown()
            self._camera.close()
            if self._visual is not None:
                self._visual.close()
