"""Freeze explicitly modelled RGB + registered measured LiDAR without simulator truth.

No Gazebo model-state, scene file, depth camera or landmark interface exists in
this module. Raw cloud decoding happens in its own bounded queue worker. Image
projection runs only when a visual request asks for an observation.
"""
from collections import deque
import copy
import hashlib
import math
import secrets
import threading
import time

import numpy as np

from .visual_geometry import resolve_bbox


class PhotoObservationPoseChanged(RuntimeError):
    """This frozen RGB predates vehicle motion; no person geometry was accepted."""
    code = "observation_pose_changed"

    def __init__(self, position_delta_m, orientation_delta_rad):
        super().__init__("观察后机体移动超过 0.15 m 或旋转超过 5°，请重新观察")
        self.evidence = dict(position_delta_m=float(position_delta_m),
                             orientation_delta_rad=float(orientation_delta_rad),
                             geometry_accepted=False)


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 0.05:
        raise ValueError("姿态四元数无效")
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def interpolate_pose(samples, stamp, max_gap=0.15):
    samples = sorted(samples, key=lambda item: item["stamp"])
    before = [item for item in samples if item["stamp"] <= stamp]
    after = [item for item in samples if item["stamp"] >= stamp]
    if not before or not after:
        raise RuntimeError("RGB 时刻缺少前后里程计，无法同步视觉定位")
    left, right = before[-1], after[0]
    gap = right["stamp"]-left["stamp"]
    if gap > max_gap:
        raise RuntimeError("RGB 前后里程计间隔过大，拒绝插值")
    alpha = 0.0 if gap <= 1e-9 else (stamp-left["stamp"])/gap
    p = (1-alpha)*np.asarray(left["position"]) + alpha*np.asarray(right["position"])
    qa, qb = np.asarray(left["quaternion"], dtype=float), np.asarray(right["quaternion"], dtype=float)
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        q = qa + alpha*(qb-qa)
    else:
        angle = math.acos(max(-1., min(1., dot)))
        q = (math.sin((1-alpha)*angle)*qa + math.sin(alpha*angle)*qb)/math.sin(angle)
    q /= np.linalg.norm(q)
    rotation = quaternion_matrix(q)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, p
    return dict(position=p, quaternion=q, transform=transform,
                yaw=math.atan2(rotation[1, 0], rotation[0, 0]))


class VisualGrounding:
    def __init__(self, camera_provider, state_provider, transform_provider, cloud_decoder,
                 world_frame="world", body_frame="base_link", optical_frame="camera_optical",
                 configured=True, disabled_reason=None, sensor_skew=0.15,
                 observation_lifetime=120.0, cloud_topic="/cloud_registered", camera_info_topic=None,
                 ros_clock=None, demo_target_options=None, allow_pinhole_approximation=False):
        if not 0.02 <= sensor_skew <= 0.2:
            raise ValueError("ros.visual_sensor_skew must be between 0.02 and 0.2 seconds")
        self.camera_provider, self.state_provider = camera_provider, state_provider
        self.transform_provider, self.cloud_decoder = transform_provider, cloud_decoder
        def frame_name(value, name, required=True):
            if not required and not value:
                return ""
            if not isinstance(value, str) or not value.lstrip("/") or any(c.isspace() for c in value):
                raise ValueError(name + " must name an explicit ROS frame")
            return value.lstrip("/")
        self.world_frame = frame_name(world_frame, "world_frame")
        self.body_frame = frame_name(body_frame, "body_frame", configured)
        self.optical_frame = frame_name(optical_frame, "optical_frame", configured)
        self.configured, self.disabled_reason = configured, disabled_reason
        self.sensor_skew, self.observation_lifetime = sensor_skew, observation_lifetime
        self.cloud_topic, self.camera_info_topic = cloud_topic, camera_info_topic
        self.ros_clock = ros_clock
        self.demo_target_options = copy.deepcopy(demo_target_options or {})
        self.allow_pinhole_approximation = allow_pinhole_approximation is True
        self.lock = threading.RLock()
        self.odom, self.infos, self.clouds = deque(maxlen=500), deque(maxlen=80), deque(maxlen=48)
        self.pending_cloud = None
        self.registry = {}
        self.rectification_maps = {}
        self.rectification_lock = threading.Lock()
        self.sensor_epoch = 0
        self.last_error = disabled_reason or "等待同步 RGB、相机标定和真实雷达回波"
        self.stop, self.event = threading.Event(), threading.Event()
        self.worker = None
        from .photo_diagnostics import PhotoDiagnostics
        self._photo_diagnostics = PhotoDiagnostics.from_environment()
        if configured:
            self.worker = threading.Thread(target=self._decode_clouds, name="diff-vla-visual-cloud", daemon=True)
            self.worker.start()

    def add_odom(self, stamp, position, quaternion, child_frame, received):
        if not self.configured or child_frame.lstrip("/") != self.body_frame:
            return
        try:
            quaternion_matrix(quaternion)
            if (not np.isfinite(position).all() or not math.isfinite(stamp) or stamp <= 0
                    or not math.isfinite(received) or received <= 0):
                return
        except (ValueError, TypeError):
            return
        q = np.asarray(quaternion, dtype=float)
        q /= np.linalg.norm(q)
        entry = dict(stamp=stamp, position=tuple(position), quaternion=tuple(q), received=received)
        reset = False
        with self.lock:
            if self.odom and stamp < self.odom[-1]["stamp"]:
                reset = True
                self.sensor_epoch += 1
                self.odom.clear()
                self.clouds.clear()
                self.infos.clear()
                self.pending_cloud = None
                self.registry.clear()
            self.odom.append(entry)
        return reset

    def add_info(self, message, received):
        if not self.configured or message.header.frame_id.lstrip("/") != self.optical_frame:
            return
        stamp = message.header.stamp.to_sec()
        if not math.isfinite(stamp) or stamp <= 0 or not math.isfinite(received) or received <= 0:
            return
        try:
            K = np.asarray(message.K, dtype=float).reshape(3, 3)
        except (ValueError, TypeError):
            return
        if (not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0., 0., 1.], atol=1e-9, rtol=0)
                or abs(K[1, 0]) > 1e-9 or message.width <= 0 or message.height <= 0):
            return
        try:
            distortion = np.asarray(message.D, dtype=float)
            source_model = message.distortion_model
            model = source_model
            lengths = {'plumb_bob': (4, 5), 'rational_polynomial': (8,)}
            valid_distortion = (distortion.ndim == 1 and np.isfinite(distortion).all()
                                and len(distortion) in lengths.get(model, ()))
            if (self.allow_pinhole_approximation and distortion.ndim == 1
                    and len(distortion) == 0 and model in ('', 'plumb_bob', 'elastic_pinhole')):
                # Existing cameras may advertise the ROS default model but
                # provide no D. Preserve their valid K, record the assumption,
                # and never replace a known non-empty calibration.
                model = 'elastic_pinhole'
                valid_distortion = True
        except (ValueError, TypeError):
            valid_distortion = False
        if not valid_distortion:
            with self.lock:
                self.last_error = 'CameraInfo 缺少已确认的畸变模型/系数，不能把未知畸变当作零畸变'
            return
        # Binning/ROI need an explicitly adjusted intrinsic matrix. Fail closed
        # instead of guessing how a driver's raw CameraInfo should be interpreted.
        roi = message.roi
        if (message.binning_x not in (0, 1) or message.binning_y not in (0, 1) or roi.x_offset or roi.y_offset
                or getattr(roi, "width", 0) not in (0, message.width)
                or getattr(roi, "height", 0) not in (0, message.height)):
            with self.lock:
                self.last_error = "相机存在未校正的 ROI/binning，不能用于 RGB–LiDAR 定位"
            return
        entry = dict(stamp=stamp, received=received, K=K.copy(),
                     D=tuple(distortion), model=model, source_model=source_model, frame=self.optical_frame,
                     width=int(message.width), height=int(message.height))
        with self.lock:
            self.infos.append(entry)

    def add_cloud(self, message, received):
        if (not self.configured or message.header.frame_id.lstrip("/") != self.world_frame
                or message.width*message.height <= 0 or not math.isfinite(received) or received <= 0):
            return
        with self.lock:
            self.pending_cloud = (message, received, self.sensor_epoch)
        self.event.set()

    def _decode_clouds(self):
        while not self.stop.is_set():
            self.event.wait(0.2)
            with self.lock:
                pending, self.pending_cloud = self.pending_cloud, None
                self.event.clear()
            if pending is None:
                continue
            message, received, epoch = pending
            try:
                points = np.asarray(self.cloud_decoder(message), dtype=float).reshape((-1, 3))
                points = points[np.isfinite(points).all(axis=1)]
                points.setflags(write=False)
                stamp = float(message.header.stamp.to_sec())
                if len(points) == 0 or not math.isfinite(stamp) or stamp <= 0:
                    continue
                with self.lock:
                    if epoch == self.sensor_epoch:
                        self.clouds.append(dict(stamp=stamp, received=received, points=points))
            except Exception as exc:
                with self.lock:
                    self.last_error = "实际雷达点云解码失败：" + str(exc)[:120]

    def status(self):
        with self.lock:
            latest = self.infos[-1] if self.infos else None
            fresh = bool(self.infos and self.clouds and self.odom
                         and time.time()-min(self.infos[-1]["received"], self.clouds[-1]["received"], self.odom[-1]["received"]) <= 2.5)
            return dict(configured=self.configured, source="rgb_lidar", reason=self.last_error,
                        sensor_streams_fresh=fresh,
                        camera_info_available=bool(self.infos), cloud_available=bool(self.clouds),
                        camera_info_topic=self.camera_info_topic, cloud_topic=self.cloud_topic,
                        projection_model=latest['model'] if latest else None,
                        distortion_calibrated=(latest['model'] != 'elastic_pinhole') if latest else None,
                        image_rectified=False,
                        pinhole_approximation_allowed=self.allow_pinhole_approximation,
                        optical_frame=self.optical_frame if self.configured else None)

    def _select(self, image_stamp):
        with self.lock:
            infos, clouds, odom, epoch = list(self.infos), list(self.clouds), list(self.odom), self.sensor_epoch
        if not infos:
            raise RuntimeError("缺少显式 CameraInfo 相机标定，无法视觉定位")
        if not clouds:
            raise RuntimeError("缺少实际 LiDAR 命中回波，无法给目标估计距离")
        info = min(infos, key=lambda item: abs(item["stamp"]-image_stamp))
        cloud = min(clouds, key=lambda item: abs(item["stamp"]-image_stamp))
        if abs(info["stamp"]-image_stamp) > self.sensor_skew:
            raise RuntimeError("CameraInfo 与 RGB 时间不同步")
        if abs(cloud["stamp"]-image_stamp) > self.sensor_skew:
            raise RuntimeError("RGB 与实际 LiDAR 时间差过大，拒绝使用旧点云")
        if time.time()-cloud["received"] > 2.5 or time.time()-info["received"] > 2.5:
            raise RuntimeError("雷达或相机标定流已过期")
        pose = interpolate_pose(odom, image_stamp)
        cloud_pose = interpolate_pose(odom, cloud["stamp"])
        if np.linalg.norm(pose["position"]-cloud_pose["position"]) > 0.08:
            raise RuntimeError("RGB 与 LiDAR 采样期间机体位移过大，先悬停再观察")
        angular_delta = 2*math.acos(min(1.0, abs(float(np.dot(pose["quaternion"], cloud_pose["quaternion"])))))
        if angular_delta > math.radians(3):
            raise RuntimeError("RGB 与 LiDAR 采样期间转动过大，先悬停再观察")
        return info, cloud, pose, epoch

    def _check_clock(self, image_stamp):
        if self.ros_clock is not None:
            now = self.ros_clock()
            if not math.isfinite(now) or image_stamp > now+0.1:
                raise RuntimeError("ROS 时钟已回退，旧观测不可执行，请重新观察")

    def freeze(self):
        if not self.configured:
            raise RuntimeError(self.disabled_reason or "视觉空间定位尚未标定")
        frame = self.camera_provider()
        if not frame.get("available") or not isinstance(frame.get("data"), bytes):
            raise RuntimeError(frame.get("reason") or "没有新鲜机载 RGB 图像")
        if time.time()-frame["stamp"] > 2.5:
            raise RuntimeError("RGB 图像已过期")
        if frame.get("frame_id", "").lstrip("/") != self.optical_frame:
            raise RuntimeError("RGB 图像光学坐标与显式相机标定不一致")
        self._check_clock(frame["ros_stamp"])
        selected = None
        # A camera timestamp may precede the next 50 Hz odometry callback by a
        # few milliseconds. Wait here in the request thread, never in control.
        for attempt in range(5):
            try:
                selected = self._select(frame["ros_stamp"])
                break
            except RuntimeError:
                if attempt == 4:
                    raise
                time.sleep(0.04)
        info, cloud, pose, epoch = selected
        if frame.get("raw_width") != info["width"] or frame.get("raw_height") != info["height"]:
            raise RuntimeError("RGB 原始尺寸与 CameraInfo 不一致，不能猜测裁剪或缩放标定")
        if frame["width"] > info["width"] or frame["height"] > info["height"]:
            raise RuntimeError("RGB 编码尺寸不符合已知的等比例降采样")
        body_T_optical = np.asarray(self.transform_provider(frame["ros_stamp"], self.body_frame, self.optical_frame), dtype=float)
        if body_T_optical.shape != (4, 4) or not np.isfinite(body_T_optical).all():
            raise RuntimeError("缺少有效的机体到相机光学坐标外参")
        if (not np.allclose(body_T_optical[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(body_T_optical[:3, :3].T @ body_T_optical[:3, :3], np.eye(3), atol=1e-5)
                or abs(np.linalg.det(body_T_optical[:3, :3])-1) > 1e-5):
            raise RuntimeError("相机外参不是有效刚体变换")
        camera_T_world = np.linalg.inv(pose["transform"] @ body_T_optical)
        K = info["K"].copy()
        scale_x, scale_y = frame["width"]/info["width"], frame["height"]/info["height"]
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        # OpenCV resize maps pixel centers, not pixel corners.
        K[0, 2] += (scale_x-1)*0.5
        K[1, 2] += (scale_y-1)*0.5
        from .rectified_observation import rectified_observation
        with self.rectification_lock:
            image_data, projected, effective_distortion, rectified = rectified_observation(
                frame, cloud['points'], camera_T_world, K, info['D'], info['model'],
                self.rectification_maps)
        token = secrets.token_urlsafe(24)
        public = dict(data=image_data, stamp=frame["stamp"], ros_stamp=frame["ros_stamp"],
                      width=frame["width"], height=frame["height"],
                      position=dict(zip(("x", "y", "z"), map(float, pose["position"]))),
                      yaw=float(pose["yaw"]), observation_id=token,
                      orientation=dict(zip(("x","y","z","w"), map(float, pose["quaternion"]))), body_frame=self.body_frame)
        calibration_hash = hashlib.sha256(K.tobytes()+body_T_optical.tobytes()
                                         +repr((info['model'], info["D"])).encode()).hexdigest()[:16]
        internal = dict(public=copy.deepcopy(public), created=time.monotonic(), projection=projected,
                        epoch=epoch,
                        K=K.copy(), body_T_optical=body_T_optical.copy(),
                        camera_T_world=camera_T_world.copy(), distortion=effective_distortion,
                        rectified=rectified, source_distortion=tuple(info['D']),
                        projection_model=info['model'], distortion_calibrated=info['model'] != 'elastic_pinhole',
                        source_camera_model=info.get('source_model', info['model']),
                        quaternion=pose["quaternion"], calibration_id=calibration_hash,
                        cloud_stamp=cloud["stamp"], camera_info_stamp=info["stamp"],
                        sync_delta_s=abs(cloud["stamp"]-frame["ros_stamp"]))
        with self.lock:
            self._check_clock(frame["ros_stamp"])
            if epoch != self.sensor_epoch:
                raise RuntimeError("传感时钟在冻结期间重置，请重新观察")
            cutoff = time.monotonic()-self.observation_lifetime
            self.registry = {key: value for key, value in self.registry.items() if value["created"] >= cutoff}
            if len(self.registry) >= 8:
                oldest = min(self.registry, key=lambda key: self.registry[key]["created"])
                del self.registry[oldest]
            self.registry[token] = internal
            self.last_error = None
        return public

    def photo_camera_context(self, observation, include_demo_targets=True):
        """Frozen calibration, optionally with same-frame demo candidates.

        Static safety monitoring needs authentic image/pose calibration only;
        its independent worker must not wait for optional target clustering.
        """
        with self.lock:
            frozen=self.registry.get(observation.get('observation_id'))
        if (frozen is None or time.monotonic()-frozen['created']>self.observation_lifetime or
                any(observation.get(k)!=v for k,v in frozen['public'].items())):
            raise RuntimeError('相机观测已失效或被修改')
        self._check_clock(observation['ros_stamp'])
        if frozen['epoch']!=self.sensor_epoch:
            raise RuntimeError('相机观测时钟发生变化')
        intrinsic=frozen['K']
        context = dict(width=observation['width'],height=observation['height'],ros_stamp=observation['ros_stamp'],
                    world_frame=self.world_frame,optical_frame=self.optical_frame,
                    optical_axes='x_right_y_down_z_forward',
                    image_rectified=bool(frozen.get('rectified', False)),
                    projection_model=frozen['projection_model'],
                    distortion_calibrated=frozen['distortion_calibrated'],
                    source_camera_model=frozen['source_camera_model'],
                    intrinsics=dict(fx=float(intrinsic[0,0]),fy=float(intrinsic[1,1]),
                                    cx=float(intrinsic[0,2]),cy=float(intrinsic[1,2])),
                    world_T_camera=np.linalg.inv(frozen['camera_T_world']).tolist())
        from uav_bridge.demo_targets import target_filter_config
        if include_demo_targets and target_filter_config(self.demo_target_options) is not None:
            from .demo_target_geometry import target_context
            if 'demo_targets' not in frozen:
                frozen['demo_targets'] = target_context(frozen['projection'], self.demo_target_options,
                                                       frozen['cloud_stamp'])
            context['demo_targets'] = copy.deepcopy(frozen['demo_targets'])
        with self.lock:
            self._check_clock(observation['ros_stamp'])
            if frozen['epoch'] != self.sensor_epoch:
                raise RuntimeError('传感时钟在目标筛选期间重置，请重新观察')
        return context

    def resolve(self, observation, bbox, standoff_m=1.2):
        if not isinstance(observation, dict):
            raise ValueError("必须使用本服务冻结的视觉观测")
        with self.lock:
            frozen = self.registry.get(observation.get("observation_id"))
        if frozen is None or time.monotonic()-frozen["created"] > self.observation_lifetime:
            raise RuntimeError("视觉观测不存在或模型等待过久，请重新观察")
        if any(observation.get(key) != value for key, value in frozen["public"].items()):
            raise ValueError("冻结观测被修改，拒绝使用")
        self._check_clock(observation["ros_stamp"])
        current = self.state_provider()
        if not current.get("connected") or time.time()-current.get("position_stamp", 0) > 2.0:
            raise RuntimeError("当前位姿不新鲜，不能复核视觉目标")
        xyz = np.array([current["position"][axis] for axis in ("x", "y", "z")])
        old = np.array([observation["position"][axis] for axis in ("x", "y", "z")])
        angle = abs(math.atan2(math.sin(current["yaw"]-observation["yaw"]), math.cos(current["yaw"]-observation["yaw"])))
        if current.get("body_frame") != self.body_frame or not isinstance(current.get("orientation"), dict):
            raise RuntimeError("当前位姿缺少同一机体坐标的完整四元数，无法复核视觉目标")
        current_q = np.asarray([current["orientation"].get(axis, float("nan")) for axis in ("x","y","z","w")], dtype=float)
        quaternion_matrix(current_q)
        current_q /= np.linalg.norm(current_q)
        angle = max(angle, 2*math.acos(min(1.0, abs(float(np.dot(current_q, frozen["quaternion"]))))))
        if np.linalg.norm(xyz-old) > 0.15 or angle > math.radians(5):
            raise RuntimeError("观察后机体移动超过 0.15 m 或旋转超过 5°，请重新观察")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox)
                or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)):
            raise ValueError("视觉框必须为归一化 [xmin,ymin,xmax,ymax]，范围 0 到 1")
        if not isinstance(standoff_m, (int, float)) or isinstance(standoff_m, bool) or not math.isfinite(standoff_m) or not 1.2 <= standoff_m <= 3.0:
            raise ValueError("视觉接近距离必须在 1.2 到 3.0 m 之间")
        pixels = [bbox[0]*observation["width"], bbox[1]*observation["height"],
                  bbox[2]*observation["width"], bbox[3]*observation["height"]]
        result = resolve_bbox(frozen["projection"], pixels, xyz, current["yaw"], standoff_m, 3.0)
        result["evidence"].update(dict(source="rgb_lidar", observation_id=observation["observation_id"],
                                       rgb_stamp=observation["ros_stamp"], cloud_stamp=frozen["cloud_stamp"],
                                       camera_info_stamp=frozen["camera_info_stamp"],
                                       calibration_id=frozen["calibration_id"], sync_delta_s=frozen["sync_delta_s"],
                                       world_frame=self.world_frame, optical_frame=self.optical_frame,
                                       cloud_topic=self.cloud_topic, bbox_normalized=list(bbox)))
        result['evidence'].update(self._projection_evidence(frozen))
        with self.lock:
            self._check_clock(observation["ros_stamp"])
            if frozen["epoch"] != self.sensor_epoch:
                raise RuntimeError("传感时钟在定位期间重置，请重新观察")
        return result

    def _photo_person(self, observation, bbox, options=None, require_level=False):
        """Revalidate the same frozen evidence; never re-read a different RGB."""
        from .photo_geometry import measure_person
        if not isinstance(observation, dict):
            raise ValueError("必须使用本服务冻结的视觉观测")
        with self.lock:
            frozen = self.registry.get(observation.get("observation_id"))
        if frozen is None or time.monotonic()-frozen["created"] > self.observation_lifetime:
            raise RuntimeError("视觉观测不存在或等待过久，请重新观察")
        if any(observation.get(key) != value for key, value in frozen["public"].items()):
            raise ValueError("冻结观测被修改，拒绝使用")
        self._check_clock(observation["ros_stamp"])
        current = self.state_provider()
        if not current.get("connected") or time.time()-current.get("position_stamp", 0) > 2.0:
            raise RuntimeError("当前位姿不新鲜，不能复核拍照人物")
        if current.get("body_frame") != self.body_frame or not isinstance(current.get("orientation"), dict):
            raise RuntimeError("拍照需要同一机体坐标的完整姿态")
        xyz = np.array([current["position"][axis] for axis in ("x", "y", "z")])
        previous = np.array([observation["position"][axis] for axis in ("x", "y", "z")])
        q = np.asarray([current["orientation"].get(axis, float("nan")) for axis in ("x", "y", "z", "w")])
        rotation = quaternion_matrix(q)
        q = q/np.linalg.norm(q)
        angle = 2*math.acos(min(1., abs(float(np.dot(q, frozen["quaternion"])))))
        displacement = float(np.linalg.norm(xyz-previous))
        if displacement > .15 or angle > math.radians(5):
            raise PhotoObservationPoseChanged(displacement, angle)
        # Composition predicts a settled hover. Read-only flight monitoring
        # instead uses the complete image-time pose already in camera_T_world;
        # requiring the moving aircraft to be level would falsely abort it.
        if require_level and math.acos(min(1., max(-1., float(rotation[2, 2])))) > math.radians(3):
            raise RuntimeError("机体尚未水平悬停，请稳定后再计算拍照构图")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)
                or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)):
            raise ValueError("人物框必须为归一化 [xmin,ymin,xmax,ymax]")
        # freeze() either corrects the exact image or explicitly retains the
        # original pinhole approximation. Nonzero uncorrected D is never used.
        if any(abs(value) > 1e-10 for value in frozen["distortion"]):
            raise RuntimeError("拍照构图目前需要已校正 RGB 和零畸变 CameraInfo")
        pixels = np.asarray(bbox)*[observation["width"], observation["height"], observation["width"], observation["height"]]
        person = measure_person(frozen["projection"], pixels, frozen["camera_T_world"], frozen["K"],
                                frozen["body_T_optical"], xyz, current["yaw"], options)
        person["evidence"].update(observation_id=observation["observation_id"],
                                  rgb_stamp=observation["ros_stamp"], cloud_stamp=frozen["cloud_stamp"],
                                  camera_info_stamp=frozen["camera_info_stamp"],
                                  calibration_id=frozen["calibration_id"], sync_delta_s=frozen["sync_delta_s"],
                                  world_frame=self.world_frame, optical_frame=self.optical_frame,
                                  cloud_topic=self.cloud_topic, bbox_normalized=list(bbox))
        person['evidence'].update(self._projection_evidence(frozen))
        person["evidence"]["person_cluster"].update(
            source_frame=self.world_frame, ros_stamp=observation["ros_stamp"],
            rgb_stamp=observation["ros_stamp"], cloud_stamp=frozen["cloud_stamp"],
            observation_id=observation["observation_id"], bbox_normalized=list(bbox))
        person["evidence"]["camera_context"] = self.photo_camera_context(observation)
        if not frozen['distortion_calibrated']:
            # The underlying geometry also serves calibrated/legacy callers.
            # Label this explicitly opted-in observation at its trust boundary;
            # retain every measured extent, raw return and association check.
            for key in ('photo_guard_envelope_v1', 'association_anchor_metadata'):
                if key in person['evidence']:
                    person['evidence'][key]['source'] = 'elastic_pinhole_rgb_lidar'
        with self.lock:
            self._check_clock(observation["ros_stamp"])
            if frozen["epoch"] != self.sensor_epoch:
                raise RuntimeError("传感时钟在人物定位期间重置，请重新观察")
        return person

    def photo_bearing(self, observation, bbox):
        """Read-only optical bearing; it authorizes no translation or identity."""
        if not isinstance(observation, dict):
            raise ValueError("必须使用本服务冻结的观测")
        with self.lock:
            frozen = self.registry.get(observation.get("observation_id"))
        if frozen is None or time.monotonic()-frozen["created"] > self.observation_lifetime:
            raise RuntimeError("候选画面已过期，请重新观察")
        if any(observation.get(key) != value for key, value in frozen["public"].items()):
            raise ValueError("冻结观测被修改")
        self._check_clock(observation["ros_stamp"])
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)
                or not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)):
            raise ValueError("候选框必须为归一化图像坐标")
        if any(abs(value) > 1e-10 for value in frozen["distortion"]):
            raise RuntimeError("候选方位需要已校正图像")
        state = self.state_provider()
        if not state.get("connected") or time.time()-state.get("position_stamp", 0) > 2:
            raise RuntimeError("当前位姿不新鲜")
        old = np.array([observation["position"][axis] for axis in ("x", "y", "z")])
        current = np.array([state["position"][axis] for axis in ("x", "y", "z")])
        quaternion = np.asarray([state.get("orientation", {}).get(axis, float("nan")) for axis in ("x", "y", "z", "w")])
        quaternion_matrix(quaternion)
        quaternion /= np.linalg.norm(quaternion)
        angle = 2*math.acos(min(1., abs(float(np.dot(quaternion, frozen["quaternion"])))))
        if state.get("body_frame") != self.body_frame or np.linalg.norm(current-old) > .15 or angle > math.radians(5):
            raise RuntimeError("候选观测后位姿变化过大，请重新观察")
        pixel = np.array([.5*(bbox[0]+bbox[2])*observation["width"],
                          .5*(bbox[1]+bbox[3])*observation["height"], 1.])
        ray = np.linalg.inv(frozen["K"]) @ pixel
        world_ray = np.linalg.inv(frozen["camera_T_world"])[:3, :3] @ ray
        forward = frozen["body_T_optical"][:3, 2]
        if np.linalg.norm(world_ray[:2]) < 1e-6 or np.linalg.norm(forward[:2]) < .70:
            raise RuntimeError("候选水平观测方向不明确")
        yaw = math.atan2(world_ray[1], world_ray[0])-math.atan2(forward[1], forward[0])
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        delta = math.atan2(math.sin(yaw-state["yaw"]), math.cos(yaw-state["yaw"]))
        with self.lock:
            if frozen["epoch"] != self.sensor_epoch:
                raise RuntimeError("传感时钟已重置")
        return dict(yaw=yaw, delta_yaw=delta,
                    source=("elastic_pinhole_rgb_bearing" if not frozen['distortion_calibrated']
                            else "calibrated_rgb_bearing"),
                    observation_id=observation["observation_id"], depth_used=False,
                    translation_authorized=False, **self._projection_evidence(frozen))

    @staticmethod
    def _projection_evidence(frozen):
        return dict(projection_model=frozen['projection_model'],
                    distortion_calibrated=frozen['distortion_calibrated'],
                    image_rectified=bool(frozen['rectified']),
                    source_camera_model=frozen['source_camera_model'])

    def measure_photo(self, observation, bbox, options=None):
        try:
            person = self._photo_person(observation, bbox, options)
            return dict(evidence=person["evidence"],
                        person_surface_position=person["evidence"]["person_surface_position"])
        except Exception as exc:
            self._photo_failure(observation, bbox, exc)
            raise

    def resolve_photo(self, observation, bbox, options=None):
        from .photo_geometry import resolve_photo
        try:
            return resolve_photo(self._photo_person(observation, bbox, options, require_level=True))
        except Exception as exc:
            self._photo_failure(observation, bbox, exc)
            raise

    def validate_proposed_photo(self, observation, bbox, goal, options=None):
        from .photo_geometry import validate_proposed_photo
        try:
            person = self._photo_person(observation, bbox, options, require_level=True)
            return validate_proposed_photo(person, goal, route_navigation=(options or {}).get('navigation_scope') in ('geofence', 'planner'))
        except Exception as exc:
            self._photo_failure(observation, bbox, exc)
            raise

    def _photo_failure(self, observation, bbox, exception):
        try:
            if self._photo_diagnostics is None or not isinstance(observation, dict):
                return
            with self.lock:
                frozen = self.registry.get(observation.get("observation_id"))
            if frozen is not None:
                self._photo_diagnostics.submit(frozen, bbox, exception)
        except Exception:
            pass

    def close(self):
        self.stop.set()
        self.event.set()
        if self.worker:
            self.worker.join(timeout=0.5)
        if self._photo_diagnostics is not None:
            self._photo_diagnostics.close()
