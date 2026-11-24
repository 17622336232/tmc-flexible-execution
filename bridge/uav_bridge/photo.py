"""User-selected stationary subject photography from fresh onboard RGB/LiDAR evidence."""
import copy
import math
import re
import time

from .photo_detector import (PersonDetector, PhotoDetectionTimeout,
                             assess_framing, assess_stationarity, _box)
from .planner import validate_plan


_FULL_CLOUD_CLUSTER = 'full_cloud_measured_cluster_rgb_association_v1'
_ASSOCIATED_CLUSTERS = ('measured_voxel_euclidean_rgb_association_v2', _FULL_CLOUD_CLUSTER)


class PhotoFramePending(ValueError):
    """No strictly newer frame yet; calibration and geometry errors are separate."""


def parse_photo_request(text):
    """Strict small offline grammar. Unsupported tails never become silent actions."""
    if not re.search(r'拍|全身照|\bphoto(?:graph)?\b', text, re.I):
        return None
    normalized = re.sub(r'[\s，,。.!！；;]', '', text)
    match = re.fullmatch(r'(?:请|麻烦)?(?:给我|帮我|为我)?拍(?:一张|张)?全身(?:照|照片)'
                         r'(?:(?:然后|拍完后?|之后|再)?(返回|回来|返航))?'
                         r'(?:(?:然后|再|并)?(降落))?', normalized)
    if not match:
        raise ValueError('当前摄影任务支持“给我拍一张全身照”，可明确加“然后返回”或“降落”；暂不支持跟拍、合影或其他附加动作。')
    return {'return_home': bool(match.group(1)), 'land': bool(match.group(2))}


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _distance(a, b):
    if not all(_number(p.get(axis)) for p in (a, b) for axis in ('x', 'y', 'z')):
        raise ValueError('人物定位缺少有限的三维坐标。')
    return math.sqrt(sum((a[axis] - b[axis]) ** 2 for axis in ('x', 'y', 'z')))


def _geometry_source_matches(source, evidence):
    """Keep original evidence valid; bind explicit approximations to their image.

    This changes provenance only. All measured-point, envelope and association
    checks remain mandatory, including for the original pinhole model.
    """
    camera = evidence.get('camera_context', {})
    if source == 'calibrated_rgb_lidar':
        # Legacy evidence has no model fields. Explicit contradictory records
        # must not relabel an approximation as a calibrated measurement.
        return not (evidence.get('projection_model') == 'elastic_pinhole'
                    or evidence.get('distortion_calibrated') is False
                    or (isinstance(camera, dict) and
                        (camera.get('projection_model') == 'elastic_pinhole'
                         or camera.get('distortion_calibrated') is False)))
    return (source == 'elastic_pinhole_rgb_lidar' and isinstance(camera, dict)
            and all(item.get('projection_model') == 'elastic_pinhole'
                    and item.get('distortion_calibrated') is False
                    and item.get('image_rectified') is False for item in (evidence, camera)))


def _guard_envelope_points(evidence):
    key = 'photo_guard_envelope_v1'
    if key not in evidence:
        return None
    proof = evidence[key]
    if (not isinstance(proof, dict) or set(proof) !=
            {'source', 'method', 'projection_xy', 'raw_xy_bounds', 'raw_point_count'} or
            not _geometry_source_matches(proof['source'], evidence) or
            proof['method'] not in ('full_bbox_depth_envelope_and_all_raw_support_aabb', 'measured_cluster_aabb') or
            type(proof['raw_point_count']) is not int or proof['raw_point_count'] < 1):
        raise ValueError('人体完整包络证明格式或来源无效。')
    projection, bounds = proof['projection_xy'], proof['raw_xy_bounds']
    def point(value):
        return isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) for v in value)
    if (not isinstance(projection, (list, tuple)) or len(projection) != 8 or
            not all(point(p) for p in projection) or not isinstance(bounds, dict) or
            set(bounds) != {'min', 'max'} or not point(bounds['min']) or not point(bounds['max']) or
            any(bounds['min'][i] > bounds['max'][i] for i in range(2))):
        raise ValueError('人体包络必须包含全部八个投影角点和完整原始回波边界。')
    # Every raw return lies in this AABB. A convex circle enclosing all four
    # corners encloses the entire rectangle, hence also every original return.
    corners = [[x, y] for x in (bounds['min'][0], bounds['max'][0])
               for y in (bounds['min'][1], bounds['max'][1])]
    return [list(p) for p in projection]+corners


def guard_covers_measurement(frozen, current, center=None, radius=None):
    """Prove current measured envelope remains in the immutable original circle.

    New complete evidence avoids the loose triangle inequality around a noisy
    surface median. Both legacy records retain that stricter old check; losing
    one side's new proof or presenting malformed evidence never downgrades it.
    """
    if not isinstance(frozen, dict) or not isinstance(current, dict):
        raise ValueError('缺少人体保护包络测量证据。')
    old, new = frozen.get('person_surface_position'), current.get('person_surface_position')
    if not isinstance(old, dict) or not isinstance(new, dict):
        raise ValueError('人体保护缺少真实表面坐标。')
    _distance(old, new)
    original_center = [old['x'], old['y']]
    original_radius = frozen.get('photo_guard_radius_m')
    required = current.get('photo_required_guard_radius_m', current.get('photo_guard_radius_m'))
    if not _number(original_radius) or original_radius <= 0 or not _number(required) or required <= 0:
        raise ValueError('人体保护范围缺少有限正数测量。')
    if center is not None and (not isinstance(center, (list, tuple)) or list(center) != original_center):
        raise ValueError('保护圈圆心与原始确认测量不一致。')
    if radius is not None and (not _number(radius) or radius != original_radius):
        raise ValueError('保护圈半径与原始确认测量不一致。')
    before_points, current_points = _guard_envelope_points(frozen), _guard_envelope_points(current)
    if (before_points is None) != (current_points is None):
        raise ValueError('人体完整包络证明在两次观测间缺失，不能降级继续。')
    if before_points is None:
        shift = math.hypot(new['x']-old['x'], new['y']-old['y'])
        return shift+required <= original_radius+1e-6
    margin = 0.
    for key, low, high in (('body_radius_m', .4, 1.), ('min_person_clearance_m', 1.5, 5.),
                           ('measurement_margin_m', .15, .5)):
        a, b = frozen.get(key), current.get(key)
        if not _number(a) or not _number(b) or not low <= a <= high or a != b:
            raise ValueError('人体保护安全余量缺失或改变，不能缩小原确认范围。')
        margin += a
    for evidence in (frozen, current):
        envelope_radius, measured_required = evidence.get('person_envelope_radius_m'), evidence.get('photo_required_guard_radius_m')
        if (not _number(envelope_radius) or envelope_radius < 0 or not _number(measured_required) or
                abs(envelope_radius+margin-measured_required) > 1e-6):
            raise ValueError('人体保护半径与原安全余量证据不一致。')
    return all(math.hypot(p[0]-old['x'], p[1]-old['y'])+margin <= original_radius+1e-6
               for p in before_points+current_points)


class PhotoSubjectChanged(ValueError):
    code = 'photo_subject_changed'


class PhotoValidationTimeout(ValueError):
    code = 'photo_validation_timeout'


class PhotoCandidateUncertain(ValueError):
    code = 'person_candidate_uncertain'

    def __init__(self, message, detection):
        super().__init__(message)
        # Internal retry evidence only. Keep every candidate, including weak
        # bystanders; a later frame may not silently drop one and pass.
        self.detection = copy.deepcopy(detection)


class PhotoService:
    def __init__(self, config, adapter, detector=None):
        self.config, self.adapter = config, adapter
        self.options = config.get('photo', {})
        if detector is None and self.options.get('person_detector') == 'yolov8n':
            from .photo_yolo import YoloPersonDetector
            detector = YoloPersonDetector()
        self.detector = detector or PersonDetector()

    def availability(self):
        enabled = self.options.get('enabled') is True
        supported = enabled and all(callable(getattr(self.adapter, name, None)) for name in
                                    ('visual_observation', 'resolve_photo_target', 'measure_photo_target',
                                     'set_photo_guard', 'clear_photo_guard', 'start_photo_search', 'stop_photo_search',
                                     'measure_photo_candidates', 'resolve_selected_photo'))
        detector = self.detector.availability()
        ready = supported and detector.get('configured') is True
        return {'supported': supported, 'ready': ready,
                'provider': 'local_person_detector',
                'reason': ('当前配置未启用全身照任务。' if not enabled else
                           '当前适配器没有经过验证的人物保护和构图定位。' if not supported else
                           '本地人物/姿态检测模型尚未就绪。' if not ready else '')}

    def _observation(self, after=None, cancelled=None, deadline=None):
        configured_deadline = time.monotonic() + max(.2, min(15., float(self.options.get('frame_wait_timeout_seconds', 8.))))
        deadline = configured_deadline if deadline is None else min(deadline, configured_deadline)
        while True:
            if cancelled is not None and cancelled.is_set():
                raise ValueError('摄影任务已取消。')
            if time.monotonic() >= deadline:
                raise PhotoFramePending('未取得新的相机帧；未复用预览或悬停前的旧图。')
            observation = self.adapter.visual_observation()
            if (not isinstance(observation, dict) or not _number(observation.get('ros_stamp')) or
                    observation['ros_stamp'] <= 0 or not _number(observation.get('stamp')) or
                    not -1 <= time.time() - observation['stamp'] <= self.config.get('camera_timeout_seconds', 2.5)):
                raise ValueError('摄影需要新鲜且带采集时刻的机载 RGB 图像。')
            if after is None or observation['ros_stamp'] > after:
                return observation
            if time.monotonic() >= deadline:
                raise PhotoFramePending('未取得新的相机帧；未复用预览或悬停前的旧图。')
            time.sleep(.03)

    def _call_detector(self, method, observation, deadline=None, cancelled=None):
        if deadline is None:
            deadline = time.monotonic()+float(self.options.get('detector_timeout_seconds', 5.))
        if cancelled is not None and cancelled.is_set():
            raise ValueError('摄影任务已取消。')
        if time.monotonic() >= deadline:
            raise PhotoValidationTimeout('人物检测作业已超过原始预算。')
        try:
            detect = getattr(self.detector, method)
            if getattr(self.detector, 'supports_detection_deadline', False) is True:
                result = detect(observation['data'], deadline=deadline, cancelled=cancelled)
            else:
                result = detect(observation['data'])
        except PhotoDetectionTimeout as error:
            raise PhotoValidationTimeout(str(error)) from None
        if cancelled is not None and cancelled.is_set():
            raise ValueError('摄影任务已取消。')
        if time.monotonic() >= deadline:
            raise PhotoValidationTimeout('人物检测作业已超过原始预算，未采用迟到结果。')
        return result

    def _detection_frame_fresh(self, observation):
        stamp = observation.get('stamp')
        return (_number(stamp) and
                -1 <= time.time()-stamp <= self.config.get('camera_timeout_seconds', 2.5))

    def _detect(self, observation, capture=False, deadline=None, cancelled=None):
        result = self._call_detector('detect', observation, deadline, cancelled)
        if result.get('width') != observation.get('width') or result.get('height') != observation.get('height'):
            raise ValueError('人物检测与相机图像尺寸不一致。')
        return result, self._framing(result, capture)

    def _framing(self, result, capture=False):
        framing = assess_framing(result, confidence=max(.55, float(self.options.get('min_confidence', .6))),
                                 capture_margin=max(.08, float(self.options.get('framing_margin', .08))))
        if not framing.get('subject_ready'):
            raise ValueError(framing.get('message', '人物不唯一、不可见或被画面裁切。'))
        quality = framing.get('quality', {})
        if not quality.get('pose_validation') or not quality.get('full_body_evidence'):
            raise ValueError('未确认头部和双脚的可见姿态证据，请完整站入镜头再预览。')
        if capture:
            if not framing.get('ready'):
                raise ValueError(framing.get('message', '人物构图未通过，未保存为成功全身照。'))
            quality = framing.get('quality', {})
            if not quality.get('pose_validation') or not quality.get('full_body_evidence'):
                raise ValueError('未确认头部和双脚的可见姿态证据，未保存为成功全身照。')
        return framing

    @staticmethod
    def _attach_retry(error, audit):
        # This small diagnostic travels with the original error code. No image,
        # raw pose, unverified coordinates, or failed geometry is accepted.
        error.photo_retry = copy.deepcopy(audit)
        return error

    def _sample(self, geometry, after=None, capture=False, cancelled=None, validate_detection=None, raw_people=False,
                budget_seconds=.5, reference_subjects=None):
        started = time.monotonic()
        deadline = started + budget_seconds
        events = []
        last_error = None
        uncertain_candidates = None
        uncertain_origin = None
        extra_weak = None
        def budget_error():
            return (uncertain_origin if uncertain_origin is not None else last_error if last_error is not None else
                    PhotoValidationTimeout('本次摄影图像验证超过 %.1f 秒预算，未采纳迟到结果。' % budget_seconds))
        def audit(exhausted=False, reason=None):
            return {'attempts': len(events), 'retry_count': max(0, len(events) - 1),
                    'budget_seconds': budget_seconds, 'elapsed_seconds': round(time.monotonic() - started, 4),
                    'exhausted': exhausted, 'stop_reason': reason, 'events': copy.deepcopy(events)}
        for attempt in range(1, 4):
            if cancelled is not None and cancelled.is_set():
                raise self._attach_retry(ValueError('摄影任务已取消。'), audit(reason='cancelled'))
            if time.monotonic() >= deadline:
                raise self._attach_retry(budget_error(), audit(True, 'wall_clock_budget'))
            try:
                observation = self._observation(after=after, cancelled=cancelled,
                                                deadline=deadline)
            except (ValueError, RuntimeError) as exc:
                if cancelled is not None and cancelled.is_set():
                    raise self._attach_retry(exc, audit(reason='cancelled'))
                if last_error is not None:
                    raise self._attach_retry(budget_error(), audit(True, 'no_new_valid_frame'))
                if time.monotonic() >= deadline:
                    raise self._attach_retry(budget_error(), audit(True, 'wall_clock_budget'))
                raise self._attach_retry(exc, audit(reason='observation_rejected'))
            event = {'attempt': attempt, 'ros_stamp': observation['ros_stamp']}
            events.append(event)
            after = observation['ros_stamp']
            # Only explicitly uncertain candidates can retry a detection.
            # Their first complete candidate set is retained across retries;
            # detector/process errors and actual subject changes still stop.
            uncertain = None
            try:
                if raw_people:
                    try:
                        detection = self._people(observation, require_candidates=True,
                                                 deadline=deadline, cancelled=cancelled)
                    except PhotoCandidateUncertain as exc:
                        uncertain, detection = exc, exc.detection
                    framing = {"bbox": None}
                else:
                    detection, framing = self._detect(observation, capture=capture,
                                                      deadline=deadline, cancelled=cancelled)
                checked_detection = detection
                reliable = []
                if raw_people and (uncertain is not None or extra_weak is not None):
                    minimum = max(.55, float(self.options.get('min_confidence', .6)))
                    reliable = [item for item in detection['detections'] if item.get('confidence', 0) >= minimum]
                if extra_weak is not None and len(reliable) != 1:
                    if reference_subjects is None and uncertain is None:
                        # All original candidates becoming clear is still a
                        # valid multi-person freeze; none is being discarded.
                        self._match_uncertain_candidates(extra_weak['detection'], detection)
                        uncertain_candidates = extra_weak['detection']
                        extra_weak = None
                    else:
                        raise PhotoSubjectChanged('复核时可靠主体丢失或出现新的可靠人物，请重新核对人物。')
                if (extra_weak is None and uncertain_candidates is None and uncertain is not None and
                        len(reliable) == 1 and len(detection['detections']) > 1 and
                        (reference_subjects is None or len(reference_subjects) == 1)):
                    extra_weak = dict(detection=copy.deepcopy(detection), anchor=None, clean_frames=0)
                    uncertain_origin = uncertain
                if extra_weak is not None:
                    # A weak additional box is unresolved, not an established
                    # extra person. Keep it in the audit while validating the
                    # reliable subject against any existing selection.
                    checked_detection = dict(detection, detections=reliable)
                if validate_detection is not None:
                    validate_detection(checked_detection, framing)
                if uncertain_candidates is not None:
                    self._match_uncertain_candidates(uncertain_candidates, detection)
                elif uncertain is not None and extra_weak is None:
                    uncertain_candidates = copy.deepcopy(detection)
                    uncertain_origin = uncertain
            except (ValueError, RuntimeError) as exc:
                event.update(error_code=getattr(exc, 'code', 'person_detection_rejected'))
                if cancelled is not None and cancelled.is_set():
                    raise self._attach_retry(exc, audit(reason='cancelled'))
                if isinstance(exc, PhotoValidationTimeout):
                    # The outer search loop can restart ordinary frame waits.
                    # Never let that discard an unresolved possible bystander.
                    error = uncertain_origin if uncertain_origin is not None else exc
                    raise self._attach_retry(error, audit(True, 'wall_clock_budget'))
                raise self._attach_retry(exc, audit(reason='detection_rejected'))
            if cancelled is not None and cancelled.is_set():
                raise self._attach_retry(ValueError('摄影任务已取消。'), audit(reason='cancelled'))
            if time.monotonic() >= deadline:
                event['error_code'] = 'validation_budget_elapsed_after_detection'
                raise self._attach_retry(budget_error(), audit(True, 'wall_clock_budget'))
            if not self._detection_frame_fresh(observation):
                # A valid but slow detection cannot authorize old pixels.
                # Re-read strictly newer RGB within this same job budget.
                event['error_code'] = 'detection_frame_expired'
                last_error = PhotoFramePending('检测完成时图像已过期，重新获取新帧。')
                if attempt == 3:
                    raise self._attach_retry(budget_error(), audit(True, 'attempt_limit'))
                continue
            if extra_weak is not None:
                clean = len(detection['detections']) == 1
                event['additional_weak_candidate_count'] = len(detection['detections']) - 1
                if extra_weak['anchor'] is None or clean:
                    try:
                        anchor = self._measure_retry_subject(observation, reliable, reference_subjects,
                                                             extra_weak['anchor'])
                    except (ValueError, RuntimeError) as exc:
                        event['error_code'] = getattr(exc, 'code', 'geometry_rejected')
                        if cancelled is not None and cancelled.is_set():
                            raise self._attach_retry(ValueError('摄影任务已取消。'), audit(reason='cancelled'))
                        if (isinstance(exc, (PhotoFramePending, PhotoValidationTimeout)) or
                                event['error_code'] in ('person_range_sparse', 'person_range_coverage',
                                'person_background_sparse', 'person_cluster_sparse', 'observation_pose_changed')):
                            # No spatial anchor means no basis for clearing an
                            # unresolved detection, including in a later job.
                            raise self._attach_retry(uncertain_origin, audit(True, 'weak_subject_geometry_unavailable'))
                        raise self._attach_retry(exc, audit(reason='non_retryable_geometry'))
                    if cancelled is not None and cancelled.is_set():
                        raise self._attach_retry(ValueError('摄影任务已取消。'), audit(reason='cancelled'))
                    if time.monotonic() >= deadline:
                        raise self._attach_retry(uncertain_origin, audit(True, 'wall_clock_budget'))
                    if not self._detection_frame_fresh(observation):
                        event['error_code'] = 'geometry_frame_expired'
                        last_error = uncertain_origin
                        if attempt == 3:
                            raise self._attach_retry(uncertain_origin, audit(True, 'attempt_limit'))
                        continue
                    if extra_weak['anchor'] is None:
                        extra_weak['anchor'] = anchor
                    elif clean:
                        extra_weak['clean_frames'] += 1
                if not clean:
                    extra_weak['clean_frames'] = 0
                event['consistent_single_subject_frames'] = extra_weak['clean_frames']
                if extra_weak['clean_frames'] < 2:
                    event['error_code'] = 'person_candidate_uncertain'
                    last_error = uncertain_origin
                    if attempt == 3:
                        raise self._attach_retry(uncertain_origin, audit(True, 'attempt_limit'))
                    continue
                # Two later fresh observations did not reproduce the extra
                # weak detections. This is bounded evidence, not proof that
                # the initial detector boxes described imaginary people.
                event['weak_candidates_not_redetected'] = len(extra_weak['detection']['detections']) - 1
                event['additional_weak_candidate_resolution'] = 'not_redetected_in_two_fresh_frames'
            if uncertain is not None:
                event['error_code'] = uncertain.code
                last_error = uncertain
                if attempt == 3:
                    raise self._attach_retry(uncertain, audit(True, 'attempt_limit'))
                continue
            try:
                measured = geometry(observation, detection if raw_people else framing['bbox'])
            except (ValueError, RuntimeError) as exc:
                event['error_code'] = getattr(exc, 'code', 'geometry_rejected')
                # Only the calibrated photo observer identifies stale SELF
                # pose before accepting any person geometry. Discard that
                # entire RGB and re-detect a strictly newer frame; never
                # relabel the old coordinates or retry actual subject motion.
                stale_self_pose = raw_people and event['error_code'] == 'observation_pose_changed'
                if not stale_self_pose and event['error_code'] not in (
                        'person_range_sparse', 'person_range_coverage', 'person_background_sparse',
                        'person_cluster_sparse'):
                    raise self._attach_retry(exc, audit(reason='non_retryable_geometry'))
                if raw_people and len(detection['detections']) > 1:
                    # A batch failure hides partial measurements: an already
                    # measured person's movement must never be lost to a retry.
                    raise self._attach_retry(exc, audit(reason='multi_person_geometry_no_retry'))
                last_error = exc
                if attempt == 3 or time.monotonic() >= deadline:
                    raise self._attach_retry(exc, audit(True, 'attempt_limit' if attempt == 3 else 'wall_clock_budget'))
                continue
            if cancelled is not None and cancelled.is_set():
                raise self._attach_retry(ValueError('摄影任务已取消。'), audit(reason='cancelled'))
            # A slow inference/map query cannot turn any late attempt into success.
            if time.monotonic() >= deadline:
                event['error_code'] = 'validation_budget_elapsed_after_geometry'
                raise self._attach_retry(budget_error(), audit(True, 'wall_clock_budget'))
            if not self._detection_frame_fresh(observation):
                event['error_code'] = 'geometry_frame_expired'
                last_error = PhotoFramePending('定位完成时图像已过期，重新获取新帧。')
                if attempt == 3:
                    raise self._attach_retry(budget_error(), audit(True, 'attempt_limit'))
                continue
            event['outcome'] = 'validated_geometry'
            return {'observation': observation, 'detection': detection, 'composition': framing,
                    'geometry': measured, 'geometry_retry': audit()}
        raise AssertionError('bounded geometry retry exhausted without a result')

    @staticmethod
    def person(resolved):
        return resolved.get('person_surface_position', resolved.get('evidence', {}).get('person_surface_position', {}))

    @staticmethod
    def _association_position(measured):
        evidence = measured.get('evidence', {})
        if not isinstance(evidence, dict):
            raise ValueError('人物关联锚点缺少有效测量证据。')
        cluster = evidence.get('person_cluster', {})
        if isinstance(cluster, dict) and cluster.get('algorithm') == _FULL_CLOUD_CLUSTER:
            anchor = evidence.get('association_anchor_world')
            metadata = evidence.get('association_anchor_metadata')
            centroid = cluster.get('centroid_world')
            surface = evidence.get('person_surface_position')
            # This is a measured component centroid, not the legacy torso-ray
            # estimate and not proof of an unseen body center. Bind every copy
            # to the same measured point; absent metadata must never downgrade.
            if (not all(isinstance(p, dict) and set(p) == {'x', 'y', 'z'} and
                        all(_number(p[axis]) for axis in 'xyz')
                        for p in (anchor, centroid, surface))
                    or not isinstance(metadata, dict)
                    or not _geometry_source_matches(metadata.get('source'), evidence)
                    or metadata.get('method') != 'measured_component_centroid_v1'
                    or metadata.get('scope') != 'cross_frame_association_estimate_only'
                    or metadata.get('algorithm') != _FULL_CLOUD_CLUSTER
                    or metadata.get('is_lidar_return') is not False
                    or metadata.get('body_center_known') is not False
                    or any(abs(anchor[axis]-p[axis]) > 1e-8 for p in (centroid, surface) for axis in 'xyz')):
                raise ValueError('实测人物簇关联锚点或来源无效，不能降级为旧测量。')
            return anchor
        present = 'association_anchor_world' in evidence or 'association_anchor_metadata' in evidence
        if not present:
            return None
        anchor, metadata = evidence.get('association_anchor_world'), evidence.get('association_anchor_metadata')
        if (not isinstance(anchor, dict) or set(anchor) != {'x', 'y', 'z'}
                or not all(_number(anchor.get(axis)) for axis in ('x', 'y', 'z'))
                or not isinstance(metadata, dict)
                or not _geometry_source_matches(metadata.get('source'), evidence)
                or metadata.get('method') != 'torso_roi_center_ray_median_optical_depth_v1'
                or metadata.get('scope') != 'cross_frame_association_estimate_only'
                or metadata.get('is_lidar_return') is not False
                or metadata.get('body_center_known') is not False
                or not _number(metadata.get('depth_m')) or metadata['depth_m'] <= .1
                or not isinstance(metadata.get('pixel'), (list, tuple)) or len(metadata['pixel']) != 2
                or not all(_number(value) for value in metadata['pixel'])):
            raise ValueError('人物关联锚点坐标或测量来源无效，不能退回表面点继续任务。')
        return anchor

    @classmethod
    def _association_distance(cls, before, current):
        surfaces = (cls.person(before), cls.person(current))
        if not all(isinstance(surface, dict) for surface in surfaces):
            raise ValueError('人物实际表面测量缺少有效三维坐标。')
        # The estimate must not conceal a missing/nonfinite actual surface,
        # which remains the independent center of the protected envelope.
        surface_distance = _distance(*surfaces)
        old, new = cls._association_position(before), cls._association_position(current)
        if (old is None) != (new is None):
            raise ValueError('人物关联锚点在两次观测间缺失，不能混用旧版表面点。')
        clusters = [item.get('evidence', {}).get('person_cluster', {}) for item in (before, current)]
        algorithms = [item.get('algorithm') if isinstance(item, dict) else None for item in clusters]
        if any(algorithm in _ASSOCIATED_CLUSTERS for algorithm in algorithms):
            if algorithms[0] != algorithms[1]:
                raise ValueError('人物三维簇来源在观测间变化，不能降级关联。')
            positions = [item.get('centroid_world') for item in clusters]
            if not all(isinstance(p, dict) and set(p) == {'x','y','z'} and all(_number(p[a]) for a in 'xyz')
                       for p in positions):
                raise ValueError('人物三维簇中心缺少有效实测坐标。')
            a, b = positions
            # Effective real lightweight profile: motion_vertical_weight=.20.
            # This is correspondence only; the frozen raw-envelope protection
            # below independently decides whether flight remains permissible.
            return math.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2+(.2*(a['z']-b['z']))**2)
        if old is None:
            # Compatibility is allowed only when BOTH measurements predate
            # the anchor contract. A lost/invalid new field must fail closed.
            return surface_distance
        return _distance(old, new)

    def _association_tolerance(self, before, current):
        clusters = [item.get('evidence', {}).get('person_cluster', {}) for item in (before, current)]
        algorithms = [item.get('algorithm') if isinstance(item, dict) else None for item in clusters]
        if algorithms[0] in _ASSOCIATED_CLUSTERS and algorithms[0] == algorithms[1]:
            return min(.95, max(.01, float(self.options.get('cluster_match_distance_m', .95))))
        return min(.20, max(.01, float(self.options.get('person_motion_tolerance_m', .20))))

    def preview(self, text):
        intent = parse_photo_request(text)
        if intent is None:
            raise ValueError('不是受支持的全身照任务。')
        available = self.availability()
        if not available['ready']:
            raise ValueError(available['reason'])
        snap = self.adapter.snapshot()
        real_ready = (snap.get('mode') == 'real' and snap.get('capabilities', {}).get('yaw_target') is True
                      and snap.get('real_execution', {}).get('backend') == 'diff_vla_real'
                      and snap.get('real_execution', {}).get('ready') is True)
        if snap.get('mode') != 'gazebo' and not real_ready:
            raise ValueError('实机任务执行服务尚未就绪，请检查设备日志。')
        sampled = self._sample(self.adapter.resolve_photo_target)
        observation, detection, framing = sampled['observation'], sampled['detection'], sampled['composition']
        resolved = sampled['geometry']
        _distance(self.person(resolved), self.person(resolved))
        steps = []
        if resolved.get('requires_motion'):
            steps.append(dict(resolved['goal'], action='goto', label='全身照构图位置'))
        steps.append({'action': 'photo'})
        if intent['return_home']:
            home = self.config.get('home')
            if not isinstance(home, dict):
                raise ValueError('未记录真实起飞位置，无法添加返回动作。')
            steps.append(dict(home, action='goto', label='返回起飞点'))
        if intent['land']:
            steps.append({'action': 'land'})
        if len(steps) > self.config.get('max_steps', 12):
            raise ValueError('摄影任务超过允许步骤数。')
        # Validate ordinary steps using the shared bounds/sequence policy. Photo
        # is only generated here; it is not added to model/control allow-lists.
        for step in steps:
            if step['action'] != 'photo':
                validate_plan({'summary': '全身照', 'steps': [step]}, self.config)
        return {'kind': 'photo', 'plan': {'summary': '拍摄一张全身照' + ('，返回起飞点' if intent['return_home'] else '') +
                                        ('并降落' if intent['land'] else '后悬停' if intent['return_home'] else '；拍完保持悬停'), 'steps': steps},
                'observation': observation, 'detection': detection, 'composition': framing,
                'photo_target': resolved, 'bbox': framing['bbox'], 'geometry_retry': sampled['geometry_retry']}

    def _validate_measurement(self, reference, measured):
        displacement = self._association_distance(reference['photo_target'], measured)
        tolerance = self._association_tolerance(reference['photo_target'], measured)
        if displacement > tolerance:
            raise ValueError('人物位置关联差异 %.2f 米，超过本次关联范围；请重新确认人物。' % displacement)
        frozen = reference['photo_target']['evidence']
        current = measured.get('evidence', {})
        if not guard_covers_measurement(frozen, current):
            raise ValueError('当前人体包络已超出已确认保护范围；请保持静止并重新预览。')
        return displacement

    def _validate_person(self, reference, sampled, capture=False, previous=None):
        observation, detection, framing = sampled['observation'], sampled['detection'], sampled['composition']
        measured = sampled['geometry']['measurement']
        displacement = self._validate_measurement(reference, measured)
        return {'observation': observation, 'detection': detection, 'composition': framing,
                'measurement': measured, 'person_displacement_m': displacement,
                'geometry_retry': sampled['geometry_retry']}

    def _observe(self, reference, after=None, capture=False, cancelled=None, previous=None, resolve=False):
        if reference.get('selection'):
            return self._observe_selected(reference, after, capture, cancelled, previous, resolve)
        def geometry(observation, bbox):
            measured = self.adapter.measure_photo_target(observation, bbox)
            # Reject measured movement/envelope change before resolving a goal;
            # a later sparse solve must never hide an already-known unsafe person.
            self._validate_measurement(reference, measured)
            result = {'measurement': measured}
            if resolve:
                result['resolved'] = self.adapter.resolve_photo_target(observation, bbox)
                self._validate_immutable_goal(reference, result['resolved'])
            return result
        def validate_detection(detection, framing):
            if capture and previous is not None:
                stationary = assess_stationarity(previous['detection'], detection)
                if not stationary.get('stable'):
                    raise ValueError('连续画面中的人物尚未保持静止，未完成拍摄。')
                framing['quality']['stationarity_validation'] = True
                framing['stationarity'] = stationary
        sampled = self._sample(geometry, after=after, capture=capture, cancelled=cancelled,
                               validate_detection=validate_detection)
        try:
            result = self._validate_person(reference, sampled, capture, previous)
            if resolve:
                result['resolved'] = sampled['geometry']['resolved']
            return result
        except (ValueError, RuntimeError) as exc:
            # Position/envelope/stationarity failures are never retried.
            raise self._attach_retry(exc, sampled['geometry_retry'])

    def observe(self, reference, after=None, capture=False, cancelled=None, previous=None):
        return self._observe(reference, after, capture, cancelled, previous)

    def _validate_immutable_goal(self, reference, current):
        old_goal, new_goal = reference['photo_target']['goal'], current['goal']
        if _distance(old_goal, new_goal) > .15:
            raise ValueError('构图目标与预览发生变化，请重新预览。')
        yaw_change = abs((old_goal.get('yaw', 0) - new_goal.get('yaw', 0) + math.pi) % (2 * math.pi) - math.pi)
        if yaw_change > math.radians(5):
            raise ValueError('构图朝向与预览发生变化，请重新预览。')
        checker = getattr(self.adapter, 'validate_photo_goal', None)
        if callable(checker):
            checker(old_goal, current['evidence'])
        elif reference['photo_target'].get('requires_motion') and _distance(old_goal, new_goal) > 1e-6:
            raise ValueError('缺少对原已确认目标的独立安全复核。')

    def revalidate(self, reference):
        # Measurement, re-resolution and the old endpoint map query share one
        # budget. No nested retries or unbounded second geometric check.
        return self._observe(reference, after=reference['observation']['ros_stamp'], resolve=True)

    def _people(self, observation, require_candidates=False, deadline=None, cancelled=None):
        result = self._call_detector('detect_people', observation, deadline, cancelled)
        if result.get('width') != observation.get('width') or result.get('height') != observation.get('height'):
            raise ValueError('候选人物检测与机载图像尺寸不一致。')
        people = result.get('detections')
        if not isinstance(people, list) or len(people) > 4:
            raise ValueError('当前支持最多四个人物候选，请减少画面内人数后重新搜索。')
        if require_candidates:
            if not people:
                raise PhotoSubjectChanged('原候选人物已离开画面，请重新搜索。')
            minimum = max(.55, float(self.options.get('min_confidence', .6)))
            if any(item.get('confidence', 0) < minimum for item in people):
                # The detector's accepted boxes remain intact; uncertainty is
                # not permission to remove a possible person from protection.
                message = ('当前唯一人物候选还不够清晰，需要新的清晰画面复核；仍不可靠时请重新搜索。'
                           if len(people) == 1 else
                           '当前多个人物候选中仍有不够清晰者，需要保留全部候选复核；不能忽略可能的旁人。')
                raise PhotoCandidateUncertain(message, result)
        return result

    def _match_uncertain_candidates(self, original, current):
        before, after = original['detections'], current['detections']
        if len(before) != len(after):
            raise PhotoSubjectChanged('复核前后人物候选数量变化，不能忽略消失的疑似旁人或替换原候选，请重新搜索。')
        old_boxes = [_box(item.get('bbox')) for item in before]
        new_boxes = [_box(item.get('bbox')) for item in after]
        # Reuse the existing image association overlap; do not lower the
        # confidence threshold or claim this box check proves identity.
        used = set()
        for box in old_boxes:
            matches = [i for i, candidate in enumerate(new_boxes) if self._overlap(box, candidate) >= .30]
            if len(matches) != 1 or matches[0] in used:
                raise PhotoSubjectChanged('复核中的人物候选交叉、变化或关联不唯一，请重新搜索并确认人物。')
            used.add(matches[0])

    def _measure_retry_subject(self, observation, reliable, reference_subjects, anchor):
        measured = self.adapter.measure_photo_candidates(observation, reliable)
        if (not isinstance(measured, list) or len(measured) != 1 or
                not isinstance(measured[0], dict) or measured[0].get('index') != 0):
            raise PhotoSubjectChanged('复核时可靠主体的实测定位不唯一。')
        current = measured[0]
        # Validate genuine measurement provenance even for the first temporary
        # anchor. Image motion is not person motion: use the existing world
        # association tolerance, not overlap with the old pixel rectangle.
        self._association_distance(current, current)
        if reference_subjects is not None:
            self._match_subjects({'selection': {'subjects': reference_subjects}}, measured)
        if anchor is not None and self._association_distance(anchor, current) > self._association_tolerance(anchor, current):
            raise PhotoSubjectChanged('复核中的可靠主体三维位置无法对应，不能替换原人物。')
        return copy.deepcopy(current)

    def scan_candidates(self, after=None, cancelled=None):
        budget = self.options.get('search_job_timeout_seconds', 5.)
        deadline = time.monotonic() + budget
        observation = self._observation(after, cancelled, deadline)
        result = self._people(observation, deadline=deadline, cancelled=cancelled)
        if cancelled is not None and cancelled.is_set():
            raise ValueError('搜索已取消。')
        if time.monotonic() > deadline:
            raise PhotoValidationTimeout('搜索图像检测超过 %.1f 秒，未采纳迟到候选。' % budget)
        if not self._detection_frame_fresh(observation):
            raise PhotoFramePending('检测完成时候选画面已过期，重新获取新帧。')
        return {'observation': observation, 'detection': result,
                'reliable_count': sum(item.get('confidence', 0) >= max(.55, self.options.get('min_confidence', .6))
                                      for item in result['detections'])}

    def freeze_candidates(self, after=None, cancelled=None):
        def geometry(observation, detection):
            return self.adapter.measure_photo_candidates(observation, detection['detections'])
        sampled = self._sample(geometry, after=after, cancelled=cancelled, raw_people=True,
                               budget_seconds=self.options.get('search_job_timeout_seconds', 5.))
        return {'observation': sampled['observation'], 'detection': sampled['detection'],
                'subjects': sampled['geometry'], 'geometry_retry': sampled['geometry_retry']}

    def _match_subjects(self, reference, current):
        previous = reference['selection']['subjects']
        if len(current) != len(previous):
            raise PhotoSubjectChanged('画面人数发生变化，不能将所选人物替换成另一人，请重新搜索。')
        result, used = {}, set()
        for before in previous:
            matches = [item for item in current if self._association_distance(before, item) <=
                       self._association_tolerance(before, item)]
            if len(matches) != 1 or matches[0]['index'] in used:
                raise PhotoSubjectChanged('所选人物或旁人移动、交叉或关联不唯一，已停止任务，请重新搜索。')
            item = matches[0]
            try:
                covered = guard_covers_measurement(before['evidence'], item['evidence'])
            except ValueError as error:
                raise PhotoSubjectChanged(str(error)) from None
            if not covered:
                raise PhotoSubjectChanged('人物包络超出已确认保护范围，请重新搜索。')
            result[before['index']] = item
            used.add(item['index'])
        return result

    @staticmethod
    def _overlap(a, b):
        area = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
        return area/union if union > 0 else 0.

    def _observe_selected(self, reference, after=None, capture=False, cancelled=None, previous=None, resolve=False):
        early_index = [None]
        def image_gate(raw, _framing):
            if len(raw['detections']) != len(reference['selection']['subjects']):
                raise PhotoSubjectChanged('画面人数发生变化，不能将所选人物替换成另一人，请重新搜索。')
            if capture and previous is not None:
                old_box = previous['detection']['detections'][0]['bbox']
                matches = [i for i, item in enumerate(raw['detections']) if self._overlap(old_box, item['bbox']) >= .30]
                if len(matches) != 1:
                    raise PhotoSubjectChanged('拍摄时人物图像交叉、消失或关联不唯一。')
                index = matches[0]
                temporary = dict(raw, detections=[raw['detections'][index]])
                if not assess_stationarity(previous['detection'], temporary).get('stable'):
                    raise PhotoSubjectChanged('拍摄时所选人物尚未保持静止。')
                early_index[0] = index
        def geometry(observation, raw):
            subjects = self.adapter.measure_photo_candidates(observation, raw['detections'])
            matched = self._match_subjects(reference, subjects)
            chosen = matched[reference['selection']['selected_index']]
            index = chosen['index']
            if early_index[0] is not None and early_index[0] != index:
                raise PhotoSubjectChanged('人物图像与三维定位关联不一致，未继续拍摄。')
            detection = self.detector.select_subject(observation['data'], raw, index)
            framing = self._framing(detection, capture)
            if capture and previous is not None:
                stationary = assess_stationarity(previous['detection'], detection)
                if not stationary.get('stable'):
                    raise PhotoSubjectChanged('拍摄时所选人物尚未保持静止。')
                framing['quality']['stationarity_validation'] = True
                framing['stationarity'] = stationary
            self._validate_measurement(reference, chosen)
            result = {'measurement': chosen, 'detection': detection, 'composition': framing,
                      'selection': {'subjects': subjects, 'selected_index': index}}
            if resolve:
                others = [item['bbox'] for i, item in enumerate(raw['detections']) if i != index]
                result['resolved'] = self.adapter.resolve_selected_photo(observation, framing['bbox'], others)
                if reference['photo_target'].get('goal'):
                    self._validate_immutable_goal(reference, result['resolved'])
            return result
        sampled = self._sample(geometry, after=after, capture=capture, cancelled=cancelled,
                               validate_detection=image_gate, raw_people=True,
                               reference_subjects=reference['selection']['subjects'])
        result = sampled['geometry']
        result.update(observation=sampled['observation'], geometry_retry=sampled['geometry_retry'],
                      person_displacement_m=_distance(self.person(result['measurement']), self.person(reference['photo_target'])))
        return result

    def preview_subject(self, text, frozen, candidate_index):
        intent = parse_photo_request(text)
        if intent is None or not isinstance(candidate_index, int) or isinstance(candidate_index, bool):
            raise ValueError('选人请求无效。')
        if not 0 <= candidate_index < len(frozen['subjects']):
            raise ValueError('候选人物编号已失效。')
        chosen = next((item for item in frozen['subjects'] if item['index'] == candidate_index), None)
        if chosen is None:
            raise ValueError('候选人物测距证据不存在。')
        reference = {'observation': frozen['observation'], 'photo_target': {'evidence': chosen['evidence']},
                     'selection': {'subjects': frozen['subjects'], 'selected_index': candidate_index}}
        fresh = self._observe_selected(reference, after=frozen['observation']['ros_stamp'], resolve=True)
        target = fresh['resolved']
        steps = [dict(target['goal'], action='goto', label='所选人物的全身照构图位置')] if target['requires_motion'] else []
        steps.append({'action': 'photo'})
        if intent['return_home']:
            if not isinstance(self.config.get('home'), dict):
                raise ValueError('未记录真实起飞位置，无法添加返回动作。')
            steps.append(dict(self.config['home'], yaw=target['goal']['yaw'], action='goto', label='返回起飞点并保持观察人物'))
        if intent['land']:
            steps.append({'action': 'land'})
        if len(steps) > self.config['max_steps']:
            raise ValueError('拍摄任务超出步骤上限。')
        for step in steps:
            if step['action'] != 'photo':
                validate_plan({'summary': '所选人物全身照', 'steps': [step]}, self.config)
        return {'kind': 'photo', 'plan': {'summary': '为你确认的人物拍一张全身照' +
                                        ('，返回起飞点' if intent['return_home'] else '') +
                                        ('并降落' if intent['land'] else '；完成后悬停'), 'steps': steps},
                'observation': fresh['observation'], 'detection': fresh['detection'],
                'composition': fresh['composition'], 'photo_target': target,
                'bbox': fresh['composition']['bbox'], 'selection': fresh['selection'],
                'geometry_retry': fresh['geometry_retry']}

    @staticmethod
    def detector_metadata(detection):
        result = {key: copy.deepcopy(detection[key]) for key in
                  ('provider', 'model_sha256', 'network_sha256', 'inference_ms', 'scope', 'pose', 'all_person_count', 'selected_detection_index') if key in detection}
        result['provider'] = 'local_person_detector'
        result['person_detector_provider'] = detection.get('provider')
        return result
