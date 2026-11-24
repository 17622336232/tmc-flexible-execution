"""Small, task-local visual episodes, never a map or a capture authorization.

Images and acquisition poses come only from the onboard observation interface.
Sampling/retrieval uses view diversity, not a task catalogue, another model call,
simulator objects, or inferred object coordinates. Historical timestamps survive.
"""
import copy
import hashlib
import math
import threading
import time
import uuid

from .vision import VisionService


def _bounded(options, key, default, low, high, integer=False):
    value = options.get(key, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not low <= value <= high or
            (integer and not isinstance(value, int))):
        raise ValueError('photo.scene_memory.%s 配置无效。' % key)
    return value


def settings(config):
    options = config.get('photo', {}).get('scene_memory', {})
    if not isinstance(options, dict) or not isinstance(options.get('enabled', True), bool):
        raise ValueError('photo.scene_memory 必须为包含 enabled 布尔值的配置。')
    return dict(enabled=options.get('enabled', True),
                max_frames=_bounded(options, 'max_frames', 12, 1, 16, True),
                max_request_frames=_bounded(options, 'max_request_frames', 3, 0, 3, True),
                max_bytes=_bounded(options, 'max_bytes', 6*1024*1024, 65536, 8*1024*1024, True),
                max_request_bytes=_bounded(options, 'max_request_bytes', 1536*1024, 65536, 4*1024*1024, True),
                max_age_seconds=_bounded(options, 'max_age_seconds', 600., 1., 900.))


def _view_distance(a, b):
    position = math.sqrt(sum((a['position'][k]-b['position'][k])**2 for k in ('x', 'y', 'z')))
    yaw = abs(math.atan2(math.sin(a['yaw']-b['yaw']), math.cos(a['yaw']-b['yaw'])))
    return max(position/.75, yaw/math.radians(20.))


def _observation(raw, config, historical=False):
    request_config = copy.deepcopy(config)
    if historical:
        request_config['camera_timeout_seconds'] = settings(config)['max_age_seconds']
    frozen = VisionService._freeze_observation(raw, request_config)
    # Never keep arbitrary private fields/raw clouds from an adapter observation.
    result = {key: copy.deepcopy(frozen[key]) for key in
              ('data', 'stamp', 'ros_stamp', 'width', 'height', 'position', 'yaw', 'observation_id')}
    result.update(source='onboard_rgb', frame_id=str(config.get('frame_id', 'world')))
    return result


def frame_metadata(frame, now=None):
    obs = frame['observation']
    result = dict(memory_id=frame['memory_id'], role='historical_scene_context', phase=frame['phase'],
                  observation_id=obs['observation_id'], image_sha256=frame['image_sha256'],
                  stamp=obs['stamp'], ros_stamp=obs['ros_stamp'],
                  age_seconds=round(max(0., (time.time() if now is None else now)-obs['stamp']), 3),
                  width=obs['width'], height=obs['height'], source='onboard_rgb',
                  frame_id=obs['frame_id'], pose=dict(position=copy.deepcopy(obs['position']), yaw=obs['yaw']))
    if frame.get('camera_context') is not None:
        result['camera_context'] = copy.deepcopy(frame['camera_context'])
    if frame.get('evidence'):
        result['evidence'] = {key: frame['evidence'][key] for key in ('id', 'url') if key in frame['evidence']}
    return result


def validate_frames(raw, current, config, camera_validator):
    """Revalidate server-selected episodes before transmitting any old pixels."""
    options = settings(config)
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > options['max_request_frames'] or (raw and not options['enabled']):
        raise ValueError('历史场景图片超过本轮允许范围。')
    result, identities, hashes, size = [], set(), set(), 0
    for item in raw:
        if (not isinstance(item, dict) or
                set(item)-{'memory_id', 'phase', 'observation', 'camera_context', 'image_sha256', 'evidence'} or
                not isinstance(item.get('memory_id'), str) or not 1 <= len(item['memory_id']) <= 100 or
                item.get('phase') not in ('search', 'confirmation', 'viewpoint')):
            raise ValueError('历史场景记录格式无效。')
        obs = _observation(item['observation'], config, historical=True)
        digest = hashlib.sha256(obs['data']).hexdigest()
        if (digest != item.get('image_sha256') or item['memory_id'] in identities or digest in hashes or
                obs['ros_stamp'] <= 0 or obs['ros_stamp'] >= current['ros_stamp'] or
                obs['stamp'] > current['stamp'] or
                item['observation'].get('frame_id') != str(config.get('frame_id', 'world'))):
            raise ValueError('历史画面的哈希、时序或坐标系不匹配。')
        camera = item.get('camera_context')
        if camera is not None:
            camera = camera_validator(camera, obs)
            if camera['world_frame'] != obs['frame_id']:
                raise ValueError('历史相机标定的世界坐标系不匹配。')
        size += len(obs['data'])
        if size > options['max_request_bytes']:
            raise ValueError('历史场景图片超过本轮字节预算。')
        identities.add(item['memory_id']); hashes.add(digest)
        frame = dict(memory_id=item['memory_id'], phase=item['phase'], observation=obs,
                     camera_context=camera, image_sha256=digest)
        if isinstance(item.get('evidence'), dict):
            frame['evidence'] = {key: item['evidence'][key] for key in ('id', 'url')
                                 if isinstance(item['evidence'].get(key), str)}
        result.append(frame)
    return result


class SceneMemory:
    def __init__(self, config):
        self.config, self.options = copy.deepcopy(config), settings(config)
        self.lock = threading.RLock()
        self.frames, self.dropped = [], 0

    def _prune(self, now):
        self.frames = [frame for frame in self.frames
                       if 0 <= now-frame['observation']['stamp'] <= self.options['max_age_seconds']]

    def add(self, observation, camera_context=None, phase='viewpoint'):
        if not self.options['enabled']:
            return False
        try:
            if phase not in ('search', 'confirmation', 'viewpoint'):
                raise ValueError('Unknown visual-memory phase')
            obs = _observation(observation, self.config)
            if obs['ros_stamp'] <= 0 or len(obs['data']) > self.options['max_bytes']:
                raise ValueError('Invalid visual-memory frame')
            # Calibration is independently screened by PhotoAgentService before
            # transmission. Do not let diagnostic sampling block normal control.
            frame = dict(memory_id='m-'+str(uuid.uuid4()), phase=phase, observation=obs,
                         camera_context=copy.deepcopy(camera_context),
                         image_sha256=hashlib.sha256(obs['data']).hexdigest())
            with self.lock:
                self._prune(time.time())
                if any(f['image_sha256'] == frame['image_sha256'] for f in self.frames):
                    return False
                nearby = [f for f in self.frames if _view_distance(f['observation'], obs) < 1.]
                if nearby and phase == 'search':
                    return False
                # A stable confirmation/review replaces a near-identical search
                # view rather than crowding out other directions with duplicates.
                self.frames = [f for f in self.frames if f not in nearby]
                self.frames.append(frame)
                while (len(self.frames) > self.options['max_frames'] or
                       sum(len(f['observation']['data']) for f in self.frames) > self.options['max_bytes']):
                    self.frames.pop(0)
            return True
        except (ValueError, TypeError, KeyError, OverflowError):
            with self.lock:
                self.dropped += 1
            return False

    def select(self, current, evidence_callback=None):
        """At most three diverse past views; no extra inference or full video."""
        with self.lock:
            self._prune(time.time())
            pool = [frame for frame in self.frames if
                    frame['observation']['ros_stamp'] < current['ros_stamp'] and
                    frame['observation']['stamp'] <= current['stamp'] and
                    _view_distance(frame['observation'], current) >= 1.]
            selected, used = [], 0
            while pool and len(selected) < self.options['max_request_frames']:
                anchors = [current]+[f['observation'] for f in selected]
                candidate = max(pool, key=lambda f: (min(_view_distance(f['observation'], a) for a in anchors),
                                                     f['observation']['stamp']))
                pool.remove(candidate)
                size = len(candidate['observation']['data'])
                if used+size > self.options['max_request_bytes']:
                    continue
                if evidence_callback and not candidate.get('evidence'):
                    try:
                        candidate['evidence'] = evidence_callback(candidate['observation'])
                    except Exception:
                        pass  # Evidence storage is diagnostic, never admission.
                selected.append(candidate); used += size
            return copy.deepcopy(sorted(selected, key=lambda f: f['observation']['ros_stamp']))

    def public(self):
        with self.lock:
            self._prune(time.time())
            return dict(enabled=self.options['enabled'], stored_frames=len(self.frames),
                        stored_bytes=sum(len(f['observation']['data']) for f in self.frames),
                        max_request_frames=self.options['max_request_frames'], dropped_frames=self.dropped,
                        scope='仅本次任务真实机载历史画面；历史不授权拍摄或证明当前空间安全')
