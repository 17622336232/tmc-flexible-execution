"""Asynchronous bounded execution of real model photography decisions.

Model, person measurement and vehicle commands have independent work slots.
Only this tick/admission state machine can commit a photo or authorize a move.
"""
import copy
import hashlib
import json
import math
import os
import secrets
from pathlib import Path
import tempfile
import threading
import time
from concurrent.futures import Future

from .photo import PhotoSubjectChanged, PhotoFramePending, guard_covers_measurement
from .photo_detector import PhotoDetectionTimeout
from .photo_agent import PhotoReviewInvalidated
from .demo_targets import target_filter_config
from .photo_policy import execution_limits
from .flight_limits import planner_scope, within_bounds, exceeds_distance
from .scene_memory import SceneMemory
from .vehicle_search import VehicleSearch, new_car_directions


LIMITS = dict(max_decisions=8, max_moves=3, max_duration_seconds=480, max_model_request_seconds=120,
              max_move_m=1.2, max_radius_m=1.5, same_altitude=True,
              review_image_max_age_seconds=130)


class _ProposalRejected(ValueError):
    """A fresh subject passed monitoring, but this exact model goal did not."""


class _MoveObservationInvalidated(ValueError):
    """Known stale evidence before adapter.goto was called; no goal was sent."""


class _ReviewObservationInvalidated(ValueError):
    """The model was not called; only its queued input frame has expired."""


from .photo_task_schema import requirements as _requirements, shots as task_shots, shot_requirements


def _validate_checks(decision, task, shot_index=0):
    requirements = shot_requirements(task, shot_index)
    checks, visual = decision.get('checks'), decision.get('visual')
    if (not isinstance(visual, dict) or set(visual) != {'subject_visible', 'certainty', 'evidence'} or
            not isinstance(visual['subject_visible'], bool) or visual['certainty'] not in ('high', 'uncertain') or
            not isinstance(visual['evidence'], str) or not visual['evidence'].strip() or len(visual['evidence']) > 1000 or
            not isinstance(checks, list) or len(checks) != len(requirements)):
        raise ValueError('模型没有提供完整的逐项摄影检查。')
    ids = []
    for item in checks:
        if (not isinstance(item, dict) or set(item) != {'requirement_id', 'state', 'evidence'} or
                not isinstance(item['requirement_id'], str) or
                item['state'] not in ('met', 'unmet', 'uncertain', 'not_observable') or
                not isinstance(item['evidence'], str) or not item['evidence'].strip() or len(item['evidence']) > 1000):
            raise ValueError('模型逐项摄影检查格式无效。')
        ids.append(item['requirement_id'])
    expected = [r['id'] for r in requirements]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('模型检查遗漏、重复或替换了已确认摄影要求。')
    return (visual['subject_visible'] is True and visual['certainty'] == 'high' and
            all(item['state'] == 'met' for item in checks))


def _distance(a, b):
    return math.sqrt(sum((float(a[k])-float(b[k]))**2 for k in ('x', 'y', 'z')))


def _yaw_delta(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def _pose_close(a, b):
    try:
        if _distance(a['position'], b['position']) > .15 or _yaw_delta(a['yaw'], b['yaw']) > math.radians(5):
            return False
        qa, qb = a.get('orientation'), b.get('orientation')
        if qa is not None and qb is not None:
            va, vb = [float(qa[k]) for k in 'xyzw'], [float(qb[k]) for k in 'xyzw']
            na, nb = math.sqrt(sum(v*v for v in va)), math.sqrt(sum(v*v for v in vb))
            if not .99 <= na <= 1.01 or not .99 <= nb <= 1.01:
                return False
            angle = 2*math.acos(min(1., abs(sum(x*y for x,y in zip(va,vb))/(na*nb))))
            if angle > math.radians(5):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _stationary(snap, maximum_speed=.15):
    try:
        speed = math.sqrt(sum(float(snap['velocity'][a])**2 for a in ('x', 'y', 'z')))
        return math.isfinite(speed) and speed <= maximum_speed and snap.get('flight_mode') in ('HOVERING', 'OFFBOARD', None)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _navigation_arrived(snap, goal, config):
    """Planner reports completion; telemetry only gates a useful new image."""
    acceptance = config.get('photo', {}).get('navigation_acceptance', {})
    return ((config.get('mode') != 'real' or snap.get('navigation_completed') is True) and
            _distance(snap['position'], goal) <= acceptance.get('position_tolerance_m', .25) and
            _yaw_delta(snap['yaw'], goal['yaw']) <= math.radians(acceptance.get('yaw_tolerance_deg', 10)) and
            _stationary(snap, acceptance.get('stable_speed_mps', .15)))


def _observation_pose_close(a, b):
    """Allow ordinary hover drift while binding images to their measured pose."""
    try:
        ya, yb = float(a['yaw']), float(b['yaw'])
        return (math.isfinite(ya) and math.isfinite(yb) and
                _yaw_delta(ya, yb) <= math.radians(5) and _pose_close(a, b))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _fresh(observation, maximum=2.5):
    try:
        age = time.time()-float(observation['stamp'])
        return math.isfinite(age) and -.1 <= age <= maximum and float(observation['ros_stamp']) > 0
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _zone(subject):
    evidence = subject['evidence']
    p = subject.get('person_surface_position', evidence['person_surface_position'])
    return dict(center=[p['x'], p['y']], radius=evidence['photo_guard_radius_m'])


def make_reference(frozen, index):
    """Freeze selected subject and every measured bystander; no fixed waypoint."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError('所选人物编号无效。')
    subjects = copy.deepcopy(frozen['subjects'])
    chosen = next((p for p in subjects if p['index'] == index), None)
    if chosen is None or not 1 <= len(subjects) <= 4:
        raise ValueError('所选人物已失效或人数超出范围。')
    evidence = copy.deepcopy(chosen['evidence'])
    evidence['zones'] = [_zone(chosen)] + [_zone(p) for p in subjects if p['index'] != index]
    return dict(observation=copy.deepcopy(frozen['observation']),
                selection=dict(subjects=subjects, selected_index=index),
                photo_target=dict(person_surface_position=copy.deepcopy(chosen['person_surface_position']), evidence=evidence))


def _record_association_failure(photo, reference, current, observation, error, once=None):
    """Opt-in, bounded diagnostics only; never changes the original rejection."""
    temporary = None
    try:
        directory = os.environ.get('VLA_PHOTO_DIAGNOSTICS_DIR')
        if not directory or not Path(directory).is_absolute() or (once is not None and once.is_set()):
            return
        if once is not None:
            once.set()  # At most one attempt per runner, including failed IO.
        evidence_keys = ('photo_guard_radius_m', 'photo_required_guard_radius_m',
                         'person_envelope_radius_m', 'person_motion_allowance_m',
                         'measured_depth_m', 'depth_spread_m', 'depth_uncertainty_m',
                         'cloud_stamp', 'rgb_stamp', 'sync_delta_s', 'torso_point_count',
                         'torso_coverage', 'background_separation', 'photo_guard_envelope_v1',
                         'body_radius_m', 'min_person_clearance_m', 'measurement_margin_m')
        cluster_keys = ('algorithm', 'tolerance_m', 'roi_point_count', 'component_count',
                        'selected_point_count', 'excluded_background_count', 'centroid_world',
                        'observed_extent_world', 'point_count', 'uncertainty_m', 'extent_scope')
        def subject(item):
            evidence = item.get('evidence', {})
            cluster = item.get('person_cluster') or evidence.get('person_cluster') or {}
            return dict(index=item['index'], bbox=list(item['bbox']),
                        person_surface_position=copy.deepcopy(item['person_surface_position']),
                        evidence={k: copy.deepcopy(evidence[k]) for k in evidence_keys if k in evidence},
                        person_cluster={k: copy.deepcopy(cluster[k]) for k in cluster_keys if k in cluster})
        previous = reference['selection']['subjects'][:4]
        measured = current[:4]
        tolerance = min(.2, max(.01, photo.options.get('person_motion_tolerance_m', .2)))
        comparisons = []
        for old in previous:
            for new in measured:
                a, b = old['person_surface_position'], new['person_surface_position']
                delta = {axis: float(b[axis])-float(a[axis]) for axis in 'xyz'}
                distance = math.sqrt(sum(v*v for v in delta.values()))
                match_distance = photo._association_distance(old, new)
                pair_tolerance = photo._association_tolerance(old, new)
                tolerance = max(tolerance, pair_tolerance)
                comparisons.append(dict(reference_index=old['index'], current_index=new['index'],
                                        delta_m=delta, distance_3d_m=distance,
                                        distance_xy_m=math.hypot(delta['x'], delta['y']),
                                        distance_for_match_m=match_distance, match_tolerance_m=pair_tolerance,
                                        within_match_tolerance=match_distance <= pair_tolerance))
        payload = dict(version=1, diagnostic_only=True, decision_changed=False,
                       simulator_object_truth_used=False, error_type=type(error).__name__,
                       error_code=getattr(error, 'code', None), error=str(error)[:500],
                       match_tolerance_m=tolerance,
                       reference=[subject(p) for p in previous], current=[subject(p) for p in measured],
                       comparisons=comparisons,
                       observation={k: copy.deepcopy(observation[k]) for k in
                                    ('stamp', 'ros_stamp', 'position', 'yaw', 'orientation', 'width', 'height')
                                    if k in observation},
                       image_sha256=hashlib.sha256(observation['data']).hexdigest())
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
        if len(encoded) > 65536:
            return
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix='.association-failure-', dir=str(path))
        with os.fdopen(descriptor, 'wb') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(encoded)
        os.replace(temporary, str(path/'association-failure-latest.json'))
        temporary = None
    except Exception:
        pass
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def measure_selected(photo, reference, after=None, cancelled=None, diagnostic_once=None):
    """Fresh spatial association only: never uses old framing or goal resolver."""
    def validate_detection(raw, _framing):
        # Check before any retryable range measurement: a sparse frame must
        # never conceal a disappeared or newly arrived person on the next try.
        if len(raw['detections']) != len(reference['selection']['subjects']):
            raise PhotoSubjectChanged('画面人数发生变化，请重新搜索并确认人物。')

    def geometry(observation, raw):
        subjects = photo.adapter.measure_photo_candidates(observation, raw['detections'])
        try:
            matched = photo._match_subjects(reference, subjects)
        except (ValueError, RuntimeError) as error:
            _record_association_failure(photo, reference, subjects, observation, error, diagnostic_once)
            raise
        chosen = matched[reference['selection']['selected_index']]
        photo._validate_measurement(reference, chosen)
        frozen_people = reference['selection']['subjects']
        ordered = ([p for p in frozen_people if p['index'] == reference['selection']['selected_index']] +
                   [p for p in frozen_people if p['index'] != reference['selection']['selected_index']])
        zones = reference['photo_target']['evidence']['zones']
        if len(zones) != len(ordered):
            raise ValueError('已确认人物与保护范围数量不一致。')
        for before, zone in zip(ordered, zones):
            current = matched[before['index']]
            if not guard_covers_measurement(before['evidence'], current['evidence'], zone['center'], zone['radius']):
                raise ValueError('当前人物包络已超出原始固定保护圈，不能累计放宽。')
        indexed = []
        for before in reference['selection']['subjects']:
            item = copy.deepcopy(matched[before['index']])
            item['reference_index'] = before['index']
            indexed.append(item)
        # Freeze the primary center/radius together with every zone. Mixing a
        # new measured primary with old zones would violate the guard contract.
        evidence = copy.deepcopy(reference['photo_target']['evidence'])
        return dict(measurement=copy.deepcopy(chosen), selection=dict(subjects=indexed, selected_index=chosen['index']),
                    resolved=dict(evidence=evidence, person_surface_position=copy.deepcopy(evidence['person_surface_position'])))
    sampled = photo._sample(geometry, after=after, cancelled=cancelled,
                            validate_detection=validate_detection, raw_people=True,
                            budget_seconds=photo.options.get('search_job_timeout_seconds', 5.),
                            reference_subjects=reference['selection']['subjects'])
    result = sampled['geometry']
    result.update(observation=sampled['observation'], detection=sampled['detection'], geometry_retry=sampled.get('geometry_retry'))
    return result


def observe_stationary_subject(photo, reference, after=None, cancelled=None):
    """Fresh RGB/pose, with the operator-confirmed STATIC geometry retained.

    No continuous detector, point-cloud re-clustering or identity tracker is
    needed in the explicitly stationary demo. Projected regions are hints for
    the VLM, never new measured boxes or evidence of a person's current pose.
    """
    observation=photo._observation(after=after,cancelled=cancelled,deadline=time.monotonic()+.5)
    camera=photo.adapter.photo_camera_context(observation,include_demo_targets=False)
    subjects=copy.deepcopy(reference['selection']['subjects'])
    transform=camera['world_T_camera'];intrinsic=camera['intrinsics']
    for item in subjects:
        cluster=item.get('person_cluster') or item['evidence']['person_cluster']
        extent=cluster['observed_extent_world'];pixels=[];depths=[]
        for x in (extent['min']['x'],extent['max']['x']):
            for y in (extent['min']['y'],extent['max']['y']):
                for z in (extent['min']['z'],extent['max']['z']):
                    delta=[v-transform[i][3] for i,v in enumerate((x,y,z))]
                    optical=[sum(transform[j][i]*delta[j] for j in range(3)) for i in range(3)]
                    depths.append(optical[2])
                    if optical[2]>.1:
                        pixels.append((intrinsic['fx']*optical[0]/optical[2]+intrinsic['cx'],
                                       intrinsic['fy']*optical[1]/optical[2]+intrinsic['cy']))
        # Never leave the initial camera's box in a later image when the
        # stationary person is outside this view. The world guard is retained
        # independently; a projection is not a fresh detection or visibility.
        item['bbox']=None
        if max(depths)<=.1:
            item['projection_status']='behind_camera'
        elif min(depths)<=.1:
            item['projection_status']='near_camera_plane'
        else:
            lo=[max(0.,min(1.,min(p[a] for p in pixels)/observation[k])) for a,k in enumerate(('width','height'))]
            hi=[max(0.,min(1.,max(p[a] for p in pixels)/observation[k])) for a,k in enumerate(('width','height'))]
            if hi[0]>lo[0] and hi[1]>lo[1]:
                item['bbox']=lo+hi
                item['projection_status']='partial' if min(lo+[1-v for v in hi])<=0 else 'inside'
            else:
                item['projection_status']='outside_image'
        item['bbox_source']='projection_of_initial_static_cluster'
        item['reference_index']=item['index']
        item['evidence']['camera_context']=copy.deepcopy(camera)
    selected=reference['selection']['selected_index']
    chosen=next(item for item in subjects if item['index']==selected)
    return dict(observation=observation,measurement=chosen,
                selection=dict(subjects=subjects,selected_index=selected),
                detection=dict(detections=[],provider='stationary_person_no_redetection'),
                resolved=copy.deepcopy(reference['photo_target']),
                subject_position_policy='confirmed_stationary_person',geometry_remeasured=False)


class PhotoAgentRunner:
    def __init__(self, config, adapter, service, photo, store, task, reference, initial, guard_token,
                 evidence_callback=None, scene_memory=None):
        self.config, self.adapter, self.service, self.photo, self.store = config, adapter, service, photo, store
        self.limits = execution_limits(config)
        self.stationary_subject = (self.limits.get('navigation_strategy')=='local_route' and
                                   self.config.get('photo',{}).get('subject_motion_mode')=='stationary')
        # The frozen image keeps its actual exposure pose; model goals are
        # absolute world coordinates. Current-image/map/control checks remain.
        self.static_world_review = (self.stationary_subject and self.config.get('mode')=='real' and
                                    self.config.get('photo',{}).get('navigation_scope')=='planner' and
                                    self.config.get('photo',{}).get('decision_mode')=='vlm')
        self.task, self.reference, self.last = copy.deepcopy(task), copy.deepcopy(reference), copy.deepcopy(initial)
        _requirements(self.task)
        self.shots = copy.deepcopy(task_shots(self.task))
        self.shot_index = 0
        self.guard_token, self.evidence_callback = guard_token, evidence_callback
        self.scene_memory = scene_memory if scene_memory is not None else SceneMemory(config)
        self.scene_memory.add(initial['observation'], initial['measurement'].get('evidence', {}).get('camera_context'),
                              phase='confirmation')
        self.memory_frames_sent = 0
        self.last_memory_frame_count = 0
        self.origin = copy.deepcopy(initial['observation']['position'])
        self.session = service.new_session(self.task, origin=self.origin)
        self.cancelled = threading.Event()
        self.diagnostic_once = threading.Event()
        self.lock, self.action_lock = threading.RLock(), threading.RLock()
        # The adapter's final publish callback must never acquire self.lock:
        # tick may hold it while reading adapter state. This independent lock
        # linearizes pending landing cancellation with its publish boundary.
        self.dispatch_lock = threading.Lock()
        self._landing_dispatched = threading.Event()
        self.started = time.monotonic()
        self.deadline = self.started + self.limits['max_duration_seconds']
        self.stage, self.revision, self.moves = 'settling', 0, 0
        self.history, self.photos, self.move_audit = [], [], []
        self.decisions = []
        self.last_decision = self.reviewed_evidence = None
        self.review = self.review_measurement = None
        self.review_changed = False
        self.review_cancelled = threading.Event()
        self.review_refreshes = []
        self.stable_pose = self.stable_since = None
        self.stable_position_stamp = None
        self.model_frame_after_wall = self.model_frame_after_ros = None
        self.model_future = self.local_future = self.action_future = self.save_future = None
        self.model_started = self.local_started = self.action_started = self.save_started = 0.
        self.local_kind = 'monitor'
        self.last_monitor = self.started
        self.next_monitor = self.started
        self.wait_until = 0.
        self.pending_decision = self.goal = None
        self.result = None
        self.frame_wait_reason = self.navigation_wait_reason = ''
        self.decision_wait_reason = ''
        self._saved_uncommitted = set()
        self._committed = set()
        self.updated_at = time.time()
        self.car_search = None
        self.car_control_future = self.car_scan_future = None
        self.car_control_started = 0.
        self.car_scan_started = self.car_scan_after = None
        self.car_scan_next = 0.
        self.car_stop_reason = ''
        self.car_cleanup_ids = set()

    @staticmethod
    def _submit(function, name):
        future = Future()
        def run():
            try:
                future.set_result(function())
            except BaseException as error:
                future.set_exception(error)
        threading.Thread(target=run, name='photo-agent-'+name, daemon=True).start()
        return future

    def _delete_orphans(self, identities):
        for identity in identities:
            try:
                self.store.remove(identity)
                with self.lock:
                    self._saved_uncommitted.discard(identity)
            except Exception:
                pass

    def cancel(self):
        with self.lock:
            # Cancellation and save admission have one linearization point.
            # Never set the event outside this lock and then commit past it.
            with self.dispatch_lock:
                self.cancelled.set()
            orphans = self._saved_uncommitted - self._committed
        self.service.cancel(self.session)
        if self.car_search is not None and self.car_search.owner is not None:
            identity = self.car_search.owner
            if identity not in self.car_cleanup_ids:
                self.car_cleanup_ids.add(identity)
                self._submit(lambda: self.adapter.stop_photo_search(identity), 'car-search-cleanup')
        if orphans:
            self._submit(lambda: self._delete_orphans(orphans), 'orphan-cleanup')

    def cancel_pending_landing(self):
        """Atomically cancel an unissued landing; False preserves active landing.

        Calling this after completion or after the adapter crossed its final
        publish boundary has no cancellation side effect.
        """
        with self.lock:
            if self.stage != 'landing' or self.result is not None:
                return False
            with self.dispatch_lock:
                if self._landing_dispatched.is_set():
                    return False
                self.cancelled.set()
            self.cancel()
            return True

    def _before_land_dispatch(self):
        # Called inside the adapter's final state lock, immediately before its
        # publish. Do not take runner.lock or call back into the adapter here.
        with self.dispatch_lock:
            if self.cancelled.is_set():
                raise ValueError('待发送的摄影降落已取消。')
            self._landing_dispatched.set()

    def public(self):
        with self.lock:
            return dict(stage=self.stage, revision=self.revision, source='llm', model=self.task.get('model'),
                        landing_dispatched=self._landing_dispatched.is_set(),
                        task=copy.deepcopy(self.task), limits=copy.deepcopy(self.limits),
                        capture_progress=dict(shot_index=self.shot_index, shot_count=len(self.shots),
                            completed_count=len(self.photos),
                            current_shot=copy.deepcopy(self.shots[self.shot_index]) if self.shot_index < len(self.shots) else None),
                        remaining_budget=dict(decisions=max(0, self.limits['max_decisions']-self.revision), moves=max(0, self.limits['max_moves']-self.moves),
                                              seconds=round(max(0., self.deadline-time.monotonic()), 2)),
                        last_decision=copy.deepcopy(self.last_decision), reviewed_evidence=copy.deepcopy(self.reviewed_evidence),
                        decisions=copy.deepcopy(self.decisions),
                        review_refreshes=copy.deepcopy(self.review_refreshes),
                        scene_memory=dict(self.scene_memory.public(), frames_sent=self.memory_frames_sent,
                                          last_request_frames=self.last_memory_frame_count),
                        waiting_reason=self.decision_wait_reason or self.frame_wait_reason or self.navigation_wait_reason,
                        vehicle_search=(self.car_search.context(True, time.monotonic()) if self.car_search else None),
                        updated_at=self.updated_at, photos=copy.deepcopy(self.photos), moves=copy.deepcopy(self.move_audit))

    def _decision_update(self, **fields):
        if self.last_decision is not None:
            self.last_decision.update(copy.deepcopy(fields))
        for record in self.decisions:
            if (self.last_decision or {}).get('preplanned'):
                if record['revision'] == self.last_decision['source_revision'] and record.get('next_shot'):
                    # Keep one record per REAL model call. Reusing its next goal
                    # must not overwrite the prior photo's committed delivery or
                    # mislabel its checks as checks of the next, unseen image.
                    record['next_shot'].update(copy.deepcopy(fields))
            elif record['revision'] == self.revision:
                record.update(copy.deepcopy(fields))

    def _fail(self, reason):
        self.stage = 'failed'
        self.result = dict(state='failed', error=str(reason))
        self.cancel()
        return self.result

    def _complete(self):
        self.stage = 'completed'
        self.result = dict(state='completed', error=None)
        self.service.cancel(self.session)
        return self.result

    def _history(self, decision, outcome, reason):
        self.history.append(dict(action=decision['action'], outcome=outcome, goal=copy.deepcopy(decision.get('goal')),
                                 reason=str(reason)[:300] or '已验证'))
        self.history = self.history[-self.limits['max_decisions']:]

    def _reject(self, decision, reason, now):
        self._history(decision, 'rejected', reason)
        if self.last_decision is None or self.last_decision.get('revision') != self.revision:
            self.last_decision = dict(copy.deepcopy(decision), revision=self.revision, received_at=time.time())
        self._decision_update(accepted=False, local_rejection=str(reason), delivery='rejected')
        self.pending_decision = None
        self.decision_wait_reason = ''
        self.stage, self.wait_until = 'waiting_pose', now+1.

    def _spatial(self, measured):
        observation = measured['observation']
        selection = measured['selection']
        def describe(subject):
            cluster = subject.get('person_cluster') or subject['evidence'].get('person_cluster')
            if not isinstance(cluster, dict):
                raise ValueError('缺少人物真实点云簇，不能调用摄影模型。')
            return dict(id=str(subject.get('reference_index', subject['index'])),
                        bbox=copy.deepcopy(subject['bbox']),
                        centroid_world=copy.deepcopy(cluster['centroid_world']),
                        observed_extent_world=copy.deepcopy(cluster['observed_extent_world']),
                        uncertainty_m=cluster['uncertainty_m'],
                        ros_stamp=subject['evidence'].get('cloud_stamp', observation['ros_stamp']))
        selected = next(p for p in selection['subjects'] if p['index'] == selection['selected_index'])
        result = dict(selected_subject=describe(selected),
                      other_people=[describe(p) for p in selection['subjects'] if p['index'] != selected['index']],
                      known_space=dict(available=False, candidates=[]),
                      guard_zones=copy.deepcopy(self.reference['photo_target']['evidence']['zones']))
        camera = selected.get('evidence', {}).get('camera_context')
        if camera is not None:
            result['camera_context'] = copy.deepcopy(camera)
        return result

    def _refresh_review(self, now, reason=None):
        """Discard only an obsolete camera decision, not the task or its guard."""
        reason = reason or '机位已变化，本轮旧图判断作废，稳定后重新取图。'
        self.review_refreshes.append(dict(revision=self.revision, reason=reason, stamp=time.time()))
        self.review_refreshes = self.review_refreshes[-self.limits['max_decisions']:]
        self.pending_decision = None
        self.decision_wait_reason = ''
        self.review_measurement = None
        self.stage = 'settling'
        self.frame_wait_reason = reason
        self._reset_observation_window()
        self.next_monitor = now

    def _defer_decision(self, now, reason):
        """Refresh only local action evidence; the frozen model decision stays."""
        self.stage = 'validating'
        self.goal = None
        self.decision_wait_reason = reason
        self._decision_update(delivery='waiting_stability', accepted=True,
                              local_rejection='', waiting_reason=reason)
        self.next_monitor = now

    def _motion_admission(self, current, observation):
        """Final read-only check, also called inside adapter's publish lock.

        No runner lock is acquired here. Only the typed transient below can be
        retried as a definitely unissued action; safety failures stay terminal.
        """
        if self.cancelled.is_set():
            raise ValueError('摄影动作已取消。')
        if current.get('command_error'):
            raise ValueError(current['command_error'])
        guard = current.get('photo_guard') or {}
        if guard.get('id') != self.guard_token or guard.get('active') is not True or guard.get('error'):
            raise ValueError('发送动作前人物保护已改变。')
        if (current.get('connected') is not True or current.get('planning_ready') is not True or
                current.get('armed') is not True or current.get('landed') is not False or
                current.get('flight_mode') != 'OFFBOARD'):
            raise ValueError('发送动作前飞行、定位或地图状态已失效。')
        if not _stationary(current):
            raise _MoveObservationInvalidated('模型决定已保留，等待飞行速度恢复稳定后执行。')
        if not _fresh(observation) or not _pose_close(current, observation):
            raise _MoveObservationInvalidated('模型决定已保留，正在更新动作发送前的本地观测。')

    def _begin_local(self, kind, now):
        reference, after = copy.deepcopy(self.reference), self.last['observation']['ros_stamp']
        if self.stage == 'observing' and self.model_frame_after_ros is not None:
            after = max(after, self.model_frame_after_ros)
        decision = copy.deepcopy(self.pending_decision)
        def worker():
            if self.stationary_subject:
                fresh=observe_stationary_subject(self.photo,reference,after=after,cancelled=self.cancelled)
            else:
                fresh = measure_selected(self.photo, reference, after=after, cancelled=self.cancelled,
                                         diagnostic_once=self.diagnostic_once)
            if self.cancelled.is_set():
                raise ValueError('摄影已取消。')
            if kind == 'move':
                goal = copy.deepcopy(decision['goal'])
                try:
                    self._goal_allowed(goal, fresh)
                except ValueError as error:
                    raise _ProposalRejected(str(error)) from None
                try:
                    if self.stationary_subject:
                        fresh['proposal_validation']=self.adapter.validate_static_photo_goal(
                            fresh['observation'],goal,self.reference['photo_target']['evidence'])
                    else:
                        selected = fresh['measurement']['index']
                        boxes = fresh['detection']['detections']
                        fresh['proposal_validation'] = self.adapter.validate_proposed_photo_goal(
                            fresh['observation'], boxes[selected]['bbox'],
                            [box['bbox'] for i, box in enumerate(boxes) if i != selected], goal)
                except (ValueError, RuntimeError) as error:
                    raise _ProposalRejected(str(error)) from None
                if fresh['proposal_validation'].get('goal') != goal:
                    raise ValueError('本地验证改写模型机位，已拒绝。')
            return fresh
        self.local_kind, self.local_started = kind, now
        self.local_future = self._submit(worker, 'measure-'+kind)
        self.next_monitor = now+.5

    def _begin_model(self, snap, now):
        if self.revision >= self.limits['max_decisions']:
            raise ValueError('摄影模型已达本次授权的决策次数上限。')
        observation = self.last['observation']
        if (self.model_frame_after_wall is None or self.model_frame_after_ros is None or
                observation['stamp'] <= self.model_frame_after_wall or
                observation['ros_stamp'] <= self.model_frame_after_ros or
                not _fresh(observation) or not _stationary(snap) or
                not _observation_pose_close(snap, observation) or
                (not self.static_world_review and not _observation_pose_close(snap, self.stable_pose))):
            return
        if self.review is not None and self.last['observation']['ros_stamp'] <= self.review['ros_stamp']:
            return
        self.review_measurement = copy.deepcopy(self.last)
        self.decision_wait_reason = ''
        self.review_changed = False
        self.review_cancelled = threading.Event()
        review_cancelled = self.review_cancelled
        self.review = copy.deepcopy(self.last['observation'])
        self.review['data'] = bytes(self.review['data'])
        spatial, history = self._spatial(self.review_measurement), copy.deepcopy(self.history)
        memory = self.scene_memory.select(self.review, self.evidence_callback)
        self.scene_memory.add(self.review, spatial.get('camera_context'), phase='viewpoint')
        self.memory_frames_sent += len(memory)
        self.last_memory_frame_count = len(memory)
        self.revision += 1
        self.reviewed_evidence = (self.evidence_callback(self.review) if self.evidence_callback else
                                 {k: self.review[k] for k in ('stamp', 'ros_stamp', 'observation_id')})
        self.reviewed_evidence = dict(self.reviewed_evidence, revision=self.revision,
                                     image_sha256=hashlib.sha256(self.review['data']).hexdigest())
        observation = copy.deepcopy(self.review)
        search_context = self._car_search_context(snap, now)
        def worker():
            if self.cancelled.is_set():
                raise ValueError('模型请求前任务已取消。')
            if not _fresh(observation):
                raise _ReviewObservationInvalidated('模型请求前图像已过期，未发送请求，重新取图。')
            options = {'scene_memory': memory} if memory else {}
            if search_context['supported'] or self.car_search is not None:
                options['search_context'] = search_context
            if self.stationary_subject and target_filter_config(self.config.get('photo', {})) is not None:
                # Optional person/car candidates belong to this exact frozen
                # review. Keep their computation off the safety-monitor slot;
                # later monitor frames must never supply this image's geometry.
                options['prepare_camera_context'] = self.adapter.photo_camera_context
            return self.service.decide(observation, spatial, copy.deepcopy(self.task), history=history, session=self.session,
                                       review_cancelled=review_cancelled, **options)
        self.model_started, self.stage = now, 'awaiting_model'
        self.model_future = self._submit(worker, 'model')

    def _car_search_context(self, snap, now):
        supported = (self.static_world_review and
            (snap.get('real_execution') or {}).get('guarded_photo_search') is True and
            callable(getattr(getattr(self.photo, 'detector', None), 'detect_cars', None)) and
            callable(getattr(self.adapter, 'start_photo_search', None)) and
            callable(getattr(self.adapter, 'stop_photo_search', None)))
        episode = self.car_search or VehicleSearch(self.shots[self.shot_index]['id'])
        return episode.context(supported, now)

    def _start_car_search(self, snap, now, fresh):
        context = self._car_search_context(snap, now)
        if not context['supported']:
            raise ValueError('当前执行器或检测器尚不支持带人物保护的车辆扫视，请更新程序。')
        if context['exhausted']:
            raise ValueError('已完成本张车辆的一圈或 30 秒原地扫视，仍未确认目标；已拍照片保留。')
        if self.car_search is None:
            self.car_search = VehicleSearch(self.shots[self.shot_index]['id'])
        # Only directions actually reviewed in this exact image are suppressed.
        camera = self.review_measurement['measurement']['evidence']['camera_context']
        self.car_search.review(camera)
        identity = secrets.token_urlsafe(18)
        sweep, duration = self.car_search.start(identity, now)
        token = self.adapter.capture_control_token()  # Before queue/command_lock.
        observation = copy.deepcopy(fresh['observation'])
        before_dispatch = lambda current: self._motion_admission(current, observation)
        def start():
            if self.cancelled.is_set():
                raise ValueError('车辆扫视已取消，未启动。')
            before_dispatch(self.adapter.snapshot())
            result = self.adapter.start_photo_search(identity, max_sweep_rad=sweep,
                timeout=duration, rate=.35, guard_token=self.guard_token, control_token=token,
                before_dispatch=before_dispatch)
            if self.cancelled.is_set():
                self.adapter.stop_photo_search(identity)
                raise ValueError('车辆扫视已取消，不采纳迟到启动。')
            return result
        self.car_control_future = self._submit(start, 'car-search-start')
        self.car_control_started = now
        self.car_scan_after = self.review['ros_stamp']
        self.car_scan_next = now
        self.car_stop_reason = ''
        self.stage = 'car_search_starting'
        self.decision_wait_reason = ''
        self.frame_wait_reason = '正在原地慢转寻找车辆，已确认人物保护保持。'

    def _stop_car_search(self, reason):
        if self.stage == 'car_search_stopping':
            return
        identity = self.car_search.owner
        self.car_stop_reason = reason
        self.stage = 'car_search_stopping'
        self.frame_wait_reason = ('发现新的车辆候选，停止转向后用新画面辨认。' if reason == 'candidate'
                                  else '原地车辆扫视结束，正在确认停止。')
        self.car_control_future = self._submit(lambda: self.adapter.stop_photo_search(identity), 'car-search-stop')
        self.car_control_started = time.monotonic()

    def _begin_car_scan(self, now):
        after = self.car_scan_after
        owner = self.car_search.owner
        reviewed = copy.deepcopy(self.car_search.views)
        def scan():
            deadline = time.monotonic()+self.config.get('photo', {}).get('search_job_timeout_seconds', 5.)
            observation = self.photo._observation(after=after, cancelled=self.cancelled, deadline=deadline)
            detected = self.photo.detector.detect_cars(observation['data'], deadline=deadline, cancelled=self.cancelled)
            if self.cancelled.is_set():
                raise ValueError('车辆扫视已取消。')
            if (time.monotonic() >= deadline or not _fresh(observation)):
                raise PhotoFramePending('车辆候选图像已过期，继续获取新画面。')
            if (detected.get('image_sha256') != hashlib.sha256(observation['data']).hexdigest() or
                    detected.get('width') != observation['width'] or detected.get('height') != observation['height']):
                raise ValueError('车辆检测未绑定当前实际相机帧。')
            detections = detected.get('detections')
            if not isinstance(detections, list):
                raise ValueError('车辆检测结果格式无效。')
            for item in detections:
                box = item.get('bbox')
                if (item.get('label') != 'car' or not isinstance(box, (list, tuple)) or len(box) != 4 or
                        not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box) or
                        not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                    raise ValueError('车辆候选框无效。')
            camera = self.adapter.photo_camera_context(observation, include_demo_targets=False)
            if self.cancelled.is_set() or not _fresh(observation):
                raise PhotoFramePending('车辆候选已过期，继续获取新画面。')
            return dict(owner=owner, observation=observation, directions=new_car_directions(detections, camera, reviewed))
        self.car_scan_started = now
        self.car_scan_future = self._submit(scan, 'car-search-scan')

    def _tick_car_search(self, snap, now):
        if self.stage not in ('car_search_starting', 'car_search_scanning', 'car_search_stopping'):
            return
        episode = self.car_search
        status = snap.get('photo_search') or {}
        episode.observe_status(status)
        if self.car_control_future is not None:
            if not self.car_control_future.done():
                if now-self.car_control_started > self.config.get('photo', {}).get('search_job_timeout_seconds', 5.):
                    raise ValueError('车辆扫视控制确认超时，已取消本次任务，已拍照片保留。')
                return  # The adapter bounds each start/stop acknowledgement.
            try:
                result = self.car_control_future.result()
            except _MoveObservationInvalidated as exc:
                if self.stage != 'car_search_starting':
                    raise
                # This typed exception is produced before the search START is
                # published (including the adapter's final guarded callback).
                self.car_control_future = None
                episode.stop(dict(id=episode.owner, angle_swept_rad=0.), now)
                self._defer_decision(now, str(exc))
                return
            self.car_control_future = None
            if self.stage == 'car_search_starting':
                self._decision_update(delivery='searching', waiting_reason='')
                self._history(self.pending_decision, 'executed', '保留原人物保护，在当前位置扫视寻找车辆候选。')
                self.pending_decision = None
                self.stage = 'car_search_scanning'
                return  # This snapshot can predate the active acknowledgement.
            acknowledgement = (result or {}).get('photo_search') or {}
            if acknowledgement.get('id') != episode.owner:
                raise ValueError('车辆扫视停止确认未绑定本次搜索，不能继续拍摄。')
            episode.stop(acknowledgement, now)
            # The old scan may still be finishing; its result is retired below.
            self.car_scan_after = max(self.car_scan_after or 0., self.last['observation']['ros_stamp'])
            if self.car_stop_reason != 'candidate':
                raise ValueError('已完成本张车辆的一圈或 30 秒原地扫视，仍未确认目标；已拍照片保留。')
            self.stage = 'settling'
            self.frame_wait_reason = '已停止转向，稳定后用新画面辨认车辆颜色和拍摄侧面。'
            self._reset_observation_window()
            self.next_monitor = now
            return
        if self.stage != 'car_search_scanning':
            return
        if status.get('id') == episode.owner and status.get('error'):
            raise ValueError('车辆扫视控制失效：'+str(status['error']))
        if self.car_scan_future is not None and self.car_scan_future.done():
            future, self.car_scan_future = self.car_scan_future, None
            try:
                result = future.result()
            except (PhotoFramePending, PhotoDetectionTimeout):
                result = None  # Same episode budget; stale pixels never stop a search.
            if result is not None and result['owner'] == episode.owner:
                observation = result['observation']
                self.car_scan_after = observation['ros_stamp']
                if _fresh(observation) and result['directions']:
                    self._stop_car_search('candidate')
                    return
            self.car_scan_next = now+.25
        # A completed, still-fresh last scan can find a candidate on the tick
        # that the controller finishes its sweep. Stop first, then review a NEW
        # settled image; completing the sweep must not discard that evidence.
        context = episode.context(True, now)
        if context['exhausted'] or (status.get('id') == episode.owner and status.get('active') is False):
            self._stop_car_search('exhausted')
            return
        if self.car_scan_future is None and now >= self.car_scan_next:
            self._begin_car_scan(now)

    def _next_viewpoint(self, metadata):
        consume = getattr(self.service, 'take_next_viewpoint', None)
        decision = consume(self.session, self.task, metadata['id']) if callable(consume) else None
        if decision is None:
            return False
        # A discarded review may fail before entering the service. Its runner
        # attempt still counts toward the budget, but consumes no model iteration.
        # Bind the proposal to the actual capture decision, not the attempt count.
        audit = self.pending_decision.get('audit')
        source_iteration = audit.get('iteration') if isinstance(audit, dict) else None
        if (decision.get('preplanned') is not True or decision.get('action') != 'move' or
                decision.get('shot_id') != self.shots[self.shot_index]['id'] or
                decision.get('shot_index') != self.shot_index or decision.get('after_photo_id') != metadata['id'] or
                type(source_iteration) is not int or source_iteration < 1 or
                type(decision.get('source_revision')) is not int or
                decision.get('source_revision') != source_iteration or
                decision.get('observation') != self.pending_decision.get('observation') or
                decision.get('goal') != (self.pending_decision.get('next_shot') or {}).get('goal')):
            raise ValueError('下一机位预案未绑定刚保存的照片、原模型响应或下一张顺序。')
        self.pending_decision = copy.deepcopy(decision)
        self.last_decision = dict(copy.deepcopy(decision), revision=self.revision, received_at=time.time(),
                                  reviewed_evidence=copy.deepcopy(self.reviewed_evidence),
                                  delivery='not_dispatched', proposal_accepted=True, latency_seconds=0.)
        self._decision_update(delivery='not_dispatched', preplanned=True,
                              source_revision=self.revision, after_photo_id=metadata['id'])
        # Retain the original reviewed pose as the admission anchor. A NEW
        # observation and route validation are still mandatory before any goto.
        self.review_changed = False
        self.stage = 'validating'
        return True

    def _reset_observation_window(self):
        self.stable_pose = self.stable_since = self.stable_position_stamp = None
        self.model_frame_after_wall = self.model_frame_after_ros = None

    def _observation_window_ready(self, snap, now):
        # position_stamp is the adapter's age-preserving wall-time mapping,
        # not a raw ROS source stamp. Repeated snapshots cannot earn dwell.
        try:
            stamp = float(snap['position_stamp'])
            valid = (math.isfinite(stamp) and stamp > 0 and
                     0 <= time.time()-stamp <= self.config['telemetry_timeout_seconds'])
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid or not _stationary(snap):
            self._reset_observation_window()
            self.stage = 'settling'
            return False
        if (self.stable_pose is None or
                (not self.static_world_review and not _observation_pose_close(snap, self.stable_pose)) or
                stamp < self.stable_position_stamp):
            self._reset_observation_window()
            self.stable_since, self.stable_pose = now, copy.deepcopy(snap)
            self.stable_position_stamp = stamp
            self.stage = 'settling'
            return False
        # The mission loop may poll faster than odometry publishes. An equal
        # source stamp contributes no source time, but does not erase already
        # observed stability; rollback, stale data and motion still reset it.
        self.stable_position_stamp = stamp
        if now-self.stable_since < .6 or stamp-float(self.stable_pose['position_stamp']) < .6:
            return False
        if self.model_frame_after_wall is None:
            # Do not reuse an in-flight monitor's pre-settle image, even if it
            # arrives later and happens to pass the general 2-degree review gate.
            self.model_frame_after_wall = time.time()
            self.model_frame_after_ros = float(self.last['observation']['ros_stamp'])
            camera_stamp = snap.get('camera_status', {}).get('ros_stamp')
            if isinstance(camera_stamp, (int, float)) and math.isfinite(camera_stamp):
                self.model_frame_after_ros = max(self.model_frame_after_ros, camera_stamp)
        self.stage = 'observing'
        return True

    def _goal_allowed(self, goal, fresh):
        if self.review_changed:
            raise ValueError('人物画面构图已变化，旧机位判断失效；请用新图重新判断。')
        if not isinstance(goal, dict) or set(goal) != {'x','y','z','yaw'}:
            raise ValueError('模型机位格式无效。')
        if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in goal.values()):
            raise ValueError('模型机位包含非有限数值。')
        obs = fresh['observation']
        routing = self.limits.get('navigation_strategy') == 'local_route'
        if (self.moves >= self.limits['max_moves'] or exceeds_distance(_distance(goal, obs['position']), self.limits['max_move_m']) or
                exceeds_distance(_distance(goal, self.origin), self.limits['max_radius_m']) or
                (self.limits['same_altitude'] and abs(goal['z']-self.origin['z']) > .05) or
                (not routing and _yaw_delta(goal['yaw'], obs['yaw']) > math.radians(25))):
            raise ValueError('模型机位超出已授权的移动次数、单步距离、范围、高度或转角限制。')
        if not within_bounds(goal, self.config['bounds'], planner_scope(self.config)):
            raise ValueError('模型机位超出飞行范围。')
        for zone in self.reference['photo_target']['evidence']['zones']:
            a, b, center = [obs['position'][k] for k in ('x','y')], [goal[k] for k in ('x','y')], zone['center']
            if routing:
                if any(math.hypot(p[0]-center[0], p[1]-center[1]) < zone['radius']+.02 for p in (a,b)):
                    raise ValueError('模型最终机位进入人物保护范围。')
                continue  # The local route, rather than the straight chord, must clear people.
            delta = [b[i]-a[i] for i in range(2)]
            norm2 = sum(v*v for v in delta)
            t = 0 if norm2 == 0 else max(0., min(1., sum((center[i]-a[i])*delta[i] for i in range(2))/norm2))
            if math.hypot(a[0]+t*delta[0]-center[0], a[1]+t*delta[1]-center[1]) < zone['radius']+.02:
                raise ValueError('模型机位或直连航段进入已确认人物保护范围。')

    def _action(self, kind, now, fresh=None):
        goal = copy.deepcopy(self.goal)
        observation = copy.deepcopy(fresh['observation']) if fresh is not None else None
        before_dispatch = (lambda current: self._motion_admission(current, observation)) if self.static_world_review else None
        capture = getattr(self.adapter, 'capture_control_token', None)
        if self.config.get('mode') == 'real' and not callable(capture):
            raise ValueError('实机执行接口缺少控制代次绑定，请更新任务适配器后重新确认。')
        # Capture before scheduling or waiting for command_lock. Acquiring a
        # new epoch inside the worker would forget an intervening RC takeover.
        control_token = capture() if callable(capture) else None
        if kind == 'land':
            self._landing_dispatched.clear()
        def worker():
            with getattr(self.adapter, 'command_lock', self.action_lock):
                if self.cancelled.is_set():
                    raise ValueError('摄影动作已取消。')
                current = self.adapter.snapshot()
                if current.get('command_error') and not (
                        kind == 'land' and current.get('landed') is True and current.get('armed') is False):
                    # A takeover may already have been latched between tick's
                    # snapshot and the scheduling-time token capture. Do not
                    # let land() clear that old-task failure as a new command.
                    # Any takeover after this read is covered by control_token.
                    raise ValueError(current['command_error'])
                if kind == 'goto':
                    if before_dispatch is not None:
                        before_dispatch(current)
                    elif not _fresh(fresh['observation']):
                        raise _MoveObservationInvalidated('发送机位前测量已过期，未下发目标，重新取图。')
                    if before_dispatch is None and not _pose_close(current, fresh['observation']):
                        raise _MoveObservationInvalidated('发送机位前无人机位置已变化，未下发目标，重新取图。')
                    guard = current.get('photo_guard', {})
                    if guard.get('id') != self.guard_token or guard.get('active') is not True or guard.get('error'):
                        raise ValueError('发送机位前人物保护已改变。')
                    if callable(capture):
                        options = {'before_dispatch': before_dispatch} if before_dispatch is not None else {}
                        self.adapter.goto(control_token=control_token, **options, **goal)
                    else:
                        self.adapter.goto(**goal)
                else:
                    if callable(capture):
                        self.adapter.land(control_token=control_token, before_dispatch=self._before_land_dispatch)
                    else:
                        # Non-hardware legacy/test adapters have no asynchronous
                        # publication interface. Real mode requires it above.
                        self._before_land_dispatch()
                        self.adapter.land()
            return kind
        self.action_started = now
        self.action_future = self._submit(worker, kind)

    def _begin_save(self, decision, now):
        observation = copy.deepcopy(self.review)
        composition = dict(passed=True, ready=True, source='llm', visual=copy.deepcopy(decision['visual']),
                           requirements=shot_requirements(self.task, self.shot_index), checks=copy.deepcopy(decision['checks']),
                           shot=copy.deepcopy(self.shots[self.shot_index]), shot_index=self.shot_index,
                           shot_count=len(self.shots),
                           reason=decision['reason'], task=copy.deepcopy(self.task),
                           image_sha256=hashlib.sha256(observation['data']).hexdigest(),
                           observation_id=observation['observation_id'],
                           fresh_revalidation=dict(stamp=self.last['observation']['stamp'],
                               ros_stamp=self.last['observation']['ros_stamp'],
                               observation_id=self.last['observation']['observation_id'],
                               pose=copy.deepcopy(self.last['observation']['position']),
                               subject_binding=('confirmed_stationary_person_initial_measurement' if self.stationary_subject else
                                                'unique_measured_cluster_within_frozen_guard'),
                               guard_id=self.guard_token, checked_at=time.time()))
        detector = dict(provider='photo_agent_llm', source='llm', model=self.task.get('model'),
                        local_safety_source='measured_rgb_lidar_map')
        def worker():
            if self.cancelled.is_set() or not _fresh(observation, self.limits['review_image_max_age_seconds']):
                raise ValueError('保存前摄影任务已取消或审查照片已过期。')
            metadata = self.store.save(observation, composition, detector)
            with self.lock:
                self._saved_uncommitted.add(metadata['id'])
                discarded = self.cancelled.is_set()
            if discarded:
                self._delete_orphans([metadata['id']])
                raise ValueError('取消后的迟到照片已删除。')
            return metadata
        self.save_started, self.stage = now, 'saving'
        self.save_future = self._submit(worker, 'save')

    def tick(self, snap, now):
        with self.lock:
            self.updated_at = time.time()
            if self.result is not None:
                return self.result
            if self.cancelled.is_set():
                return self._fail('摄影任务已取消。')
            try:
                if self.stage == 'landing':
                    if self.action_future and self.action_future.done():
                        self.action_future.result()
                        self.action_future = None
                    if snap.get('landed') is True and snap.get('armed') is False:
                        return self._complete()
                    if not self._landing_dispatched.is_set():
                        error = snap.get('command_error')
                        if (not error and self.config.get('mode') == 'real' and
                                snap.get('flight_mode') != 'OFFBOARD'):
                            error = '实机已退出 OFFBOARD，待发送的摄影降落已取消。'
                        if not error and (snap.get('armed') is not True or snap.get('landed') is not False):
                            error = '飞机飞行状态已改变，待发送的摄影降落已取消。'
                        if error and self.cancel_pending_landing():
                            raise ValueError(error)
                    if now-self.action_started > 30:
                        raise ValueError('原地降落未在时限内确认完成。')
                    return None
                if now >= self.deadline:
                    raise ValueError('摄影任务超过本次授权的总时长。')
                if snap.get('armed') is not True or snap.get('landed') is not False:
                    raise ValueError('摄影期间飞机不再处于飞行状态。')
                if self.config.get('mode') == 'real' and snap.get('flight_mode') != 'OFFBOARD':
                    raise ValueError('实机已退出 OFFBOARD，摄影任务已取消；恢复模式后需重新确认任务。')
                guard = snap.get('photo_guard', {})
                if guard.get('active') is not True or guard.get('id') != self.guard_token or guard.get('error'):
                    raise ValueError('已确认人物保护失效。')
                if snap.get('command_error'):
                    raise ValueError(snap['command_error'])
                if snap.get('planning_ready') is False:
                    raise ValueError('规划心跳或注册点云已失效。')
                if not self.static_world_review and self.stage == 'saving' and (
                        not _stationary(snap) or not _pose_close(snap, self.review)):
                    # The JPEG is already frozen. Finish the bounded disk job,
                    # keep safety monitoring live, and discard its uncommitted
                    # result before reviewing a new frame. Never advance shots.
                    self.review_changed = True
                    self.review_cancelled.set()
                    self.frame_wait_reason = '保存期间机位变化，当前照片不提交，完成清理后重新取图。'
                if not self.static_world_review and self.stage in ('awaiting_model', 'validating') and (
                        not _stationary(snap) or not _pose_close(snap, self.review)):
                    # Legacy profiles retain their historical review policy.
                    # Static world-frame reviews are unaffected by later motion.
                    self.review_changed = True
                    self.review_cancelled.set()
                    self.frame_wait_reason = '机位已变化，等待本轮判断结束后重新取图。'

                if self.local_future is not None:
                    monitor_limit = 2.5 if self.stationary_subject else 1.
                    if not self.local_future.done() and now-self.local_started > monitor_limit:
                        raise ValueError('本地人物监测超时。')
                    if self.local_future.done():
                        kind, future = self.local_kind, self.local_future
                        self.local_future = None
                        try:
                            fresh = future.result()
                        except Exception as exc:
                            if isinstance(exc, _ProposalRejected):
                                self._reject(self.pending_decision, str(exc), now)
                                fresh = None
                            elif isinstance(exc, PhotoFramePending) and self.stationary_subject:
                                # Retry only the lack of a NEW frame, within the
                                # existing image-freshness window. No old image
                                # can authorize a move or a new photo save.
                                self.frame_wait_reason = '等待下一张新鲜相机帧，短暂延迟后自动重试。'
                                self.next_monitor = now+.1
                                fresh = None
                            else:
                                raise
                        if fresh is not None:
                            if not _fresh(fresh['observation']) or fresh['observation']['ros_stamp'] <= self.last['observation']['ros_stamp']:
                                raise ValueError('本地人物观测已过期或复用了旧帧。')
                            self.last, self.last_monitor = fresh, now
                            self.frame_wait_reason = ''
                            if (not self.stationary_subject and self.review_measurement is not None and
                                    self.stage in ('awaiting_model', 'validating', 'saving')):
                                # Geometry/identity already passed. A visible change only
                                # invalidates this historical frame, even if the box later
                                # returns; a new model review can continue the same task.
                                old_box = self.review_measurement['measurement']['bbox']
                                if self.photo._overlap(old_box, fresh['measurement']['bbox']) < .8:
                                    self.review_changed = True
                            if (kind in ('move','capture','search_car') and
                                    fresh['observation']['stamp'] <= self.last_decision['received_at']):
                                # Recheck with a frame actually acquired after
                                # the model returned, never relabel an old one.
                                kind = 'monitor'
                            if kind in ('move','capture','search_car'):
                                defer_motion = (self.static_world_review and kind in ('move','search_car') and
                                    (not _stationary(snap) or not _pose_close(snap, fresh['observation'])))
                                if (not self.static_world_review and
                                        (not _pose_close(snap, fresh['observation']) or not _pose_close(snap, self.review))):
                                    self.review_changed = True
                                    self.review_cancelled.set()
                                if defer_motion:
                                    self._defer_decision(now, '模型决定已保留，等待稳定并更新本地观测后执行。')
                                elif self.review_cancelled.is_set():
                                    self._decision_update(delivery='discarded', accepted=False,
                                                          local_rejection='机位变化，重新取图。')
                                    self._refresh_review(now)
                                elif kind == 'move':
                                    try:
                                        self._goal_allowed(self.pending_decision['goal'], fresh)
                                    except ValueError as exc:
                                        self._reject(self.pending_decision, str(exc), now)
                                    else:
                                        self.decision_wait_reason = ''
                                        self.goal = copy.deepcopy(self.pending_decision['goal'])
                                        audit = dict(revision=self.revision, model_goal=copy.deepcopy(self.goal), executed_goal=None, delivery='pending',
                                            local_validation=copy.deepcopy(fresh['proposal_validation'].get('evidence', {})),
                                            preplanned=self.pending_decision.get('preplanned', False),
                                            after_photo_id=self.pending_decision.get('after_photo_id'),
                                            guard_id=self.guard_token, observed_stamp=fresh['observation']['stamp'])
                                        if (self.move_audit and self.move_audit[-1].get('revision') == self.revision and
                                                self.move_audit[-1].get('model_goal') == self.goal and
                                                self.move_audit[-1].get('delivery') == 'waiting_stability'):
                                            self.move_audit[-1].update(audit)
                                        else:
                                            self.move_audit.append(audit)
                                        self._decision_update(delivery='pending', waiting_reason='', local_validation=copy.deepcopy(fresh['proposal_validation'].get('evidence', {})))
                                        self._action('goto', now, fresh)
                                        self.stage, self.stable_since = 'moving', None
                                elif kind == 'search_car':
                                    self._start_car_search(snap, now, fresh)
                                else:
                                    if not _fresh(self.review, self.limits['review_image_max_age_seconds']):
                                        self._reject(self.pending_decision, '模型审查照片已超过 130 秒，必须用新图重新判断。', now)
                                    else:
                                        old = self.review_measurement['measurement']['bbox']
                                        new = fresh['measurement']['bbox']
                                        if self.review_changed or (not self.stationary_subject and self.photo._overlap(old, new) < .8):
                                            self._reject(self.pending_decision,
                                                         '人物身份与安全范围复核通过，但画面构图已变化；未保存旧图，请用新图重新判断。', now)
                                        else:
                                            self.decision_wait_reason = ''
                                            self._begin_save(self.pending_decision, now)
                if now-self.last_monitor > (2.5 if self.stationary_subject else 1.25) or not _fresh(self.last['observation']):
                    raise ValueError('等待模型时人物监测未保持新鲜。')

                self._tick_car_search(snap, now)
                if (self.stage not in ('car_search_starting','car_search_scanning','car_search_stopping') and
                        self.car_scan_future is not None and self.car_scan_future.done()):
                    # A stopped sweep owns no later candidate result.
                    self.car_scan_future = None

                if self.action_future is not None and self.action_future.done():
                    future, self.action_future = self.action_future, None
                    try:
                        action = future.result()
                    except _MoveObservationInvalidated as exc:
                        # This exception is raised only before adapter.goto.
                        # Unknown adapter failures still terminate the task;
                        # never retry a goal that might already have been sent.
                        reason = str(exc)
                        if self.static_world_review:
                            self.move_audit[-1].update(delivery='waiting_stability', local_rejection=reason)
                            self._defer_decision(now, reason)
                            return None
                        self.move_audit[-1].update(delivery='discarded', local_rejection=reason)
                        self._decision_update(delivery='discarded', accepted=False, local_rejection=reason)
                        self._history(self.pending_decision, 'rejected', reason)
                        self.review_changed = True
                        self.review_cancelled.set()
                        self.goal = None
                        self._refresh_review(now, reason)
                        return None
                    if action == 'goto':
                        self.move_audit[-1].update(executed_goal=copy.deepcopy(self.goal), delivery='delivered', delivered_at=time.time())
                        self._decision_update(delivery='delivered', waiting_reason='', executed_goal=copy.deepcopy(self.goal), delivered_at=time.time())
                        self.moves += 1
                        self._history(self.pending_decision, 'executed', '原模型机位通过当前地图和固定人物保护后已下发。')
                if self.stage == 'moving':
                    execution = snap.get('real_execution') or {}
                    self.navigation_wait_reason = ('等待可执行路径：'+str(execution['error'])
                                                   if execution.get('error') else
                                                   '等待原轨迹服务确认接收。' if execution.get('awaiting_native_receipt') else '')
                    navigation_timeout = (180. if self.limits.get('navigation_strategy') == 'local_route' else
                                          min(40., float(self.config.get('step_timeout_seconds', 30))))
                    if now-self.action_started > navigation_timeout:
                        raise ValueError('模型提出的机位未能按时到达。'+self.navigation_wait_reason)
                    if self.action_future is None and _navigation_arrived(snap, self.goal, self.config):
                        trajectory_id = execution.get('validated_trajectory_id', guard.get('validated_trajectory_id', 0))
                        self.move_audit[-1]['validated_trajectory_id'] = trajectory_id
                        self.move_audit[-1]['delivery'] = 'arrived'
                        self._decision_update(delivery='arrived', validated_trajectory_id=trajectory_id)
                        self.stage = 'settling'
                        self.navigation_wait_reason = ''
                        self._reset_observation_window()
                        self.pending_decision = None

                if self.model_future is not None:
                    if (now-self.model_started > self.limits['max_model_request_seconds'] and
                            not (self.model_future.done() and self.review_cancelled.is_set())):
                        raise ValueError('真实摄影模型超过 120 秒，迟到结果不会执行。')
                    if self.model_future.done():
                        future, self.model_future = self.model_future, None
                        try:
                            decision = future.result()
                        except PhotoReviewInvalidated:
                            self._refresh_review(now)
                            return None
                        except _ReviewObservationInvalidated as exc:
                            self._refresh_review(now, str(exc))
                            return None
                        except Exception:
                            session_cancelled = getattr(self.session, 'cancelled', None)
                            review_ended_session = (getattr(self.session, 'terminal_review_cancelled', None)
                                                    is self.review_cancelled)
                            if (self.review_cancelled.is_set() and not self.cancelled.is_set() and
                                    (not getattr(self.session, 'terminal', False) or review_ended_session) and
                                    not (session_cancelled is not None and session_cancelled.is_set())):
                                self._refresh_review(now)
                                return None
                            raise
                        if self.review_cancelled.is_set():
                            self._refresh_review(now)
                            return None
                        expected = hashlib.sha256(self.review['data']).hexdigest()
                        binding = decision.get('observation', {})
                        if binding.get('image_sha256') != expected or binding.get('observation_id') != self.review['observation_id']:
                            raise ValueError('模型结果与所审查的原始 JPEG 不一致。')
                        requirements_met = _validate_checks(decision, self.task, self.shot_index)
                        if self.task['schema_version'] == 3 and (
                                decision.get('shot_id') != self.shots[self.shot_index]['id'] or
                                decision.get('shot_index') != self.shot_index):
                            raise ValueError('模型决策绑定的照片顺序不匹配。')
                        self.last_decision = dict(copy.deepcopy(decision), revision=self.revision, received_at=time.time(),
                            reviewed_evidence=copy.deepcopy(self.reviewed_evidence), delivery='not_dispatched',
                            proposal_accepted=decision.get('accepted'), latency_seconds=round(now-self.model_started, 4))
                        self.decisions.append(copy.deepcopy(self.last_decision))
                        self.decisions = self.decisions[-self.limits['max_decisions']:]
                        self.pending_decision = decision
                        action = decision['action']
                        if action == 'abort':
                            raise ValueError(decision['reason'])
                        if action == 'search_car':
                            if decision.get('goal') is not None or decision.get('next_shot') is not None:
                                raise ValueError('车辆扫视不得携带位置目标或越过当前照片。')
                            self.stage = 'validating'
                        if action == 'move' and decision.get('accepted') is False:
                            self._reject(decision, decision.get('rejection', {}).get('reason', decision['reason']), now)
                        elif action == 'wait':
                            self._decision_update(delivery='waiting')
                            self._history(decision, 'waiting', decision['reason'])
                            self.stage, self.wait_until = 'waiting_pose', now+1.
                            self.pending_decision = None
                        elif action in ('move','capture','search_car'):
                            if action == 'capture':
                                if not requirements_met:
                                    raise ValueError('模型没有确认全部拍摄条件。')
                            if self.review_changed:
                                self._reject(decision, '人物画面构图已变化，旧判断失效；请用新图重新判断。', now)
                            else:
                                self.stage = 'validating'
                        else:
                            raise ValueError('模型动作不受支持。')

                if self.save_future is not None:
                    if (not self.save_future.done() and
                            now-self.save_started > self.config.get('photo', {}).get('save_timeout_seconds', 10.)):
                        raise ValueError('照片持久化超过时限。')
                    if self.save_future.done():
                        metadata = self.save_future.result()
                        if now-self.last_monitor > (2.5 if self.stationary_subject else 1.25):
                            raise ValueError('保存提交前安全监测已失效，未保留照片。')
                        if (self.review_changed or not _fresh(self.review, self.limits['review_image_max_age_seconds']) or
                                (not self.static_world_review and not _pose_close(snap, self.review))):
                            self.save_future = None
                            self._submit(lambda: self._delete_orphans([metadata['id']]), 'discard-photo')
                            reason = '保存期间机位或构图变化，未提交照片已作废，稳定后重新取图。'
                            self._history(self.pending_decision, 'rejected', reason)
                            self._decision_update(delivery='discarded', accepted=False, local_rejection=reason)
                            self._refresh_review(now, reason)
                            return None
                        if metadata.get('image_sha256') != hashlib.sha256(self.review['data']).hexdigest():
                            raise ValueError('保存照片与模型审查原图不同。')
                        if self.task['schema_version'] == 3:
                            advanced = self.service.commit_capture(self.session, self.task, metadata)
                            if advanced != self.shot_index+1:
                                raise ValueError('模型会话推进的照片顺序不一致。')
                        self._committed.add(metadata['id'])
                        self._decision_update(delivery='photo_committed', photo_id=metadata['id'])
                        self.photos.append(copy.deepcopy(metadata))
                        self._history(self.pending_decision, 'executed',
                                      '照片 %s 已按本轮原图保存：%s' % (self.shots[self.shot_index]['id'], metadata['id']))
                        self.shot_index += 1
                        self.car_search = None  # A different shot gets its own finite search.
                        self.save_future = None
                        if self.shot_index < len(self.shots):
                            if self._next_viewpoint(metadata):
                                return None
                            self.stage, self.pending_decision = 'settling', None
                            self.review_measurement = None
                            self._reset_observation_window()
                            return None
                        if self.task['finish'] == 'land_here':
                            self.stage = 'landing'
                            self._action('land', now)
                            return None
                        return self._complete()

                if self.stage == 'waiting_pose' and now >= self.wait_until:
                    self.stage = 'settling'
                    self._reset_observation_window()
                if (self.stage in ('settling', 'observing') and self.model_future is None and
                        self._observation_window_ready(snap, now)):
                    self._begin_model(snap, now)
                if self.local_future is None and (not self.frame_wait_reason or now >= self.next_monitor):
                    if self.stage == 'validating' and self.pending_decision is not None:
                        kind = self.pending_decision['action']
                        if self.static_world_review and kind in ('move','search_car') and not _stationary(snap):
                            self._defer_decision(now, '模型决定已保留，等待飞行速度恢复稳定后执行。')
                            self._begin_local('monitor', now)
                        else:
                            self._begin_local(kind, now)
                    elif self.stage not in ('landing','completed','failed') and now >= self.next_monitor:
                        self._begin_local('monitor', now)
                return None
            except Exception as exc:
                return self._fail(str(exc))
