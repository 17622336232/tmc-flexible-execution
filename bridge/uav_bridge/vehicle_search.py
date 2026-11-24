"""Task-local car search accounting and reviewed-direction suppression.

Directions describe already reviewed camera views, never persistent car identity
or unobserved geometry. No vehicle command or model request occurs here.
"""
import math


def angle_delta(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def image_bearing(camera, u, v):
    k, transform = camera['intrinsics'], camera['world_T_camera']
    ray = [(u-k['cx'])/k['fx'], (v-k['cy'])/k['fy'], 1.]
    direction = [sum(transform[i][j]*ray[j] for j in range(3)) for i in range(2)]
    if not all(math.isfinite(x) for x in direction) or math.hypot(*direction) < 1e-6:
        raise ValueError('当前相机投影不能提供有效水平方位，未启动车辆扫视。')
    return math.atan2(direction[1], direction[0])


def reviewed_view(camera):
    """The actual reviewed horizontal field of view, including clipped objects.

    Suppression is not a pose gate. Its only purpose is to avoid stopping again
    for the same previously inspected direction after a negative model review.
    """
    width, height = camera['width'], camera['height']
    center = image_bearing(camera, width/2, height/2)
    edges = [image_bearing(camera, width*f, height/2) for f in (0., 1.)]
    limits = [math.atan2(math.sin(edge-center), math.cos(edge-center)) for edge in edges]
    if not min(limits) < 0 < max(limits) or max(abs(v) for v in limits) >= math.pi/2:
        raise ValueError('当前相机视野方位无效，未启动车辆扫视。')
    return dict(bearing=center, min_delta=min(limits), max_delta=max(limits))


def new_car_directions(detections, camera, reviewed):
    result = []
    for item in detections:
        box = item['bbox']
        bearing = image_bearing(camera, (box[0]+box[2])*camera['width']/2,
                                (box[1]+box[3])*camera['height']/2)
        if not any(view['min_delta'] <= math.atan2(math.sin(bearing-view['bearing']),
                   math.cos(bearing-view['bearing'])) <= view['max_delta'] for view in reviewed):
            result.append(bearing)
    return result


class VehicleSearch:
    MAX_SWEEP = 2*math.pi
    MAX_SCAN_SECONDS = 30.

    def __init__(self, shot_id):
        self.shot_id = shot_id
        self.sweep_used = self.seconds_used = 0.
        self.views = []
        self.state = 'idle'
        self.owner = None
        self.started = None
        self.segment_angle = 0.
        self.segment_limit = 0.

    def context(self, supported, now):
        active_seconds = 0. if self.started is None else max(0., now-self.started)
        seconds = min(self.MAX_SCAN_SECONDS, self.seconds_used+active_seconds)
        sweep = min(self.MAX_SWEEP, self.sweep_used+self.segment_angle)
        remaining_seconds, remaining_sweep = self.MAX_SCAN_SECONDS-seconds, self.MAX_SWEEP-sweep
        exhausted = remaining_sweep <= 1e-6 or remaining_seconds < 1.
        return dict(supported=bool(supported), shot_id=self.shot_id,
                    state='exhausted' if exhausted else self.state,
                    sweep_used_rad=sweep, remaining_sweep_rad=remaining_sweep,
                    scan_seconds_used=seconds, remaining_scan_seconds=remaining_seconds,
                    reviewed_view_count=len(self.views), exhausted=exhausted)

    def review(self, camera):
        view = reviewed_view(camera)
        if not any(angle_delta(view['bearing'], old['bearing']) < 1e-4 for old in self.views):
            self.views.append(view)
        # Existing model-decision budget bounds this list as well.
        self.views = self.views[-32:]

    def start(self, owner, now):
        context = self.context(True, now)
        if context['exhausted'] or self.owner is not None:
            raise ValueError('本张车辆原地扫视的一圈或 30 秒预算已用完。')
        self.owner, self.started = owner, now
        self.segment_angle = 0.
        self.segment_limit = context['remaining_sweep_rad']
        self.state = 'scanning'
        return context['remaining_sweep_rad'], context['remaining_scan_seconds']

    def observe_status(self, status):
        if status.get('id') != self.owner:
            return
        value = status.get('angle_swept_rad')
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            self.segment_angle = max(self.segment_angle, min(self.segment_limit, max(0., value)))

    def stop(self, status, now):
        self.observe_status(status)
        self.sweep_used = min(self.MAX_SWEEP, self.sweep_used+self.segment_angle)
        self.seconds_used = min(self.MAX_SCAN_SECONDS, self.seconds_used+max(0., now-self.started))
        self.owner, self.started, self.segment_angle = None, None, 0.
        self.state = 'candidate_review'
