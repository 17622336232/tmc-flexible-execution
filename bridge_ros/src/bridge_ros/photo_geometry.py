"""Single, stationary person photography from measured RGB/LiDAR only.

The detector supplies a visible-person box; geometry cannot establish a
person's identity or prove that limbs are unoccluded.  No scene/model poses,
assumed person height, semantic map, or simulator truth are accepted here.
"""
import math

import numpy as np

from .visual_geometry import (VisualGeometryError, _array, _independent,
                              _points, _ray_count)
from .person_cluster import (associate_torso_cluster, support,
                             exterior_strips, PERSON_BBOX_UNCERTAINTY_PIXELS)


def _reject(code, message, evidence=None):
    raise VisualGeometryError(code, message, evidence)


def _xyz(point):
    return dict(zip(("x", "y", "z"), map(float, point)))


def _person_box_size_supported(width_pixels, height_pixels, min_width=24., max_aspect=6.):
    return (width_pixels >= min_width and height_pixels >= 80
            and 1.25 <= height_pixels/width_pixels <= max_aspect)


def _guard_envelope_evidence(envelope_world, raw_supported_world):
    """Bound every current protected point without moving the frozen circle.

    Eight calibrated silhouette/range corners bound the projected envelope.
    The XY AABB contains *all* original in-box range-supported returns, before
    ray de-duplication or connected-cluster filtering. Its four vertices are
    sufficient for conservative circle containment because a disk is convex.
    This is measured support, not a completed body or simulator truth.
    """
    envelope = _array(envelope_world, (8, 3), "person_envelope_world")
    raw = _points(raw_supported_world, "raw_supported_person_points")
    if not len(raw) or not np.isfinite(raw).all():
        _reject("invalid_input", "人物保护包络必须包含实际支持回波")
    return dict(source="calibrated_rgb_lidar",
                method="full_bbox_depth_envelope_and_all_raw_support_aabb",
                projection_xy=envelope[:, :2].tolist(),
                raw_xy_bounds=dict(min=raw[:, :2].min(axis=0).tolist(),
                                   max=raw[:, :2].max(axis=0).tolist()),
                raw_point_count=int(len(raw)))


def _association_anchor(bbox, median_depth, K, camera_T_world):
    """Estimate a repeatable image/range anchor, not a physical return/center.

    The torso's sampling density changes with the LiDAR scan pattern. Its
    per-axis return median is therefore not a repeatable body landmark. Use
    the center of the already-supported torso ROI and its measured optical
    depth for association only; raw surfaces/envelopes still guard motion.
    """
    box = _array(bbox, (4,), "bbox")
    intrinsic = _array(K, (3, 3), "K")
    camera_from_world = _array(camera_T_world, (4, 4), "camera_T_world")
    if (isinstance(median_depth, (bool, np.bool_))
            or not isinstance(median_depth, (int, float, np.floating))
            or not math.isfinite(median_depth) or median_depth <= .1
            or box[2] <= box[0] or box[3] <= box[1]):
        _reject("invalid_input", "人物关联锚点需要有效人物框和实测前景深度")
    pixel = np.array([box[0]+.5*(box[2]-box[0]),
                      box[1]+.39*(box[3]-box[1]), 1.])
    try:
        optical = np.linalg.solve(intrinsic, pixel)*median_depth
        world_from_camera = np.linalg.inv(camera_from_world)
    except np.linalg.LinAlgError:
        _reject("invalid_input", "人物关联锚点需要可逆的相机标定和外参")
    anchor = optical @ world_from_camera[:3, :3].T+world_from_camera[:3, 3]
    if not np.isfinite(anchor).all():
        _reject("invalid_input", "人物关联锚点计算结果无效")
    return dict(association_anchor_world=_xyz(anchor),
                association_anchor_metadata=dict(
                    source="calibrated_rgb_lidar", method="torso_roi_center_ray_median_optical_depth_v1",
                    scope="cross_frame_association_estimate_only", pixel=list(map(float, pixel[:2])),
                    depth_m=float(median_depth), is_lidar_return=False, body_center_known=False))


def photo_limits(options=None):
    options = options or {}
    limits = {}
    for key, default, low, high in (
            ("framing_margin", .08, .08, .15),
            ("target_height_fraction", .50, .45, .60),
            ("max_move_m", 1.20, .01, 1.20),
            ("min_person_clearance_m", 2.50, 1.50, 5.0),
            ("drone_radius_m", .40, .40, 1.0),
            ("measurement_margin_m", .15, .15, .50),
            ("person_motion_tolerance_m", .20, .01, .20),
            ("viewpoint_surface_allowance_m", 0., 0., .50),
            ("min_person_bbox_width_px", 24., 12., 24.),
            ("max_person_bbox_aspect", 6., 6., 12.)):
        value = options.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high):
            _reject("invalid_photo_limits", "拍照参数 %s 不在已验证安全范围内" % key)
        limits[key] = float(value)
    return limits


def measure_person(projected, bbox, camera_T_world, K, body_T_optical,
                   vehicle_position, vehicle_yaw, options=None):
    """Associate a detected person with measured spatial support.

    The deployed measured-cluster mode accepts a partial image silhouette:
    the full cloud, not image height/aspect, determines position and extent.
    The legacy torso-range mode retains its complete-box requirements.
    """
    limits = photo_limits(options)
    box = _array(bbox, (4,), "bbox")
    position = _array(vehicle_position, (3,), "vehicle_position")
    intrinsic = _array(K, (3, 3), "K")
    camera_from_world = _array(camera_T_world, (4, 4), "camera_T_world")
    body_from_camera = _array(body_T_optical, (4, 4), "body_T_optical")
    width, height = int(projected["width"]), int(projected["height"])
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    measured_cluster = (options or {}).get('person_localization') == 'measured_cluster'
    if measured_cluster:
        # Selection may happen before the aircraft has room to frame a full
        # body. Keep valid image bounds and the existing width threshold;
        # do not infer anatomical completeness from a box touching an edge.
        if (not math.isfinite(vehicle_yaw) or bw < limits['min_person_bbox_width_px']
                or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            _reject('invalid_person_box', '人物检测框无效或过小，无法关联实测位置')
    elif (not math.isfinite(vehicle_yaw) or not _person_box_size_supported(
            bw, bh, limits['min_person_bbox_width_px'], limits['max_person_bbox_aspect'])
            or x1 < .025*width or y1 < .025*height
            or x2 > .975*width or y2 > .975*height):
        _reject("person_not_complete", "人物框过小、触边或形状不支持完整站立单人，请重新观察")
    points = _points(projected["points_world"], "points_world")
    uv = _array(projected["uv"], (len(points), 2), "uv")
    depth = _array(projected["depth"], (len(points),), "depth")
    if not np.isfinite(points).all() or np.any(depth <= .1):
        _reject("invalid_input", "人物点云投影包含无效测量")
    if measured_cluster:
        from .cluster_person_geometry import measure_cluster_person
        return measure_cluster_person(projected, box, camera_from_world, intrinsic,
                                      body_from_camera, position, vehicle_yaw, limits, options)
    raw_uv, raw_depth, raw_points = uv, depth, points
    independent = _independent(uv, depth)
    uv, depth, points = uv[independent], depth[independent], points[independent]
    rel = (uv-[x1, y1])/[bw, bh]
    # Head/legs are excluded from range estimation, not from composition.
    torso = ((rel[:, 0] >= 1./6) & (rel[:, 0] <= 5./6)
             & (rel[:, 1] >= .20) & (rel[:, 1] <= .58))
    tuv, td, tp = uv[torso], depth[torso], points[torso]
    evidence = {"bbox_pixels": list(map(float, box)), "torso_point_count": _ray_count(tuv),
                "torso_roi_relative": [1./6, .20, 5./6, .58],
                "source": "rgb_lidar", "pose_validation": False,
                "full_body_proven": False, "stationary_person_required": True}
    evidence["torso_coverage"] = dict(columns=len(np.unique(np.floor(tuv[:, 0]/2))) if len(tuv) else 0,
                                        rows=len(np.unique(np.floor(tuv[:, 1]/2))) if len(tuv) else 0,
                                        horizontal_fraction=float(np.ptp(tuv[:, 0])/((2./3)*bw)) if len(tuv) else 0.,
                                        vertical_fraction=float(np.ptp(tuv[:, 1])/(.38*bh)) if len(tuv) else 0.)
    evidence.update(measured_depth_m=float(np.median(td)) if len(td) else None,
                    depth_spread_m=float(np.percentile(td, 95)-np.percentile(td, 5)) if len(td) else None)
    if evidence["torso_point_count"] < 3:
        _reject("person_range_sparse", "人物上身的独立雷达回波不足，不能借用背景距离", evidence)
    unique_x = len(np.unique(np.floor(tuv[:, 0]/2)))
    unique_y = len(np.unique(np.floor(tuv[:, 1]/2)))
    span = np.ptp(tuv, axis=0)/[(2./3)*bw, .38*bh]
    evidence["torso_coverage"] = dict(columns=unique_x, rows=unique_y,
                                       horizontal_fraction=float(span[0]), vertical_fraction=float(span[1]))
    # Use the deployed 3-D cluster's point/extent support. A narrow torso can
    # contain just one column of a finite angular scan even though the full
    # measured object has adequate horizontal and vertical extent.
    try:
        selected, cluster = associate_torso_cluster(
            uv, depth, points, box, torso, projected["points_world_all"], options)
    except VisualGeometryError as exc:
        evidence.update(exc.evidence)
        _reject(exc.code, str(exc), evidence)
    tuv, td, tp = uv[selected], depth[selected], points[selected]
    selected_support = support(tuv, td, box)
    evidence.update(torso_point_count=selected_support["point_count"], person_cluster=cluster)
    evidence["torso_coverage"] = {key: selected_support[key] for key in
                                  ("columns", "rows", "horizontal_fraction", "vertical_fraction")}
    median = float(np.median(td))
    tolerance = max(.25, .065*median)
    spread = float(np.percentile(td, 95)-np.percentile(td, 5))
    if (_ray_count(tuv[np.abs(td-median) > tolerance]) >= 2
            or spread > max(.35, .10*median)):
        _reject("person_range_ambiguous", "人物上身混入多个距离表面，拒绝生成拍摄点", evidence)
    in_box = ((rel >= 0) & (rel <= 1)).all(axis=1)
    raw_rel = (raw_uv-[x1, y1])/[bw, bh]
    raw_in_box = ((raw_rel >= 0) & (raw_rel <= 1)).all(axis=1)
    # Ray de-duplication establishes independent support only. Within a .10 m
    # range bin the nearest original return may cross the occlusion boundary;
    # it must not disappear because a slightly farther sample was seen first.
    if np.any(raw_in_box & (raw_depth < median-tolerance)):
        _reject("person_occluded", "人物框内有更近的遮挡物，无法安全确认完整人物", evidence)
    background = {}
    for name, strip in exterior_strips(uv, box).items():
        # A detected box can cut a few pixels into an arm/shoulder. Returns
        # inside its existing 4px uncertainty band are neither affirmative
        # background nor contradictory foreground. Exclude near/far alike;
        # do not expand the strip or alter full-box occlusion/raw support.
        behind = strip & (depth >= median+max(.35, .10*median))
        n, b = _ray_count(uv[strip]), _ray_count(uv[behind])
        fraction = b/max(1, b+_ray_count(uv[strip & ~behind]))
        background[name] = {"point_count": n, "behind_count": b, "behind_fraction": fraction}
    evidence["background_separation"] = background
    evidence["bbox_uncertainty_pixels"] = PERSON_BBOX_UNCERTAINTY_PIXELS
    surface = np.median(tp, axis=0)
    evidence.update(_association_anchor(box, median, intrinsic, camera_from_world))
    supported = in_box & (np.abs(depth-median) <= tolerance)
    raw_supported = raw_in_box & (np.abs(raw_depth-median) <= tolerance)
    target_points = raw_points[raw_supported]
    nearest = float(np.min(np.linalg.norm(target_points[:, :2]-position[:2], axis=1)))
    uncertainty = max(.12, spread*.5+.06)
    # Expand the RGB envelope by 4 pixels and propagate measured range error.
    expanded = [x1-PERSON_BBOX_UNCERTAINTY_PIXELS, y1-PERSON_BBOX_UNCERTAINTY_PIXELS,
                x2+PERSON_BBOX_UNCERTAINTY_PIXELS, y2+PERSON_BBOX_UNCERTAINTY_PIXELS]
    pixels = np.array([[expanded[ix], expanded[iy], 1.]
                       for ix, iy in ((0, 1), (0, 3), (2, 1), (2, 3))])
    rays = pixels @ np.linalg.inv(intrinsic).T
    optical = np.concatenate((rays*(median-uncertainty), rays*(median+uncertainty)))
    world_from_camera = np.linalg.inv(camera_from_world)
    envelope = optical @ world_from_camera[:3, :3].T+world_from_camera[:3, 3]
    frame_pixels = np.array([[box[ix], box[iy], 1.]
                             for ix, iy in ((0, 1), (0, 3), (2, 1), (2, 3))])
    frame_rays = frame_pixels @ np.linalg.inv(intrinsic).T
    frame_optical = np.concatenate((frame_rays*(median-uncertainty), frame_rays*(median+uncertainty)))
    framing_envelope = frame_optical @ world_from_camera[:3, :3].T+world_from_camera[:3, 3]
    # The horizontal circle encloses the full detected silhouette envelope,
    # not merely the torso samples; add body radius and measurement margin.
    person_radius = max(float(np.max(np.linalg.norm(envelope[:, :2]-surface[:2], axis=1))),
                        float(np.max(np.linalg.norm(target_points[:, :2]-surface[:2], axis=1))))
    required_guard_radius = (person_radius+limits["drone_radius_m"]
                             +limits["min_person_clearance_m"]+limits["measurement_margin_m"])
    # A changing viewpoint measures different physical body surfaces. Reserve
    # extra room in the ORIGINAL circle before flight, rather than moving or
    # shrinking that circle as the camera travels. This is safety margin only;
    # it does not invent missing points or a body center for the VLM.
    guard_radius = (required_guard_radius+limits["person_motion_tolerance_m"]
                    +limits['viewpoint_surface_allowance_m'])
    measured_distance = float(np.linalg.norm(position[:2]-surface[:2]))
    cluster.update(uncertainty_m=uncertainty)
    evidence.update(measured_depth_m=median, depth_spread_m=spread,
                    depth_uncertainty_m=uncertainty, person_surface_position=_xyz(surface),
                    photo_guard_envelope_v1=_guard_envelope_evidence(envelope, target_points),
                    nearest_person_return_body_distance_m=nearest,
                    measured_body_distance_m=measured_distance,
                    person_envelope_radius_m=person_radius,
                    body_radius_m=limits["drone_radius_m"],
                    min_person_clearance_m=limits["min_person_clearance_m"],
                    measurement_margin_m=limits["measurement_margin_m"], photo_guard_radius_m=guard_radius,
                    photo_required_guard_radius_m=required_guard_radius,
                    person_motion_allowance_m=limits["person_motion_tolerance_m"],
                    viewpoint_surface_allowance_m=limits['viewpoint_surface_allowance_m'],
                    full_body_evidence="detector_bbox_only")
    return dict(evidence=evidence, envelope=envelope, framing_envelope=framing_envelope, surface=surface,
                body_T_optical=body_from_camera, K=intrinsic, bbox=box,
                position=position, yaw=float(vehicle_yaw), width=width, height=height, limits=limits,
                points_world_all=_points(projected["points_world_all"], "points_world_all"))


def _camera_at(person, position, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    world_from_body = np.array([[c, -s, 0, position[0]], [s, c, 0, position[1]],
                                [0, 0, 1, position[2]], [0, 0, 0, 1.]])
    return world_from_body @ person["body_T_optical"]


def _predicted_box(person, position, yaw, envelope=None):
    camera_from_world = np.linalg.inv(_camera_at(person, position, yaw))
    envelope = person["envelope"] if envelope is None else envelope
    optical = envelope @ camera_from_world[:3, :3].T+camera_from_world[:3, 3]
    if np.any(optical[:, 2] <= .1):
        return None
    uv = (optical/optical[:, 2:3]) @ person["K"].T
    return [float(np.min(uv[:, 0])/person["width"]), float(np.min(uv[:, 1])/person["height"]),
            float(np.max(uv[:, 0])/person["width"]), float(np.max(uv[:, 1])/person["height"])]


def resolve_photo(person, margin=None, target_height=None, max_move=None):
    """Keep altitude; compose by an optional short approach and body yaw only.

    No automatic backwards motion, climbing, orbiting, or hidden-space search.
    Existing satisfactory framing can produce a zero-distance target.  This
    function proves neither route observability nor trajectory safety; callers
    must query the map and install the complete-trajectory person guard.
    """
    limits = person.get("limits", photo_limits())
    margin = limits["framing_margin"] if margin is None else margin
    target_height = limits["target_height_fraction"] if target_height is None else target_height
    max_move = limits["max_move_m"] if max_move is None else max_move
    if not (.08 <= margin <= .15 and .45 <= target_height <= .60 and 0 <= max_move <= 1.2):
        _reject("invalid_photo_limits", "拍照构图参数不在已验证范围内")
    surface, position = person["surface"], person["position"]
    evidence = dict(person["evidence"])
    direction = surface[:2]-position[:2]
    distance = float(np.linalg.norm(direction))
    if distance <= evidence["photo_guard_radius_m"]:
        _reject("person_too_close", "当前人与旋翼的保守间距不足，禁止自动拍摄移动", evidence)
    direction /= distance
    forward = person["body_T_optical"][:3, 2]
    if np.linalg.norm(forward[:2]) < .70:
        _reject("unsupported_camera_orientation", "当前拍照仅支持朝前的固定相机")
    camera_heading = math.atan2(forward[1], forward[0])
    candidates = []
    for movement in np.linspace(0, max_move, max(2, int(math.ceil(max_move/.025))+1)):
        endpoint = position.copy()
        endpoint[:2] += movement*direction
        yaw = math.atan2(direction[1], direction[0])-camera_heading
        # Account for lateral camera translation when centering the subject.
        for _ in range(3):
            camera = _camera_at(person, endpoint, yaw)[:3, 3]
            yaw = math.atan2(surface[1]-camera[1], surface[0]-camera[0])-camera_heading
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        if abs(math.atan2(math.sin(yaw-person["yaw"]), math.cos(yaw-person["yaw"]))) > math.radians(25):
            continue
        body_distance = float(np.linalg.norm(endpoint[:2]-surface[:2]))
        if body_distance < evidence["photo_guard_radius_m"]+.025:
            continue
        safety_box = _predicted_box(person, endpoint, yaw)
        if safety_box is None or min(safety_box[0], safety_box[1], 1-safety_box[2], 1-safety_box[3]) < margin:
            continue
        clearance = float(np.min(np.linalg.norm(person["points_world_all"]-endpoint, axis=1)))
        if clearance < .6:
            continue
        predicted = _predicted_box(person, endpoint, yaw, person["framing_envelope"])
        range_boxes = [_predicted_box(person, endpoint, yaw, person["framing_envelope"][index:index+4])
                       for index in (0, 4)]
        # Pixel uncertainty is reserved for margins, never counted as a larger
        # subject. Use the smaller height across the measured depth interval.
        height = min(item[3]-item[1] for item in range_boxes)
        if height > .80:
            continue
        candidates.append((movement, endpoint, yaw, predicted, height, clearance, body_distance, safety_box))
        if height >= target_height:
            break
    if not candidates:
        _reject("photo_framing_unreachable", "保持当前高度与安全间距时无法完整构图，请重新布置拍摄位置", evidence)
    candidate = candidates[-1]
    if candidate[4] < target_height:
        _reject("person_too_small", "安全范围内人物仍太小，无法获得已验收的全身取景", evidence)
    movement, endpoint, yaw, box, height, clearance, body_distance, safety_box = candidate
    yaw_change = abs(math.atan2(math.sin(yaw-person["yaw"]), math.cos(yaw-person["yaw"])))
    requires_motion = bool(movement > .025 or yaw_change > math.radians(2))
    evidence.update(photo=True, predicted_bbox=box, predicted_height_fraction=height,
                    predicted_safety_bbox=safety_box,
                    framing_margin=margin, target_height_fraction=target_height,
                    move_m=float(movement), max_move_m=max_move,
                    endpoint_clearance_m=clearance, endpoint_person_body_distance_m=body_distance,
                    requires_motion=requires_motion, automatic_backwards_motion=False,
                    altitude_preserved=True, complete_trajectory_guard_required=True)
    return dict(goal=dict(_xyz(endpoint), yaw=float(yaw)), evidence=evidence,
                requires_motion=requires_motion)


def validate_proposed_photo(person, goal, route_navigation=False):
    """Validate a model's exact world goal; never replace it with our own goal.

    Translation may be forward, backward or sideways within the same bounded
    measured corridor. This geometric test is not a map or trajectory proof;
    the adapter must check the full envelope and arm the trajectory guard.
    Composition belongs to the model/task: local validation preserves enough
    complete-person visibility for measurement, without imposing centering,
    portrait size or an aesthetic border on the requested composition.
    """
    if (not isinstance(goal, dict) or set(goal) != {"x", "y", "z", "yaw"}
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in goal.values()) or not -math.pi <= goal["yaw"] <= math.pi):
        _reject("invalid_photo_goal", "模型机位必须为有限 world x/y/z/yaw，不能包含额外控制字段")
    position = person["position"]
    endpoint = np.asarray([goal[axis] for axis in ("x", "y", "z")], dtype=float)
    movement = float(np.linalg.norm(endpoint-position))
    limits, evidence = person["limits"], dict(person["evidence"])
    if route_navigation:
        # A distant view is a navigation intention, not a straight corridor.
        # Old-view silhouette projections cannot certify a new body aspect.
        # Current person geometry is measured above; the local route planner
        # and controller validate all actual motion against map/people.
        distance = float(np.linalg.norm(endpoint[:2]-person['surface'][:2]))
        if movement > 70.:
            _reject('photo_goal_out_of_bounds', '最终机位必须在场地范围内', evidence)
        if distance < evidence['photo_guard_radius_m']+.025:
            _reject('person_too_close', '最终摄影机位进入人物保护范围', evidence)
        evidence.update(photo=True, model_goal_used=True, goal_replaced=False,
                        navigation_strategy='local_route', move_m=movement,
                        altitude_preserved=abs(endpoint[2]-position[2]) <= .05, endpoint_person_body_distance_m=distance,
                        local_route_required=True, complete_trajectory_guard_required=True)
        return dict(goal=dict(goal), evidence=evidence, requires_motion=True)
    if movement > limits["max_move_m"]+1e-9 or abs(endpoint[2]-position[2]) > .05:
        _reject("photo_goal_out_of_bounds", "模型机位超过单次短程范围或未保持当前高度", evidence)
    yaw_delta = math.atan2(math.sin(goal["yaw"]-person["yaw"]), math.cos(goal["yaw"]-person["yaw"]))
    if abs(yaw_delta) > math.radians(25):
        _reject("photo_goal_loses_subject", "模型转向过大，不能保持已选人物可见", evidence)
    for ratio in np.linspace(0, 1, max(2, int(math.ceil(movement/.075))+1)):
        point = position+ratio*(endpoint-position)
        distance = float(np.linalg.norm(point[:2]-person["surface"][:2]))
        if distance < evidence["photo_guard_radius_m"]+.025:
            _reject("person_too_close", "模型候选航段侵入人物完整保护范围", evidence)
        box = _predicted_box(person, point, person["yaw"]+ratio*yaw_delta)
        if box is None or min(box[0], box[1], 1-box[2], 1-box[3]) < .025:
            _reject("photo_goal_loses_subject", "模型候选航段无法保持已选人物完整可见", evidence)
        # Check the unexpanded silhouette at BOTH measured depth bounds. The
        # four-pixel safety border must not inflate a too-small measurement.
        # These are the same sensor-association minima as measure_person,
        # not a portrait-size preference or a guarantee of future returns.
        for i in (0, 4):
            measured_box = _predicted_box(person, point, person["yaw"]+ratio*yaw_delta,
                                           person["framing_envelope"][i:i+4])
            if (measured_box is None or not _person_box_size_supported(
                    (measured_box[2]-measured_box[0])*person["width"],
                    (measured_box[3]-measured_box[1])*person["height"],
                    limits['min_person_bbox_width_px'], limits['max_person_bbox_aspect'])):
                _reject("photo_goal_not_measurable", "模型候选航段人物框过小或比例不支持可靠测距，测量条件不足",
                        dict(evidence, predicted_measurement_bbox=measured_box,
                             minimum_measured_bbox_pixels=[limits['min_person_bbox_width_px'], 80],
                             measured_bbox_aspect_range=[1.25, limits['max_person_bbox_aspect']]))
    safety_box = _predicted_box(person, endpoint, goal["yaw"])
    predicted = _predicted_box(person, endpoint, goal["yaw"], person["framing_envelope"])
    range_boxes = [_predicted_box(person, endpoint, goal["yaw"], person["framing_envelope"][i:i+4])
                   for i in (0, 4)]
    height = min(box[3]-box[1] for box in range_boxes)
    center = [.5*(predicted[0]+predicted[2]), .5*(predicted[1]+predicted[3])]
    clearance = float(np.min(np.linalg.norm(person["points_world_all"]-endpoint, axis=1)))
    if clearance < .6:
        _reject("photo_goal_obstructed", "模型机位靠近实测障碍物", evidence)
    requires_motion = movement > .025 or abs(yaw_delta) > math.radians(2)
    evidence.update(photo=True, model_goal_used=True, goal_replaced=False,
                    predicted_bbox=predicted, predicted_safety_bbox=safety_box,
                    predicted_height_fraction=height, predicted_center=center,
                    subject_visibility_margin=.025, composition_evaluation="model_task_requirements",
                    minimum_measured_bbox_pixels=[limits['min_person_bbox_width_px'], 80],
                    measured_bbox_aspect_range=[1.25, limits['max_person_bbox_aspect']],
                    move_m=movement, max_move_m=limits["max_move_m"], requires_motion=requires_motion,
                    endpoint_clearance_m=clearance,
                    endpoint_person_body_distance_m=float(np.linalg.norm(endpoint[:2]-person["surface"][:2])),
                    altitude_preserved=True, observed_subject_required_along_corridor=True,
                    complete_trajectory_guard_required=True)
    return dict(goal=dict(goal), evidence=evidence, requires_motion=requires_motion)
