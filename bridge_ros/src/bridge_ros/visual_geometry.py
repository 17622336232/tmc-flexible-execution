"""Conservative RGB bounding-box grounding using only measured point clouds.

No simulator object poses, scene files, or synthetic depth are used here.  These
checks establish geometric support for a surface *inside* a visual detection;
they cannot prove its semantic identity or clear the flight path.  An obstacle
aware planner and fresh, explicitly modelled, time-aligned measurements remain required.
"""

import math

import numpy as np


class VisualGeometryError(ValueError):
    """A rejected observation, with a stable code and JSON-compatible evidence."""

    def __init__(self, code, message, evidence=None):
        super().__init__(message)
        self.code = str(code)
        self.evidence = dict(evidence or {})


def _reject(code, message, evidence=None):
    raise VisualGeometryError(code, message, evidence)


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        _reject("invalid_input", "%s 必须为有效数值数组" % name)
    if result.shape != shape or not np.isfinite(result).all():
        _reject("invalid_input", "%s 的形状或数值无效" % name)
    return result


def _dimension(value, name):
    if isinstance(value, (bool, np.bool_)):
        _reject("invalid_input", "%s 必须为正整数" % name)
    try:
        valid = math.isfinite(float(value)) and int(value) == value and int(value) > 0
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        _reject("invalid_input", "%s 必须为正整数" % name)
    return int(value)


def _points(value, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        _reject("invalid_input", "%s 必须为 N×3 数值数组" % name)
    if result.ndim != 2 or result.shape[1] != 3:
        _reject("invalid_input", "%s 必须为 N×3 数值数组" % name)
    return result


def _distortion_radius_limit(coefficients):
    """Conservative injective disk of the Brown-Conrady projection.

    A polynomial lens model is only meaningful on its branch around the
    optical axis: e.g. barrel distortion can fold a far off-axis ray back into
    an otherwise valid pixel. The radial Jacobian eigenvalues are N/D and
    [ND+r(N'D-ND')]/D**2. Tangential distortion perturbs either eigenvalue by
    at most 6*r*hypot(p1,p2). The first positive zero of either lower bound (or
    D) therefore bounds a disk on which the full Jacobian is positive definite.
    In particular, checking just the derivative at the final ray is not enough:
    a polynomial may have folded twice before that ray.
    """
    poly = np.polynomial.polynomial
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    numerator = np.array([1., 0, k1, 0, k2, 0, k3])
    denominator = np.array([1., 0, k4, 0, k5, 0, k6])
    perturbation = np.array([0., 6*math.hypot(p1, p2)])
    with np.errstate(over="ignore", invalid="ignore"):
        first_bound = poly.polysub(numerator, poly.polymul(perturbation, denominator))
        second_bound = poly.polyadd(
            poly.polymul(numerator, denominator),
            poly.polymul([0., 1.], poly.polysub(
                poly.polymul(poly.polyder(numerator), denominator),
                poly.polymul(numerator, poly.polyder(denominator)))))
        second_bound = poly.polysub(second_bound,
                                    poly.polymul(perturbation,
                                                 poly.polymul(denominator, denominator)))
    limit = float("inf")
    for polynomial in (denominator, first_bound, second_bound):
        if not np.isfinite(polynomial).all():
            _reject("invalid_calibration", "相机畸变系数数值不稳定")
        trimmed = poly.polytrim(polynomial)
        if len(trimmed) <= 1:
            continue
        try:
            roots = poly.polyroots(trimmed)
        except np.linalg.LinAlgError:
            _reject("invalid_calibration", "无法验证相机畸变的有效投影范围")
        if not np.isfinite(roots).all():
            _reject("invalid_calibration", "无法验证相机畸变的有效投影范围")
        for root in roots:
            # Treat near-real complex pairs conservatively as boundaries;
            # repeated polynomial roots can acquire a small imaginary part.
            if root.real > 0 and abs(root.imag) <= 1e-3*(1+abs(root.real)):
                limit = min(limit, float(root.real))
    return limit*(1-1e-6)


def build_projection(points_world, camera_T_world, K, width, height,
                     distortion=None, distortion_model="plumb_bob"):
    """Project measured world-frame points into an explicitly modelled RGB image.

    ``camera_T_world`` maps world coordinates to camera OPTICAL coordinates
    (+x right, +y down, +z forward), not the inverse or a ROS camera-link frame.
    ``K`` must be the intrinsic matrix for this exact image size. Brown-Conrady
    ``plumb_bob`` distortion accepts 0, 4 or 5 coefficients (k1,k2,p1,p2,k3);
    ``rational_polynomial`` accepts 8 (then k4,k5,k6). ``elastic_pinhole``
    accepts only empty D: it explicitly retains the original uncorrected
    pinhole approximation, not a claim of calibrated zero distortion.
    Unknown models fail closed. For a rectified image pass zero distortion
    and rectified intrinsics.
    Extrinsics, frame/time validity and cloud provenance are caller duties.

    Invalid returns, depths <= 0.1 m and rays beyond the distortion model
    injective main branch are removed from the projection.
    All finite measured points are retained separately for endpoint clearance.
    The returned pixel ``uv``, positive optical ``depth``, and ``points_world``
    rows correspond one-to-one.  No raster interpolation or invented depth is
    performed.  Arrays are owned copies, so caller mutation cannot alter them.
    """
    width, height = _dimension(width, "width"), _dimension(height, "height")
    points = _points(points_world, "points_world")
    transform = _array(camera_T_world, (4, 4), "camera_T_world")
    intrinsic = _array(K, (3, 3), "K")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0)):
        _reject("invalid_calibration", "相机外参必须为有效刚体变换")
    if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-9, rtol=0)
            or abs(intrinsic[1, 0]) > 1e-9):
        _reject("invalid_calibration", "相机内参必须为有效针孔相机矩阵")
    try:
        coefficients = np.asarray([] if distortion is None else distortion, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        _reject("invalid_calibration", "畸变系数必须为有限数值数组")
    permitted = {"plumb_bob": (0, 4, 5), "rational_polynomial": (8,),
                 "elastic_pinhole": (0,)}
    if (distortion_model not in permitted or coefficients.ndim != 1
            or len(coefficients) not in permitted[distortion_model]
            or not np.isfinite(coefficients).all()):
        _reject("unsupported_distortion", "不支持此相机畸变模型或系数格式")
    all_points = points[np.isfinite(points).all(axis=1)].copy()
    camera_points = all_points @ rotation.T + transform[:3, 3]
    in_front = camera_points[:, 2] > 0.1
    camera_points, visible_points = camera_points[in_front], all_points[in_front]
    xy = camera_points[:, :2] / camera_points[:, 2:3]
    if len(coefficients):
        distortion_terms = np.zeros(8, dtype=np.float64)
        distortion_terms[:len(coefficients)] = coefficients
        k1, k2, p1, p2, k3, k4, k5, k6 = distortion_terms
        radius_limit = _distortion_radius_limit(distortion_terms)
        valid_branch = np.linalg.norm(xy, axis=1) < radius_limit
        xy, camera_points = xy[valid_branch], camera_points[valid_branch]
        visible_points = visible_points[valid_branch]
        x, y = xy[:, 0], xy[:, 1]
        radius2 = x*x+y*y
        radius4, radius6 = radius2*radius2, radius2*radius2*radius2
        denominator = 1+k4*radius2+k5*radius4+k6*radius6
        if np.any(denominator <= 1e-6):
            _reject("invalid_calibration", "相机畸变模型在观测射线上无有效投影")
        radial = (1+k1*radius2+k2*radius4+k3*radius6)/denominator
        xy = np.column_stack((x*radial+2*p1*x*y+p2*(radius2+2*x*x),
                              y*radial+p1*(radius2+2*y*y)+2*p2*x*y))
    homogeneous = np.column_stack((xy, np.ones(len(xy)))) @ intrinsic.T
    uv = homogeneous[:, :2]
    in_frame = (np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    return {"uv": uv[in_frame].copy(),
            "depth": camera_points[in_frame, 2].copy(),
            "points_world": visible_points[in_frame].copy(),
            "points_world_all": all_points,
            "width": width, "height": height}


def _independent(uv, depth):
    """Count unique 2-pixel rays; retain different ranges on the same ray.

    Replayed/duplicated points must not create support.  Range bins preserve
    depth conflicts instead of a z-buffer silently hiding a second surface.
    """
    if len(uv) == 0:
        return np.empty(0, dtype=np.int64)
    bins = np.column_stack((np.floor(uv / 2.0), np.floor(depth / 0.10)))
    return np.unique(bins, axis=0, return_index=True)[1]


def _ray_count(uv):
    if len(uv) == 0:
        return 0
    return int(len(np.unique(np.floor(uv / 2.0), axis=0)))


def _position_dict(point):
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}


def resolve_bbox(projected, bbox, vehicle_position, yaw, standoff_m=1.2,
                 max_move_m=3.0):
    """Resolve PIXEL ``[x1, y1, x2, y2]`` bbox to a conservative world goal.

    Normalized boxes are deliberately NOT guessed or converted.  Boxes must be
    in bounds and at least 16 pixels in each dimension.  ``vehicle_position``
    is world-frame [x,y,z], and ``yaw`` is its finite current heading in radians.

    Acceptance needs >=12 independent central rays, 2-D center coverage,
    coherent central depth, no nearer conflicting box surface, and >=3 rays
    on EACH left/right exterior strip, of which >=75% are demonstrably behind
    the core surface.  Thus missing foreground returns in front of a flat wall
    cannot pass by borrowing wall depth.  Missing background evidence also
    fails closed; thin, occluded, edge-of-frame or sparse targets may be refused.

    The goal keeps the aircraft's altitude and faces the supported surface.
    It leaves the requested horizontal standoff (at least 1.2 m), plus 0.15 m
    measurement margin, from every accepted target return.  Travel over 3 m
    is rejected, never silently clamped.  Any measured point within 0.6 m of
    the endpoint also rejects it.  Clearance is only of OBSERVED points, not a
    claim that unobserved space or the path is free.  The caller must use its
    normal obstacle-aware planner and revalidate the observation before flight.

    Returns JSON-compatible {goal: {x,y,z,yaw}, evidence: {...}}.  Rejections
    raise VisualGeometryError with ``code`` and partial ``evidence``.
    """
    if not isinstance(projected, dict):
        _reject("invalid_input", "点云投影格式无效")
    try:
        width = _dimension(projected["width"], "width")
        height = _dimension(projected["height"], "height")
        world = _points(projected["points_world"], "points_world")
        uv = _array(projected["uv"], (len(world), 2), "uv")
        depth = _array(projected["depth"], (len(world),), "depth")
        all_world = _points(projected["points_world_all"], "points_world_all")
    except KeyError as exc:
        _reject("invalid_input", "点云投影缺少字段 %s" % str(exc))
    if (not np.isfinite(world).all() or not np.isfinite(all_world).all()
            or np.any(depth <= 0.1)
            or np.any(uv < 0) or np.any(uv[:, 0] >= width)
            or np.any(uv[:, 1] >= height)):
        _reject("invalid_input", "点云投影包含无效坐标")
    box = _array(bbox, (4,), "bbox (pixel xyxy)")
    position = _array(vehicle_position, (3,), "vehicle_position")
    try:
        yaw, standoff_m, max_move_m = float(yaw), float(standoff_m), float(max_move_m)
    except (ValueError, TypeError, OverflowError):
        _reject("invalid_input", "航向和安全距离必须为有限数值")
    if (not all(math.isfinite(v) for v in (yaw, standoff_m, max_move_m))
            or standoff_m < 1.2 or not 0 < max_move_m <= 3.0):
        _reject("invalid_limits", "站距至少为 1.2 米，单次移动上限不得超过 3 米")
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or bw < 16 or bh < 16:
        _reject("invalid_bbox", "目标框必须在图像内且宽高至少为 16 像素")
    center = np.array([(x1+x2)/2, (y1+y2)/2])
    dimensions = np.array([bw, bh])
    raw_uv, raw_depth, raw_world = uv, depth, world
    index = _independent(uv, depth)
    uv, depth, world = uv[index], depth[index], world[index]
    relative = (uv-center)/dimensions
    in_box = (np.abs(relative) <= 0.5).all(axis=1)
    in_core = (np.abs(relative) <= 0.30).all(axis=1)
    core_uv, core_depth, core_world = uv[in_core], depth[in_core], world[in_core]
    evidence = {"bbox_pixels": [float(v) for v in box],
                "point_count": _ray_count(core_uv),
                "bbox_point_count": _ray_count(uv[in_box]),
                "standoff_m": standoff_m, "measurement_margin_m": 0.15,
                "max_move_m": max_move_m,
                "endpoint_clearance_required_m": 0.6}
    if evidence["point_count"] < 12:
        _reject("insufficient_core_points", "目标框中心的独立雷达回波不足", evidence)
    core_relative = relative[in_core]
    grid = np.clip(np.floor((core_relative+0.30)/0.20), 0, 2).astype(int)
    occupied = np.unique(grid, axis=0)
    span = np.ptp(core_relative, axis=0)/0.60
    central_count = _ray_count(core_uv[(np.abs(core_relative) <= 0.15).all(axis=1)])
    quadrants = [bool(np.any((core_relative[:, 0]*sx > 0)
                            & (core_relative[:, 1]*sy > 0)))
                 for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
    evidence["core_coverage"] = {
        "occupied_cells": int(len(occupied)), "total_cells": 9,
        "horizontal_fraction": float(span[0]), "vertical_fraction": float(span[1]),
        "center_ray_count": central_count, "quadrants": int(sum(quadrants))}
    if (len(occupied) < 5 or np.any(span < 0.45)
            or central_count < 3 or not all(quadrants)):
        _reject("insufficient_center_coverage", "雷达回波没有充分覆盖目标中心", evidence)
    median_depth = float(np.median(core_depth))
    depth_tolerance = max(0.35, 0.08*median_depth)
    depth_span = float(np.percentile(core_depth, 95)-np.percentile(core_depth, 5))
    evidence.update({"robust_depth_m": median_depth, "depth_span_m": depth_span,
                     "depth_tolerance_m": depth_tolerance})
    # Two independent rays at another depth are a conflict, even a minority.
    far_core = np.abs(core_depth-median_depth) > depth_tolerance
    if (_ray_count(core_uv[far_core]) >= 2
            or depth_span > max(0.45, 0.12*median_depth)):
        _reject("split_depth", "目标中心存在多个深度表面，无法可靠关联物体", evidence)
    # A nearer object near the bbox edge cannot be ignored in favour of a wall
    # that happens to cover the central pixels.
    nearer = in_box & (depth < median_depth-depth_tolerance)
    if _ray_count(uv[nearer]) >= 1:
        _reject("foreground_conflict", "目标框内存在更近的回波，中心深度可能属于背景", evidence)
    background_gap = max(0.35, 0.10*median_depth)
    background = {}
    for label, sign in (("left", -1), ("right", 1)):
        strip = ((relative[:, 0]*sign > 0.50) & (relative[:, 0]*sign <= 0.80)
                 & (np.abs(relative[:, 1]) <= 0.40))
        total = _ray_count(uv[strip])
        behind = strip & (depth >= median_depth+background_gap)
        behind_count = _ray_count(uv[behind])
        # Count non-background rays independently too, so multiple returns on
        # one ray cannot create a misleading fraction above 100%.
        not_behind_count = _ray_count(uv[strip & ~behind])
        fraction = behind_count / max(1, behind_count+not_behind_count)
        background[label] = {"point_count": total, "behind_count": behind_count,
                             "behind_fraction": float(fraction)}
    evidence["background_separation"] = background
    evidence["background_gap_required_m"] = background_gap
    if any(side["behind_count"] < 3 or side["behind_fraction"] < 0.75
           for side in background.values()):
        _reject("unproven_foreground", "目标两侧缺少更远背景回波，不能排除把后墙当作目标", evidence)
    supported = in_box & (np.abs(depth-median_depth) <= depth_tolerance)
    # Use every measured target return for distance guarantees. Deduplication
    # above is only for evidence counts and must not hide a closer raw sample.
    raw_relative = (raw_uv-center)/dimensions
    raw_supported = ((np.abs(raw_relative) <= 0.5).all(axis=1)
                     & (np.abs(raw_depth-median_depth) <= depth_tolerance))
    target_points = raw_world[raw_supported]
    target = np.median(core_world[~far_core], axis=0)
    direction = target[:2]-position[:2]
    horizontal_range = float(np.linalg.norm(direction))
    evidence.update({"target_position": _position_dict(target),
                     "range_m": float(np.linalg.norm(target-position)),
                     "horizontal_range_m": horizontal_range,
                     "target_surface_ray_count": _ray_count(uv[supported]),
                     "target_surface_return_count": int(len(target_points))})
    if horizontal_range < 1e-6:
        _reject("target_too_close", "目标水平方向不明确或过近", evidence)
    direction /= horizontal_range
    # Reverse triangle inequality guarantees the horizontal standoff from all
    # supported target returns, including returns closer than their median.
    nearest_range = float(np.min(np.linalg.norm(target_points[:, :2]-position[:2], axis=1)))
    movement = nearest_range-standoff_m-0.15
    evidence["nearest_target_horizontal_range_m"] = nearest_range
    evidence["move_m"] = movement
    if movement < 0.05:
        _reject("target_too_close", "目标已接近最小站距，拒绝继续靠近", evidence)
    if movement > max_move_m+1e-9:
        _reject("move_too_far", "视觉目标需要移动超过单次安全上限，请先缩短距离后重新观测", evidence)
    endpoint = position.copy()
    endpoint[:2] += direction*movement
    clearance = float(np.min(np.linalg.norm(all_world-endpoint, axis=1))) if len(all_world) else 0.0
    evidence["endpoint_clearance_m"] = clearance
    if clearance < 0.6:
        _reject("unsafe_endpoint", "目标停靠点距离实际点云障碍不足 0.6 米", evidence)
    actual_standoff = float(np.min(np.linalg.norm(target_points[:, :2]-endpoint[:2], axis=1)))
    evidence["target_standoff_m"] = actual_standoff
    goal = _position_dict(endpoint)
    goal["yaw"] = float(math.atan2(target[1]-endpoint[1], target[0]-endpoint[0]))
    return {"goal": goal, "evidence": evidence}
