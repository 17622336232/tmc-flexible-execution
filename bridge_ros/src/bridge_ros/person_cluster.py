"""Measured 3-D components for a frozen RGB person's torso.

The .34 m connectivity follows perception_display's real-flight settings.
Voxel size, connectivity, point count and size filters follow the effective
lightweight/person_scene deployment. RGB selects a measured 3-D object; it does
not require a wall behind that object.
No ground completion, body-height prior, scene pose, or synthetic point exists.
"""
import itertools
import math

import numpy as np

from .visual_geometry import VisualGeometryError, _ray_count
from .measured_cluster import cluster_cloud, labels_for


TOLERANCE_M = .34
# Shared with photo_geometry's silhouette envelope. Pixels inside this band
# cannot establish independent exterior background or contradict it.
PERSON_BBOX_UNCERTAINTY_PIXELS = 4.


def exterior_strips(uv, bbox):
    """Both unchanged outer strips, outside the detector boundary uncertainty."""
    rel = (uv-np.asarray(bbox[:2]))/(np.asarray(bbox[2:])-bbox[:2])
    side_y = (rel[:, 1] >= .15) & (rel[:, 1] <= .65)
    return {
        "left": side_y & (rel[:, 0] >= -.5)
                & (uv[:, 0] < bbox[0]-PERSON_BBOX_UNCERTAINTY_PIXELS),
        "right": side_y & (rel[:, 0] <= 1.5)
                 & (uv[:, 0] > bbox[2]+PERSON_BBOX_UNCERTAINTY_PIXELS),
    }


def euclidean_components(points, tolerance=TOLERANCE_M):
    """Deterministic spatial-hash BFS; returned rows always index input points."""
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) > 9000
            or not np.isfinite(points).all() or not math.isfinite(tolerance) or tolerance <= 0):
        raise VisualGeometryError("invalid_input", "聚类需要有限真实三维点和有界输入")
    cells = np.floor(points/tolerance).astype(np.int64)
    buckets = {}
    for i, cell in enumerate(cells):
        buckets.setdefault(tuple(cell), []).append(i)
    unseen, groups = set(range(len(points))), []
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    for seed in range(len(points)):
        if seed not in unseen:
            continue
        unseen.remove(seed)
        group, queue = [], [seed]
        while queue:
            current = queue.pop()
            group.append(current)
            for offset in offsets:
                key = tuple(cells[current]+offset)
                for candidate in buckets.get(key, ()):
                    if candidate in unseen and np.linalg.norm(points[current]-points[candidate]) <= tolerance:
                        unseen.remove(candidate)
                        queue.append(candidate)
        groups.append(np.asarray(sorted(group), dtype=np.int64))
    return groups


def support(uv, depth, bbox):
    width, height = bbox[2]-bbox[0], bbox[3]-bbox[1]
    count = _ray_count(uv)
    span = np.ptp(uv, axis=0)/[(2./3)*width, .38*height] if len(uv) else [0., 0.]
    result = dict(point_count=count, columns=len(np.unique(np.floor(uv[:, 0]/2))),
                  rows=len(np.unique(np.floor(uv[:, 1]/2))), horizontal_fraction=float(span[0]),
                  vertical_fraction=float(span[1]))
    if len(depth):
        median = float(np.median(depth))
        spread = float(np.percentile(depth, 95)-np.percentile(depth, 5))
        result.update(depth_min_m=float(min(depth)), depth_median_m=median,
                      depth_max_m=float(max(depth)), depth_spread_m=spread,
                      coherent=(_ray_count(uv[np.abs(depth-median) > max(.25, .065*median)]) < 2
                                and spread <= max(.35, .10*median)))
    else:
        result["coherent"] = False
    result["supported"] = bool(count >= 3)
    return result


def associate_torso_cluster(uv, depth, points, bbox, torso, all_points=None, options=None):
    """Match the RGB torso to an independently clustered compact 3-D object.

    Cluster the complete measured cloud before applying the RGB box so that a
    cropped section of a wall cannot masquerade as a compact object. Multiple
    supported compact objects in the same torso frustum remain ambiguous.
    Background returns may be absent. All raw nearer returns are still checked
    by the caller, including returns sharing a de-duplication bin.
    """
    cloud = cluster_cloud(points if all_points is None else all_points, options)
    labels = labels_for(points, cloud)
    diagnostics = dict(algorithm="measured_voxel_euclidean_rgb_association_v2",
                       tolerance_m=cloud['parameters']['tolerance_m'],
                       voxel_size_m=cloud['parameters']['voxel_size_m'],
                       parameters=cloud['parameters'], input_point_count=cloud['input_point_count'],
                       voxel_count=cloud['voxel_count'], roi_point_count=_ray_count(uv[torso]),
                       component_count=len(cloud['groups']), components=[],
                       background_returns_required=False)
    candidates, incoherent = [], False
    for group in cloud['groups']:
        mask = torso & (labels == group['label'])
        if not mask.any():
            continue
        stats = support(uv[mask], depth[mask], bbox)
        stats.update(label=group['label'], compact=group['compact'],
                     voxel_count=group['voxel_count'], extent_world_m=(group['maximum']-group['minimum']).tolist())
        diagnostics['components'].append(stats)
        if group['compact'] and stats['supported']:
            if stats['coherent']:
                candidates.append((mask, group))
            else:
                incoherent = True
    if len(candidates) != 1 or incoherent:
        if len(candidates) > 1 or incoherent:
            code, message = 'person_range_ambiguous', '人物框同时对应多个有效三维簇，不能唯一确认人物位置'
        elif not any(c['supported'] for c in diagnostics['components']):
            code, message = 'person_range_sparse', '人物对应的有效三维簇回波不足，请重新观察'
        else:
            code, message = 'person_cluster_unmatched', '人物框未匹配到尺寸合适且测距一致的三维簇，请重新观察'
        raise VisualGeometryError(code, message, {'person_cluster': diagnostics})
    mask, group = candidates[0]
    median = float(np.median(depth[mask]))
    residual = torso & ~mask
    # A nearer residual can be an occluder even when too small to become a
    # supported object. Never select the larger/farther candidate over it.
    if np.any(residual & (depth < median-max(.25, .065*median))):
        raise VisualGeometryError('person_occluded', '人物框内有更近的遮挡回波，无法安全确认人物',
                                  {'person_cluster': diagnostics})
    def xyz(value):
        return dict(zip(('x', 'y', 'z'), map(float, value)))
    diagnostics.update(selected_point_count=_ray_count(uv[mask]),
                       excluded_background_count=_ray_count(uv[residual]),
                       centroid_world=xyz(group['centroid']),
                       observed_extent_world={'min': xyz(group['minimum']), 'max': xyz(group['maximum'])},
                       point_count=group['point_count'], extent_scope='visible_connected_returns_only',
                       complete_body_size_known=False, ground_completion=False, full_body_semantics=False)
    return mask, diagnostics
