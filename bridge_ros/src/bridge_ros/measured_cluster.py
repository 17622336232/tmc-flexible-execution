"""Compact objects from the measured cloud, using the real-flight profile.

The ground interval is expressed in the registered cloud's gravity-aligned
world frame. It must be calibrated for a different real-flight frame. No
ground completion or simulator object pose is used.
"""
import itertools
import json
import math
import threading

import numpy as np

from .visual_geometry import VisualGeometryError
from uav_bridge.demo_targets import target_filter_config


DEFAULTS = dict(voxel_size_m=.08, tolerance_m=.34, min_points=3,
                max_points=8000, min_horizontal_extent_m=.06,
                min_vertical_extent_m=.03, min_volume_m3=.0005, max_extent_m=2.10,
                ground_height_m=0., min_height_m=.16, max_height_m=2.03)

_CACHE_CREATION_LOCK = threading.Lock()


def parameters(options=None):
    supplied = (options or {}).get('person_clustering', {})
    if not isinstance(supplied, dict) or set(supplied)-set(DEFAULTS):
        raise VisualGeometryError('invalid_photo_limits', '人物聚类参数格式无效')
    result = dict(DEFAULTS, **supplied)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise VisualGeometryError('invalid_photo_limits', '人物聚类参数必须为有限数值')
        if key != 'ground_height_m' and value <= 0:
            raise VisualGeometryError('invalid_photo_limits', '人物聚类阈值必须为正数')
    if (not .02 <= result['voxel_size_m'] <= .15
            or not result['voxel_size_m'] <= result['tolerance_m'] <= .5
            or type(result['min_points']) is not int or not 3 <= result['min_points'] <= 100
            or type(result['max_points']) is not int or not result['min_points'] <= result['max_points'] <= 8000
            or not result['min_height_m'] < result['max_height_m'] <= 3.
            or not .1 <= result['max_extent_m'] <= 2.10):
        raise VisualGeometryError('invalid_photo_limits', '人物聚类参数超出支持范围')
    return result


def components(points, tolerance):
    """Spatial-hash BFS without a new numerical-library dependency."""
    cells = np.floor(points/tolerance).astype(np.int64)
    buckets = {}
    for index, cell in enumerate(cells.tolist()):
        buckets.setdefault(tuple(cell), []).append(index)
    visited = np.zeros(len(points), dtype=bool)
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    squared = tolerance*tolerance
    for seed in range(len(points)):
        if visited[seed]:
            continue
        visited[seed] = True
        queue, group = [seed], []
        while queue:
            current = queue.pop()
            group.append(current)
            x, y, z = cells[current]
            px, py, pz = points[current]
            for dx, dy, dz in offsets:
                for candidate in buckets.get((x+dx, y+dy, z+dz), ()):
                    if visited[candidate]:
                        continue
                    qx, qy, qz = points[candidate]
                    if (px-qx)**2+(py-qy)**2+(pz-qz)**2 <= squared:
                        visited[candidate] = True
                        queue.append(candidate)
        yield np.asarray(sorted(group), dtype=np.int64)


def cluster_cloud(points, options=None):
    """Return voxel labels and true point extents; retain oversized labels.

    In particular a wall is clustered BEFORE the RGB box is applied. Cropping
    the wall to that box would manufacture a small, person-sized object.
    """
    cfg = parameters(options)
    demo = target_filter_config(options)
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) > 30000
            or not np.isfinite(points).all()):
        raise VisualGeometryError('invalid_input', '人物聚类需要有界、有限的实测三维点云')
    height = points[:, 2]-cfg['ground_height_m']
    # Demo size rejection must see the measured top of an oversized component.
    # Cropping a tall wall/box to person height would manufacture a valid size.
    height_mask = height >= cfg['min_height_m']
    if demo is None:
        height_mask &= height <= cfg['max_height_m']
    indices = np.flatnonzero(height_mask)
    keys = np.floor(points[indices]/cfg['voxel_size_m']).astype(np.int64)
    if len(keys):
        unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    else:
        # Ubuntu 20.04's NumPy cannot reshape an empty array in unique(axis=0).
        # No measured points means no clusters, not a fabricated range.
        unique = np.empty((0,3), dtype=np.int64)
        inverse = counts = np.empty(0, dtype=np.int64)
    centers = np.column_stack([np.bincount(inverse, weights=points[indices, axis], minlength=len(unique))
                               for axis in range(3)])/counts[:, None] if len(unique) else np.empty((0, 3))
    voxel_labels = np.full(len(unique), -1, dtype=np.int64)
    groups = []
    for label, group in enumerate(components(centers, cfg['tolerance_m'])):
        voxel_labels[group] = label
        raw = points[indices[np.isin(inverse, group)]]
        low, high = raw.min(axis=0), raw.max(axis=0)
        extent = high-low
        compact = bool(cfg['min_points'] <= len(group) <= cfg['max_points']
                       and max(extent[:2]) >= cfg['min_horizontal_extent_m']
                       and extent[2] >= cfg['min_vertical_extent_m']
                       and np.prod(extent) >= cfg['min_volume_m3']
                       and max(extent) <= cfg['max_extent_m'])
        item = dict(label=label, voxel_count=len(group), point_count=len(raw), compact=compact,
                    centroid=np.mean(raw, axis=0), minimum=low, maximum=high)
        if demo is not None:
            from .demo_target_geometry import classify_component
            item.update(classify_component(raw, compact,
                        cfg['min_points'] <= len(group) <= cfg['max_points'], demo, cfg['voxel_size_m']))
            # Existing person association must never choose a car-sized group.
            item['compact'] = compact and 'person' in item['size_classes']
        groups.append(item)
    lookup = {tuple(key): int(label) for key, label in zip(unique.tolist(), voxel_labels)}
    return dict(parameters=cfg, lookup=lookup, groups=groups, input_point_count=len(points),
                cropped_point_count=len(indices), voxel_count=len(unique),
                maximum_height_applied=demo is None)


def clustered_projection(projected, options=None):
    """Share clustering only within one private, frozen sensor projection.

    People association and demo context use the same measured cloud. The key
    includes every effective clustering/size parameter; a new frame owns a new
    cache, and a configuration change replaces rather than reuses old geometry.
    """
    key = json.dumps([parameters(options), target_filter_config(options)],
                     sort_keys=True, separators=(',', ':'), allow_nan=False)
    with _CACHE_CREATION_LOCK:
        lock = projected.setdefault('_measured_cluster_lock', threading.RLock())
    with lock:
        cached = projected.get('_measured_cluster_cache')
        if cached is None or cached[0] != key:
            cloud = cluster_cloud(projected['points_world_all'], options)
            projected['_measured_cluster_cache'] = (key, cloud)
            return cloud
        return cached[1]


def labels_for(points, clustered):
    cfg = clustered['parameters']
    points = np.asarray(points)
    keys = np.floor(points/cfg['voxel_size_m']).astype(np.int64)
    height = points[:, 2]-cfg['ground_height_m']
    # Match the same height policy used when building this cloud's labels.
    # Demo classification needs the entire component, including its top.
    maximum = cfg['max_height_m'] if clustered.get('maximum_height_applied', True) else math.inf
    return np.asarray([clustered['lookup'].get(tuple(key), -1)
                       if cfg['min_height_m'] <= z <= maximum else -1
                       for key, z in zip(keys.tolist(), height)], dtype=np.int64)
