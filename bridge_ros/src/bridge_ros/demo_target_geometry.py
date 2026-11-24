"""Rotation-independent size filtering of measured components, not obstacles."""
import numpy as np

from uav_bridge.demo_targets import target_filter_config


def oriented_size(points):
    """Minimum-area horizontal rectangle plus measured vertical extent.

    Test convex-hull edge directions so a 45-degree car does not acquire the
    much larger width of its world-axis bounding box. No completion is used.
    """
    xy = sorted(set(map(tuple, points[:, :2])))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    chains = []
    for values in (xy, reversed(xy)):
        chain = []
        for p in values:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain)
    hull = np.asarray(chains[0][:-1] + chains[1][:-1] or xy)
    if len(hull) < 2:
        return np.array([0., 0., float(np.ptp(points[:, 2]))])
    edges = np.roll(hull, -1, axis=0) - hull
    best = None
    for edge in edges:
        norm = np.linalg.norm(edge)
        if norm < 1e-10:
            continue
        axis = edge / norm
        size = np.ptp(hull @ np.array([axis, [-axis[1], axis[0]]]).T, axis=0)
        key = (float(np.prod(size)), float(sum(size)))
        if best is None or key < best[0]:
            best = (key, size)
    length, width = sorted(best[1], reverse=True)
    return np.array([length, width, float(np.ptp(points[:, 2]))])


def classify_component(points, compact_person, sample_supported, config, resolution):
    size = oriented_size(points)
    matches = {}
    person = config['person']
    # One voxel is a measurement allowance, never an enlarged measured box.
    if (compact_person and np.all(size + resolution >= person['minimum_observed_size_m']) and
            np.all(size <= np.asarray(person['maximum_size_m']) + resolution)):
        matches['person'] = 'compatible'
    car = config['car']
    nominal, tolerance = np.asarray(car['nominal_size_m']), np.asarray(car['tolerance_ratio'])
    if (sample_supported and np.all(size + resolution >= car['minimum_observed_size_m']) and
            np.all(size <= nominal*(1+tolerance) + resolution)):
        matches['car'] = ('within_nominal_range' if np.all(size + resolution >= nominal*(1-tolerance))
                          else 'partial')
    return dict(observed_size_m=size.tolist(), size_classes=list(matches), size_match=matches)


def target_context(projected, options, stamp):
    """Only visible person/car size candidates; retain all raw points elsewhere."""
    from .measured_cluster import clustered_projection, labels_for
    config = target_filter_config(options)
    if config is None:
        return None
    cloud = clustered_projection(projected, options)
    labels = labels_for(projected['points_world'], cloud)
    candidates = []
    def xyz(p):
        return dict(zip(('x', 'y', 'z'), map(float, p)))
    for group in cloud['groups']:
        if not group.get('size_classes'):
            continue
        selected = labels == group['label']
        if np.count_nonzero(selected) < 3:
            continue
        uv = projected['uv'][selected]
        low = np.maximum(0., uv.min(axis=0)-1.) / [projected['width'], projected['height']]
        high = np.minimum([projected['width'], projected['height']], uv.max(axis=0)+1.) / [projected['width'], projected['height']]
        if np.any(high <= low):
            continue
        item = dict(id='size_cluster_' + str(group['label']), size_classes=group['size_classes'],
                    size_match=group['size_match'], bbox=list(map(float, np.r_[low, high])),
                    centroid_world=xyz(group['centroid']),
                    observed_extent_world=dict(min=xyz(group['minimum']), max=xyz(group['maximum'])),
                    observed_size_m=group['observed_size_m'], point_count=group['point_count'],
                    full_extent_observed=False)
        candidates.append((float(np.median(projected['depth'][selected])), item))
    candidates.sort(key=lambda row: row[0])
    return dict(source='measured_cloud_size_filter_v1', ros_stamp=float(stamp), semantic_confirmed=False,
                candidates=[item for _, item in candidates[:config['max_candidates']]])
