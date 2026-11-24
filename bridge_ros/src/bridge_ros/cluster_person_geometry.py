"""RGB-selected measured cluster: location and extent, with no torso quota.

Cluster the full cloud first. A camera crop cannot turn a wall into a person.
The initial person detector/pose check remains independent of this geometry.
"""
import itertools

import numpy as np

from uav_bridge.demo_targets import target_filter_config

from .measured_cluster import clustered_projection, labels_for
from .visual_geometry import _points, _ray_count, VisualGeometryError


def measure_cluster_person(projected, box, camera_from_world, intrinsic,
                           body_from_camera, position, yaw, limits, options):
    from .photo_geometry import _guard_envelope_evidence, _xyz

    points = _points(projected['points_world'], 'points_world')
    all_points = _points(projected['points_world_all'], 'points_world_all')
    uv, depth = np.asarray(projected['uv']), np.asarray(projected['depth'])
    inside = ((uv >= box[:2]) & (uv <= box[2:])).all(axis=1)
    cloud = clustered_projection(projected, options)
    labels = labels_for(points, cloud)
    hit_labels = set(labels[inside].tolist())
    matching = [g for g in cloud['groups'] if g['label'] in hit_labels]
    candidates = [g for g in matching if g['compact']]
    diagnostic = dict(algorithm='full_cloud_measured_cluster_rgb_association_v1',
                      parameters=cloud['parameters'], input_point_count=cloud['input_point_count'],
                      voxel_count=cloud['voxel_count'], component_count=len(cloud['groups']),
                      matching_compact_clusters=len(candidates), background_returns_required=False,
                      torso_return_quota_required=False, complete_body_size_known=False,
                      ground_completion=False, extent_scope='visible_connected_returns_only')
    if len(candidates) != 1:
        code = 'person_range_ambiguous' if candidates else 'person_cluster_unmatched'
        if not candidates and len(matching) == 1:
            group, cfg = matching[0], cloud['parameters']
            hit_count = int(np.count_nonzero(inside))
            # Retry only a few actual returns from one undersampled FULL
            # component. Empty boxes, separate components, excluded-height
            # returns, and sufficient-but-invalid geometry remain rejected.
            sparse = (hit_labels == {group['label']}
                      and 0 < hit_count < cfg['min_points']
                      and 0 < group['voxel_count'] < cfg['min_points']
                      and max(group['maximum']-group['minimum']) <= cfg['max_extent_m'])
            demo = target_filter_config(options)
            if sparse and demo is not None:
                sparse = bool(np.all(np.asarray(group['observed_size_m']) <=
                    np.asarray(demo['person']['maximum_size_m']) + cfg['voxel_size_m']))
            if sparse:
                code = 'person_cluster_sparse'
                diagnostic.update(selected_point_count=_ray_count(uv[inside]),
                                  cluster_voxel_count=group['voxel_count'],
                                  required_cluster_voxels=cfg['min_points'])
        raise VisualGeometryError(code, '人物框没有唯一匹配到尺寸合适的实测三维簇，请换个角度观察',
                                  {'person_cluster': diagnostic})
    group = candidates[0]
    selected = inside & (labels == group['label'])
    median = float(np.median(depth[selected]))
    if np.any(inside & ~selected & (depth < median-max(.25, .065*median))):
        raise VisualGeometryError('person_occluded', '人物框内有更近的独立遮挡物，请重新观察',
                                  {'person_cluster': diagnostic})
    center = group['centroid']
    low, high = group['minimum'], group['maximum']
    envelope = np.asarray(list(itertools.product(*zip(low, high))), dtype=float)
    # Keep every measured cluster point, including outside the RGB box. This
    # extent is separate from the silhouette projection used for composition.
    raw = all_points[labels_for(all_points, cloud) == group['label']]
    extent_radius = float(np.max(np.linalg.norm(envelope[:, :2]-center[:2], axis=1)))
    required = extent_radius+limits['drone_radius_m']+limits['min_person_clearance_m']+limits['measurement_margin_m']
    radius = required+limits['person_motion_tolerance_m']+limits['viewpoint_surface_allowance_m']
    proof = _guard_envelope_evidence(envelope, raw)
    proof['method'] = 'measured_cluster_aabb'
    diagnostic.update(centroid_world=_xyz(center),
                      observed_extent_world=dict(min=_xyz(low), max=_xyz(high)),
                      point_count=group['point_count'], selected_point_count=_ray_count(uv[selected]),
                      cluster_voxel_count=group['voxel_count'])
    uncertainty = limits['measurement_margin_m']
    diagnostic['uncertainty_m'] = uncertainty
    evidence = dict(source='rgb_lidar', localization_method='measured_cluster',
                    bbox_pixels=box.tolist(), person_cluster=diagnostic,
                    person_position_source='measured_cluster_centroid', person_surface_position=_xyz(center),
                    association_anchor_world=_xyz(center),
                    association_anchor_metadata=dict(
                        source='calibrated_rgb_lidar', method='measured_component_centroid_v1',
                        scope='cross_frame_association_estimate_only',
                        algorithm=diagnostic['algorithm'],
                        is_lidar_return=False, body_center_known=False),
                    measured_depth_m=median, depth_spread_m=float(np.ptp(depth[selected])),
                    depth_uncertainty_m=uncertainty, photo_guard_envelope_v1=proof,
                    nearest_person_return_body_distance_m=float(np.min(np.linalg.norm(raw[:, :2]-position[:2], axis=1))),
                    measured_body_distance_m=float(np.linalg.norm(center[:2]-position[:2])),
                    person_envelope_radius_m=extent_radius, body_radius_m=limits['drone_radius_m'],
                    min_person_clearance_m=limits['min_person_clearance_m'], measurement_margin_m=uncertainty,
                    photo_guard_radius_m=radius, photo_required_guard_radius_m=required,
                    person_motion_allowance_m=limits['person_motion_tolerance_m'],
                    viewpoint_surface_allowance_m=limits['viewpoint_surface_allowance_m'],
                    full_body_proven=False, full_body_evidence='detector_bbox_only',
                    pose_validation=False, stationary_person_required=True)
    # This is only a framing estimate, never a larger obstacle/protection
    # envelope and never a claim that unseen body parts have been measured.
    pixels = np.array([[box[ix], box[iy], 1.] for ix, iy in ((0, 1), (0, 3), (2, 1), (2, 3))])
    rays = pixels @ np.linalg.inv(intrinsic).T
    optical = np.concatenate((rays*max(.11, median-uncertainty), rays*(median+uncertainty)))
    world_from_camera = np.linalg.inv(camera_from_world)
    framing = optical @ world_from_camera[:3, :3].T+world_from_camera[:3, 3]
    return dict(evidence=evidence, envelope=envelope, framing_envelope=framing, surface=center,
                body_T_optical=body_from_camera, K=intrinsic, bbox=box, position=position, yaw=float(yaw),
                width=int(projected['width']), height=int(projected['height']), limits=limits,
                points_world_all=all_points)
