"""Explicit photography scope shared by previews, model and executor.

The legacy profile remains available for regression and hardware bring-up.
Simulation navigation uses the configured geofence rather than a 1.5 m bubble.
Neither profile changes people clearance, obstacle checks or controller limits.
"""
import copy
import math
from .flight_limits import validated_bounds, require_native_backend

LOCAL_LIMITS = dict(max_decisions=8, max_moves=3, max_duration_seconds=480,
    max_model_request_seconds=120, max_move_m=1.2, max_radius_m=1.5,
    same_altitude=True, review_image_max_age_seconds=130)


def execution_limits(config):
    limits = dict(LOCAL_LIMITS)
    scope = config.get('photo', {}).get('navigation_scope', 'local_adjustment')
    if scope == 'local_adjustment':
        return limits
    if scope == 'planner':
        require_native_backend(config)
        limits.update(schema_version=4, navigation_scope='planner', navigation_strategy='local_route',
                      bounds=validated_bounds(config), bounds_source='diff_planner',
                      virtual_wall_enabled=config['planner_limits']['enable_virtual_wall'],
                      subject_motion_mode=config.get('photo', {}).get('subject_motion_mode', 'tracked'),
                      max_move_m=None, max_radius_m=None, same_altitude=False,
                      max_moves=24, max_decisions=32, max_duration_seconds=1800)
        return limits
    if scope != 'geofence':
        raise ValueError('photo.navigation_scope must be local_adjustment, geofence or planner')
    if (config.get('mode') not in ('gazebo', 'mock') and not
            (config.get('mode') == 'real' and config.get('ros', {}).get('execution_backend') == 'diff_vla_real_v1')):
        raise ValueError('Expanded photography requires a simulation adapter or the configured real task execution backend')
    bounds = copy.deepcopy(config['bounds'])
    for axis in 'xyz':
        values = bounds.get(axis)
        if (not isinstance(values, list) or len(values) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                or not 0 < values[1]-values[0] <= 40):
            raise ValueError('Invalid photography geofence')
    diagonal = math.sqrt(sum((bounds[axis][1]-bounds[axis][0])**2 for axis in 'xyz'))
    limits.update(schema_version=3, navigation_scope='geofence', navigation_strategy='local_route', bounds=bounds,
                  subject_motion_mode=config.get('photo',{}).get('subject_motion_mode','tracked'),
                  max_move_m=diagonal,
                  max_radius_m=diagonal,
                  same_altitude=False, max_moves=24, max_decisions=32, max_duration_seconds=1800)
    return limits
