"""Shared bounds semantics; native flight uses the planner's loaded map limits."""
import copy
import math


def planner_scope(config):
    return config.get('photo', {}).get('navigation_scope') == 'planner'


def require_native_backend(config):
    if config.get('mode') != 'real' or config.get('ros', {}).get('execution_backend') != 'diff_vla_real_v1':
        raise ValueError('Planner limits require the configured real task execution backend')


def apply_planner_limits(config, loaded):
    """Bind a snapshot exported by the running C++ map, never a second YAML."""
    require_native_backend(config)
    if not isinstance(loaded, dict) or type(loaded.get('enable_virtual_wall')) is not bool:
        raise ValueError('Original planner limits are unavailable')
    for key in ('virtual_ground', 'virtual_ceil', 'local_map_half_x', 'local_map_half_y', 'local_map_half_z'):
        value = loaded.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid original planner limit: '+key)
        if key.startswith('local_map_half_') and value <= 0:
            raise ValueError('Invalid original local map range')
    enabled = loaded['enable_virtual_wall']
    if enabled and loaded['virtual_ground'] >= loaded['virtual_ceil']:
        raise ValueError('Invalid original virtual wall range')
    config['planner_limits'] = copy.deepcopy(loaded)
    config['bounds'] = dict(x=None, y=None,
        z=[loaded['virtual_ground'], loaded['virtual_ceil']] if enabled else None)


def validated_bounds(config):
    native = planner_scope(config)
    if native:
        require_native_backend(config)
        # Do not fall back to mock bounds or proceed before the map has started.
        expected = dict(mode='real', ros=config['ros'])
        apply_planner_limits(expected, config.get('planner_limits'))
        if config.get('bounds') != expected['bounds']:
            raise ValueError('Bounds differ from the loaded original planner')
        return copy.deepcopy(expected['bounds'])
    bounds = config.get('bounds')
    if not isinstance(bounds, dict):
        raise ValueError('Missing bounds configuration')
    for axis in 'xyz':
        pair = bounds.get(axis)
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pair) or
                pair[0] >= pair[1] or (axis == 'z' and pair[0] <= 0)):
            raise ValueError('Invalid bounds.'+axis)
    return copy.deepcopy(bounds)


def within_interval(value, interval, exclusive=False):
    return interval is None or (interval[0] < value < interval[1] if exclusive
                                else interval[0] <= value <= interval[1])


def within_bounds(point, bounds, native=False):
    return all(within_interval(point[axis], bounds[axis], native and axis == 'z') for axis in 'xyz')


def exceeds_distance(distance, limit):
    return limit is not None and distance > limit+1e-6
