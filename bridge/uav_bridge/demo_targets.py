"""Demo target size profiles; these never filter the planner's point cloud."""
import copy
import math


DEFAULT_TARGET_FILTER = {
    'enabled': True,
    'max_candidates': 12,
    'person': {
        'nominal_size_m': [.60, .45, 1.70],
        'minimum_observed_size_m': [.06, .02, .03],
        'maximum_size_m': [1.20, 1.00, 2.10],
    },
    'car': {
        'nominal_size_m': [4.78, 1.898, 1.67],
        'tolerance_ratio': [.25, .25, .30],
        'minimum_observed_size_m': [1.20, .02, .50],
    },
}


def _vector(value, name, lower=0., upper=100.):
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or not lower < x <= upper for x in value)):
        raise ValueError(name + ' must contain three positive finite values')
    return [float(x) for x in value]


def target_filter_config(options=None):
    raw = (options or {}).get('demo_target_filter')
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT_TARGET_FILTER):
        raise ValueError('photo.demo_target_filter has invalid fields')
    result = copy.deepcopy(DEFAULT_TARGET_FILTER)
    for key, value in raw.items():
        if key in ('person', 'car'):
            if not isinstance(value, dict) or set(value) - set(result[key]):
                raise ValueError('Invalid demo target profile: ' + key)
            result[key].update(copy.deepcopy(value))
        else:
            result[key] = value
    if not isinstance(result['enabled'], bool):
        raise ValueError('demo_target_filter.enabled must be boolean')
    if type(result['max_candidates']) is not int or not 1 <= result['max_candidates'] <= 12:
        raise ValueError('demo target candidate limit must be 1..12')
    for name in ('person', 'car'):
        profile = result[name]
        for key, value in profile.items():
            profile[key] = _vector(value, name + '.' + key,
                                   upper=.75 if key == 'tolerance_ratio' else 20.)
        nominal = profile['nominal_size_m']
        maximum = (profile['maximum_size_m'] if name == 'person' else
                   [n * (1. + t) for n, t in zip(nominal, profile['tolerance_ratio'])])
        if (nominal[0] < nominal[1] or
                any(lo > n or n > hi for lo, n, hi in
                    zip(profile['minimum_observed_size_m'], nominal, maximum))):
            raise ValueError('Invalid minimum/nominal/maximum demo size: ' + name)
    return result if result['enabled'] else None


def validated_target_context(raw, observation_stamp):
    """Copy measured, per-image candidates into the model's spatial context."""
    if (not isinstance(raw, dict) or raw.get('source') != 'measured_cloud_size_filter_v1' or
            raw.get('semantic_confirmed') is not False or
            not isinstance(raw.get('candidates'), list) or len(raw['candidates']) > 12):
        raise ValueError('Invalid demo target evidence')
    stamp = raw.get('ros_stamp')
    if (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or
            not math.isfinite(stamp) or stamp <= 0 or abs(stamp-observation_stamp) > .2):
        raise ValueError('Demo target evidence does not match the current observation')
    candidates, ids = [], set()
    def xyz(value):
        if not isinstance(value, dict) or set(value) != {'x', 'y', 'z'}:
            raise ValueError('Invalid target coordinates')
        values = [value[k] for k in ('x', 'y', 'z')]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            raise ValueError('Invalid target coordinates')
        return dict(zip(('x', 'y', 'z'), map(float, values)))
    for item in raw['candidates']:
        if not isinstance(item, dict):
            raise ValueError('Invalid target candidate')
        identity, classes = item.get('id'), item.get('size_classes')
        if (not isinstance(identity, str) or not 1 <= len(identity) <= 80 or identity in ids or
                not isinstance(classes, list) or not 1 <= len(classes) <= 2 or
                any(c not in ('person', 'car') for c in classes) or len(set(classes)) != len(classes)):
            raise ValueError('Invalid target identity or size class')
        ids.add(identity)
        matches = item.get('size_match')
        if (not isinstance(matches, dict) or set(matches) != set(classes) or
                any(x not in ('compatible', 'partial', 'within_nominal_range') for x in matches.values())):
            raise ValueError('Invalid target size match')
        bbox = item.get('bbox')
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox) or
                not 0 <= bbox[0] < bbox[2] <= 1 or not 0 <= bbox[1] < bbox[3] <= 1):
            raise ValueError('Invalid target image bounds')
        count = item.get('point_count')
        if type(count) is not int or count < 3 or item.get('full_extent_observed') is not False:
            raise ValueError('Nominal dimensions cannot certify unseen target surfaces')
        extent = item.get('observed_extent_world', {})
        low, high, center = xyz(extent.get('min')), xyz(extent.get('max')), xyz(item.get('centroid_world'))
        if any(low[k] > high[k] or not low[k]-1e-8 <= center[k] <= high[k]+1e-8 for k in low):
            raise ValueError('Invalid measured target envelope')
        size = item.get('observed_size_m')
        # A visible flat face can have a zero thickness; it remains partial.
        if (not isinstance(size, list) or len(size) != 3 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in size)):
            raise ValueError('Invalid measured target dimensions')
        candidates.append(dict(id=identity, size_classes=list(classes), size_match=dict(matches),
                               bbox=list(bbox), point_count=count, full_extent_observed=False,
                               centroid_world=center, observed_extent_world=dict(min=low, max=high),
                               observed_size_m=list(size)))
    return dict(source=raw['source'], ros_stamp=float(stamp), semantic_confirmed=False, candidates=candidates)
