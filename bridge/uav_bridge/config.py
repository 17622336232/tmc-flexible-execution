"""Configuration shared by the bridge, mock and ROS adapter."""

import copy
import json
import math
from .flight_limits import planner_scope, require_native_backend
from .demo_targets import target_filter_config


DEFAULT_CONFIG = {
    "mode": "mock",
    "navigation_mode": "preset",
    "bounds": {"x": [-8.0, 8.0], "y": [-5.0, 5.0], "z": [0.6, 2.5]},
    "home": {"x": -6.0, "y": -3.0, "z": 1.2},
    "takeoff_height": 1.2,
    "landmarks": {},
    "max_steps": 12,
    "telemetry_timeout_seconds": 2.0,
    "step_timeout_seconds": 120.0,
    "arrival_tolerance_m": 0.25,
    "arrival_speed_mps": 0.2,
    "arrival_dwell_seconds": 0.5,
    "plan_ttl_seconds": 300,
    "request_cache_size": 2048,
    "client_timeout_seconds": 15.0,
    "camera_timeout_seconds": 2.5,
    "llm": {"base_url": "", "model": "", "api_key_env": "UAV_LLM_API_KEY", "timeout_seconds": 20},
    "vision": {"enabled": True, "provider": "chat_completions", "base_url": "", "model": "", "api_key_env": "UAV_VLM_API_KEY", "timeout_seconds": 20},
    "photo": {
        "enabled": False,
        "decision_mode": "local",
        "require_pose_validation": True,
        "min_confidence": 0.6,
        "framing_margin": 0.08,
        "target_height_fraction": 0.5,
        "min_person_clearance_m": 2.5,
        "drone_radius_m": 0.4,
        "measurement_margin_m": 0.15,
        "max_move_m": 1.2,
        "subject_motion_mode": "tracked",
        "preview_ttl_seconds": 30,
        "stationary_dwell_seconds": 0.6,
        "monitor_interval_seconds": 0.5,
        "frame_wait_timeout_seconds": 8,
        "person_motion_tolerance_m": 0.2,
        "viewpoint_surface_allowance_m": 0.,
        "cluster_match_distance_m": .95,
        "min_person_bbox_width_px": 24.,
        "max_person_bbox_aspect": 6.,
        "detector_timeout_seconds": 5,
        "search_job_timeout_seconds": 5,
        "save_timeout_seconds": 10,
        "store_directory": None,
        "max_photos": 100,
        "max_photo_bytes": 5242880
    },
}


def load_config(path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            override = json.load(handle)
        if not isinstance(override, dict):
            raise ValueError("Configuration must be a JSON object")
        for key, value in override.items():
            if key in ("llm", "vision", "bounds", "photo") and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        if "home" not in override and isinstance(override.get("landmarks", {}).get("home"), dict):
            config["home"] = {key: value for key, value in override["landmarks"]["home"].items()
                              if key in ("x", "y", "z", "yaw")}
    if planner_scope(config):
        require_native_backend(config)
        if path and 'bounds' in override:
            raise ValueError('Planner scope reads original planner limits; remove duplicate bounds')
        config['bounds'] = dict(x=None, y=None, z=None)  # Resolved after the C++ map starts.
    for axis in (() if planner_scope(config) else ("x", "y", "z")):
        interval = config["bounds"].get(axis)
        if (not isinstance(interval, list) or len(interval) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in interval) or
                interval[0] >= interval[1]):
            raise ValueError("bounds.%s must contain two finite increasing numbers" % axis)
    for key in ("telemetry_timeout_seconds", "step_timeout_seconds", "arrival_tolerance_m", "arrival_speed_mps", "arrival_dwell_seconds", "plan_ttl_seconds", "client_timeout_seconds", "camera_timeout_seconds"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("%s must be positive and finite" % key)
    photo = config["photo"]
    if not isinstance(photo, dict) or not isinstance(photo.get("enabled"), bool):
        raise ValueError("photo.enabled must be a boolean")
    target_filter_config(photo)
    if photo.get("decision_mode") not in ("local", "vlm"):
        raise ValueError("photo.decision_mode must be local or vlm")
    if photo.get('person_detector', 'mobilenet_ssd') not in ('mobilenet_ssd', 'yolov8n'):
        raise ValueError('photo.person_detector must be mobilenet_ssd or yolov8n')
    if photo.get('subject_motion_mode','tracked') not in ('stationary','tracked'):
        raise ValueError('photo.subject_motion_mode must be stationary or tracked')
    if photo.get('person_localization', 'torso_cluster') not in ('torso_cluster', 'measured_cluster'):
        raise ValueError('photo.person_localization must be torso_cluster or measured_cluster')
    if photo.get("require_pose_validation") is not True:
        raise ValueError("photo requires full-body pose validation")
    photo_limits = {
        "min_confidence": (0.55, 0.95), "framing_margin": (0.08, 0.15),
        "target_height_fraction": (0.45, 0.60), "min_person_clearance_m": (1.5, 5.0),
        "drone_radius_m": (0.4, 1.0), "measurement_margin_m": (0.15, 0.5),
        "max_move_m": (0.01, 1.2), "preview_ttl_seconds": (1, 30),
        "stationary_dwell_seconds": (0.5, 3), "monitor_interval_seconds": (0.1, 1),
        "frame_wait_timeout_seconds": (1, 15), "person_motion_tolerance_m": (0.01, 0.2),
        "viewpoint_surface_allowance_m": (0., .5), "cluster_match_distance_m": (.01, .95),
        "min_person_bbox_width_px": (12., 24.), "max_person_bbox_aspect": (6., 12.),
        "detector_timeout_seconds": (0.1, 10), "max_photos": (1, 500),
        "search_job_timeout_seconds": (1, 10), "save_timeout_seconds": (2, 30),
        "max_photo_bytes": (1024, 5 * 1024 * 1024),
    }
    for key, (minimum, maximum) in photo_limits.items():
        value = photo.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not minimum <= value <= maximum):
            raise ValueError("photo.%s must be within [%s, %s]" % (key, minimum, maximum))
    for key in ("max_photos", "max_photo_bytes"):
        if not isinstance(photo[key], int):
            raise ValueError("photo.%s must be an integer" % key)
    acceptance = photo.get('navigation_acceptance', {})
    ranges = {'position_tolerance_m': (.1, .5), 'yaw_tolerance_deg': (3., 15.),
              'stable_speed_mps': (.05, .2)}
    if not isinstance(acceptance, dict) or set(acceptance)-set(ranges):
        raise ValueError('photo.navigation_acceptance contains unsupported fields')
    for key, value in acceptance.items():
        lo, hi = ranges[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('photo.navigation_acceptance.%s must be within [%s, %s]' % (key, lo, hi))
    if photo.get("store_directory") is not None and not isinstance(photo["store_directory"], str):
        raise ValueError("photo.store_directory must be a path string or null")
    return config
