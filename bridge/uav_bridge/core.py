"""Thread-safe mission engine. Acceptance and measured completion are distinct."""

import copy
import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future

from .planner import Planner, validate_plan
from .vision import VisionService
from .photo import PhotoService, parse_photo_request, PhotoFramePending, PhotoValidationTimeout
from .photo_store import PhotoStore
from .photo_agent import PhotoAgentService, PhotoAgentError
from .photo_agent_runner import PhotoAgentRunner, make_reference, measure_selected, LIMITS as PHOTO_RUNNER_LIMITS
from .photo_task_schema import shots as task_shots
from .task_log import TaskLog
from .evidence_store import EvidenceStore
from .scene_memory import SceneMemory
from .mobile_settings import MobileState, LatestStateWriter, mission_summary, model_name
from .photo_policy import execution_limits


PHOTO_AGENT_LIMITS = copy.deepcopy(PHOTO_RUNNER_LIMITS)
PHOTO_MEASUREMENT_SCOPE = '首次选人和安全测距需可识别的人物图像和可关联的实测点云，实测聚类模式支持局部人体入镜。静止人物模式保留初始位置和防碰撞范围，后续各张是否需要人物入镜按该张要求判断；实时关联模式仍需持续可测。'


class BridgeError(Exception):
    def __init__(self, code, message, status=400, details=None):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status
        self.details = details


def _finite(value):
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Bridge:
    def __init__(self, adapter, config, planner=None):
        self.adapter = adapter
        self.config = copy.deepcopy(config)
        self.photo_limits = execution_limits(self.config)
        if self.config.get("navigation_mode") == "perception":
            self.config["home"] = None
            self.config["landmarks"] = {}
        planner_config = copy.deepcopy(self.config)
        if self.config.get("navigation_mode") == "perception":
            planner_config["llm"] = {"enabled": False}
        self.planner = planner or Planner(planner_config)
        self.vision = VisionService(self.config, adapter)
        self.photo = PhotoService(self.config, adapter)
        self.photo_store = PhotoStore(self.config.get("photo", {}))
        self.photo_agent = PhotoAgentService(self.config, adapter)
        self._photo_agent_runner = None
        self._photo_context = None
        self._search_context = None
        self._search_slots = threading.BoundedSemaphore(1)
        self._subject_preview_slots = threading.BoundedSemaphore(1)
        self._photo_slots = threading.BoundedSemaphore(1)
        self._photo_confirm_slots = threading.BoundedSemaphore(1)
        self._command_epoch = 0
        self._photo_confirming = False
        self._home = None
        self._home_stamp = None
        self._evidence = OrderedDict()
        self._inspection_slots = threading.BoundedSemaphore(2)
        self._inspection_wait_started = None
        self._inspection_ready_since = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._mission = None
        self._previous_mission = None
        self._mobile_state = MobileState(self.config.get('mobile_state_directory'))
        self._history_writer = LatestStateWriter(self._mobile_state, 'missions.json')
        self.task_log = TaskLog(self._mobile_state)
        self._evidence_store = EvidenceStore(self.config.get('mobile_state_directory'))
        records = self._mobile_state.read('missions.json', [])
        self._mission_history = ([item for item in records if isinstance(item, dict)][-20:]
                                 if isinstance(records, list) else [])
        self._plans = OrderedDict()
        self._requests = OrderedDict()
        self._client_sequences = OrderedDict()
        self._last_error = None
        self._last_client = time.monotonic()
        self._watchdog_latched = False
        self._snapshot = {}
        self._dispatch = None

    def missions(self):
        with self._lock:
            records = copy.deepcopy(self._mission_history)
            current = mission_summary(self._mission)
            if current:
                records = [item for item in records if item.get('id') != current.get('id')]
                records.append(current)
            return {'missions': list(reversed(records[-20:]))}

    def models(self):
        with self._lock:
            vision = copy.deepcopy(self.config.get('vision', {}))
        if vision.get('provider') != 'chatgpt_plan':
            return {'models': [], 'current': vision.get('model', '')}
        catalog = VisionService._chatgpt_store(vision).cached_models()
        return {'models': [{'slug': item['slug'], 'display_name': item.get('display_name', item['slug'])}
                           for item in catalog if isinstance(item, dict) and model_name(item.get('slug'))],
                'current': vision.get('model', '')}

    def logs(self):
        with self._lock:
            self.task_log.sync(self._mission)
            return self.task_log.read()

    def select_model(self, body):
        if not isinstance(body, dict) or set(body) != {'model'} or not model_name(body['model']):
            raise BridgeError('model_invalid', '请选择账户目录中的模型。')
        with self._lock:
            snap = self._read()
            if (not self._fresh(snap) or
                    math.sqrt(sum(snap.get('velocity', {}).get(a, 0.)**2 for a in ('x','y','z'))) > .12
                    or self._mission and self._mission['state'] == 'running' or self._photo_confirming):
                raise BridgeError('model_busy', '当前任务正在执行或飞行器未停稳；结束任务并悬停后可以切换模型。', 409)
            if body['model'] not in {item['slug'] for item in self.models()['models']}:
                raise BridgeError('model_unavailable', '模型不在当前账户已授权的目录内，请在 Ubuntu 刷新目录。', 422)
            self._mobile_state.write(self.config.get('model_settings_file', 'model.json'), {'model': body['model']})
            self.config['vision']['model'] = body['model']
            self.vision = VisionService(self.config, self.adapter)
            self.photo_agent = PhotoAgentService(self.config, self.adapter)
            self._command_epoch += 1
            self._plans.clear()
            self.task_log.event('settings', 'model_changed', '下一个任务使用模型 '+body['model'])
            return {'model': body['model'], 'changed': True}

    def start(self):
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="mission-loop", daemon=True)
                self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        with self._lock:
            if self._mission and self._mission["state"] == "running":
                self._abort("Bridge shutting down; requested hold", "cancelled")
        detector_close = getattr(self.photo.detector, 'close', None)
        if detector_close:
            detector_close()
        close = getattr(self.adapter, "close", None)
        if close:
            close()
        self.task_log.close()
        self._history_writer.close()
        self._evidence_store.close()

    def note_client(self):
        with self._lock:
            self._last_client = time.monotonic()

    def _read(self):
        try:
            snap = self.adapter.snapshot()
            if not isinstance(snap, dict):
                raise ValueError("Adapter snapshot must be a dictionary")
            self._snapshot = copy.deepcopy(snap)
        except Exception as exc:
            self._snapshot = {"mode": self.config.get("mode", "unknown"), "connected": False}
            self._last_error = "Telemetry adapter failed: %s" % exc
        if self._home is None and self._fresh(self._snapshot) and self._snapshot["landed"] and not self._snapshot["armed"]:
            p = self._snapshot["position"]
            height = self.config["takeoff_height"] + (p["z"] if self.config.get("takeoff_relative") else 0)
            self._home = {"x": p["x"], "y": p["y"], "z": height, "yaw": self._snapshot["yaw"]}
            self._home_stamp = self._snapshot["position_stamp"]
            if self.config.get("navigation_mode") == "perception":
                self.config["home"] = copy.deepcopy(self._home)
                if hasattr(self.planner, "config"):
                    self.planner.config["home"] = copy.deepcopy(self._home)
        return self._snapshot

    def _fresh(self, snap, require_position=True):
        if snap.get("connected") is not True:
            return False
        now = time.time()
        for key in (["state_stamp", "position_stamp"] if require_position else ["state_stamp"]):
            stamp = snap.get(key)
            if not _finite(stamp) or not -1.0 <= now - stamp <= self.config["telemetry_timeout_seconds"]:
                return False
        if not isinstance(snap.get("armed"), bool) or not isinstance(snap.get("landed"), bool):
            return False
        if require_position:
            if not all(_finite(snap.get("position", {}).get(axis)) for axis in ("x", "y", "z")):
                return False
            if not all(_finite(snap.get("velocity", {}).get(axis)) for axis in ("x", "y", "z")):
                return False
            if not _finite(snap.get("yaw")):
                return False
        return True

    def status(self):
        with self._lock:
            snap = self._read()
            llm = self.config.get("llm", {})
            api_ready = bool(llm.get("enabled") is not False and llm.get("base_url") and llm.get("model") and
                             os.environ.get(llm.get("api_key_env", "UAV_LLM_API_KEY")))
            camera_status = snap.get("camera_status", {})
            camera_available = (isinstance(camera_status, dict) and camera_status.get("available") is True and
                                self._camera_stamp_fresh(camera_status.get("stamp")))
            vision_info = (self.vision.availability() if hasattr(self.vision, "availability") else
                           {"configured": self.vision.configured, "provider": "fixture", "auth_state": "fixture"})
            vision_ready = vision_info["configured"]
            photo_info = self.photo.availability()
            agent_mode = self.config.get("photo", {}).get("decision_mode") == "vlm"
            agent_info = self.photo_agent.availability() if agent_mode else {}
            if agent_mode:
                photo_info = dict(photo_info, ready=photo_info["ready"] and agent_info.get("configured") is True,
                                  provider="photo_vlm",
                                  reason=photo_info["reason"] or ("请先完成摄影模型授权并选择模型。" if not agent_info.get("configured") else ""))
            return {"server_time": time.time(), "mode": snap.get("mode", self.config.get("mode", "unknown")),
                    "connected": snap.get("connected") is True,
                    "telemetry_fresh": self._fresh(snap), "vehicle": copy.deepcopy(snap),
                    "mission": copy.deepcopy(self._mission),
                    "previous_mission": copy.deepcopy(self._previous_mission),
                    "capabilities": {"natural_language": True, "vision": vision_ready,
                                     "photo_supported": photo_info["supported"],
                                     "photo_ready": photo_info["ready"] and camera_available,
                                     "photo_reason": photo_info["reason"] or ("机载相机尚未就绪。" if not camera_available else ""),
                                     "photo_provider": photo_info["provider"],
                                     "photo_decision_mode": "vlm" if agent_mode else "local",
                                     "photo_agent_configured": agent_info.get("configured") is True,
                                     "photo_agent_model": self.config.get("vision", {}).get("model") if agent_mode else None,
                                     "photo_agent_reasoning_effort": self.config.get("photo", {}).get("reasoning_effort") if agent_mode else None,
                                     "photo_task_schema_version": 3 if agent_mode else None,
                                     "photo_max_shots": 6 if agent_mode else None,
                                     "photo_measurement_scope": PHOTO_MEASUREMENT_SCOPE if agent_mode else None,
                                     "photo_agent_limits": copy.deepcopy(self.photo_limits) if agent_mode else None,
                                     "photo_search_supported": callable(getattr(self.adapter, "start_photo_search", None)) and callable(getattr(self.adapter, "stop_photo_search", None)),
                                     "vision_configured": vision_ready,
                                     "vision_provider": vision_info["provider"],
                                     "vision_auth_state": vision_info["auth_state"],
                                     "vision_model": self.config.get("vision", {}).get("model") if vision_ready else None,
                                     "vision_actions": ["inspect", "approach"],
                                     "text_api_configured": api_ready,
                                     "navigation_mode": self.config.get("navigation_mode", "preset"),
                                     "home_captured": self._home is not None,
                                     "home": copy.deepcopy(self._home),
                                     "home_source": "grounded_odometry" if self._home is not None else None,
                                     "home_stamp": self._home_stamp,
                                     "camera": callable(getattr(self.adapter, "camera_frame", None)),
                                     "camera_available": camera_available,
                                     "yaw_target": snap.get("capabilities", {}).get("yaw_target", True),
                                     "parser_mode": "offline_then_api" if api_ready else "offline",
                                     "model": llm.get("model") if api_ready else None,
                                     "takeoff_height": snap.get("takeoff_height", self.config["takeoff_height"]),
                                     "bounds": self.config["bounds"], "landmarks": self.config["landmarks"]},
                    "photo_cleanup_pending": bool(getattr(self, "_photo_cleanup", None) is not None and not self._photo_cleanup.done()),
                    "photo_confirming": self._photo_confirming,
                    "photo_search_cleanup_pending": bool(getattr(self, "_search_cleanup", None) is not None and not self._search_cleanup.done()),
                    "last_error": self._last_error}

    def _store_evidence(self, observation):
        jpeg = observation.get("data")
        if not isinstance(jpeg, bytes) or len(jpeg) > 5 * 1024 * 1024:
            raise ValueError("视觉证据缺少有效 JPEG。")
        evidence_id = str(uuid.uuid4())
        metadata = {"id": evidence_id, "url": "/api/evidence/" + evidence_id + ".jpg",
                    "stamp": observation["stamp"], "source": "onboard_rgb",
                    "image_sha256": hashlib.sha256(jpeg).hexdigest(),
                    "frame_id": observation.get("observation_id"),
                    "pose": {"position": copy.deepcopy(observation["position"]), "yaw": observation["yaw"]}}
        for field in ("width", "height", "ros_stamp"):
            if field in observation:
                metadata[field] = observation[field]
        for field in ("orientation", "body_frame"):
            if field in observation:
                metadata["pose"][field] = copy.deepcopy(observation[field])
        with self._lock:
            self._evidence[evidence_id] = (jpeg, metadata)
            while len(self._evidence) > 32 or sum(len(value[0]) for value in self._evidence.values()) > 32 * 1024 * 1024:
                self._evidence.popitem(last=False)
        try:
            self._evidence_store.submit(jpeg, metadata)
        except (OSError, ValueError):
            pass  # Diagnostics storage cannot authorize or block a flight.
        return copy.deepcopy(metadata)

    def evidence_frame(self, evidence_id):
        with self._lock:
            stored = self._evidence.get(evidence_id)
        if stored is None:
            try:
                stored = self._evidence_store.read(evidence_id)
            except (OSError, ValueError):
                raise BridgeError("evidence_missing", "审查图片已超出缓存保留范围；文字日志仍保留。", 404) from None
        return stored[0], copy.deepcopy(stored[1])

    def _require_stationary_airborne(self, snap):
        if not self._fresh(snap) or not snap["armed"] or snap["landed"]:
            raise BridgeError("vision_state", "请先起飞并稳定悬停，再基于当前画面规划或观察。", 409)
        if math.sqrt(sum(snap["velocity"][a] ** 2 for a in ("x", "y", "z"))) > 0.15:
            raise BridgeError("vision_state", "视觉任务需要无人机先稳定悬停。", 409)

    @staticmethod
    def _pose_matches(snap, observation):
        position = observation.get("position", {})
        if not all(_finite(position.get(a)) for a in ("x", "y", "z")) or not _finite(observation.get("yaw")):
            return False
        distance = math.sqrt(sum((snap["position"][a] - position[a]) ** 2 for a in ("x", "y", "z")))
        yaw = abs((snap["yaw"] - observation["yaw"] + math.pi) % (2 * math.pi) - math.pi)
        if distance > 0.15 or yaw > math.radians(5):
            return False
        if "orientation" in observation:
            qa, qb = snap.get("orientation", {}), observation["orientation"]
            if not isinstance(qa, dict) or not isinstance(qb, dict) or not all(_finite(q.get(k)) for q in (qa, qb) for k in ("x", "y", "z", "w")):
                return False
            na = math.sqrt(sum(qa[k] ** 2 for k in ("x", "y", "z", "w")))
            nb = math.sqrt(sum(qb[k] ** 2 for k in ("x", "y", "z", "w")))
            if na < 1e-6 or nb < 1e-6:
                return False
            cosine = min(1.0, abs(sum(qa[k] * qb[k] for k in ("x", "y", "z", "w")) / (na * nb)))
            if 2 * math.acos(cosine) > math.radians(5):
                return False
        if observation.get("body_frame") and snap.get("body_frame") != observation["body_frame"]:
            return False
        return True

    def _validate_executable_plan(self, proposed):
        if not isinstance(proposed, dict) or set(proposed) != {"summary", "steps"} or not isinstance(proposed["steps"], list):
            raise ValueError("视觉计划结构无效。")
        if not 1 <= len(proposed["steps"]) <= self.config["max_steps"]:
            raise ValueError("视觉计划步骤数量无效。")
        clean = []
        for step in proposed["steps"]:
            if isinstance(step, dict) and step.get("action") == "inspect":
                if set(step) != {"action", "question"} or not isinstance(step["question"], str) or not 1 <= len(step["question"].strip()) <= 1000:
                    raise ValueError("inspect 需要 1 到 1000 字符的 question。")
                clean.append({"action": "inspect", "question": step["question"].strip()})
            else:
                clean.extend(validate_plan({"summary": proposed["summary"], "steps": [step]}, self.config)["steps"])
        if not isinstance(proposed["summary"], str) or not 1 <= len(proposed["summary"].strip()) <= 2000:
            raise ValueError("视觉任务摘要无效。")
        return {"summary": proposed["summary"].strip(), "steps": clean}

    def _validate_visual_execution(self, visual, snap):
        self._require_stationary_airborne(snap)
        if not self._pose_matches(snap, visual["observation"]):
            raise BridgeError("visual_pose_changed", "无人机已移动超过 0.15 米或转向超过 5 度，请重新预览。", 409)
        try:
            target = visual.get("visual_target")
            if target:
                # Recheck the original registered sensor bundle as well as the
                # fresh image. This validates its full orientation and lifetime,
                # not just a new frame that might hide intervening roll/pitch.
                self.adapter.resolve_visual_target(visual["observation"], target["bbox"], target["standoff_m"])
            current = self.adapter.visual_observation()
            if not self._pose_matches(snap, current) or not self._pose_matches(current, visual["observation"]):
                raise ValueError("新画面位姿与已确认预览不一致。")
            if target:
                resolved = self.adapter.resolve_visual_target(current, target["bbox"], target["standoff_m"])
                goal = resolved["goal"]
                validate_plan({"summary": "视觉目标复核", "steps": [dict(goal, action="goto")]}, self.config)
                difference = math.sqrt(sum((goal[a] - target["goal"][a]) ** 2 for a in ("x", "y", "z")))
                if difference > 0.25:
                    raise ValueError("当前 RGB/雷达重新定位与预览目标相差超过 0.25 米。")
                yaw_difference = abs((goal["yaw"] - target["goal"]["yaw"] + math.pi) % (2 * math.pi) - math.pi)
                if yaw_difference > math.radians(5):
                    raise ValueError("当前目标朝向与预览相差超过 5 度。")
                # The confirmed endpoint remains immutable. A newly safe endpoint
                # does not prove the old one safe. Distance to a measured point
                # set is 1-Lipschitz, so subtracting endpoint displacement gives
                # a conservative lower bound for the old endpoint's clearance.
                evidence = resolved.get("evidence", {})
                fields = ("target_standoff_m", "measurement_margin_m", "endpoint_clearance_m", "endpoint_clearance_required_m")
                if all(_finite(evidence.get(field)) for field in fields):
                    horizontal = math.hypot(goal["x"] - target["goal"]["x"], goal["y"] - target["goal"]["y"])
                    if evidence["target_standoff_m"] - horizontal + 1e-6 < target["standoff_m"] + evidence["measurement_margin_m"]:
                        raise ValueError("新回波无法证明已确认目标仍保留所需站距与测量余量。")
                    if evidence["endpoint_clearance_m"] - difference + 1e-6 < evidence["endpoint_clearance_required_m"]:
                        raise ValueError("新回波无法证明已确认目标仍保留所需障碍净空。")
                elif difference > 1e-6:
                    raise ValueError("目标发生变化且缺少验证原目标站距/净空的证据。")
        except (ValueError, RuntimeError, KeyError, AttributeError) as exc:
            raise BridgeError("visual_target_changed", "视觉目标复核失败，请重新预览：%s" % exc, 409)

    def _start_inspection(self, question):
        if not self._inspection_slots.acquire(blocking=False):
            raise RuntimeError("视觉检查仍在处理此前请求，请稍后再试。")
        future = Future()
        def worker():
            try:
                observation = self.adapter.visual_observation()
                answer = self.vision.inspect(observation, question)
                future.set_result({"answer": answer, "observation": observation})
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self._inspection_slots.release()
        threading.Thread(target=worker, name="visual-inspection", daemon=True).start()
        return future

    def _camera_stamp_fresh(self, stamp):
        return _finite(stamp) and -1.0 <= time.time() - stamp <= self.config.get("camera_timeout_seconds", 2.5)

    def camera_frame(self):
        """Read an already encoded camera frame without taking the flight lock.

        The adapter owns capture/encoding and exposes a cheap immutable snapshot.
        This path must never render, decode, substitute an overview, or retain an
        old image as if it were live camera feedback.
        """
        provider = getattr(self.adapter, "camera_frame", None)
        if not callable(provider):
            raise BridgeError("camera_unavailable", "当前模式没有可用的机载 RGB 相机。", 503)
        try:
            frame = provider()
        except Exception:
            raise BridgeError("camera_unavailable", "无法读取机载 RGB 相机帧。", 503)
        if not isinstance(frame, dict):
            raise BridgeError("camera_unavailable", "机载 RGB 相机尚未提供图像。", 503)
        stamp = frame.get("stamp")
        if not self._camera_stamp_fresh(stamp):
            if _finite(stamp) and stamp > 0:
                raise BridgeError("camera_stale", "机载 RGB 图像已过期；未返回缓存旧帧。", 503)
            raise BridgeError("camera_unavailable", "机载 RGB 相机尚未提供图像。", 503)
        if frame.get("available") is not True:
            raise BridgeError("camera_unavailable", str(frame.get("reason") or "机载 RGB 相机当前不可用。")[:300], 503)
        jpeg = frame.get("data")
        if (not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= 5 * 1024 * 1024 or
                not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9") or
                frame.get("content_type", "image/jpeg") != "image/jpeg"):
            raise BridgeError("camera_unavailable", "机载 RGB 图像编码无效。", 503)
        metadata = {"stamp": stamp}
        for field in ("width", "height"):
            value = frame.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 16384:
                metadata[field] = value
        return jpeg, metadata

    def preview(self, body):
        self._fields(body, {"text"}, {"text"})
        text = body["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise BridgeError("invalid_text", "text must contain 1–2000 characters")
        with self._lock:
            self._read()  # Capture a genuine launch origin before parsing return.
        if self.config.get("photo", {}).get("decision_mode") == "vlm":
            # This mode parses the complete user request with the real model.
            # No keyword or legacy photography grammar substitutes for it.
            return self._preview_photo_agent(text)
        if re.search(r"拍|全身照|\bphoto(?:graph)?\b", text, re.I):
            return self._preview_photo(text)
        visual = None
        # Network calls may take seconds; never hold the flight lock while parsing.
        try:
            try:
                rules_config = copy.deepcopy(self.config)
                rules_config["llm"] = {"enabled": False}
                proposed, source = Planner(rules_config).preview_with_source(text)
                plan = validate_plan(proposed, self.config)
            except ValueError as exc:
                # Known invalid numeric actions remain errors; never ask a model
                # to silently repair an out-of-bounds or malformed coordinate.
                if not str(exc).startswith(("无法完整解析", "当前版本尚未接入视觉识别")):
                    raise
                visual_intent = re.search(r"观察|检查|看|画面|镜头|视觉|靠近|接近|眼前|前方|识别|辨认|跟随|inspect|look|see|camera|visible|approach|recogniz|detect", text, re.IGNORECASE)
                if self.config.get("navigation_mode") != "perception" and not visual_intent:
                    proposed, source = self.planner.preview_with_source(text)
                    plan = validate_plan(proposed, self.config)
                else:
                    with self._lock:
                        self._require_stationary_airborne(self._read())
                        if self._mission and self._mission["state"] == "running":
                            raise BridgeError("mission_busy", "请先等待当前任务完成或悬停，再冻结视觉预览。", 409)
                    visual = self.vision.preview(text)
                    plan, source = self._validate_executable_plan(visual["plan"]), "vision_api"
                    with self._lock:
                        snap = self._read()
                        self._require_stationary_airborne(snap)
                        if not self._pose_matches(snap, visual["observation"]):
                            raise BridgeError("visual_pose_changed", "解析期间位姿已改变，请重新获取画面并预览。", 409)
        except (ValueError, RuntimeError) as exc:
            raise BridgeError("invalid_plan", str(exc), 422, details=getattr(exc, "diagnostics", None))
        plan_id = str(uuid.uuid4())
        digest = hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()
        ttl = min(30, self.config["plan_ttl_seconds"]) if visual else self.config["plan_ttl_seconds"]
        expires = time.time() + ttl
        result = dict(plan, plan_id=plan_id, plan_hash=digest, expires_at=expires,
                      expires_in_seconds=ttl, parser=source,
                      vision={"configured": self.vision.configured, "used": visual is not None}, evidence=[], grounding=None)
        if visual:
            result["evidence"] = [self._store_evidence(visual["observation"])]
            target = visual.get("visual_target")
            if target:
                result["grounding"] = dict(copy.deepcopy(target["evidence"]), bbox=target["bbox"],
                                           standoff_m=target["standoff_m"], goal=target["goal"],
                                           label=plan["steps"][0].get("label", "视觉目标"))
        with self._lock:
            self._plans[plan_id] = {"plan": copy.deepcopy(plan), "hash": digest,
                                    "expires": time.monotonic() + ttl, "used": False,
                                    "visual": visual, "evidence": result["evidence"], "parser": source}
            while len(self._plans) > 128:
                self._plans.popitem(last=False)
        return result

    def photo_frame(self, photo_id):
        try:
            return self.photo_store.get(photo_id)
        except (OSError, ValueError, KeyError):
            raise BridgeError("photo_missing", "照片不存在或完整性校验失败。", 404)

    def _require_photo_state(self, snap):
        real_ready = (snap.get('mode') == 'real' and snap.get('capabilities', {}).get('yaw_target') is True
                      and snap.get('real_execution', {}).get('backend') == 'diff_vla_real'
                      and snap.get('real_execution', {}).get('ready') is True)
        if snap.get("mode") != "gazebo" and not real_ready:
            raise BridgeError("photo_mode", "实机任务执行服务尚未就绪，请检查设备日志。", 409)
        if not self._fresh(snap) or not snap["armed"] or snap["landed"]:
            raise BridgeError("photo_state", "请先点击起飞并稳定悬停，再预览人物摄影任务。", 409)
        if math.sqrt(sum(snap["velocity"][a] ** 2 for a in ("x", "y", "z"))) > .15:
            raise BridgeError("photo_state", "请等待无人机稳定悬停，再预览人物摄影任务。", 409)

    def _preview_capture(self, text, visual=None, search_binding=None):
        with self._lock:
            self._require_photo_state(self._read())
            if self._mission and self._mission["state"] == "running" and search_binding is None:
                raise BridgeError("mission_busy", "请先等待当前任务完成或悬停，再预览拍照任务。", 409)
        try:
            visual = self.photo.preview(text) if visual is None else visual
            plan = visual["plan"]
            with self._lock:
                snap = self._read()
                self._require_photo_state(snap)
                if (not self._camera_stamp_fresh(visual["observation"]["stamp"]) or
                        not self._pose_matches(snap, visual["observation"])):
                    raise BridgeError("photo_pose_changed", "识别人像期间无人机位姿变化，请重新预览。", 409)
        except (ValueError, RuntimeError) as exc:
            raise BridgeError("photo_unavailable", str(exc), 422,
                              details={"geometry_retry": getattr(exc, "photo_retry", None), "geometry_code": getattr(exc, "code", None)})
        plan_id = str(uuid.uuid4())
        digest = hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()
        ttl = min(30., float(self.config.get("photo", {}).get("preview_ttl_seconds", 30)), self.config["plan_ttl_seconds"])
        if search_binding is not None:
            ttl = min(ttl, self._subject_context(search_binding['search_id'], search_binding['candidate_id'])['expires'] - time.monotonic())
        evidence = [self._store_evidence(visual["observation"])]
        target = visual["photo_target"]
        result = dict(plan, plan_id=plan_id, plan_hash=digest, expires_at=time.time() + ttl,
                      expires_in_seconds=ttl, parser="photo_local",
                      vision={"configured": True, "used": True, "provider": "local_person_detector"},
                      evidence=evidence,
                      grounding=dict(copy.deepcopy(target["evidence"]), bbox=visual["bbox"], goal=target["goal"]),
                      photo={"phase": "capture", "subject_selection": copy.deepcopy(search_binding),
                             "search_id": search_binding.get('search_id') if search_binding else None,
                             "candidate_id": search_binding.get('candidate_id') if search_binding else None,
                             "composition": dict(copy.deepcopy(visual["composition"]), passed=bool(visual["composition"].get("ready"))),
                             "geometry_retry": copy.deepcopy(visual.get("geometry_retry")),
                             "movement": {"required": bool(target["requires_motion"]), "goal": target["goal"],
                                          "distance_m": math.sqrt(sum((target['goal'][a] - visual['observation']['position'][a])**2 for a in ('x','y','z')))},
                             "detector": self.photo.detector_metadata(visual["detection"]),
                             "scope": "仅拍摄你确认的静止候选；不识别身份，不追踪移动或交叉人物。"})
        with self._lock:
            self._plans[plan_id] = {"plan": copy.deepcopy(plan), "hash": digest,
                                    "expires": time.monotonic() + ttl, "used": False,
                                    "visual": visual, "evidence": evidence, "parser": "photo_local", "search_binding": copy.deepcopy(search_binding)}
            while len(self._plans) > 128:
                self._plans.popitem(last=False)
        return result

    def _preview_photo_agent(self, text):
        """Parse without the control lock, then bind the complete task to search."""
        with self._lock:
            self._require_photo_state(self._read())
            self._photo_admission_check('execute')
            if self._mission and self._mission['state'] == 'running':
                raise BridgeError('mission_busy', '请先完成当前任务或点击悬停，再解析摄影任务。', 409)
            local = self.photo.availability()
            if not local['ready']:
                raise BridgeError('photo_unavailable', local['reason'], 422)
            if not self.photo_agent.availability().get('configured'):
                raise BridgeError('photo_agent_unavailable', '请先完成摄影模型授权并选择模型；不会改用固定规则。', 422)
            epoch = self._command_epoch
        task = self._parse_photo_agent_text(text)
        with self._lock:
            self._require_photo_state(self._read())
            self._photo_admission_check('execute')
            if epoch != self._command_epoch or (self._mission and self._mission['state'] == 'running'):
                raise BridgeError('photo_parse_changed', '解析期间收到其他指令，未采用迟到任务；请重新预览。', 409)
            plan = {'summary': task['summary'] + '；先原地搜索，再由你确认人物',
                    'steps': [{'action': 'search_person'}], 'photo_task': copy.deepcopy(task),
                    'photo_limits': copy.deepcopy(self.photo_limits)}
            plan_id = str(uuid.uuid4())
            digest = hashlib.sha256(_canonical(plan).encode('utf-8')).hexdigest()
            ttl = min(180., self.config['plan_ttl_seconds'])
            self._plans[plan_id] = {'plan': copy.deepcopy(plan), 'hash': digest,
                                    'expires': time.monotonic() + ttl, 'used': False,
                                    'visual': None, 'evidence': [], 'parser': 'photo_vlm',
                                    'search_request': {'text': text, 'task': copy.deepcopy(task)}}
            while len(self._plans) > 128:
                self._plans.popitem(last=False)
        return dict(plan, plan_id=plan_id, plan_hash=digest, expires_at=time.time() + ttl,
                    expires_in_seconds=ttl, parser='photo_vlm', evidence=[], grounding=None,
                    vision={'configured': True, 'used': False, 'provider': 'photo_vlm'},
                    photo={'phase': 'search', 'task': copy.deepcopy(task),
                           'limits': copy.deepcopy(self.photo_limits),
                           'measurement_scope': PHOTO_MEASUREMENT_SCOPE,
                           'scope': '模型已解析完整文字任务；先原地搜人，选人确认后才开始视觉机位判断。'})

    def _parse_photo_agent_text(self, text):
        """Use one model contract for both read-only parsing and flight preview."""
        log_id = str(uuid.uuid4())
        self.task_log.event(log_id, 'parsing', '正在解析：'+text.strip(), model=self.config.get('vision',{}).get('model'))
        try:
            task = self.photo_agent.parse_task(text)
        except PhotoAgentError as exc:
            self.task_log.event(log_id, 'parse_failed', exc.message, error_code=exc.code)
            raise BridgeError(exc.code, exc.message, 422, details=exc.details) from None
        # Reject mixed-version providers rather than reinterpret an old task
        # through a fixed framing/view/gesture template. The service validates
        # requirement semantics; this boundary binds its complete frozen result.
        if (not isinstance(task, dict) or task.get('schema_version') not in (2,3) or
                task.get('user_request') != text.strip() or not isinstance(task.get('requirements'), list) or
                not task['requirements']):
            raise BridgeError('photo_task_contract', '模型任务缺少完整的开放要求或原话绑定，请检查摄影服务版本。', 422)
        task_content = {key: value for key, value in task.items() if key != 'task_sha256'}
        try:
            expected_task_hash = hashlib.sha256(_canonical(task_content).encode('utf-8')).hexdigest()
        except (ValueError, TypeError, OverflowError):
            raise BridgeError('photo_task_contract', '模型任务包含无效数据，未生成执行计划。', 422) from None
        if (not isinstance(task.get('task_sha256'), str) or
                not hmac.compare_digest(task['task_sha256'].encode('utf-8'), expected_task_hash.encode('ascii'))):
            raise BridgeError('photo_task_binding', '模型任务内容与任务摘要校验不一致，未生成执行计划。', 422)
        try:
            sequence = task_shots(task)
        except ValueError as error:
            raise BridgeError('photo_task_contract', str(error), 422) from None
        self.task_log.event(log_id, 'parsed', '模型已解析 %d 张照片：%s' % (len(sequence), task['summary']),
                            task_id=task['id'], shots=sequence)
        return copy.deepcopy(task)

    def parse_photo_task(self, body):
        """Return model understanding only; this route cannot create a plan."""
        self._fields(body, {'text'}, {'text'})
        text = body['text']
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise BridgeError('invalid_text', 'text must contain 1–2000 characters')
        with self._lock:
            if (self.config.get('photo', {}).get('decision_mode') != 'vlm' or
                    not self.photo_agent.availability().get('configured')):
                raise BridgeError('photo_agent_unavailable', '请先启用摄影模型并完成授权；只解析入口不会改用规则模板。', 422)
            if self._mission and self._mission['state'] == 'running':
                raise BridgeError('mission_busy', '当前任务正在执行，请结束后再单独解析新任务。', 409)
        task = self._parse_photo_agent_text(text)
        return dict(parser='photo_vlm', task=task, executable=False,
                    limits=copy.deepcopy(self.photo_limits), measurement_scope=PHOTO_MEASUREMENT_SCOPE,
                    vision={'configured': True, 'used': False, 'provider': 'photo_vlm'},
                    scope='仅展示模型对完整原话的理解；未创建可执行计划，也未启动搜索或飞行。')

    def _stationary_subject_confirmation(self):
        """Real demo choices bind a measured person, not an old camera pose.

        This only removes historical-pose admission: each preview/confirmation
        still measures a new frame, associates the original world geometry,
        and checks its current pose before installing the original guard.
        """
        photo = self.config.get('photo', {})
        return (self.config.get('mode') == 'real' and photo.get('navigation_scope') == 'planner' and
                photo.get('subject_motion_mode') == 'stationary' and photo.get('decision_mode') == 'vlm')

    def _preview_photo_agent_selection(self, context, visual, binding):
        snap = self._read()
        self._require_photo_state(snap)
        if (not self._camera_stamp_fresh(visual['observation']['stamp']) or
                not self._pose_matches(snap, visual['observation'])):
            raise BridgeError('photo_pose_changed', '选人复核期间位姿变化，请重新选择。', 409)
        task = context['task']
        plan = {'summary': task['summary'], 'steps': [{'action': 'photo_agent'}],
                'photo_task': copy.deepcopy(task), 'photo_limits': copy.deepcopy(self.photo_limits),
                'subject_selection': copy.deepcopy(binding)}
        plan_id = str(uuid.uuid4())
        digest = hashlib.sha256(_canonical(plan).encode('utf-8')).hexdigest()
        ttl = min(90., self.config['plan_ttl_seconds'], context['expires'] - time.monotonic())
        if ttl <= 0:
            raise BridgeError('photo_subject_expired', '人物候选已过期，请重新搜索。', 409)
        evidence = [self._store_evidence(visual['observation'])]
        self._plans[plan_id] = {'plan': copy.deepcopy(plan), 'hash': digest,
                                'expires': time.monotonic() + ttl, 'used': False,
                                'visual': visual, 'evidence': evidence, 'parser': 'photo_vlm',
                                'search_binding': copy.deepcopy(binding)}
        while len(self._plans) > 128:
            self._plans.popitem(last=False)
        return dict(plan, plan_id=plan_id, plan_hash=digest, expires_at=time.time() + ttl,
                    expires_in_seconds=ttl, parser='photo_vlm', evidence=evidence, grounding=None,
                    vision={'configured': True, 'used': False, 'provider': 'photo_vlm'},
                    photo={'phase': 'agent', 'task': copy.deepcopy(task),
                           'limits': copy.deepcopy(self.photo_limits),
                           'measurement_scope': PHOTO_MEASUREMENT_SCOPE,
                           'search_id': binding['search_id'], 'candidate_id': binding['candidate_id'],
                           'subject_selection': copy.deepcopy(binding),
                           'scope': '授权模型在约定范围内调整机位；原样验证每次提议，保存模型审查的原图，再执行约定结束动作。'})

    def _preview_photo(self, text):
        try:
            intent = parse_photo_request(text)
            if intent is None:
                raise ValueError('不是受支持的摄影指令。')
            if not self.photo.availability()['ready']:
                raise ValueError(self.photo.availability()['reason'])
        except (ValueError, RuntimeError) as exc:
            raise BridgeError('photo_unavailable', str(exc), 422)
        with self._lock:
            self._require_photo_state(self._read())
            self._photo_admission_check('execute')
            if self._mission and self._mission['state'] == 'running':
                raise BridgeError('mission_busy', '请先完成当前任务或点击悬停，再开始搜索。', 409)
            if not all(callable(getattr(self.adapter, name, None)) for name in ('start_photo_search', 'stop_photo_search')):
                raise BridgeError('photo_search_unavailable', '当前控制器没有经过验证的原地搜索接口。', 409)
            plan = {'summary': '原地慢转寻找拍照候选；找到后等待你确认人物', 'steps': [{'action': 'search_person'}]}
            plan_id = str(uuid.uuid4())
            digest = hashlib.sha256(_canonical(plan).encode('utf-8')).hexdigest()
            ttl = min(30., self.config['plan_ttl_seconds'])
            self._plans[plan_id] = {'plan': plan, 'hash': digest, 'expires': time.monotonic() + ttl,
                                    'used': False, 'visual': None, 'evidence': [], 'parser': 'photo_local',
                                    'search_request': {'text': text, 'intent': intent}}
            while len(self._plans) > 128:
                self._plans.popitem(last=False)
        return dict(plan, plan_id=plan_id, plan_hash=digest, expires_at=time.time() + ttl,
                    expires_in_seconds=ttl, parser='photo_local', evidence=[], grounding=None,
                    vision={'configured': True, 'used': False, 'provider': 'local_person_detector'},
                    photo={'phase': 'search', 'scope': '最多原地转一圈；只搜索可见候选，不判断身份，不横移找人。'})

    def _subject_context(self, search_id, candidate_id=None):
        context = self._search_context
        if (context is None or context['id'] != search_id or context['cancelled'].is_set() or
                context['phase'] != 'awaiting_subject' or not self._mission or self._mission['state'] != 'running'):
            raise BridgeError('photo_search_missing', '搜索已取消、失效或尚未找到可确认人物，请重新搜索。', 409)
        if time.monotonic() >= context['expires']:
            raise BridgeError('photo_subject_expired', '人物候选已过期，请重新搜索。', 409)
        if candidate_id is not None and candidate_id not in context['candidates']:
            raise BridgeError('photo_candidate_missing', '候选编号不属于本次搜索。', 404)
        return context

    def _search_binding_valid(self, record):
        binding = record.get('search_binding') if record else None
        if not binding:
            return False
        context = self._subject_context(binding['search_id'], binding['candidate_id'])
        if binding['revision'] != context['selection_revision']:
            raise BridgeError('photo_selection_changed', '已有更新的人物选择预览，请确认最新计划。', 409)
        return True

    def subject_preview(self, body):
        self._fields(body, {'search_id', 'candidate_id'}, {'search_id', 'candidate_id'})
        if any(not isinstance(body[key], str) or not re.fullmatch(r'[a-f0-9-]{36}', body[key]) for key in body):
            raise BridgeError('invalid_subject', 'search_id 与 candidate_id 必须来自当前候选画面。')
        if not self._subject_preview_slots.acquire(blocking=False):
            raise BridgeError('photo_busy', '正在复核另一项人物选择，请稍后重试。', 409)
        try:
            with self._lock:
                context = self._subject_context(body['search_id'], body['candidate_id'])
                self._require_photo_state(self._read())
                if self._photo_confirming:
                    raise BridgeError('photo_busy', '已开始执行摄影确认，请勿重复选择。', 409)
                index = context['candidates'][body['candidate_id']]
                epoch = self._command_epoch
                # Refresh may retire this frame while measurement is running.
                # Keep an immutable local reference; final admission still
                # checks the live candidate ID and command epoch.
                frozen = context['frozen']
            try:
                if context.get('task'):
                    reference = make_reference(frozen, index)
                    fresh = measure_selected(self.photo, reference,
                                             after=frozen['observation']['ros_stamp'],
                                             cancelled=context['cancelled'])
                    # Keep the original per-person guard centres and association
                    # baseline. Re-basing them on each fresh sample would let
                    # several individually small movements escape the guard.
                    visual = dict(reference, kind='photo_agent', observation=fresh['observation'],
                                  task=copy.deepcopy(context['task']), initial=fresh)
                else:
                    visual = self.photo.preview_subject(context['text'], frozen, index)
            except (ValueError, RuntimeError) as exc:
                with self._lock:
                    if (self._search_context is context and epoch == self._command_epoch and
                            getattr(exc, 'code', None) == 'photo_subject_changed'):
                        self._abort('人物选择关联失效：%s' % exc)
                raise BridgeError('photo_subject_unavailable', str(exc), 422,
                                  details={'geometry_code': getattr(exc, 'code', None), 'geometry_retry': getattr(exc, 'photo_retry', None)})
            with self._lock:
                self._subject_context(body['search_id'], body['candidate_id'])
                if self._search_context is not context or epoch != self._command_epoch:
                    raise BridgeError('photo_search_changed', '选择期间搜索已取消或状态改变。', 409)
                if self._photo_confirming:
                    raise BridgeError('photo_busy', '已开始执行摄影确认，请勿重复选择。', 409)
                context['selection_revision'] += 1
                self.task_log.event(context['id'], 'subject_selected', '已选择人物，正在复核新鲜测量；确认后执行摄影。',
                                    candidate_id=body['candidate_id'])
                binding = {'search_id': context['id'], 'candidate_id': body['candidate_id'],
                           'revision': context['selection_revision']}
                if context.get('task'):
                    return self._preview_photo_agent_selection(context, visual, binding)
                return self._preview_capture(context['text'], visual=visual, search_binding=binding)
        finally:
            self._subject_preview_slots.release()

    def _schedule_search_cleanup(self, search_id):
        future = Future()
        self._search_cleanup = future
        def cleanup():
            try:
                future.set_result(self.adapter.stop_photo_search(search_id))
            except Exception as exc:
                future.set_exception(exc)
        threading.Thread(target=cleanup, name='photo-search-cleanup', daemon=True).start()

    def _release_search_context(self):
        context = self._search_context
        self._search_context = None
        if context is not None:
            context['cancelled'].set()
            self._schedule_search_cleanup(context['id'])

    def _search_job(self, kind, context):
        if not self._search_slots.acquire(blocking=False):
            raise RuntimeError('前一次搜索检测仍在结束，请稍后重试。')
        future = Future()
        def worker():
            try:
                if context['cancelled'].is_set():
                    raise ValueError('搜索已取消。')
                if kind == 'start':
                    result = self.adapter.start_photo_search(context['id'], max_sweep_rad=2*math.pi, timeout=30., rate=.35)
                elif kind == 'stop':
                    result = self.adapter.stop_photo_search(context['id'])
                elif kind == 'scan':
                    result = self.photo.scan_candidates(after=context.get('last_ros_stamp'), cancelled=context['cancelled'])
                else:
                    result = self.photo.freeze_candidates(after=context.get('last_ros_stamp'), cancelled=context['cancelled'])
                if kind in ('scan', 'freeze') and context.get('scene_memory') is not None:
                    camera = None
                    try:
                        # Search history needs calibration/pose only. Optional
                        # full-frame target clustering must not age this scan.
                        camera = self.adapter.photo_camera_context(
                            result['observation'], include_demo_targets=False)
                    except (AttributeError, ValueError, RuntimeError):
                        pass
                    context['scene_memory'].add(result['observation'], camera, phase='search')
                if context['cancelled'].is_set():
                    # The adapter remembers stop-before-start tombstones, so a
                    # delayed start cannot resurrect a cancelled rotation.
                    if kind == 'start':
                        self.adapter.stop_photo_search(context['id'])
                    raise ValueError('搜索已取消，未采纳迟到结果。')
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self._search_slots.release()
        context.update(future=future, job_kind=kind, job_started=time.monotonic())
        threading.Thread(target=worker, name='photo-search-' + kind, daemon=True).start()

    def _refresh_photo_subjects(self, context):
        """Retire old evidence, then reobserve in hover without a new sweep.

        The first candidate's absolute confirmation deadline is retained.
        Incrementing the epoch also invalidates confirmation work that is
        already measuring outside the mission lock.
        """
        self._command_epoch += 1
        context['selection_revision'] += 1
        context.update(phase='settling', future=None, frozen=None, candidates={},
                       stable_since=None, stable_pose=None, refreshing=True)
        public = self._mission['photo_search']
        public.update(state='refreshing_subject', candidates=[], evidence=None,
                      waiting_reason='候选画面的机位已变化，正在悬停更新画面，请重新确认人物。')
        # Retain plan records for the normal expired/revision error response,
        # but ensure no previous choice can authorize a fresh candidate.
        for record in self._plans.values():
            if (record.get('search_binding') or {}).get('search_id') == context['id']:
                record['expires'] = 0.

    def _tick_photo_search(self, snap, now):
        context = self._search_context
        if context is None:
            self._abort('人物搜索上下文已失效。')
            return
        if not snap.get('armed') or snap.get('landed'):
            self._abort('搜索期间无人机不再处于已解锁飞行状态。')
            return
        if self.config.get('mode') == 'real' and snap.get('flight_mode') != 'OFFBOARD':
            self._abort('实机已退出 OFFBOARD，人物搜索已取消；恢复模式后需重新确认任务。')
            return
        public = self._mission['photo_search']
        controller = snap.get('photo_search') or {}
        if controller.get('id') == context['id']:
            public['angle_swept_rad'] = controller.get('angle_swept_rad', public['angle_swept_rad'])
        if context['phase'] != 'starting' and snap.get('command_error'):
            self._abort('搜索或人物确认已取消：%s' % snap['command_error'])
            return
        if 'expires' in context:
            public['expires_in_seconds'] = max(0., context['expires'] - now)
            if now >= context['expires']:
                public['state'] = 'expired'
                self._abort('人物确认等待已过期，已保持悬停；请重新搜索。')
                return
        if context['phase'] == 'awaiting_subject':
            if (not (context.get('task') and self._stationary_subject_confirmation()) and
                    not self._pose_matches(snap, context['frozen']['observation'])):
                self._refresh_photo_subjects(context)
            return
        if (('expires' not in context and now - context['started'] > 30.) or
                public['angle_swept_rad'] > 2*math.pi + .02):
            self._abort('原地搜索已到时限或一圈，未找到可确认人物，已保持悬停。')
            return
        future = context.get('future')
        if future is not None:
            job_limit = (2. if context['job_kind'] in ('start', 'stop') else
                         self.config.get('photo', {}).get('search_job_timeout_seconds', 5.))
            if not future.done() and now - context['job_started'] > job_limit:
                self._abort('人物搜索检测或控制确认超时，已停止转向。')
                return
            if not future.done():
                return
            try:
                result = future.result()
                kind = context['job_kind']
                context['future'] = None
                if kind == 'start':
                    context['phase'] = 'searching'
                    return  # The snapshot may predate the active ACK just received.
                elif kind == 'stop':
                    context['phase'] = 'settling'
                    context['stable_since'] = context['stable_pose'] = None
                    return  # Start the dwell from fresh post-stop telemetry.
                else:
                    observation = result['observation']
                    if not self._camera_stamp_fresh(observation['stamp']):
                        raise PhotoFramePending('检测期间候选画面已过期，重新获取新帧。')
                    context['last_ros_stamp'] = observation['ros_stamp']
                    context['last_scan'] = now
                    public.pop('waiting_reason', None)
                    context.pop('frame_wait_started', None)
                    if kind == 'scan':
                        # Search establishes a selectable person, not the final
                        # composition. A reliable candidate anywhere in the
                        # actual image is enough to stop and take a fresh frame.
                        visible = any(item['confidence'] >= max(.55, self.config.get('photo', {}).get('min_confidence', .6))
                                      for item in result['detection']['detections'])
                        if visible:
                            context['phase'] = 'stopping'
                            self._search_job('stop', context)
                    else:
                        if (not self._photo_search_stable_pose(snap, context['stable_pose']) or
                                not self._pose_matches(snap, observation) or observation['stamp'] <= context['stable_wall']):
                            # A stale pose invalidates this frame, not the whole
                            # choice workflow. No movement is reauthorized.
                            self._refresh_photo_subjects(context)
                            return
                        context['frozen'] = result
                        context['candidates'] = {str(uuid.uuid4()): index for index in range(len(result['detection']['detections']))}
                        context.setdefault('expires', now + 180.)
                        context['phase'] = 'awaiting_subject'
                        context['refreshing'] = False
                        evidence = self._store_evidence(observation)
                        public.update(state='awaiting_subject', evidence=evidence,
                                      expires_at=time.time() + context['expires'] - now,
                                      expires_in_seconds=context['expires'] - now,
                                      candidates=[{'id': identity, 'bbox': result['detection']['detections'][index]['bbox'],
                                                   'confidence': result['detection']['detections'][index]['confidence']}
                                                  for identity, index in context['candidates'].items()])
                        self._mission['vision_used'] = not bool(context.get('task'))
                        self._mission['evidence'].append(evidence)
                        self._record_photo_audit('candidate_freeze', True, result.get('geometry_retry'), observation)
                        return
            except (PhotoFramePending, PhotoValidationTimeout) as exc:
                # Keep the original sweep/time cap and physical protections.
                # A slow completed job may retry; its stale image is never used.
                context['future'] = None
                context['last_scan'] = now
                public['waiting_reason'] = str(exc)
                since = context.setdefault('frame_wait_started', now)
                if now-since >= self.config.get('photo', {}).get('frame_wait_timeout_seconds', 8.):
                    self._abort('搜索持续未取得可用的新鲜画面，已停止转向。')
                return
            except Exception as exc:
                self._abort('人物搜索失败：%s' % exc)
                return
        phase = context['phase']
        public['state'] = ('refreshing_subject' if context.get('refreshing') else 'searching')
        if phase == 'starting' and context['future'] is None:
            self._search_job('start', context)
        elif phase == 'searching':
            if controller.get('id') == context['id'] and (controller.get('error') or controller.get('active') is False):
                self._abort('原地搜索已停止：%s' % (controller.get('error') or '已完成一圈但未找到人物'))
                return
            if context['future'] is None and now - context.get('last_scan', 0) >= .25:
                self._search_job('scan', context)
        elif phase == 'settling':
            stamp = snap.get('position_stamp')
            if (not self._photo_search_stable_pose(snap, context.get('stable_pose')) or
                    not _finite(stamp) or not 0 <= time.time()-stamp <= self.config['telemetry_timeout_seconds']):
                context['stable_since'] = context['stable_pose'] = None
                return
            if context.get('stable_since') is None:
                context['stable_since'], context['stable_wall'] = now, time.time()
                context['stable_pose'] = copy.deepcopy(snap)
            if (context['future'] is None and now - context['stable_since'] >= .6 and
                    stamp-context['stable_pose']['position_stamp'] >= .6):
                self._search_job('freeze', context)

    def command(self, action, body):
        validation = None
        token = None
        accepted = False
        if action == "execute" and isinstance(body, dict):
            with self._lock:
                # Repeated accepted request IDs retain exactly-once behaviour.
                cached = isinstance(body.get("request_id"), str) and body["request_id"] in self._requests
                record = self._plans.get(body.get("plan_id")) if isinstance(body.get("plan_id"), str) else None
                photographic = bool(not cached and record and (record.get("visual") or {}).get("kind") in ("photo", "photo_agent"))
            if photographic:
                if not self._photo_confirm_slots.acquire(blocking=False):
                    raise BridgeError("photo_busy", "另一项摄影确认仍在复核，请稍后重试。", 409)
                try:
                    with self._lock:
                        self._fields(body, {"request_id", "plan_id", "plan_hash", "client_id", "client_seq"},
                                     {"request_id", "plan_id", "plan_hash"})
                        if (not isinstance(body.get("request_id"), str) or
                                not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", body["request_id"])):
                            raise BridgeError("invalid_request_id", "request_id must be a unique 1–128 character identifier")
                        if (not isinstance(body.get("plan_hash"), str) or
                                not hmac.compare_digest(record["hash"].encode("utf-8"), body["plan_hash"].encode("utf-8"))):
                            raise BridgeError("plan_changed", "Plan hash does not match the preview", 409)
                        client_fields = {"client_id", "client_seq"}
                        if set(body) & client_fields:
                            if (not client_fields <= set(body) or not isinstance(body["client_id"], str) or
                                    not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", body["client_id"]) or
                                    isinstance(body["client_seq"], bool) or not isinstance(body["client_seq"], int) or
                                    not 1 <= body["client_seq"] <= 9007199254740991):
                                raise BridgeError("invalid_client_sequence", "Provide client_id and a positive integer client_seq together")
                            if body["client_seq"] <= self._client_sequences.get(body["client_id"], 0):
                                raise BridgeError("stale_command", "A newer command from this client was already received", 409)
                        # Sequence reservation remains in _command's final lock;
                        # this side-effect-free admission also prevents invalid
                        # confirmation fields from activating controller guards.
                        if record["used"] or record["expires"] < time.monotonic():
                            raise BridgeError("plan_expired", "摄影预览已使用或过期，请重新预览。", 409)
                        self._require_photo_state(self._read())
                        self._photo_admission_check(action)
                        if record.get('search_binding'):
                            self._search_binding_valid(record)
                        if self._mission and self._mission["state"] == "running" and not self._search_binding_valid(record):
                            raise BridgeError("mission_busy", "A mission is running; hold or land before starting another", 409)
                        epoch = self._command_epoch
                        self._photo_confirming = True
                    try:
                        is_agent = record['visual'].get('kind') == 'photo_agent'
                        fresh = (measure_selected(self.photo, record['visual'],
                                                  after=record['visual']['observation']['ros_stamp'])
                                 if is_agent else self.photo.revalidate(record["visual"]))
                        with self._lock:
                            if epoch != self._command_epoch:
                                raise BridgeError("photo_changed", "复核期间收到其他指令，请重新预览。", 409)
                            # Inference releases the mission lock. Re-admit the
                            # same live choice before any controller side effect.
                            if record["used"] or record["expires"] < time.monotonic():
                                raise BridgeError("plan_expired", "摄影预览已使用或过期，请重新预览。", 409)
                            if record.get('search_binding'):
                                self._search_binding_valid(record)
                            snap = self._read()
                            self._require_photo_state(snap)
                            if (not self._camera_stamp_fresh(fresh["observation"]["stamp"]) or
                                    not self._pose_matches(snap, fresh["observation"]) or
                                    (not (is_agent and self._stationary_subject_confirmation()) and
                                     not self._pose_matches(snap, record["visual"]["observation"]))):
                                raise BridgeError("photo_pose_changed", "摄影确认期间无人机位姿变化，请重新预览。", 409)
                        token = self.adapter.set_photo_guard(fresh["resolved"]["evidence"])
                        validation = {"fresh": fresh, "token": token, "epoch": epoch,
                                      "kind": "photo_agent" if is_agent else "photo"}
                    except (ValueError, RuntimeError) as exc:
                        raise BridgeError("photo_target_changed", "摄影目标复核失败：%s" % exc, 409,
                                          details={"geometry_retry": getattr(exc, "photo_retry", None), "geometry_code": getattr(exc, "code", None)})
                    result = self._command(action, body, validation)
                    accepted = True
                    return result
                finally:
                    if token is not None and not accepted:
                        self._schedule_photo_cleanup(token)
                    with self._lock:
                        self._photo_confirming = False
                    self._photo_confirm_slots.release()
        return self._command(action, body)

    def _photo_admission_check(self, action, own_confirmation=False):
        if self._photo_confirming and not own_confirmation and action not in ("land", "hold"):
            raise BridgeError("photo_busy", "摄影确认正在复核；可随时悬停或降落，其他移动请稍后。", 409)
        search_cleanup = getattr(self, "_search_cleanup", None)
        if action not in ("land", "hold") and search_cleanup is not None:
            if not search_cleanup.done():
                raise BridgeError("photo_search_cleanup", "正在结束原地搜索，请稍后再发送任务。", 409)
            try:
                search_cleanup.result()
            except Exception:
                raise BridgeError("photo_search_cleanup", "搜索停止未获确认，请先降落并检查仿真。", 409)
        cleanup = getattr(self, "_photo_cleanup", None)
        if action not in ("land", "hold") and cleanup is not None:
            if not cleanup.done():
                raise BridgeError("photo_cleanup", "正在结束人物保护，请稍后再发送移动任务。", 409)
            try:
                cleanup.result()
            except Exception:
                raise BridgeError("photo_cleanup", "人物保护结束未获确认，请先降落并检查仿真。", 409)

    def _schedule_photo_cleanup(self, token):
        if token is None:
            return
        future = Future()
        with self._lock:
            self._photo_cleanup = future
        def cleanup():
            try:
                future.set_result(self.adapter.clear_photo_guard(token))
            except Exception as exc:
                future.set_exception(exc)
        threading.Thread(target=cleanup, name="photo-guard-cleanup", daemon=True).start()

    def _release_photo_agent(self):
        runner = self._photo_agent_runner
        self._photo_agent_runner = None
        if runner is not None:
            runner.cancel()
            self._schedule_photo_cleanup(runner.guard_token)

    def _release_photo_context(self):
        context = self._photo_context
        self._photo_context = None
        if context is None:
            return
        context["cancelled"].set()
        self._schedule_photo_cleanup(context["guard_token"])
        future = context.get("future")
        if future is not None and future.done():
            try:
                result = future.result()
                if result.get("photo"):
                    self.photo_store.remove(result["photo"]["id"])
            except Exception:
                pass

    def _photo_job(self, kind, context):
        if not self._photo_slots.acquire(blocking=False):
            raise RuntimeError("此前人物检测仍在结束，请稍后重新预览。")
        future = Future()
        after = context["last"]["observation"]["ros_stamp"]
        baseline = context.get("baseline")
        if kind == "capture":
            after = max(after, baseline["observation"]["ros_stamp"])
        def worker():
            result = None
            try:
                result = self.photo.observe(context["reference"], after=after,
                                            capture=kind in ("baseline", "capture"),
                                            cancelled=context["cancelled"],
                                            previous=baseline if kind == "capture" else None)
                if context["cancelled"].is_set():
                    raise ValueError("摄影任务已取消，未采纳检测结果。")
                if kind in ("baseline", "capture") and result["observation"]["stamp"] <= context["stable_wall"]:
                    raise ValueError("拍摄图像不晚于稳定悬停开始时刻。")
                if kind == "capture":
                    composition = copy.deepcopy(result["composition"])
                    composition.update(passed=True, message="所选人物全身构图、头脚姿态与连续静止检查通过。",
                                       geometry_retry=copy.deepcopy(result.get("geometry_retry")))
                    result["photo"] = self.photo_store.save(result["observation"], composition,
                                                           self.photo.detector_metadata(result["detection"]))
                if context["cancelled"].is_set():
                    if result.get("photo"):
                        self.photo_store.remove(result["photo"]["id"])
                    raise ValueError("摄影任务已取消，未采纳照片。")
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self._photo_slots.release()
        def discard_cancelled(completed):
            if context["cancelled"].is_set():
                try:
                    orphan = completed.result().get("photo")
                    if orphan:
                        self.photo_store.remove(orphan["id"])
                except Exception:
                    pass
        future.add_done_callback(discard_cancelled)
        context.update(future=future, job_kind=kind, job_started=time.monotonic())
        threading.Thread(target=worker, name="photo-" + kind, daemon=True).start()

    def _record_photo_audit(self, stage, accepted, audit, observation=None):
        if not self._mission or not isinstance(audit, dict):
            return
        entry = dict(copy.deepcopy(audit), stage=stage, accepted=bool(accepted))
        if observation is not None:
            entry.update(stamp=observation.get("stamp"), ros_stamp=observation.get("ros_stamp"))
        entries = self._mission.setdefault("photo_observations", [])
        entries.append(entry)
        if len(entries) > 64:
            del entries[:-64]
            self._mission["photo_observations_truncated"] = self._mission.get("photo_observations_truncated", 0) + 1

    def _tick_photo_monitor(self, snap, now):
        context = self._photo_context
        current_action = self._mission["steps"][self._mission["step_index"]]["action"]
        # Landing is an explicit safety transition; fresh vision must never be
        # required to finish it. Ordinary hold/land preemption releases context.
        if context["captured"] and current_action == "land":
            return True
        guard = snap.get("photo_guard")
        if isinstance(guard, dict) and (guard.get("error") or guard.get("active") is not True):
            self._abort("人物轨迹保护失效；未继续摄影任务。")
            return False
        if not snap.get("armed") or snap.get("landed"):
            self._abort("摄影任务期间飞机不再处于已解锁飞行状态。")
            return False
        future = context["future"]
        if future is not None:
            maximum = min(15., float(self.config.get("photo", {}).get("frame_wait_timeout_seconds", 8))) + min(10., float(self.config.get("photo", {}).get("detector_timeout_seconds", 5)))
            if now - context["job_started"] > maximum:
                self._abort("摄影图像检测超时；已悬停，未完成拍照。")
                return False
            if future.done():
                result = None
                try:
                    result = future.result()
                    if not self._camera_stamp_fresh(result["observation"]["stamp"]):
                        raise ValueError("摄影检测结果已过期。")
                    if result["observation"]["ros_stamp"] <= context["last"]["observation"]["ros_stamp"]:
                        raise ValueError("摄影检测复用了旧帧。")
                    kind = context["job_kind"]
                    if kind in ("baseline", "capture"):
                        self._require_photo_state(snap)
                        if (not self._pose_matches(snap, result["observation"]) or
                                not self._photo_stable_pose(snap, context.get("stable_pose"))):
                            raise ValueError("拍摄期间无人机没有保持构图位置。")
                    self._record_photo_audit(kind, True, result.get("geometry_retry"), result["observation"])
                    context["last"] = result
                    context["last_poll"] = now
                    if kind == "baseline":
                        context["baseline"] = result
                    if kind == "capture":
                        self._mission["photos"].append(copy.deepcopy(result["photo"]))
                        context["captured"] = True
                        self._mission["photo_state"] = "available"
                        self._mission["step_index"] += 1
                        context["future"] = None
                        self._dispatch = None
                        if self._mission["step_index"] == len(self._mission["steps"]):
                            self._finish("completed")
                        return False
                    context["future"] = None
                except Exception as exc:
                    audit = getattr(exc, "photo_retry", None) or (result.get("geometry_retry") if result else None)
                    self._record_photo_audit(context.get("job_kind", "unknown"), False, audit)
                    self._abort("拍照任务失败：%s" % exc)
                    return False
        if not self._camera_stamp_fresh(context["last"]["observation"]["stamp"]):
            self._abort("人物观测已过期；检测等待期间停止移动，未继续使用旧人物位置。")
            return False
        current_action = self._mission["steps"][self._mission["step_index"]]["action"]
        interval = max(.1, float(self.config.get("photo", {}).get("monitor_interval_seconds", .5)))
        if context["future"] is None and current_action != "photo" and now - context["last_poll"] >= interval:
            try:
                self._photo_job("monitor", context)
            except Exception as exc:
                self._abort("人物监测未能启动：%s" % exc)
                return False
        return True

    def _photo_search_stable_pose(self, snap, origin):
        speed = math.sqrt(sum(snap['velocity'][axis] ** 2 for axis in ('x', 'y', 'z')))
        # The real stationary-person profile uses a new measured frame after
        # stopping. Ordinary drift since the beginning of the dwell is not a
        # composition requirement; freshness and new-frame/current-pose
        # matching remain enforced by the caller before exposing candidates.
        return speed <= .15 and (self._stationary_subject_confirmation() or
                                  origin is None or Bridge._pose_matches(snap, origin))

    @staticmethod
    def _photo_stable_pose(snap, origin):
        speed = math.sqrt(sum(snap["velocity"][a] ** 2 for a in ("x", "y", "z")))
        if speed > .05:
            return False
        if origin is None:
            return True
        distance = math.sqrt(sum((snap["position"][a] - origin["position"][a]) ** 2 for a in ("x", "y", "z")))
        yaw = abs((snap["yaw"] - origin["yaw"] + math.pi) % (2 * math.pi) - math.pi)
        return distance <= .03 and yaw <= math.radians(2)

    def _tick_photo_step(self, snap, now):
        context = self._photo_context
        if self._dispatch is None:
            # The preceding measured arrival (or guard activation at the current
            # pose) already holds position. An explicit emergency hold cancels
            # the controller's photo guard and must not be sent here.
            self._dispatch = {"time": now, "wall": time.time(), "dwell": None}
            self._mission["photo_state"] = "settling"
        if now - self._dispatch["time"] > self.config["step_timeout_seconds"]:
            self._abort("未能在任务时限内完成稳定构图和拍照。")
            return
        if not self._photo_stable_pose(snap, context.get("stable_pose")):
            context["stable_since"] = context["stable_wall"] = None
            context["baseline"] = context["stable_pose"] = None
            if context["future"] is not None and context.get("job_kind") in ("baseline", "capture"):
                self._abort("拍摄期间无人机发生移动，未完成拍照。")
            return
        if context["stable_since"] is None:
            context["stable_since"], context["stable_wall"] = now, time.time()
            context["stable_pose"] = {"position": copy.deepcopy(snap["position"]), "yaw": snap["yaw"]}
        dwell = max(.5, float(self.config.get("photo", {}).get("stationary_dwell_seconds", .6)))
        if now - context["stable_since"] < dwell or context["future"] is not None:
            return
        try:
            if context["baseline"] is None:
                self._photo_job("baseline", context)
            elif now - context["last_poll"] >= .2:
                self._mission["photo_state"] = "capturing"
                self._photo_job("capture", context)
        except Exception as exc:
            self._abort("无法开始拍照：%s" % exc)

    @staticmethod
    def _fields(body, allowed, required):
        if not isinstance(body, dict) or set(body) - allowed or required - set(body):
            raise BridgeError("invalid_body", "Required fields: %s; allowed fields: %s" %
                              (", ".join(sorted(required)), ", ".join(sorted(allowed))))

    def _command(self, action, body, photo_validation=None):
        with self._lock:
            if not isinstance(body, dict):
                raise BridgeError("invalid_body", "Expected a JSON object")
            request_id = body.get("request_id")
            if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id):
                raise BridgeError("invalid_request_id", "request_id must be a unique 1–128 character identifier")
            try:
                fingerprint = _canonical({"action": action, "body": body})
            except (ValueError, TypeError):
                raise BridgeError("invalid_body", "JSON must contain finite values")
            previous = self._requests.get(request_id)
            if previous:
                if not hmac.compare_digest(previous[0], fingerprint):
                    raise BridgeError("request_conflict", "request_id was already used with different content", 409)
                return copy.deepcopy(previous[1])
            client_fields = {"client_id", "client_seq"}
            has_client = bool(set(body) & client_fields)
            if has_client:
                if (not client_fields <= set(body) or not isinstance(body["client_id"], str) or
                        not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", body["client_id"]) or
                        isinstance(body["client_seq"], bool) or not isinstance(body["client_seq"], int) or
                        not 1 <= body["client_seq"] <= 9007199254740991):
                    raise BridgeError("invalid_client_sequence", "Provide client_id and a positive integer client_seq together")
            plan_record = None
            if action == "execute":
                self._fields(body, {"request_id", "plan_id", "plan_hash"} | client_fields, {"request_id", "plan_id", "plan_hash"})
                if not isinstance(body["plan_id"], str) or not isinstance(body["plan_hash"], str):
                    raise BridgeError("invalid_plan", "plan_id and plan_hash must be strings")
                plan_record = self._plans.get(body["plan_id"])
                if not plan_record:
                    raise BridgeError("plan_missing", "Preview is unavailable; preview again", 404)
                if plan_record["expires"] < time.monotonic():
                    raise BridgeError("plan_expired", "Preview expired; preview again", 409)
                if plan_record["used"]:
                    raise BridgeError("plan_used", "This plan has already been accepted; preview again for a new mission", 409)
                if not hmac.compare_digest(plan_record["hash"].encode("utf-8"), body["plan_hash"].encode("utf-8")):
                    raise BridgeError("plan_changed", "Plan hash does not match the preview", 409)
                plan = copy.deepcopy(plan_record["plan"])
            else:
                allowed = {"takeoff": {"height"}, "goto": {"x", "y", "z", "yaw"}, "land": set(), "hold": set()}
                if action not in allowed:
                    raise BridgeError("unknown_action", "Unsupported command", 404)
                required = {"x", "y", "z"} if action == "goto" else set()
                self._fields(body, allowed[action] | {"request_id"} | client_fields, required | {"request_id"})
                step = {k: v for k, v in body.items() if k not in ({"request_id"} | client_fields)}
                step["action"] = action
                if action == "takeoff":
                    step.setdefault("height", self.config["takeoff_height"])
                plan = {"summary": {"takeoff": "起飞", "goto": "前往目标点", "land": "降落", "hold": "悬停 / 中止任务"}[action], "steps": [step]}
                try:
                    plan = validate_plan(plan, self.config)
                except ValueError as exc:
                    raise BridgeError("invalid_plan", str(exc), 422)
            if plan_record and plan_record.get('search_request'):
                # Search admission is read-only until the airborne, stationary
                # precondition passes; a rejection must not leave a ghost mission.
                self._require_photo_state(self._read())
            # Sequence admission shares the mission lock. An emergency click
            # reaching the server first prevents a delayed earlier goto/takeoff
            # from starting after the emergency hold has completed.
            if has_client:
                client_id, sequence = body["client_id"], body["client_seq"]
                if sequence <= self._client_sequences.get(client_id, 0):
                    raise BridgeError("stale_command", "A newer command from this client was already received", 409)
                self._client_sequences[client_id] = sequence
                self._client_sequences.move_to_end(client_id)
                while len(self._client_sequences) > 1024:
                    self._client_sequences.popitem(last=False)
            self._photo_admission_check(action, own_confirmation=photo_validation is not None)
            snap = self._read()
            if snap.get("capabilities", {}).get("yaw_target") is False and any("yaw" in step for step in plan["steps"]):
                raise BridgeError("yaw_unsupported", "This controller does not support a measured final yaw target; use position-only commands", 422)
            safety_action = action in ("land", "hold")
            if not self._fresh(snap, require_position=not safety_action):
                raise BridgeError("telemetry_unavailable", "Fresh vehicle telemetry is required before accepting commands", 503)
            if plan_record and plan_record.get('search_request'):
                self._require_photo_state(snap)
            if plan_record and plan_record.get("visual"):
                if plan_record["visual"].get("kind") in ("photo", "photo_agent"):
                    if photo_validation is None or photo_validation["epoch"] != self._command_epoch:
                        raise BridgeError("photo_changed", "摄影确认期间收到其他指令，请重新预览。", 409)
                    self._require_photo_state(snap)
                    if plan_record.get('search_binding'):
                        self._search_binding_valid(plan_record)
                    if (not self._camera_stamp_fresh(photo_validation["fresh"]["observation"]["stamp"]) or
                            not self._pose_matches(snap, photo_validation["fresh"]["observation"]) or
                            (not (plan_record["visual"].get('kind') == 'photo_agent' and self._stationary_subject_confirmation()) and
                             not self._pose_matches(snap, plan_record["visual"]["observation"]))):
                        raise BridgeError("photo_pose_changed", "摄影确认期间无人机位姿变化，请重新预览。", 409)
                else:
                    self._validate_visual_execution(plan_record["visual"], snap)
            active_step = None
            if self._mission and self._mission["state"] == "running" and self._dispatch is not None:
                active_step = self._mission["steps"][self._mission["step_index"]]["action"]
            if self._photo_agent_runner is not None and self._photo_agent_runner.public().get('stage') == 'landing':
                active_step = 'land'
            grounded = snap["landed"] and not snap["armed"]
            if (action == "hold" and not grounded and snap.get("mode") in ("gazebo", "real") and
                    (active_step == "land" or snap.get("flight_mode") == "LANDING")):
                # Neither the simulation pause topic nor the real trajectory
                # gate transitions an already-running automatic landing to hover.
                # Reject before cancelling the tracked landing mission.
                pending = self._photo_agent_runner
                cancel_pending = getattr(pending, 'cancel_pending_landing', None)
                if (snap.get('flight_mode') == 'LANDING' or not callable(cancel_pending) or
                        not cancel_pending()):
                    raise BridgeError("landing_in_progress", "降落进行中，此版本不支持中断降落切换悬停；原降落任务继续执行。", 409)
            if action in ("hold", "land") and snap.get("mode") == "real" and active_step == "takeoff":
                raise BridgeError("takeoff_in_progress", "实机自动起飞进行中，当前指令门控不能中断自动起飞；请使用既有遥控器接管流程。", 409)
            if photo_validation is not None and photo_validation.get('kind') == 'photo_agent':
                # Constructor is local and side-effect free: no model request or
                # vehicle command runs before final atomic mission admission.
                reference = copy.deepcopy(plan_record['visual'])
                reference['observation'] = copy.deepcopy(photo_validation['fresh']['observation'])
                photo_validation['runner'] = PhotoAgentRunner(
                    self.config, self.adapter, self.photo_agent, self.photo, self.photo_store,
                    plan_record['visual']['task'], reference, photo_validation['fresh'],
                    photo_validation['token'], evidence_callback=self._store_evidence,
                    scene_memory=(self._search_context.get('scene_memory') if self._search_context and
                                  plan_record.get('search_binding', {}).get('search_id') == self._search_context['id'] else None))
            if self._mission and self._mission["state"] == "running":
                selected_search = bool(photo_validation is not None and self._search_binding_valid(plan_record))
                if not safety_action and not selected_search:
                    raise BridgeError("mission_busy", "A mission is running; hold or land before starting another", 409)
                if selected_search:
                    self._mission['photo_search']['state'] = 'selected'
                    self._finish('completed')
                else:
                    self._finish("cancelled", "Interrupted by %s command" % action)
            self._validate_sequence(plan["steps"], snap)
            mission_id = str(uuid.uuid4())
            self._previous_mission = copy.deepcopy(self._mission)
            self._mission = {"id": mission_id, "state": "running", "summary": plan["summary"],
                             "steps": plan["steps"], "step_index": 0, "error": None,
                             "started_at": time.time(), "finished_at": None,
                             "parser": plan_record.get("parser", "offline") if plan_record else "offline",
                             "vision_used": bool(plan_record and plan_record.get("visual") and
                                                 plan_record['visual'].get('kind') != 'photo_agent'),
                             "evidence": copy.deepcopy(plan_record.get("evidence", [])) if plan_record else [],
                             "inspections": [], "photos": [], "photo_observations": []}
            self._command_epoch += 1
            self._photo_context = None
            self._photo_agent_runner = None
            self._search_context = None
            if plan_record and plan_record.get('search_request'):
                self._search_context = {'id': mission_id, 'text': plan_record['search_request']['text'],
                                        'cancelled': threading.Event(), 'phase': 'starting', 'started': time.monotonic(),
                                        'future': None, 'selection_revision': 0,
                                        'scene_memory': SceneMemory(self.config),
                                        'task': copy.deepcopy(plan_record['search_request'].get('task'))}
                self._mission['photo_search'] = {'id': mission_id, 'state': 'searching', 'angle_swept_rad': 0.,
                                                'candidates': [], 'evidence': None, 'expires_at': None,
                                                'expires_in_seconds': None}
                if self._search_context.get('task'):
                    self._mission['photo_task'] = copy.deepcopy(self._search_context['task'])
                    self._mission['photo_search']['task'] = copy.deepcopy(self._search_context['task'])
            if photo_validation is not None and photo_validation.get('kind') == 'photo_agent':
                self._photo_agent_runner = photo_validation['runner']
                self._mission['photo_task'] = copy.deepcopy(plan_record['visual']['task'])
                self._mission['subject_selection'] = copy.deepcopy(plan_record['search_binding'])
                self._mission['photo_agent'] = self._photo_agent_runner.public()
            elif photo_validation is not None:
                confirmed_target = copy.deepcopy(photo_validation["fresh"]["resolved"])
                confirmed_target["goal"] = copy.deepcopy(plan_record["visual"]["photo_target"]["goal"])
                confirmed_reference = dict(plan_record["visual"], photo_target=confirmed_target)
                if photo_validation['fresh'].get('selection'):
                    confirmed_reference['selection'] = copy.deepcopy(photo_validation['fresh']['selection'])
                if plan_record.get('search_binding'):
                    self._mission['subject_selection'] = copy.deepcopy(plan_record['search_binding'])
                self._photo_context = {"reference": confirmed_reference,
                                       "guard_token": photo_validation["token"],
                                       "cancelled": threading.Event(), "future": None,
                                       "last": photo_validation["fresh"], "last_poll": time.monotonic(),
                                       "captured": False, "stable_since": None, "stable_wall": None,
                                       "baseline": None, "stable_pose": None}
                self._mission["photo_state"] = "moving" if plan["steps"][0]["action"] == "goto" else "settling"
                self._record_photo_audit("preview", True, plan_record["visual"].get("geometry_retry"), plan_record["visual"]["observation"])
                self._record_photo_audit("confirm", True, photo_validation["fresh"].get("geometry_retry"), photo_validation["fresh"]["observation"])
            self._dispatch = None
            self._inspection_wait_started = self._inspection_ready_since = None
            self._last_error = None
            self._last_client = time.monotonic()
            self._watchdog_latched = False
            if plan_record:
                plan_record["used"] = True
            result = {"accepted": True, "request_id": request_id, "mission_id": mission_id, "state": "accepted"}
            self._requests[request_id] = (fingerprint, result)
            while len(self._requests) > self.config["request_cache_size"]:
                self._requests.popitem(last=False)
            return copy.deepcopy(result)

    @staticmethod
    def _validate_sequence(steps, snap):
        airborne = not snap["landed"]
        if airborne and not snap["armed"]:
            # Landing and hold remain available in unusual autopilot states.
            if any(s["action"] not in ("land", "hold") for s in steps):
                raise BridgeError("not_armed", "Vehicle is not armed", 409)
        for step in steps:
            if step["action"] == "takeoff":
                if airborne:
                    raise BridgeError("already_airborne", "Cannot take off while already airborne", 409)
                airborne = True
            elif step["action"] in ("goto", "inspect", "photo", "search_person", "photo_agent") and not airborne:
                raise BridgeError("takeoff_required", "Take off before flying to a target, or include takeoff as the first plan step", 409)
            elif step["action"] == "land":
                airborne = False

    def _finish(self, state, error=None):
        self._release_search_context()
        self._release_photo_context()
        self._release_photo_agent()
        if self._mission:
            self._mission["state"] = state
            self._mission["error"] = error
            self._mission["finished_at"] = time.time()
            self.task_log.sync(self._mission)
            record = mission_summary(self._mission)
            self._mission_history = [item for item in self._mission_history if item.get('id') != record['id']]
            self._mission_history.append(record)
            self._mission_history = self._mission_history[-20:]
            self._history_writer.submit(self._mission_history)
        if error:
            self._last_error = error
        self._dispatch = None
        self._inspection_wait_started = self._inspection_ready_since = None

    def _abort(self, reason, state="failed"):
        if self._photo_agent_runner is not None:
            # Invalidate asynchronous command workers before obtaining the
            # adapter command lock for hold, so late goto cannot follow hold.
            self._photo_agent_runner.cancel()
        try:
            self.adapter.hold()
        except Exception as exc:
            reason += "; hold could not be delivered: %s" % exc
        if self._mission and self._mission["state"] == "running":
            self._finish(state, reason)
        else:
            self._last_error = reason
        self._watchdog_latched = True

    def _run(self):
        while not self._stop.wait(0.05):
            try:
                self.tick()
            except Exception as exc:
                with self._lock:
                    self._abort("Mission engine error: %s" % exc)

    def tick(self):
        """One control iteration; also useful with deterministic test adapters."""
        with self._lock:
            snap = self._read()
            self.task_log.sync(self._mission)
            active = self._mission and self._mission["state"] == "running"
            airborne = snap.get("armed") is True and snap.get("landed") is False
            client_lost = time.monotonic() - self._last_client > self.config["client_timeout_seconds"]
            if (active or airborne) and client_lost and not self._watchdog_latched:
                self._abort("Phone heartbeat timed out; requested hold")
                return
            if airborne and not self._fresh(snap) and not self._watchdog_latched:
                self._abort("Vehicle telemetry is stale or disconnected; requested hold")
                return
            if not active:
                return
            step = self._mission["steps"][self._mission["step_index"]]
            agent_landing = (step['action'] == 'photo_agent' and self._photo_agent_runner is not None and
                             self._photo_agent_runner.public().get('stage') == 'landing')
            if not self._fresh(snap, require_position=step["action"] not in ("land",) and not agent_landing):
                self._abort("Vehicle telemetry is stale or disconnected; requested hold")
                return
            goal_reached = step["action"] == "goto" and self._goal_geometry_reached(step, snap)
            if step["action"] == "goto" and snap.get("planning_ready") is False and not goal_reached:
                self._abort("Planner heartbeat or point cloud is unavailable; requested hold")
                return
            now = time.monotonic()
            if step['action'] == 'search_person':
                self._tick_photo_search(snap, now)
                return
            if step['action'] == 'photo_agent':
                if self._photo_agent_runner is None:
                    self._abort('摄影智能体执行上下文不存在，未执行任何模型动作。')
                    return
                result = self._photo_agent_runner.tick(snap, now)
                public = self._photo_agent_runner.public()
                self._mission['photo_agent'] = public
                self._mission['photos'] = copy.deepcopy(public.get('photos', []))
                self._mission['vision_used'] = public.get('last_decision') is not None
                if result:
                    if result.get('state') == 'completed':
                        self._mission['step_index'] += 1
                        self._finish('completed')
                    else:
                        self._abort(result.get('error') or '摄影智能体已停止；未完成拍摄任务。')
                return
            if self._photo_context is not None:
                if not self._tick_photo_monitor(snap, now):
                    return
                if step["action"] == "photo":
                    self._tick_photo_step(snap, now)
                    return
            if self._dispatch is None:
                # Recheck actual flight state at every boundary, including between
                # natural-language steps. A successful publish proves no motion.
                self._validate_sequence([step], snap)
                if step["action"] == "inspect":
                    if self._inspection_wait_started is None:
                        self._inspection_wait_started = now
                    speed = math.sqrt(sum(snap["velocity"][a] ** 2 for a in ("x", "y", "z")))
                    if speed > 0.15:
                        self._inspection_ready_since = None
                    elif self._inspection_ready_since is None:
                        self._inspection_ready_since = now
                    if now - self._inspection_wait_started > self.config["step_timeout_seconds"]:
                        self._abort("无人机未能稳定悬停，未开始视觉检查。")
                        return
                    if self._inspection_ready_since is None or now - self._inspection_ready_since < self.config["arrival_dwell_seconds"]:
                        return
                    self._inspection_wait_started = self._inspection_ready_since = None
                self._dispatch = {"time": now, "wall": time.time(), "dwell": None,
                                  "origin": copy.deepcopy(snap.get("position", {})),
                                  "armed_seen": snap.get("armed") is True}
                try:
                    if step["action"] == "goto":
                        if goal_reached:
                            # A zero-distance request may produce no PolyTraj.
                            # Cancel any old stream and still require new feedback
                            # plus the normal stable dwell before completion.
                            if self._photo_context is None:
                                self.adapter.hold()
                        else:
                            self.adapter.goto(step["x"], step["y"], step["z"], step.get("yaw"))
                    elif step["action"] == "takeoff":
                        self.adapter.takeoff(step["height"])
                    elif step["action"] == "land":
                        self.adapter.land()
                    elif step["action"] == "inspect":
                        self._require_stationary_airborne(snap)
                        self.adapter.hold()
                        self._dispatch["inspection_pose"] = {"position": copy.deepcopy(snap["position"]), "yaw": snap["yaw"]}
                        for field in ("orientation", "body_frame"):
                            if field in snap:
                                self._dispatch["inspection_pose"][field] = copy.deepcopy(snap[field])
                        self._dispatch["future"] = self._start_inspection(step["question"])
                    else:
                        self.adapter.hold()
                except Exception as exc:
                    self._abort("Command delivery failed: %s" % exc)
                return
            if step["action"] == "inspect":
                if (not snap["armed"] or snap["landed"] or
                        not self._pose_matches(snap, self._dispatch["inspection_pose"]) or
                        math.sqrt(sum(snap["velocity"][a] ** 2 for a in ("x", "y", "z"))) > 0.15):
                    self._abort("视觉检查期间无人机未保持静止；未采纳检查结果。")
                    return
                future = self._dispatch["future"]
                if now - self._dispatch["time"] > float(self.config.get("vision", {}).get("timeout_seconds", 20)) + 3:
                    self._abort("视觉检查 API 超时；未生成观察结论。")
                    return
                if future.done():
                    try:
                        completed = future.result()
                        if not self._pose_matches(snap, completed["observation"]):
                            raise ValueError("检查证据的拍摄位姿已变化。")
                        evidence = self._store_evidence(completed["observation"])
                        entry = dict(completed["answer"], step_index=self._mission["step_index"], evidence=evidence)
                        self._mission["inspections"].append(entry)
                        self._mission["evidence"].append(evidence)
                        self._mission["step_index"] += 1
                        self._dispatch = None
                        if self._mission["step_index"] == len(self._mission["steps"]):
                            self._finish("completed")
                    except Exception as exc:
                        self._abort("视觉检查失败：%s" % exc)
                return
            timeout = self.config["step_timeout_seconds"] + (step.get("duration", 0) if step["action"] == "hold" else 0)
            if now - self._dispatch["time"] > timeout:
                self._abort("Step %d timed out without measured completion; requested hold" % (self._mission["step_index"] + 1))
                return
            if step['action'] == 'takeoff' and snap['armed']:
                self._dispatch['armed_seen'] = True
            if step["action"] in ("goto", "takeoff") and not snap["armed"]:
                # Arm transitions take time; allow a bounded takeoff startup window.
                if step["action"] != "takeoff" or self._dispatch.get('armed_seen'):
                    self._abort("Vehicle became disarmed during flight")
                elif now - self._dispatch["time"] > 5:
                    self._abort('起飞请求后仍未确认解锁，请查看原飞控的起飞拒绝原因；停止后续任务不代表原自动起飞已取消。')
                return
            if step["action"] == "goto" and snap["landed"]:
                self._abort("Vehicle reported landed before reaching the destination")
                return
            arrived = self._arrived(step, snap, now)
            if step["action"] == "goto" and snap.get("command_error") and not arrived:
                self._abort("Planner command failed: %s; requested hold" % str(snap["command_error"])[:500])
                return
            if arrived:
                if self._dispatch["dwell"] is None:
                    self._dispatch["dwell"] = now
                if now - self._dispatch["dwell"] >= self.config["arrival_dwell_seconds"]:
                    if step['action'] == 'goto' and self._photo_context is None:
                        # Release the adapter's completed navigation ownership;
                        # otherwise its stale active flag blocks the next search.
                        self.adapter.hold()
                    self._mission["step_index"] += 1
                    self._dispatch = None
                    if self._mission["step_index"] == len(self._mission["steps"]):
                        self._finish("completed")
            else:
                self._dispatch["dwell"] = None

    def _goal_geometry_reached(self, target, snap):
        speed = math.sqrt(sum(snap["velocity"][axis] ** 2 for axis in ("x", "y", "z")))
        if speed > self.config["arrival_speed_mps"]:
            return False
        distance = math.sqrt(sum((snap["position"][axis] - target[axis]) ** 2 for axis in ("x", "y", "z")))
        # A photographic adjustment can be only a few centimetres. The general
        # navigation tolerance would silently declare it reached before sending
        # any trajectory, including an explicitly requested short return.
        photo_goal = self._photo_context is not None and target.get("action") == "goto"
        position_tolerance = min(self.config["arrival_tolerance_m"], .02) if photo_goal else self.config["arrival_tolerance_m"]
        yaw_tolerance = math.radians(1) if photo_goal else .2
        if distance > position_tolerance:
            return False
        if target.get("yaw") is not None:
            difference = (snap["yaw"] - target["yaw"] + math.pi) % (2 * math.pi) - math.pi
            if abs(difference) > yaw_tolerance:
                return False
        return True

    def _arrived(self, step, snap, now):
        if snap.get("state_stamp", 0) < self._dispatch["wall"]:
            return False
        if step["action"] == "land":
            return snap["landed"] is True and snap["armed"] is False
        if snap.get("position_stamp", 0) < self._dispatch["wall"]:
            return False
        speed = math.sqrt(sum(snap["velocity"][axis] ** 2 for axis in ("x", "y", "z")))
        if speed > self.config["arrival_speed_mps"]:
            return False
        if step["action"] == "hold":
            return now - self._dispatch["time"] >= step.get("duration", 0)
        takeoff_z = step.get("height", 0) + (self._dispatch["origin"].get("z", 0) if self.config.get("takeoff_relative", False) else 0)
        if _finite(snap.get("takeoff_target_z")):
            takeoff_z = snap["takeoff_target_z"]
        target = ({"x": self._dispatch["origin"]["x"], "y": self._dispatch["origin"]["y"], "z": takeoff_z}
                  if step["action"] == "takeoff" else step)
        if step["action"] == "takeoff" and snap["landed"]:
            return False
        if step["action"] == "takeoff" and snap.get("mode") == "gazebo" and snap.get("flight_mode") not in ("HOVERING", "FLYING"):
            return False
        return self._goal_geometry_reached(target, snap)
