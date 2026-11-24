"""Local RGB person detection and conservative portrait framing checks.

The pretrained, generic detector sees JPEG pixels only. It never receives scene
metadata, object names, colors, depth, poses, or world coordinates. A bounding
box with margins is a *framing check*, not proof that head/feet are unoccluded.
This module has no flight control, networking, credential or model fallback.
"""

import copy
import hashlib
import hmac
import json
import math
from pathlib import Path
import secrets
import threading
import time

from .planner import PlanError


class PhotoDetectionError(PlanError):
    pass


class PhotoDetectionTimeout(PhotoDetectionError):
    """The caller's bounded detection job ended without a usable result."""
    code = 'photo_detection_timeout'


class PhotoDetectionCancelled(PhotoDetectionError):
    code = 'photo_detection_cancelled'


MODEL_SHA256 = "52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7"
NETWORK_SHA256 = "2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "photo_person_mobilenet_ssd"


def _valid_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value))


def _box(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or
            not all(_valid_number(item) for item in value)):
        raise PhotoDetectionError("人物边界框格式无效。")
    if not (0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1):
        raise PhotoDetectionError("人物边界框超出图像或没有面积。")
    return [float(item) for item in value]


def _iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


class PersonDetector:
    """MobileNet-SSD/VOC person (class 15), OpenCV CPU inference.

    Loading is lazy; weights are checksum verified before use. One inference
    runs at a time. OpenCV's process-wide CPU thread budget is explicitly set to
    one by default to avoid competing with Gazebo software rendering.
    """

    def __init__(self, model_dir=None, confidence_threshold=.30, num_threads=1, verify_pose=True):
        if not _valid_number(confidence_threshold) or not 0 < confidence_threshold < 1:
            raise PhotoDetectionError("人物检测阈值必须在 0 与 1 之间。")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or not 1 <= num_threads <= 4:
            raise PhotoDetectionError("人物检测 CPU 线程数必须为 1 至 4。")
        self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self.threshold = float(confidence_threshold)
        self.num_threads = num_threads
        self._lock = threading.Lock()
        self._net = None
        self._verify_pose = bool(verify_pose)
        self._pose = None
        self._candidate_key = secrets.token_bytes(32)

    def availability(self):
        # Existence only, no inference / download / runtime initialization.
        from .photo_pose import POSE_PATH, RUNTIME_PATH
        person_ready = all((self.model_dir / name).is_file() for name in
                           ("deploy.prototxt", "mobilenet_iter_73000.caffemodel"))
        pose_ready = POSE_PATH.is_file()
        runtime_ready = (RUNTIME_PATH / "onnxruntime" / "__init__.py").is_file()
        return {"configured": person_ready and (not self._verify_pose or (pose_ready and runtime_ready)),
                "person_model_ready": person_ready, "pose_model_ready": pose_ready,
                "pose_runtime_ready": runtime_ready,
                "provider": "local_opencv_mobilenet_ssd", "pose_validation_enabled": self._verify_pose}

    def _load(self, cv2):
        if self._net is not None:
            return
        files = (("deploy.prototxt", NETWORK_SHA256), ("mobilenet_iter_73000.caffemodel", MODEL_SHA256))
        for name, expected in files:
            path = self.model_dir / name
            try:
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                raise PhotoDetectionError("本地人物检测模型尚未安装，请运行 scripts/download_photo_detector.py。") from None
            if checksum != expected:
                raise PhotoDetectionError("本地人物检测模型校验失败，请重新下载官方模型。")
        cv2.setNumThreads(self.num_threads)
        try:
            net = cv2.dnn.readNetFromCaffe(str(self.model_dir / "deploy.prototxt"),
                                         str(self.model_dir / "mobilenet_iter_73000.caffemodel"))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error:
            raise PhotoDetectionError("OpenCV 无法加载本地人物检测模型。") from None
        self._net = net

    def detect(self, jpeg):
        """Backward-compatible strict single-person detection plus pose."""
        return self._detect(jpeg, with_pose=self._verify_pose)

    def detect_people(self, jpeg):
        """Candidates only: no full-body/clearance/identity claim or auto choice."""
        return self._detect(jpeg, with_pose=False)

    def _binding(self, result):
        # Bind boxes to the exact image and detector instance. This is not a
        # person identity or a freshness proof; the service owns ROS stamps and
        # server-issued candidate IDs. Client-supplied/reordered boxes fail.
        payload = {key: result.get(key) for key in
                   ("image_sha256","width","height","provider","model_sha256",
                    "network_sha256","preprocessing","detections")}
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",",":"), allow_nan=False).encode("utf-8")
        except (ValueError,TypeError):
            raise PhotoDetectionError("冻结人物候选数据无效。") from None
        return hmac.new(self._candidate_key,encoded,hashlib.sha256).hexdigest()

    def _pose_for(self, jpeg, item):
        pose = {"validated": False, "full_body_evidence": False, "reason": "pose_not_requested"}
        if not self._verify_pose or item["confidence"] < .55:
            return pose
        from .photo_pose import PoseError, PoseVerifier
        try:
            with self._lock:
                if self._pose is None:
                    self._pose = PoseVerifier()
                verifier = self._pose
            return verifier.verify(jpeg,item["bbox"])
        except PoseError as error:
            return {"validated": False, "full_body_evidence": False,
                    "reason": "pose_unavailable", "message": str(error)}

    def select_subject(self, jpeg, frozen_detection, candidate_index):
        """Validate ONLY the caller's explicitly selected frozen candidate.

        The caller must map an opaque server candidate_id to this index. A new
        frame must first be detected and uniquely matched by observed geometry;
        this API does not infer identity or silently choose another person.
        """
        if not isinstance(jpeg,(bytes,bytearray)) or not isinstance(frozen_detection,dict):
            raise PhotoDetectionError("选人需要原始冻结图像和服务器候选。")
        binding = frozen_detection.get("candidate_binding")
        if (frozen_detection.get("image_sha256") != hashlib.sha256(jpeg).hexdigest() or
                not isinstance(binding,str) or len(binding)!=64 or
                any(character not in "0123456789abcdef" for character in binding) or
                not hmac.compare_digest(binding,self._binding(frozen_detection))):
            raise PhotoDetectionError("人物候选与冻结图像不匹配或已改变，请重新找人。")
        if "selected_detection_index" in frozen_detection:
            raise PhotoDetectionError("不能在已缩减的选人结果中重新选择人物。")
        people = frozen_detection.get("detections")
        if (not isinstance(people,list) or type(candidate_index) is not int or
                not 0 <= candidate_index < len(people)):
            raise PhotoDetectionError("所选人物候选不存在，请重新选择。")
        chosen = copy.deepcopy(people[candidate_index])
        pose = self._pose_for(jpeg,chosen)
        result = copy.deepcopy(frozen_detection)
        result.update(detections=[chosen],all_detections=copy.deepcopy(people),
                      all_person_count=len(people),selected_detection_index=candidate_index,pose=pose,
                      selection_binding={"image_sha256":frozen_detection["image_sha256"],
                                         "candidate_binding":frozen_detection["candidate_binding"],
                                         "candidate_index":candidate_index,"identity_verified":False})
        result["quality"] = {"detection":True,"frame_margin_check":False,
                             "pose_validation":bool(pose["validated"]),
                             "full_body_evidence":bool(pose["full_body_evidence"]),
                             "scene_single_person":len(people)==1}
        result["scope"] = "只验证手机明确选中的冻结人物候选；保留所有其他检测人物，不识别身份，不证明旁人安全。"
        return result

    def _detect(self, jpeg, with_pose):
        """Return normalized boxes and pixel-only evidence, never an XYZ goal."""
        if (not isinstance(jpeg, (bytes, bytearray)) or not 4 <= len(jpeg) <= 4 * 1024 * 1024 or
                not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
            raise PhotoDetectionError("拍照检测需要有效且不超过 4 MiB 的 JPEG。")
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise PhotoDetectionError("本地人物检测需要 OpenCV 与 NumPy。") from None
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise PhotoDetectionError("相机 JPEG 无法解码。")
        height, width = image.shape[:2]
        if not (64 <= width <= 4096 and 64 <= height <= 4096 and width * height <= 8_388_608):
            raise PhotoDetectionError("拍照检测的图像尺寸超出支持范围。")
        with self._lock:
            self._load(cv2)
            started = time.monotonic()
            # Preserve the camera aspect ratio. Directly stretching a 16:9
            # frame to a square makes upright people unnaturally thin.
            side = max(width, height)
            pad_x, pad_y = (side-width)//2, (side-height)//2
            square = np.full((side, side, 3), 127, dtype=np.uint8)
            square[pad_y:pad_y+height, pad_x:pad_x+width] = image
            blob = cv2.dnn.blobFromImage(square, scalefactor=.007843, size=(300, 300),
                                         mean=(127.5, 127.5, 127.5), swapRB=False, crop=False)
            try:
                self._net.setInput(blob)
                output = self._net.forward()
            except cv2.error:
                raise PhotoDetectionError("本地人物检测推理失败，未生成拍照任务。") from None
            elapsed = (time.monotonic() - started) * 1000.
        if output.ndim != 4 or output.shape[0:2] != (1, 1) or output.shape[3] != 7:
            raise PhotoDetectionError("人物检测模型输出格式不正确。")
        candidates = []
        for row in output[0, 0]:
            if not np.isfinite(row).all() or int(row[1]) != 15 or not self.threshold <= float(row[2]) <= 1:
                continue
            original = [(float(row[3])*side-pad_x)/width, (float(row[4])*side-pad_y)/height,
                        (float(row[5])*side-pad_x)/width, (float(row[6])*side-pad_y)/height]
            clipped = [max(0., min(1., value)) for value in original]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            candidates.append({"bbox": clipped, "confidence": float(row[2]), "label": "person",
                               "raw_bbox": original, "clipped": clipped != original})
        # The SSD network already performs NMS; deduplicate near-identical boxes
        # conservatively without merging side-by-side people.
        detections = []
        for item in sorted(candidates, key=lambda value: value["confidence"], reverse=True):
            if not any(_iou(item["bbox"], prior["bbox"]) >= .70 for prior in detections):
                detections.append(item)
        pose = {"validated": False, "full_body_evidence": False, "reason": "not_single_confident_person"}
        if with_pose and len(detections) == 1 and detections[0]["confidence"] >= .55:
            pose = self._pose_for(jpeg,detections[0])
        if not with_pose:
            pose["reason"] = "candidate_detection_only"
        result = {"detections": detections, "width": width, "height": height,
                "image_sha256": hashlib.sha256(jpeg).hexdigest(), "candidate_count":len(detections),
                "provider": "local_opencv_mobilenet_ssd", "model_sha256": MODEL_SHA256,
                "network_sha256": NETWORK_SHA256, "inference_ms": round(elapsed, 3),
                "preprocessing": "aspect_preserving_square_padding_300_bgr",
                "quality": {"detection": True, "frame_margin_check": False,
                            "pose_validation": bool(pose["validated"]),
                            "full_body_evidence": bool(pose["full_body_evidence"])},
                "pose": pose,
                "scope": "通用人物外观检测；不识别身份，不证明身体无遮挡。"}
        result["candidate_binding"] = self._binding(result)
        return result


def assess_framing(result, confidence=.55, edge_margin=.025, min_height_fraction=.45,
                   max_height_fraction=.80, center_tolerance=.16, capture_margin=.08):
    """Conservative single-person composition gate, separate from detection.

    `subject_ready` permits geometry planning; `ready` permits the *image
    composition* part of capture only. Caller must separately verify fresh
    imagery, stationary aircraft/person, valid geometry and no collision risk.
    No bounding box can prove the anatomical body is fully visible.
    """
    if not isinstance(result, dict) or not isinstance(result.get("detections"), list):
        raise PhotoDetectionError("人物检测结果无效。")
    if not (all(_valid_number(value) for value in (confidence, edge_margin, min_height_fraction,
                                                   max_height_fraction, center_tolerance, capture_margin)) and
            0 < confidence < 1 and 0 < edge_margin < .2 and
            0 < min_height_fraction < max_height_fraction < 1 and 0 < center_tolerance < .5 and
            edge_margin <= capture_margin < .25):
        raise PhotoDetectionError("全身照构图阈值无效。")
    input_quality = result.get("quality", {})
    selected = "selected_detection_index" in result
    actual_count = len(result["detections"])
    if selected:
        all_people = result.get("all_detections")
        index = result["selected_detection_index"]
        if (not isinstance(all_people,list) or type(index) is not int or not 0<=index<len(all_people) or
                len(result["detections"])!=1 or result["detections"][0]!=all_people[index] or
                result.get("all_person_count")!=len(all_people)):
            raise PhotoDetectionError("选中人物与全部候选记录不一致。")
        actual_count = len(all_people)
    quality = {"detection": True, "frame_margin_check": False,
               "pose_validation": input_quality.get("pose_validation") is True,
               "full_body_evidence": input_quality.get("full_body_evidence") is True,
               "full_body_proven": False, "stationarity_validation": False}
    report = {"ready": False, "subject_ready": False, "reason": "no_person", "quality": quality,
              "person_count": actual_count,"selected_person_count":len(result["detections"]) if selected else 0,
              "explicit_selection":selected}
    quality["scene_single_person"] = actual_count==1
    if not result["detections"]:
        report["message"] = "画面中尚未检测到人物，请站到镜头前。"
        return report
    if len(result["detections"]) != 1:
        report.update(reason="multiple_people", message="画面中有多个人物，不能确定要为谁拍照。")
        return report
    item = result["detections"][0]
    if not isinstance(item, dict) or not _valid_number(item.get("confidence")) or not 0 <= item["confidence"] <= 1:
        raise PhotoDetectionError("人物检测置信度无效。")
    bbox = _box(item.get("bbox"))
    report.update(bbox=bbox, confidence=item["confidence"])
    if item["confidence"] < confidence:
        report.update(reason="low_confidence", message="人物检测不够可靠，请保持站立并稍等。")
        return report
    left, top, right, bottom = bbox
    margins = {"left": left, "top": top, "right": 1 - right, "bottom": 1 - bottom}
    report["bbox_margins"] = dict(margins)
    report["margins"] = margins
    if item.get("clipped") or min(margins.values()) < edge_margin:
        report.update(reason="edge_truncation", message="人物靠近画面边缘，暂不能确认完整入镜；请站远一些。")
        return report
    quality["subject_margin_check"] = True
    report["subject_ready"] = True
    # A foot predicted just outside the detector box still belongs to the final
    # composition. Require capture margins around the union, not merely the box.
    if quality["pose_validation"] and isinstance(result.get("pose"), dict):
        from .photo_pose import KEYPOINT_NAMES
        keypoints = result["pose"].get("keypoints", [])
        if isinstance(keypoints, list) and len(keypoints) == 33:
            indices = result["pose"].get("validated_keypoint_indices", list(KEYPOINT_NAMES))
            if not isinstance(indices,list) or not indices or not all(type(index) is int and index in KEYPOINT_NAMES for index in indices):
                raise PhotoDetectionError("已验证人体关键点索引无效。")
            visible_points = [keypoints[index] for index in indices]
            if all(isinstance(point, dict) and _valid_number(point.get("x")) and
                   _valid_number(point.get("y")) for point in visible_points):
                margins = {"left": min(left, min(p["x"] for p in visible_points)),
                           "top": min(top, min(p["y"] for p in visible_points)),
                           "right": 1-max(right, max(p["x"] for p in visible_points)),
                           "bottom": 1-max(bottom, max(p["y"] for p in visible_points))}
                report["margins"] = margins
    occupied = bottom - top
    report.update(height_fraction=occupied, center=[(left + right) / 2, (top + bottom) / 2])
    if min(margins.values()) < capture_margin:
        report.update(reason="insufficient_margin", message="人物虽未触边，但最终照片的四边留白不足 {:.0%}。".format(capture_margin))
    elif occupied < min_height_fraction:
        report.update(reason="too_small", message="人物在画面中偏小，需要在已观测安全空间内调整构图。")
    elif occupied > max_height_fraction:
        report.update(reason="too_large", message="人物在画面中偏大，需要保留头顶和脚下空间。")
    elif (abs(report["center"][0] - .5) > center_tolerance or
          abs(report["center"][1] - .5) > center_tolerance):
        report.update(reason="off_center", message="人物未居中，需要调整取景朝向。")
    else:
        report.update(ready=True, reason="framing_ready", message=("已选人物" if selected else "单人画面")+
                      "留白和构图达标；仍需确认悬停稳定和其他人安全后拍摄。")
    quality["frame_margin_check"] = min(margins.values()) >= capture_margin
    return report


def assess_stationarity(previous, current, min_iou=.80, max_center_change=.025):
    """Two-frame image stability only; caller provides distinct fresh frames."""
    if not (0 < min_iou < 1 and 0 < max_center_change < .5):
        raise PhotoDetectionError("人物图像稳定阈值无效。")
    before, after = assess_framing(previous), assess_framing(current)
    if not before["subject_ready"] or not after["subject_ready"]:
        return {"stable": False, "reason": "subject_not_ready"}
    first, second = before["bbox"], after["bbox"]
    change = max(abs((first[0] + first[2] - second[0] - second[2]) / 2),
                 abs((first[1] + first[3] - second[1] - second[3]) / 2))
    overlap = _iou(first, second)
    return {"stable": overlap >= min_iou and change <= max_center_change,
            "iou": overlap, "center_change": change,
            "scope": "两张不同新帧中的人物框稳定；不是身份匹配或绝对静止证明。"}
