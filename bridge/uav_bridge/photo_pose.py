"""Pixel-only head/feet evidence for one upright stationary portrait subject.

Uses the Apache-2.0 OpenCV Zoo MediaPipe pose model. This deliberately narrow
upright crop differs from upstream's rotated person-keypoint crop: it is only
valid for the standing portrait task. No predicted world-space pose is used.
"""
import hashlib
import math
from pathlib import Path
import sys
import threading
import time

from .planner import PlanError


ROOT = Path(__file__).resolve().parents[2]
POSE_SHA256 = "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f"
POSE_PATH = ROOT / "models" / "photo_pose_mediapipe" / "pose_estimation_mediapipe_2023mar.onnx"
RUNTIME_PATH = ROOT / ".runtime" / "photo_pose_runtime"
KEYPOINT_NAMES = {0: "nose", 11: "left_shoulder", 12: "right_shoulder",
                  23: "left_hip", 24: "right_hip", 27: "left_ankle", 28: "right_ankle",
                  29: "left_heel", 30: "right_heel", 31: "left_foot_index", 32: "right_foot_index"}
REQUIRED_BODY_POINTS = (0,11,12,23,24,27,28)
FOOT_GROUPS = (("left",27,29,31),("right",28,30,32))
POSTURE_POINT_NAMES = {0:"nose",2:"left_eye",5:"right_eye",9:"left_mouth",10:"right_mouth",
                       11:"left_shoulder",12:"right_shoulder",13:"left_elbow",14:"right_elbow",
                       15:"left_wrist",16:"right_wrist",23:"left_hip",24:"right_hip"}


class PoseError(PlanError):
    pass


def posture_diagnostics(landmarks, confidence, width, height, image_sha256, bbox,
                        visibility_threshold=.70, presence_threshold=.80):
    """Same-image 2-D measurements, deliberately NOT a gesture/yaw classifier.

    Image x/y coordinates have different scales on a non-square camera. Convert
    to original pixels first, then normalize by the shoulder-to-hip center
    distance. Missing points produce unavailable measurements, not a negative
    semantic decision. The VLM must inspect the bound JPEG for meaning.
    """
    if (not isinstance(landmarks,list) or len(landmarks)!=33 or type(width) is not int or
            type(height) is not int or width<1 or height<1 or
            isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or
            not math.isfinite(confidence) or not 0<=confidence<=1):
        raise PoseError("姿态辅助测量的图像或关键点输入无效。")
    if (not isinstance(image_sha256,str) or len(image_sha256)!=64 or
            any(character not in "0123456789abcdef" for character in image_sha256)):
        raise PoseError("姿态辅助测量缺少有效的原图校验值。")
    if (not isinstance(bbox,(tuple,list)) or len(bbox)!=4 or
            not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in bbox) or
            not (0<=bbox[0]<bbox[2]<=1 and 0<=bbox[1]<bbox[3]<=1)):
        raise PoseError("姿态辅助测量需要有效人物框。")
    points={};quality={};trusted=[]
    for index,name in POSTURE_POINT_NAMES.items():
        point=landmarks[index]
        if (not isinstance(point,dict) or any(isinstance(point.get(key),bool) or
                not isinstance(point.get(key),(int,float)) or not math.isfinite(point[key])
                for key in ("x","y","visibility","presence")) or
                not 0<=point["visibility"]<=1 or not 0<=point["presence"]<=1):
            raise PoseError("姿态辅助关键点包含无效数值。")
        points[index]=(float(point["x"]*width),float(point["y"]*height))
        usable=(confidence>=.75 and point["visibility"]>=visibility_threshold and
                point["presence"]>=presence_threshold and 0<=point["x"]<=1 and 0<=point["y"]<=1)
        quality[name]={"index":index,"visibility":float(point["visibility"]),
                       "presence":float(point["presence"]),"usable":usable}
        if usable:trusted.append(index)
    def subtract(a,b): return (a[0]-b[0],a[1]-b[1])
    def norm(vector): return math.hypot(vector[0],vector[1])
    def dot(a,b): return a[0]*b[0]+a[1]*b[1]
    def middle(a,b): return ((a[0]+b[0])/2,(a[1]+b[1])/2)
    def missing(indices): return [POSTURE_POINT_NAMES[index] for index in indices if index not in trusted]
    def elbow_angle(shoulder,elbow,wrist):
        first,second=subtract(shoulder,elbow),subtract(wrist,elbow)
        divisor=norm(first)*norm(second)
        if min(norm(first),norm(second))<2 or divisor<=0:return None
        return math.degrees(math.acos(max(-1.,min(1.,dot(first,second)/divisor))))
    shoulders=middle(points[11],points[12]);hips=middle(points[23],points[24])
    torso=subtract(hips,shoulders);length=norm(torso)
    torso_missing=missing((11,12,23,24))
    torso_available=not torso_missing and length>=16
    axis=(torso[0]/length,torso[1]/length) if length>0 else (0.,0.)
    lateral=(axis[1],-axis[0])
    shoulder_span=norm(subtract(points[11],points[12]))
    evidence={"coordinate_space":"original_image_pixels","image_size":[width,height],
              "torso_length_px":float(length),"torso_axis":list(axis),
              "shoulder_span_px":float(shoulder_span),"point_quality":quality}
    result={"version":1,"role":"auxiliary_measurements_only","image_sha256":image_sha256,
            "input_bbox":list(map(float,bbox)),"model_sha256":POSE_SHA256,
            "pose_confidence":float(confidence),"identity_verified":False,
            "trusted_keypoint_indices":trusted,
            "scope":"同一张JPEG的二维关节测量，不判定已经叉腰或面向镜头；不提供身份或真实朝向，语义由VLM看图判断。"}
    face_indices=(0,2,5,9,10,11,12,23,24)
    face_missing=missing(face_indices);eye_mid=middle(points[2],points[5]);eye_span=norm(subtract(points[2],points[5]))
    face_available=not face_missing and torso_available and eye_span>=2 and shoulder_span>=4
    face_evidence=dict(evidence)
    face_evidence.update(eye_span_px=float(eye_span),
                         limitation="脸部关键点即使置信度高也可能在背面被模型推测，不能作为正面视角真值。")
    if face_available:
        face_evidence.update(
            nose_lateral_offset_from_eye_midpoint_over_eye_span=dot(subtract(points[0],eye_mid),lateral)/eye_span,
            nose_lateral_offset_from_shoulder_midline_over_shoulder_span=dot(subtract(points[0],shoulders),lateral)/shoulder_span,
            eye_span_over_shoulder_span=eye_span/shoulder_span,
            shoulder_span_over_torso_length=shoulder_span/length,
            mouth_lateral_offset_from_eye_midpoint_over_eye_span=dot(subtract(middle(points[9],points[10]),eye_mid),lateral)/eye_span)
    result["frontal"]={"available":face_available,"missing_points":face_missing,
                       "unavailable_reason":None if face_available else
                       ("unreliable_face_or_torso_points" if face_missing or torso_missing else "insufficient_pixel_resolution"),
                       "evidence":face_evidence}
    hands_evidence=dict(evidence);hands_missing=missing((11,12,13,14,15,16,23,24));side_results={}
    for side,shoulder,elbow,wrist,hip in (("left",11,13,15,23),("right",12,14,16,24)):
        absent=missing((shoulder,elbow,wrist,hip));side_lateral=dot(subtract(points[hip],hips),lateral)
        angle=elbow_angle(points[shoulder],points[elbow],points[wrist])
        available=not absent and torso_available and abs(side_lateral)>=1 and angle is not None
        row={"available":available,"missing_points":absent,
             "landmark_indices":{"shoulder":shoulder,"elbow":elbow,"wrist":wrist,"hip":hip},
             "unavailable_reason":None if available else
             ("unreliable_arm_or_torso_points" if absent or torso_missing else "degenerate_or_low_resolution_projection")}
        if available:
            sign=1 if side_lateral>0 else -1
            wrist_hip=subtract(points[wrist],points[hip]);elbow_hip=subtract(points[elbow],points[hip])
            row.update(wrist_to_same_hip_over_torso=norm(wrist_hip)/length,
                       wrist_height_offset_over_torso=dot(wrist_hip,axis)/length,
                       wrist_lateral_offset_from_hip_over_torso=sign*dot(wrist_hip,lateral)/length,
                       elbow_lateral_extension_over_torso=sign*dot(elbow_hip,lateral)/length,
                       elbow_angle_deg=angle)
        side_results[side]=row
    hands_evidence.update(side_results)
    hands_evidence["measurement_conventions"]={
        "wrist_height_offset":"positive towards the hips/feet along the shoulder-to-hip axis",
        "elbow_lateral_extension":"positive away from the same-side hip/torso",
        "elbow_angle":"180 degrees is straight; measurements have no gesture pass/fail threshold"}
    result["hands_on_hips"]={"available":all(row["available"] for row in side_results.values()),
                             "missing_points":hands_missing,"evidence":hands_evidence,
                             "limitation":"手腕靠近髋部本身不能证明叉腰；手下垂、遮挡和透视可能产生相似坐标，需VLM检查真实图像。"}
    return result


def assess_pose(landmarks, confidence, bbox, visibility_threshold=.70, presence_threshold=.80):
    """Return conservative model evidence; it remains an estimate, not proof."""
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or
            not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise PoseError("人体姿态置信度无效。")
    result = {"validated": False, "full_body_evidence": False,
              "reason": "pose_low_confidence", "confidence": float(confidence),
              "scope": "仅关键点模型估计的头和双脚可见证据；不识别身份，不证明无遮挡或人员安全。"}
    if confidence < .75:
        return result
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        raise PoseError("人体关键点数量无效。")
    left, top, right, bottom = bbox
    for index in KEYPOINT_NAMES:
        point = landmarks[index]
        if (not isinstance(point, dict) or
                any(isinstance(point.get(key), bool) or not isinstance(point.get(key), (int, float)) or
                    not math.isfinite(point[key]) for key in ("x", "y", "visibility", "presence"))):
            raise PoseError("人体关键点数据无效。")
        if not 0 <= point["visibility"] <= 1 or not 0 <= point["presence"] <= 1:
            raise PoseError("人体关键点概率无效。")

    def visible(index):
        point = landmarks[index]
        return point["visibility"] >= visibility_threshold and point["presence"] >= presence_threshold

    def within_frame_and_person(index):
        point = landmarks[index]
        if not .025 <= point["x"] <= .975 or not .025 <= point["y"] <= .975:
            result.update(reason="head_or_feet_uncertain", uncertain_keypoint=KEYPOINT_NAMES[index])
            return False
        # Associate the pose with the detected person, not a nearby pattern.
        if not (left - .10 * (right-left) <= point["x"] <= right + .10 * (right-left) and
                top - .05 * (bottom-top) <= point["y"] <= bottom + .05 * (bottom-top)):
            result.update(reason="pose_outside_person", uncertain_keypoint=KEYPOINT_NAMES[index])
            return False
        return True

    selected = list(REQUIRED_BODY_POINTS)
    for index in REQUIRED_BODY_POINTS:
        if not visible(index):
            result.update(reason="head_or_feet_uncertain", uncertain_keypoint=KEYPOINT_NAMES[index])
            return result
        if not within_frame_and_person(index):
            return result
    nose = landmarks[0]["y"]
    shoulders = (landmarks[11]["y"] + landmarks[12]["y"]) / 2
    hips = (landmarks[23]["y"] + landmarks[24]["y"]) / 2
    ankles = (landmarks[27]["y"] + landmarks[28]["y"]) / 2
    if not (nose < shoulders < hips < ankles and ankles - nose >= .55 * (bottom-top)):
        result["reason"] = "not_upright_full_body"
        return result
    # High logits do not rescue anatomically disconnected hallucinations.
    # A front-facing foot often self-occludes its heel. Require the ankle and at
    # least one visible endpoint per foot, with the SAME probability thresholds.
    # Every credible endpoint must be anatomically consistent and in-frame;
    # selecting a good toe must not hide a confidently cropped/displaced heel.
    body_height = bottom-top
    if abs(landmarks[27]["y"]-landmarks[28]["y"]) > .15*body_height:
        result["reason"] = "inconsistent_feet"
        return result
    foot_evidence = {}
    for side, ankle_index, heel_index, toe_index in FOOT_GROUPS:
        ankle = landmarks[ankle_index]
        if ankle["y"] < hips + .15*body_height:
            result["reason"] = "inconsistent_feet"
            return result
        accepted = []
        for foot_index in (heel_index, toe_index):
            if not visible(foot_index):
                continue
            if not within_frame_and_person(foot_index):
                return result
            foot = landmarks[foot_index]
            if (abs(foot["y"]-ankle["y"]) > .15*body_height or
                    abs(foot["x"]-ankle["x"]) > .65*(right-left) or
                    foot["y"] < hips + .15*body_height):
                result.update(reason="inconsistent_feet", uncertain_keypoint=KEYPOINT_NAMES[foot_index])
                return result
            accepted.append(foot_index)
        if not accepted:
            result.update(reason="foot_endpoint_uncertain", uncertain_foot=side,
                          uncertain_keypoints=[KEYPOINT_NAMES[heel_index],KEYPOINT_NAMES[toe_index]])
            return result
        selected.extend(accepted)
        foot_evidence[side] = {"ankle_index": ankle_index, "endpoint_indices": accepted,
                               "endpoint_names": [KEYPOINT_NAMES[index] for index in accepted]}
    result.update(validated=True, full_body_evidence=True, reason="head_and_both_feet_visible",
                  validated_keypoint_indices=sorted(selected), foot_evidence=foot_evidence,
                  minimum_visibility=min(landmarks[i]["visibility"] for i in selected),
                  minimum_presence=min(landmarks[i]["presence"] for i in selected))
    return result


class PoseVerifier:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else POSE_PATH
        self._session = None
        self._lock = threading.Lock()

    def _load(self):
        if self._session is not None:
            return
        try:
            digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        except OSError:
            raise PoseError("全身姿态模型未安装，请运行 scripts/download_photo_pose.py。") from None
        if digest != POSE_SHA256:
            raise PoseError("全身姿态模型校验失败。")
        if str(RUNTIME_PATH) not in sys.path:
            sys.path.insert(0, str(RUNTIME_PATH))
        try:
            import onnxruntime as ort
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                                 providers=["CPUExecutionProvider"])
        except Exception:
            raise PoseError("本地全身姿态推理依赖不可用，请运行 scripts/download_photo_pose.py。") from None

    def verify(self, jpeg, bbox):
        import cv2
        import numpy as np
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or
                not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox) or
                not (0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)):
            raise PoseError("姿态检测需要有效人物框。")
        if (not isinstance(jpeg, (bytes, bytearray)) or not 4 <= len(jpeg) <= 4*1024*1024 or
                not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
            raise PoseError("姿态检测需要有效 JPEG。")
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise PoseError("姿态检测无法解码 JPEG。")
        height, width = image.shape[:2]
        left, top, right, bottom = (bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height)
        # A 25% enlarged square for an upright standing person, with no fabricated
        # body keypoints or depth. Black padding is only outside the real image.
        side = max(right-left, bottom-top) * 1.25
        center = ((left+right)/2, (top+bottom)/2)
        origin = (center[0] - side/2, center[1] - side/2)
        scale = 256. / side
        transform = np.array([[scale, 0, -origin[0]*scale],
                              [0, scale, -origin[1]*scale]], dtype=np.float32)
        crop = cv2.warpAffine(image, transform, (256, 256), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        tensor = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None, :, :, :] / 255.
        with self._lock:
            self._load()
            started = time.monotonic()
            output = self._session.run(["Identity", "Identity_1"], {"input_1": tensor})
            elapsed = (time.monotonic() - started) * 1000.
        if output[0].shape != (1,195) or output[1].shape != (1,1) or not np.isfinite(output[0]).all():
            raise PoseError("人体姿态模型输出无效。")
        rows = output[0].reshape(39,5)
        # 33 anatomical landmarks; remaining 6 auxiliary points are discarded.
        landmarks = []
        for index, row in enumerate(rows[:33]):
            landmarks.append({"index": index,
                              "x": float((row[0]/256.*side + origin[0])/width),
                              "y": float((row[1]/256.*side + origin[1])/height),
                              "visibility": float(1./(1.+math.exp(-max(-80., min(80., float(row[3])))))),
                              "presence": float(1./(1.+math.exp(-max(-80., min(80., float(row[4]))))))})
        confidence = float(output[1][0,0])
        report = assess_pose(landmarks, confidence, bbox)
        image_digest = hashlib.sha256(jpeg).hexdigest()
        report.update(keypoints=landmarks, provider="local_mediapipe_pose_onnx",
                      model_sha256=POSE_SHA256, inference_ms=round(elapsed,3),
                      image_sha256=image_digest,input_bbox=list(map(float,bbox)),
                      posture_diagnostics=posture_diagnostics(landmarks,confidence,width,height,image_digest,bbox),
                      preprocessing="upright_bbox_square_1.25_rgb_256")
        return report
