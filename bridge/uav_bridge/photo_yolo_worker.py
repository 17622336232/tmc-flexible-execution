"""Local, persistent YOLO inference process; no ROS, network or flight control."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'.runtime/photo_detection_runtime'))


def main():
    import cv2
    import numpy as np
    import onnxruntime as ort
    path = ROOT/'models/photo_person_yolov8n/yolov8n.onnx'
    expected = '1ef48c76ce4efe56afda1d9fb1391c35d0866b4f272cedf97bc9a523e6975877'
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('model_checksum')
    cv2.setNumThreads(1)
    ort.disable_telemetry_events()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape != [1, 3, 640, 640] or session.get_outputs()[0].shape != [1, 84, 8400]:
        raise ValueError('model_shape')
    session.run(None, {'images': np.zeros((1, 3, 640, 640), dtype=np.float32)})
    print(json.dumps({'ready': True, 'model_sha256': expected}), flush=True)
    while True:
        line = sys.stdin.buffer.readline(6*1024*1024)
        if not line:
            return
        try:
            if not line.endswith(b'\n'):
                return
            request = json.loads(line)
            jpeg = base64.b64decode(request['jpeg'], validate=True)
            threshold = request['threshold']
            target_class = request.get('target_class', 0)
            if type(target_class) is not int or target_class not in (0, 2):
                raise ValueError('invalid_target_class')
            label = 'person' if target_class == 0 else 'car'
            if (not 4 <= len(jpeg) <= 4*1024*1024 or type(threshold) not in (int, float)
                    or not .01 <= threshold <= .99):
                raise ValueError('invalid_input')
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('invalid_image')
            height, width = image.shape[:2]
            if not 64 <= width <= 4096 or not 64 <= height <= 4096 or width*height > 8388608:
                raise ValueError('invalid_dimensions')
            start = time.monotonic()
            scale = min(640/width, 640/height)
            rw, rh = round(width*scale), round(height*scale)
            left, top = (640-rw)//2, (640-rh)//2
            padded = np.full((640, 640, 3), 114, dtype=np.uint8)
            padded[top:top+rh, left:left+rw] = cv2.resize(image, (rw, rh))
            tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32)/255.
            output = session.run(None, {'images': tensor})[0]
            if output.shape != (1, 84, 8400) or not np.isfinite(output).all():
                raise ValueError('invalid_output')
            rows = output[0].T
            classes = np.argmax(rows[:, 4:], axis=1)
            score_column = 4 + target_class
            rows = rows[(classes == target_class) & (rows[:, score_column] >= threshold)]
            candidates = []
            for row in rows[np.argsort(-rows[:, score_column])]:
                cx, cy, bw, bh = map(float, row[:4])
                confidence = float(row[score_column])
                raw = [(cx-bw/2-left)/scale/width, (cy-bh/2-top)/scale/height,
                       (cx+bw/2-left)/scale/width, (cy+bh/2-top)/scale/height]
                box = [max(0., min(1., value)) for value in raw]
                if box[0] >= box[2] or box[1] >= box[3] or not 0 <= confidence <= 1:
                    continue
                duplicate = False
                for prior in candidates:
                    other = prior['bbox']
                    overlap = max(0., min(box[2], other[2])-max(box[0], other[0]))*max(0., min(box[3], other[3])-max(box[1], other[1]))
                    union = (box[2]-box[0])*(box[3]-box[1])+(other[2]-other[0])*(other[3]-other[1])-overlap
                    if union > 0 and overlap/union >= .7:
                        duplicate = True
                        break
                if not duplicate:
                    candidates.append(dict(bbox=box, raw_bbox=raw, confidence=confidence,
                                           clipped=box != raw, label=label))
                if len(candidates) >= 100:
                    break
            result = dict(id=request['id'], image_sha256=hashlib.sha256(jpeg).hexdigest(),
                          width=width, height=height, detections=candidates, target_class=target_class,
                          inference_ms=round((time.monotonic()-start)*1000, 3))
        except Exception:
            result = {'error': 'local_yolo_inference_failed'}
        print(json.dumps(result, separators=(',', ':'), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
