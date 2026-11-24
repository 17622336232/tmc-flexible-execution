"""Image-only YOLO candidates, isolated from ROS NumPy and pose runtimes."""
import base64
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from .photo_detector import (PersonDetector, PhotoDetectionError, PhotoDetectionTimeout,
                             PhotoDetectionCancelled, _box, _valid_number)

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'models/photo_person_yolov8n/yolov8n.onnx'
MODEL_SHA256 = '1ef48c76ce4efe56afda1d9fb1391c35d0866b4f272cedf97bc9a523e6975877'
WORKER = Path(__file__).with_name('photo_yolo_worker.py')
RUNTIME = ROOT/'.runtime/photo_detection_runtime'
# Cold imports, ONNX session construction and the worker's real warmup are
# startup work, not the per-frame inference budget used during a live task.
STARTUP_TIMEOUT_SECONDS = 60.
STDERR_LIMIT_BYTES = 4096


class YoloPersonDetector(PersonDetector):
    supports_detection_deadline = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._process = None
        self._buffer = b''
        self._sequence = 0
        self._ready = False
        self._stderr = {'tail': b'', 'stop': threading.Event()}
        self._stderr_thread = None

    def availability(self):
        result = super().availability()
        ready = MODEL.is_file() and (RUNTIME/'onnxruntime/__init__.py').is_file()
        result.update(person_model_ready=MODEL.is_file(), detection_runtime_ready=ready,
                      configured=ready and (not self._verify_pose or result['pose_model_ready'] and result['pose_runtime_ready']),
                      provider='local_yolov8n_onnx_cpu_isolated')
        return result

    @staticmethod
    def _drain_stderr(stream, capture):
        """Bound memory while draining even when the parent writes an image.

        Waiting until stdout is read can deadlock: a worker blocked on stderr
        cannot read the large JPEG the parent is concurrently writing to stdin.
        Each worker owns its capture so a retired reader cannot alter a new one.
        """
        try:
            while not capture['stop'].is_set():
                if not select.select([stream], [], [], .1)[0]:
                    continue
                data = os.read(stream.fileno(), 65536)
                if not data:
                    return
                capture['tail'] = (capture['tail']+data)[-STDERR_LIMIT_BYTES:]
        except (OSError, ValueError):
            # Closing a retired worker's descriptors also ends its reader.
            return

    @staticmethod
    def _remaining(deadline, cancelled=None, startup=False, timeout=0.):
        if cancelled is not None and cancelled.is_set():
            raise PhotoDetectionCancelled('本地 YOLO 人物检测已取消，未采用结果。')
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise PhotoDetectionTimeout(('本地 YOLO 冷启动超过 %.1f 秒，尚未完成模型加载与预热。' % timeout)
                                        if startup else '本地 YOLO 人物检测超时，未采用迟到结果。')
        return remaining

    def _read(self, timeout, startup=False, *, deadline=None, cancelled=None):
        deadline = time.monotonic()+timeout if deadline is None else deadline
        while b'\n' not in self._buffer:
            remaining = self._remaining(deadline, cancelled, startup, timeout)
            process = self._process
            if not select.select([process.stdout], [], [], min(.05, remaining))[0]:
                continue
            data = os.read(process.stdout.fileno(), 65536)
            if not data or len(self._buffer)+len(data) > 262144:
                raise PhotoDetectionError('本地 YOLO 冷启动进程退出或输出无效。' if startup
                                          else '本地 YOLO 检测进程不可用。')
            self._buffer += data
        line, self._buffer = self._buffer.split(b'\n', 1)
        result = json.loads(line)
        self._remaining(deadline, cancelled, startup, timeout)
        return result

    def _write(self, data, deadline, cancelled):
        """Bound pipe backpressure too; never block before the read deadline."""
        descriptor = self._process.stdin.fileno()
        pending = memoryview(data)
        while pending:
            remaining = self._remaining(deadline, cancelled)
            if not select.select([], [descriptor], [], min(.05, remaining))[1]:
                continue
            try:
                written = os.write(descriptor, pending[:65536])
            except BlockingIOError:
                continue
            if written <= 0:
                raise PhotoDetectionError('本地 YOLO 检测请求无法写入。')
            pending = pending[written:]

    def _start(self, timeout=5., *, deadline=None, cancelled=None):
        deadline = time.monotonic()+timeout if deadline is None else deadline
        self._remaining(deadline, cancelled, True, timeout)
        if self._ready and self._process is not None and self._process.poll() is None:
            return
        # Never mistake a still-running, half-initialized worker for readiness.
        self.close()
        if not self.availability()['configured'] or hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
            raise PhotoDetectionError('本地 YOLO 模型或隔离运行时未准备好，请运行 scripts/install_photo_yolo.py。')
        self._buffer = b''
        self._stderr = {'tail': b'', 'stop': threading.Event()}
        logging.info('正在加载本地 YOLO 人物检测模型（启动等待上限 %.0f 秒）。', timeout)
        try:
            self._remaining(deadline, cancelled, True, timeout)
            self._process = subprocess.Popen([sys.executable, str(WORKER)], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True,
                                             bufsize=0)
            process = self._process
            os.set_blocking(process.stdin.fileno(), False)
            self._stderr_thread = threading.Thread(target=self._drain_stderr,
                args=(self._process.stderr, self._stderr), name='diff-vla-yolo-stderr', daemon=True)
            self._stderr_thread.start()
            ready = self._read(timeout, startup=True, deadline=deadline, cancelled=cancelled)
            if (not isinstance(ready, dict) or set(ready) != {'ready', 'model_sha256'}
                    or ready['ready'] is not True or ready['model_sha256'] != MODEL_SHA256
                    or self._process is not process or process.poll() is not None):
                raise PhotoDetectionError('本地 YOLO 冷启动确认无效，未完成受校验模型的初始化。')
            self._ready = True
        except BaseException as error:
            self.close()
            if not isinstance(error, Exception):
                raise
            if isinstance(error, (PhotoDetectionTimeout, PhotoDetectionCancelled)):
                raise
            # Only local worker startup diagnostics, never image requests or
            # model/API credentials. Bound both retained bytes and displayed text.
            detail = ' '.join(self._stderr['tail'].decode('utf-8', errors='replace').split())[-800:]
            message = str(error)[:300] if isinstance(error, PhotoDetectionError) else '本地 YOLO 冷启动失败（%s）。' % type(error).__name__
            raise PhotoDetectionError(message+(' 本地诊断：'+detail if detail else '')) from None
        logging.info('本地 YOLO 人物检测模型已完成加载与预热。')

    def warmup(self):
        with self._lock:
            self._start(STARTUP_TIMEOUT_SECONDS)

    def close(self):
        process, self._process = self._process, None
        reader, self._stderr_thread = self._stderr_thread, None
        capture = self._stderr
        self._ready = False
        self._buffer = b''
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=1.)
            finally:
                if reader is not None and reader.ident is not None:
                    reader.join(timeout=.2)  # collect the exited worker's last diagnostic
                capture['stop'].set()
                if reader is not None and reader.is_alive():
                    reader.join(timeout=.2)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except (OSError, ValueError):
                            pass

    def detect(self, jpeg, *, deadline=None, cancelled=None):
        return self._detect(jpeg, with_pose=self._verify_pose, deadline=deadline, cancelled=cancelled)

    def detect_people(self, jpeg, *, deadline=None, cancelled=None):
        return self._detect(jpeg, with_pose=False, deadline=deadline, cancelled=cancelled)

    def detect_cars(self, jpeg, *, deadline=None, cancelled=None):
        """COCO car candidates only; no colour, view, identity or 3-D claim."""
        try:
            return self._detect(jpeg, with_pose=False, deadline=deadline, cancelled=cancelled, target_class=2)
        except (PhotoDetectionTimeout, PhotoDetectionCancelled) as error:
            raise type(error)(str(error).replace('YOLO 人物检测', 'YOLO 车辆检测')) from None

    def select_subject(self, jpeg, frozen_detection, candidate_index):
        # The inherited operation verifies a selected PERSON and can request
        # human pose evidence. Car candidates must not enter that contract.
        if isinstance(frozen_detection, dict):
            target_class = frozen_detection.get('target_class', 0)
            candidates = frozen_detection.get('detections')
            if (type(target_class) is not int or target_class != 0 or
                    (isinstance(candidates, list) and any(not isinstance(item, dict) or item.get('label') != 'person'
                                                        for item in candidates))):
                raise PhotoDetectionError('车辆候选不能作为人物进行选人或姿态验证。')
        return super().select_subject(jpeg, frozen_detection, candidate_index)

    def _detect(self, jpeg, with_pose, *, deadline=None, cancelled=None, target_class=0):
        if type(target_class) is not int or target_class not in (0, 2):
            raise PhotoDetectionError('本地 YOLO 检测类别无效。')
        label = 'person' if target_class == 0 else 'car'
        if (not isinstance(jpeg, (bytes, bytearray)) or not 4 <= len(jpeg) <= 4*1024*1024
                or not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9')):
            raise PhotoDetectionError(('人物' if target_class == 0 else '车辆')+'检测需要有效且不超过 4 MiB 的 JPEG。')
        deadline = time.monotonic()+5. if deadline is None else deadline
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise PhotoDetectionError(('人物' if target_class == 0 else '车辆')+'检测需要有效的作业截止时间。')
        while not self._lock.acquire(timeout=min(.05, self._remaining(deadline, cancelled))):
            pass
        try:
            try:
                self._start(self._remaining(deadline, cancelled), deadline=deadline, cancelled=cancelled)
                digest = hashlib.sha256(jpeg).hexdigest()
                self._sequence += 1
                request = dict(id=self._sequence, jpeg=base64.b64encode(jpeg).decode('ascii'), threshold=self.threshold)
                if target_class != 0:
                    request['target_class'] = target_class
                self._write(json.dumps(request, separators=(',', ':')).encode()+b'\n', deadline, cancelled)
                measured = self._read(self._remaining(deadline, cancelled), deadline=deadline, cancelled=cancelled)
                if measured.get('id') != self._sequence or measured.get('image_sha256') != digest or 'error' in measured:
                    raise PhotoDetectionError('YOLO 检测结果与原始 JPEG 不一致。')
                returned_class = measured.get('target_class', 0)
                if type(returned_class) is not int or returned_class != target_class:
                    raise PhotoDetectionError('YOLO 检测结果类别与本次请求不一致。')
                detections = measured.get('detections')
                if (not isinstance(detections, list) or len(detections) > 100
                        or type(measured.get('width')) is not int or type(measured.get('height')) is not int
                        or not 64 <= measured['width'] <= 4096 or not 64 <= measured['height'] <= 4096):
                    raise PhotoDetectionError('YOLO 检测结果格式无效。')
                for item in detections:
                    _box(item.get('bbox'))
                    if not _valid_number(item.get('confidence')) or not self.threshold <= item['confidence'] <= 1 or item.get('label') != label:
                        raise PhotoDetectionError('YOLO 检测候选类别或置信度无效。')
                self._remaining(deadline, cancelled)
            except BaseException as error:
                self.close()
                if not isinstance(error, Exception) or isinstance(error, PhotoDetectionError):
                    raise
                raise PhotoDetectionError('本地 YOLO 检测进程失败，未改用其他结果。') from None
        finally:
            self._lock.release()
        pose = {'validated': False, 'full_body_evidence': False, 'reason': 'candidate_detection_only'}
        if target_class == 0 and with_pose and len(detections) == 1:
            pose = self._pose_for(jpeg, detections[0])
        self._remaining(deadline, cancelled)
        result = dict(measured, candidate_count=len(detections), provider='local_yolov8n_onnx_cpu_isolated',
                      model_sha256=MODEL_SHA256, network_sha256=MODEL_SHA256,
                      preprocessing='aspect_preserving_letterbox_640_rgb', pose=pose,
                      quality={'detection': True, 'frame_margin_check': False,
                               'pose_validation': bool(pose['validated']), 'full_body_evidence': bool(pose['full_body_evidence'])},
                      scope=('通用人物候选检测；不判断身份、摄影条件或世界坐标。' if target_class == 0 else
                             '通用车辆候选检测；允许局部可见，不判断颜色、侧面、身份、完整车身或世界坐标。'))
        result.pop('id', None)
        result['candidate_binding'] = self._binding(result)
        self._remaining(deadline, cancelled)
        return result
