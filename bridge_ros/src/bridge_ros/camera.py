"""Latest-frame JPEG buffer, isolated from every flight/control lock.

ROS callbacks only replace one pending reference. A separate bounded-rate
worker encodes it; slow consumers never build a queue or receive a stale image
labelled as live. This module deliberately has no ROS/OpenCV dependency.
"""
import math
import threading
import time


class CameraBuffer:
    def __init__(self, encoder=None, fps=3.0, timeout=2.5, reason="尚未收到机载相机图像", topic=None):
        if not math.isfinite(fps) or not 0.5 <= fps <= 5.0:
            raise ValueError("ros.camera_fps must be between 0.5 and 5")
        if not math.isfinite(timeout) or not 0.5 <= timeout <= 3.0:
            raise ValueError("ros.camera_timeout must be between 0.5 and 3 seconds")
        self.encoder, self.period, self.timeout = encoder, 1.0/fps, timeout
        self.topic = topic
        self.lock = threading.Lock()
        self.event, self.stop = threading.Event(), threading.Event()
        self.pending = self.latest = None
        self.latest_source_stamp = 0.0
        self.generation = 0
        self.reason = reason
        self.worker = None
        if encoder is not None:
            self.worker = threading.Thread(target=self._run, name="diff-vla-camera-jpeg", daemon=True)
            self.worker.start()

    def push(self, message, stamp, ros_stamp):
        if self.encoder is None or not all(math.isfinite(v) and v > 0 for v in (stamp, ros_stamp)):
            return
        # Repeated delivery of a retained/stalled frame must not refresh its age.
        with self.lock:
            if ros_stamp <= self.latest_source_stamp:
                return
            self.latest_source_stamp = ros_stamp
            header = getattr(message, "header", None)
            metadata = dict(frame_id=getattr(header, "frame_id", "").lstrip("/"),
                            raw_width=int(getattr(message, "width", 0)), raw_height=int(getattr(message, "height", 0)))
            self.pending = (message, stamp, ros_stamp, time.monotonic(), metadata, self.generation)
        self.event.set()

    def _run(self):
        next_encode = 0.0
        while not self.stop.is_set():
            self.event.wait(0.2)
            delay = next_encode - time.monotonic()
            if delay > 0 and self.stop.wait(delay):
                break
            if self.stop.is_set():
                break
            with self.lock:
                pending, self.pending = self.pending, None
                self.event.clear()
            if pending is None:
                continue
            message, stamp, ros_stamp, received_mono, metadata, generation = pending
            if time.time()-stamp > self.timeout or time.monotonic()-received_mono > self.timeout:
                continue
            next_encode = time.monotonic() + self.period
            try:
                # Do not hold even the camera buffer lock during conversion.
                data, width, height = self.encoder(message)
                if (not isinstance(data, bytes) or not data.startswith(b"\xff\xd8")
                        or not data.endswith(b"\xff\xd9") or len(data) > 5*1024*1024
                        or width <= 0 or height <= 0):
                    raise ValueError("相机 JPEG 编码结果无效")
                frame = dict(data=data, stamp=stamp, ros_stamp=ros_stamp,
                             width=int(width), height=int(height), received_mono=received_mono)
                frame.update(metadata)
                with self.lock:
                    if generation == self.generation:
                        self.latest = frame
                        self.reason = None
            except Exception as exc:
                # Camera failure must never stop the control or mission threads.
                with self.lock:
                    if generation == self.generation:
                        self.reason = "机载图像处理失败：" + str(exc)[:160]
                        self.latest = None

    def reset(self, reason="传感时钟重置，等待新机载图像"):
        with self.lock:
            self.generation += 1
            self.latest_source_stamp = 0.0
            self.pending = self.latest = None
            self.reason = reason
            self.event.clear()

    def frame(self, include_data=True):
        with self.lock:
            cached = dict(self.latest) if self.latest is not None else None
            reason = self.reason
        result = dict(available=False, stamp=0.0, ros_stamp=0.0, width=0, height=0,
                      reason=reason, topic=self.topic, frame_id="", raw_width=0, raw_height=0)
        if cached:
            result.update({key: cached[key] for key in ("stamp", "ros_stamp", "width", "height", "frame_id", "raw_width", "raw_height")})
            age = max(time.time()-cached["stamp"], time.monotonic()-cached["received_mono"])
            result["available"] = not self.stop.is_set() and age <= self.timeout
            result["reason"] = None if result["available"] else "机载相机图像已过期，等待新画面"
        if include_data:
            result["content_type"] = "image/jpeg"
            result["data"] = cached["data"] if cached and result["available"] else None
        return result

    def close(self):
        self.stop.set()
        self.event.set()
        if self.worker is not None:
            self.worker.join(timeout=0.5)
