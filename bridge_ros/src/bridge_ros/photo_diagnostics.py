"""Opt-in bounded local failure evidence, outside flight admission/control."""
import copy
import json
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time

import numpy as np


class PhotoDiagnostics:
    LIMIT = 12

    @classmethod
    def from_environment(cls):
        directory = os.environ.get("VLA_PHOTO_DIAGNOSTICS_DIR")
        if not directory:
            return None
        try:
            return cls(directory)
        except Exception:
            return None

    def __init__(self, directory):
        self.directory = Path(directory)
        if not self.directory.is_absolute():
            raise ValueError("Diagnostics require an explicit absolute local directory")
        self.lock = threading.Lock()
        self.pending = queue.Queue(maxsize=self.LIMIT)
        self.accepted = 0
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self._run, name="photo-local-diagnostics", daemon=True)
        self.worker.start()

    def submit(self, frozen, bbox, exception):
        # Retain immutable frozen image/array references. Compression and file
        # IO never run on the mission thread or delay its original rejection.
        try:
            failure = dict(type=type(exception).__name__, code=getattr(exception, "code", None),
                           message=str(exception), evidence=copy.deepcopy(getattr(exception, "evidence", {})))
            with self.lock:
                if self.stopped.is_set() or self.accepted >= self.LIMIT:
                    return
                self.pending.put_nowait((frozen, list(bbox), failure))
                self.accepted += 1
        except Exception:
            pass

    def _run(self):
        while not self.stopped.is_set() or not self.pending.empty():
            try:
                item = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self._write(*item)
            except Exception:
                pass  # Disk failure cannot replace the original safety error.
            finally:
                self.pending.task_done()

    def _write(self, frozen, bbox, failure):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for index in range(1, self.LIMIT+1):
            target = self.directory/("photo-failure-%02d" % index)
            reservation = self.directory/(".photo-failure-%02d.lock" % index)
            if target.exists():
                continue
            try:
                descriptor = os.open(str(reservation), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                continue
            os.close(descriptor)
            temporary = None
            try:
                if target.exists():
                    continue
                temporary = Path(tempfile.mkdtemp(prefix=".photo-failure-", dir=str(self.directory)))
                public = {key: value for key, value in frozen["public"].items() if key != "data"}
                metadata = dict(version=1, local_diagnostic_only=True, decision_changed=False,
                                simulator_object_truth_used=False, failure=failure,
                                bbox_normalized=bbox, observation=public,
                                cloud_stamp=frozen["cloud_stamp"], camera_info_stamp=frozen["camera_info_stamp"],
                                sync_delta_s=frozen["sync_delta_s"], calibration_id=frozen["calibration_id"])
                (temporary/"onboard.jpg").write_bytes(frozen["public"]["data"])
                projected = frozen["projection"]
                np.savez_compressed(str(temporary/"projection.npz"),
                                    uv=projected["uv"], depth=projected["depth"],
                                    points_world=projected["points_world"], points_world_all=projected["points_world_all"],
                                    width=projected["width"], height=projected["height"],
                                    K=frozen["K"], body_T_optical=frozen["body_T_optical"],
                                    camera_T_world=frozen["camera_T_world"], bbox_normalized=np.asarray(bbox))
                (temporary/"failure.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
                os.rename(str(temporary), str(target))  # Publish complete bundle atomically.
                temporary = None
                return
            finally:
                if temporary is not None:
                    shutil.rmtree(str(temporary), ignore_errors=True)
                try:
                    reservation.unlink()
                except OSError:
                    pass

    def flush(self, timeout=2.):
        deadline = time.monotonic()+timeout
        while self.pending.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(.005)
        return self.pending.unfinished_tasks == 0

    def close(self):
        self.stopped.set()
        self.worker.join(timeout=.5)
