"""Small durable cache of reviewed onboard JPEGs, addressed by generated UUIDs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from collections import deque

from .photo_store import PhotoStore


class EvidenceStore:
    def __init__(self, state_directory):
        self.directory = Path(state_directory)/'evidence' if state_directory else None
        self.lock = threading.RLock()
        self._pending = deque()
        self._pending_bytes = 0
        self._queue_lock = threading.Lock()
        self._writer = None
        self._closed = False
        self.dropped_writes = 0

    def submit(self, jpeg, metadata):
        """Best-effort persistence; pixels remain immediately in Bridge's cache.

        One writer and at most eight pending frames / 16 MiB. A slow disk may
        evict old pending diagnostics, never block observation or flight control.
        """
        if self.directory is None:
            return False
        if not isinstance(jpeg, bytes) or len(jpeg) > 5 * 1024 * 1024:
            raise ValueError('审查图片超过缓存范围。')
        frozen = copy.deepcopy(metadata)
        with self._queue_lock:
            if self._closed:
                return False
            while self._pending and (len(self._pending) >= 8 or
                    self._pending_bytes + len(jpeg) > 16 * 1024 * 1024):
                old, _ = self._pending.popleft()
                self._pending_bytes -= len(old)
                self.dropped_writes += 1
            self._pending.append((jpeg, frozen))
            self._pending_bytes += len(jpeg)
            if self._writer is None:
                self._writer = threading.Thread(target=self._drain, name='evidence-cache', daemon=True)
                try:
                    self._writer.start()
                except RuntimeError:
                    self._writer = None
                    return False  # Retry bounded pending diagnostics on a later submission.
        return True

    def _drain(self):
        while True:
            with self._queue_lock:
                if not self._pending:
                    self._writer = None
                    return
                jpeg, metadata = self._pending.popleft()
                self._pending_bytes -= len(jpeg)
            try:
                self.save(jpeg, metadata)
            except Exception:
                # Diagnostic persistence is optional; a failed write must not
                # prevent subsequent frames from being cached or saved.
                pass

    def close(self):
        with self._queue_lock:
            self._closed = True
            writer = self._writer
        if writer is not None and writer is not threading.current_thread():
            writer.join(timeout=.2)

    def _path(self, identity, suffix):
        if not isinstance(identity, str) or not re.fullmatch(r'[a-f0-9-]{36}', identity):
            raise ValueError('审查图片编号无效。')
        if self.directory is None:
            raise FileNotFoundError('未配置持久证据缓存。')
        if any(p.is_symlink() for p in (self.directory, self.directory.parent)):
            raise ValueError('审查图片目录不可为符号链接。')
        return self.directory/(identity+suffix)

    def save(self, jpeg, metadata):
        with self.lock:
            if self.directory is None:
                return
            path = self._path(metadata['id'], '.jpg')
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(str(self.directory), 0o700)
            PhotoStore._atomic(path, jpeg)
            PhotoStore._atomic(self._path(metadata['id'], '.json'),
                               json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode())
            # Up to 128 frames / 64 MiB. Only our UUID cache files are evicted.
            entries = sorted((p for p in self.directory.glob('*.jpg')
                              if re.fullmatch(r'[a-f0-9-]{36}\.jpg', p.name)), key=lambda p:p.stat().st_mtime)
            size = sum(p.stat().st_size for p in entries)
            while len(entries) > 128 or size > 64*1024*1024:
                old = entries.pop(0)
                size -= old.stat().st_size
                old.unlink()
                try:
                    old.with_suffix('.json').unlink()
                except FileNotFoundError:
                    pass

    def read(self, identity):
        with self.lock:
            data = PhotoStore._read(self._path(identity, '.jpg'), 5*1024*1024)
            meta = json.loads(PhotoStore._read(self._path(identity, '.json'), 16384).decode())
            if (meta.get('id') != identity or meta.get('source') != 'onboard_rgb' or
                    meta.get('url') != '/api/evidence/'+identity+'.jpg' or
                    meta.get('image_sha256') != hashlib.sha256(data).hexdigest()):
                raise ValueError('审查图片校验失败。')
            return data, meta
