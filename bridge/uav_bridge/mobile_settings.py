"""Small project-local settings/history store, containing no account credentials."""
import copy
import json
import os
from pathlib import Path
import re
import tempfile
import threading


class MobileState:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else None

    def read(self, name, fallback):
        if self.directory is None:
            return copy.deepcopy(fallback)
        try:
            path=self.directory/name
            if self.directory.is_symlink() or path.is_symlink() or path.stat().st_size>262144:
                return copy.deepcopy(fallback)
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            return copy.deepcopy(fallback)

    def write(self, name, data):
        if self.directory is None:
            return
        if self.directory.is_symlink():
            raise ValueError('项目设置目录不可为符号链接')
        self.directory.mkdir(mode=0o700,parents=True,exist_ok=True)
        raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
        if len(raw)>262144:
            raise ValueError('项目设置或记录超过大小限制')
        fd,temporary=tempfile.mkstemp(prefix='.mobile-',dir=str(self.directory))
        try:
            with os.fdopen(fd,'wb') as handle:
                os.fchmod(handle.fileno(),0o600);handle.write(raw)
            os.replace(temporary,str(self.directory/name))
        finally:
            if os.path.exists(temporary):os.unlink(temporary)


class LatestStateWriter:
    """Best-effort history persistence with one worker and one pending snapshot.

    The caller submits an already bounded public record. Disk I/O never runs
    under the caller's mission/control lock and older writes cannot overtake
    newer ones. Configuration changes still use synchronous MobileState.write.
    """
    def __init__(self, state, name):
        self.state, self.name = state, name
        self._lock = threading.Lock()
        self._pending = self._worker = None
        self._closed = False

    def submit(self, value):
        if self.state.directory is None:
            return
        snapshot = copy.deepcopy(value)
        with self._lock:
            if self._closed:
                return
            self._pending = snapshot
            if self._worker is None:
                self._worker = threading.Thread(target=self._drain, name='mission-history-cache', daemon=True)
                try:
                    self._worker.start()
                except RuntimeError:
                    self._worker = None

    def _drain(self):
        while True:
            with self._lock:
                if self._pending is None:
                    self._worker = None
                    return
                value, self._pending = self._pending, None
            try:
                self.state.write(self.name, value)
            except Exception:
                pass  # History failure cannot alter flight state or block cancellation.

    def flush(self, timeout=.2):
        with self._lock:
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)
        with self._lock:
            return self._worker is None and self._pending is None

    def close(self):
        with self._lock:
            self._closed = True
        return self.flush()


def mission_summary(mission):
    """Keep the final explanation even after another goto/land replaces status."""
    if not isinstance(mission,dict):return None
    fields=('id','state','summary','started_at','finished_at','error')
    item={key:copy.deepcopy(mission[key]) for key in fields if key in mission}
    for key in ('summary','error'):
        if isinstance(item.get(key),str):item[key]=item[key][:2400]
    agent=mission.get('photo_agent')
    if isinstance(agent,dict):
        item['photo_agent']={key:copy.deepcopy(agent[key]) for key in ('stage','model','revision') if key in agent}
        decision=agent.get('last_decision')
        if isinstance(decision,dict):
            item['photo_agent']['last_decision']={key:copy.deepcopy(decision[key]) for key in
                ('action','reason','goal','accepted','local_rejection') if key in decision}
    if isinstance(mission.get('photo_search'),dict):
        item['photo_search']={'state':mission['photo_search'].get('state')}
    return item


def model_name(value):
    return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}',value) is not None
