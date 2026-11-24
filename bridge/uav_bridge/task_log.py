"""Bounded, authenticated, human-readable task events; no transport or credentials.

Disk diagnostics are best effort and never gate flight control. Only explicit
public task fields are copied, not requests, model transport bodies or ROS state.
"""
import copy
import json
import threading
import time


_STAGES = {
    'settling': '正在稳定悬停', 'observing': '正在取得新的机载画面',
    'awaiting_model': '已发送本轮图像，等待模型判断',
    'validating': '正在验证模型提议', 'moving': '正在飞往模型选择的机位',
    'waiting_pose': '等待后重新观察', 'saving': '正在保存通过审查的原图',
    'landing': '所有要求的照片已保存，正在原地降落',
    'completed': '任务已完成', 'failed': '任务未完成', 'cancelled': '任务已中止',
    'searching': '正在原地慢转搜索人物', 'awaiting_subject': '已停稳，等待手机确认人物',
    'refreshing_subject': '候选画面已失效，正在悬停更新画面，请重新确认人物',
    'expired': '人物确认已过期', 'selected': '人物已确认',
}


class TaskLog:
    def __init__(self, state):
        self.state, self.lock = state, threading.RLock()
        rows = state.read('task-events.json', [])
        self.events = [row for row in rows if isinstance(row, dict)][-180:] if isinstance(rows, list) else []
        self.keys = {}
        self.sequence = max((row.get('sequence', 0) for row in self.events), default=0)
        self._queue_lock = threading.Lock()
        self._pending = None
        self._writer = None
        self._closed = False

    def _submit_snapshot(self):
        # Called with self.lock held, so an older event cannot enqueue after a
        # newer one. Keep only the latest complete, already bounded log snapshot.
        if getattr(self.state, 'directory', False) is None:
            return  # Explicitly disabled MobileState still keeps in-memory logs.
        snapshot = copy.deepcopy(self.events)
        with self._queue_lock:
            if self._closed:
                return
            self._pending = snapshot
            if self._writer is None:
                self._writer = threading.Thread(target=self._drain, name='task-event-cache', daemon=True)
                try:
                    self._writer.start()
                except RuntimeError:
                    # Resource pressure must not fail the mission. Keep the
                    # latest bounded snapshot for a subsequent event to retry.
                    self._writer = None

    def _drain(self):
        while True:
            with self._queue_lock:
                if self._pending is None:
                    self._writer = None
                    return
                snapshot, self._pending = self._pending, None
            try:
                # No task-log/Bridge/queue lock is held during filesystem I/O.
                self.state.write('task-events.json', snapshot)
            except Exception:
                pass  # Diagnostic failures cannot block subsequent snapshots.

    def flush(self, timeout=.2):
        """Best effort bounded flush for normal shutdown; never wait for disk forever."""
        with self._queue_lock:
            writer = self._writer
        if writer is not None and writer is not threading.current_thread():
            writer.join(timeout=timeout)
        with self._queue_lock:
            return self._writer is None and self._pending is None

    def close(self):
        with self._queue_lock:
            self._closed = True
        return self.flush()

    def event(self, context, phase, message, **details):
        with self.lock:
            self.sequence += 1
            row = dict(sequence=self.sequence, stamp=time.time(), context=str(context)[:128],
                       phase=str(phase)[:80], message=str(message)[:2400])
            # Callers supply only a public, small typed record. Never copy a
            # generic exception/request/configuration object into diagnostics.
            allowed = ('model', 'task_id', 'shots', 'candidate_id', 'error_code', 'evidence',
                       'revision', 'shot_id', 'shot_index', 'shot_count', 'action', 'goal',
                       'checks', 'delivery', 'photo_id', 'photo_count', 'task_state')
            for key in allowed:
                if key in details:
                    row[key] = copy.deepcopy(details[key])
            if isinstance(row.get('checks'), list):
                row['checks'] = [dict(requirement_id=str(c.get('requirement_id',''))[:20],
                                     state=str(c.get('state',''))[:30],
                                     evidence=str(c.get('evidence',''))[:400])
                                 for c in row['checks'][:12] if isinstance(c, dict)]
            self.events.append(row)
            self.events = self.events[-180:]
            while len(json.dumps(self.events, ensure_ascii=False).encode()) > 220000:
                self.events.pop(0)
            self._submit_snapshot()

    def sync(self, mission):
        if not isinstance(mission, dict) or not mission.get('id'):
            return
        with self.lock:
            identity, state = mission['id'], mission.get('state')
            agent, search = mission.get('photo_agent') or {}, mission.get('photo_search') or {}
            decision, progress = agent.get('last_decision') or {}, agent.get('capture_progress') or {}
            stage = (agent.get('stage') or search.get('state') or state) if state == 'running' else state
            delivery = decision.get('delivery')
            key = (state, stage, agent.get('revision'), delivery, progress.get('shot_index'),
                   mission.get('error'), len(mission.get('photos') or []))
            if self.keys.get(identity) == key:
                return
            self.keys[identity] = key
            if len(self.keys) > 64:
                self.keys.pop(next(iter(self.keys)))
            message = _STAGES.get(stage, mission.get('summary') or '任务状态已更新')
            if state in ('failed', 'cancelled'):
                message = mission.get('error') or _STAGES[state]
            elif decision and stage != 'awaiting_model':
                message += '；模型：'+str(decision.get('reason', ''))
                if decision.get('local_rejection'):
                    message += '；本地核验：'+str(decision['local_rejection'])
                if decision.get('preplanned'):
                    message += '；复用上一张验收时模型给出的下一机位，仍执行新鲜地图验证'
            if stage == 'awaiting_model':
                count = agent.get('scene_memory', {}).get('stored_frames', 0)
                if count:
                    message += '；从本任务关键帧中选取不同视角作为参考，当前图像单独验收'
            details = dict(task_state=state, model=agent.get('model'), revision=agent.get('revision'),
                           shot_index=progress.get('shot_index'), shot_count=progress.get('shot_count'),
                           photo_count=len(mission.get('photos') or []))
            evidence = agent.get('reviewed_evidence') or search.get('evidence')
            if isinstance(evidence, dict):
                details['evidence'] = {k: copy.deepcopy(evidence[k]) for k in
                    ('id', 'url', 'source', 'stamp', 'ros_stamp', 'width', 'height', 'image_sha256', 'frame_id', 'revision', 'pose')
                    if k in evidence}
            if decision and stage != 'awaiting_model':
                for k in ('action', 'goal', 'checks', 'delivery', 'shot_id', 'photo_id'):
                    if k in decision:
                        details[k] = decision[k]
            self.event(identity, stage, message, **details)

    def read(self):
        with self.lock:
            return dict(events=copy.deepcopy(self.events), max_events=180,
                        scope='最近的任务解析、选人、模型提议、本地执行与停止记录；不包含账户凭据。')
