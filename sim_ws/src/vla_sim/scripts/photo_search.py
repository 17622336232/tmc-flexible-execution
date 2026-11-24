#!/usr/bin/env python3
"""Finite, owner-bound in-place yaw search; contains no ROS dependencies.

A start is only pending.  The controller must obtain current map evidence before
calling activate(), and must hold the copied position throughout the lease.
"""
import math
import re
import time


class PhotoSearchError(ValueError):
    pass


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


class PhotoSearch:
    TOKEN = re.compile(r'^[A-Za-z0-9_-]{16,96}$')

    def __init__(self, monotonic=None):
        self._clock = monotonic or time.monotonic
        self.id = self.nonce = ''
        self.active = self.pending = False
        self.position = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.rate_rad_s = self.max_sweep_rad = self.timeout_s = 0.0
        self.angle_swept_rad = 0.0
        self.error = self.reason = ''
        self.expires = self._ros_deadline = self._wall_deadline = 0.0
        self._last_ros = self._last_wall = None
        self._last_step = None
        self._last_stamp = 0.0
        # IDs include cancellation tombstones. Never evict a cancelled ID:
        # delayed worker starts must not resurrect it later in this process.
        self._seen_ids = set()
        self._seen_nonces = set()

    def _finish(self, reason, error=''):
        self.active = self.pending = False
        self.reason, self.error = reason, error
        self._last_step = None

    def cancel(self, reason, now):
        """Unconditionally stop; external cancellation never grants new motion."""
        self._finish(reason or 'cancelled', reason or '')
        if _finite(now):
            self._last_ros = now if self._last_ros is None else max(now, self._last_ros)
        else:
            self.reason = self.error = 'photo_search_invalid_clock'

    def _check_time(self, now):
        wall = self._clock()
        if not _finite(now) or not _finite(wall):
            self._finish('photo_search_invalid_clock', 'photo_search_invalid_clock')
            return False
        if ((self._last_ros is not None and now < self._last_ros)
                or (self._last_wall is not None and wall < self._last_wall)):
            self._finish('photo_search_clock_rollback', 'photo_search_clock_rollback')
            return False
        self._last_ros, self._last_wall = now, wall
        if self.active or self.pending:
            if now >= self.expires:
                self._finish('photo_search_lease_expired', 'photo_search_lease_expired')
                return False
            if now >= self._ros_deadline or wall >= self._wall_deadline:
                self._finish('photo_search_timeout', 'photo_search_timeout')
                return False
        return True

    def _parameters(self, message):
        values = tuple(message.get(k) for k in ('rate_rad_s', 'max_sweep_rad', 'timeout_s'))
        if (not all(_finite(v) for v in values) or not .1 <= values[0] <= .35
                or not 0 < values[1] <= 2*math.pi or not 1 <= values[2] <= 35):
            raise PhotoSearchError('invalid_photo_search_parameters')
        return values

    def apply(self, message, now, position, yaw):
        if not isinstance(message, dict) or type(message.get('version')) is not int or message['version'] != 1:
            raise PhotoSearchError('invalid_photo_search_version')
        op, owner, nonce = message.get('op'), message.get('id'), message.get('nonce')
        if (op not in ('start', 'refresh', 'stop') or not isinstance(owner, str)
                or not self.TOKEN.fullmatch(owner) or not isinstance(nonce, str)
                or not self.TOKEN.fullmatch(nonce) or nonce in self._seen_nonces):
            raise PhotoSearchError('invalid_or_replayed_photo_search')
        stamp, ttl = message.get('stamp'), message.get('ttl')
        if (not _finite(now) or not _finite(stamp) or stamp <= 0 or not 0 <= now-stamp <= .6
                or not _finite(ttl) or not .2 <= ttl <= 3 or stamp+ttl <= now):
            raise PhotoSearchError('stale_photo_search_message')
        # An early STOP is a cancellation tombstone, even if START is still in
        # a worker queue. A foreign STOP must never stop/modify the live owner.
        if op == 'stop' and owner != self.id and (self.active or self.pending):
            self._seen_ids.add(owner)
            self._seen_nonces.add(nonce)
            return 'ignored_stop'
        if op == 'refresh' and owner != self.id:
            raise PhotoSearchError('photo_search_id_mismatch')
        if op != 'stop' and stamp < self._last_stamp:
            raise PhotoSearchError('stale_photo_search_message')
        parameters = self._parameters(message) if op != 'stop' else None
        if op == 'start':
            if owner in self._seen_ids:
                raise PhotoSearchError('reused_photo_search_id')
            if (not isinstance(position, (tuple, list)) or len(position) != 3
                    or not all(_finite(v) for v in position) or not _finite(yaw)):
                raise PhotoSearchError('invalid_photo_search_pose')
        if op == 'refresh' and parameters != (self.rate_rad_s, self.max_sweep_rad, self.timeout_s):
            raise PhotoSearchError('photo_search_parameters_changed')
        time_ok = self._check_time(now)
        if not time_ok and self.error in ('photo_search_clock_rollback', 'photo_search_invalid_clock'):
            raise PhotoSearchError(self.error)
        if op == 'start':
            if self.active or self.pending:
                raise PhotoSearchError('photo_search_already_owned')
            self.id = owner
            self.position, self.yaw = list(position), float(yaw)
            self.rate_rad_s, self.max_sweep_rad, self.timeout_s = parameters
            self.angle_swept_rad = 0.0
            self.error, self.reason = '', 'awaiting_map_evidence'
            self.pending, self.active = True, False
            self._ros_deadline = stamp+self.timeout_s
            self._wall_deadline = self._last_wall+self.timeout_s
            self._last_step = None
            self._seen_ids.add(owner)
        elif op == 'refresh':
            if not time_ok or not (self.active or self.pending):
                raise PhotoSearchError('photo_search_refresh_after_completion')
        else:
            self._seen_ids.add(owner)
            self.id = owner
            self._finish('stopped')
        self.nonce, self._last_stamp = nonce, max(stamp, self._last_stamp)
        self._seen_nonces.add(nonce)
        if op != 'stop':
            self.expires = stamp+ttl
        return op

    def activate(self, now):
        """Activate once after controller map checks; repeated calls do not reset."""
        if not self._check_time(now):
            return False
        if self.active:
            return True
        if not self.pending:
            return False
        self.pending, self.active = False, True
        self.reason = 'searching'
        self._last_step = now
        return True

    def step(self, now, dt):
        """Return positive yaw change, bounded by rate, elapsed ROS time and sweep."""
        if not _finite(dt) or dt < 0:
            self.cancel('invalid_photo_search_step', now)
            raise PhotoSearchError('invalid_photo_search_step')
        if not self._check_time(now) or not self.active:
            return 0.0
        elapsed = max(0.0, now-self._last_step)
        self._last_step = now
        # Repeated calls at one ROS instant cannot synthesize extra sweep.
        duration = min(dt, elapsed)
        remaining = max(0.0, self.max_sweep_rad-self.angle_swept_rad)
        delta = min(self.rate_rad_s*duration, remaining)
        self.angle_swept_rad += delta
        self.yaw += delta
        if delta >= remaining:
            self.angle_swept_rad = self.max_sweep_rad
            self._finish('sweep_complete')
        return delta

    def status(self):
        return dict(supported=True, id=self.id, nonce=self.nonce, active=self.active,
                    pending=self.pending, angle_swept_rad=self.angle_swept_rad,
                    error=self.error, holding_position=self.active or self.pending,
                    position=list(self.position), rate_rad_s=self.rate_rad_s,
                    reason=self.reason, yaw=self.yaw, expires_ros_stamp=self.expires,
                    max_sweep_rad=self.max_sweep_rad, timeout_s=self.timeout_s)
