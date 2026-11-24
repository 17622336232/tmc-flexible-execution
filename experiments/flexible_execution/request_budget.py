"""Persistent reservations for actual transport attempts in one logical job."""

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from experiments.task_attack.model import ModelAttemptLimitExceeded


class RequestBudget:
    def __init__(self, path, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("The request cap must be a positive integer")
        self.path = Path(path)
        self.limit = limit

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf8") as ledger:
            fcntl.flock(ledger, fcntl.LOCK_EX)
            ledger.seek(0)
            rows = [json.loads(line) for line in ledger]
            if len(rows) > self.limit or any(
                row.get("limit") != self.limit or row.get("reservation") != sequence
                for sequence, row in enumerate(rows, 1)
            ):
                raise ValueError("Request ledger does not match the locked job budget")
            yield ledger, len(rows)

    def check(self):
        with self._locked() as (_, used):
            if used >= self.limit:
                raise ModelAttemptLimitExceeded("Logical job reached its model request cap")

    def reserve(self, *, model, effort, image_sha256):
        with self._locked() as (ledger, used):
            if used >= self.limit:
                raise ModelAttemptLimitExceeded("Logical job reached its model request cap")
            reservation = used + 1
            row = {"reservation": reservation, "limit": self.limit, "wall": time.time(),
                   "pid": os.getpid(), "model": model, "reasoning_effort": effort,
                   "image_sha256": image_sha256}
            ledger.write(json.dumps(row, allow_nan=False) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            return reservation

    def summary(self):
        with self._locked() as (_, used):
            return {"ledger": str(self.path), "limit": self.limit, "reserved": used, "remaining": self.limit - used}
