"""Monotonic request, observation and action intervals for clean simulations."""

import json
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from time import monotonic


class Timeline:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self.lock = Lock()

    def event(self, kind, step, **fields):
        with self.lock, self.path.open("a", encoding="utf8") as log:
            at = monotonic()
            row = {"event": kind, "step": step, "monotonic_seconds": at, **fields}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        return at

    @contextmanager
    def measure(self, kind, step, **fields):
        self.event(kind + "_started", step, **fields)
        try:
            yield
        except BaseException as error:
            self.event(kind + "_finished", step, ok=False, error_type=type(error).__name__, **fields)
            raise
        else:
            self.event(kind + "_finished", step, ok=True, **fields)


def read_trace_rows(path, *, allow_truncated=False):
    rows = []
    with Path(path).open("rb") as log:
        for line in log:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if allow_truncated and not line.endswith(b"\n"):
                    return rows, True
                raise
    return rows, False


def summarize(path, *, allow_truncated=False):
    starts = {}
    intervals = {}
    report = None
    rows, truncated = read_trace_rows(path, allow_truncated=allow_truncated)
    for row in rows:
        event = row["event"]
        at = row["monotonic_seconds"]
        step = row["step"]
        if event == "report_written":
            report = row["outcome"]
            continue
        if not event.endswith(("_started", "_finished")):
            continue
        kind, phase = event.rsplit("_", 1)
        if kind not in {"action", "observation", "model_request", "search"} or (
                event == "search_finished" and "ok" not in row):
            continue
        key = kind, step
        if phase == "started":
            if key in starts:
                raise ValueError("Interval started twice: " + repr(key))
            starts[key] = at
            continue
        if key not in starts or at < starts[key]:
            raise ValueError("Timing interval lacks a matching start: " + repr(key))
        start = starts.pop(key)
        intervals.setdefault(kind, []).append({
            "step": step, "start": start, "finish": at, "seconds": at - start, "ok": row["ok"],
        })
    requests = intervals.get("model_request", [])
    actions = intervals.get("action", [])
    return {
        "report_outcome": report,
        "intervals": intervals,
        "unfinished_intervals": [{"kind": kind, "step": step} for kind, step in starts],
        "truncated_tail": truncated,
        "model_request_seconds": sum(row["seconds"] for row in requests),
        "model_action_overlap_seconds": sum(
            max(0, min(request["finish"], action["finish"]) - max(request["start"], action["start"]))
            for request in requests for action in actions
        ),
    }
