import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.flexible_execution.acceleration_audit import _union
from experiments.flexible_execution.acceleration_results import _attempt_history, _wall_seconds
from experiments.flexible_execution.adapters import v2_execution
from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.timeline import Timeline


class AccelerationTests(unittest.TestCase):
    def test_shared_frontend_hit_uses_zero_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registration = root / "reference.json"
            photo = Frame("frame", "scene", 1, 0.0, b"fixed camera bytes", "simulated_onboard_rgb")
            model = Mock()
            model.request.side_effect = [{
                "program": {"scope": {"mode": "all", "description": "visible equipment"},
                            "fields": {"cracked": "flag"}, "filter": {"op": "keep"},
                            "extreme": None, "report": ["cracked"]},
                "conditional": {"status": "supported", "object_scope": "visible equipment",
                                "question": "Is the object cracked?", "materials": []}},
                {"targets": [{"description": "visible housing", "bbox": [0.1, 0.2, 0.4, 0.6],
                              "view_sufficient": True}]}]
            trace = Timeline(root / "timeline.jsonl")
            with patch.object(v2_execution, "_take_photo", return_value=(photo, {}, {})):
                cold = v2_execution._front_end(None, root, trace, model, 0., "Inspect visible equipment",
                                               registration, "registered-scene")
                cached = v2_execution._front_end(None, root, trace, model, 0., "Inspect visible equipment",
                                                 registration, "registered-scene")
            self.assertEqual(cold[:3], cached[:3])
            self.assertEqual(len(cold[2]), 1)
            self.assertEqual(model.request.call_count, 2)

    def test_worker_wall_excludes_idle_time_between_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control"
            control.mkdir()
            first = {"id": "job", "stage": "acceleration-v2-development", "attempt": 1,
                     "started_at": "2026-11-25T07:26:43+00:00", "finished_at": "2026-11-25T07:29:56+00:00"}
            second = {**first, "attempt": 2, "started_at": "2026-11-26T04:04:39+00:00",
                      "finished_at": "2026-11-26T04:09:43+00:00"}
            final = {**second, "finished_at": "2026-11-26T04:20:00+00:00"}
            (control / "jobs.jsonl").write_text(
                "\n".join([json.dumps(first), '{"incomplete":', json.dumps(second), json.dumps(final)]) + "\n",
                encoding="utf8")
            history, unreadable = _attempt_history(root / "development")
            self.assertEqual(unreadable, 1)
            self.assertEqual(sum(_wall_seconds(row) for row in history["job"].values()), 497)

    def test_spans_count_overlap_once(self):
        self.assertEqual(_union([(0, 4), (2, 7), (3, 6), (10, 12)]), 9)
