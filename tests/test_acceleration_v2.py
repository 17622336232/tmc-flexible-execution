import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.flexible_execution.acceleration_audit import _union
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
            model.request.return_value = {
                "program": {"scope": {"mode": "all", "description": "visible equipment"},
                            "fields": {"cracked": "flag"}, "filter": {"op": "keep"},
                            "extreme": None, "report": ["cracked"]}, "conditional": None}
            trace = Timeline(root / "timeline.jsonl")
            with patch.object(v2_execution, "_take_photo", return_value=(photo, {}, {})):
                cold = v2_execution._front_end(None, root, trace, model, 0., "Inspect visible equipment",
                                               registration, "registered-scene")
                cached = v2_execution._front_end(None, root, trace, model, 0., "Inspect visible equipment",
                                                 registration, "registered-scene")
            self.assertEqual(cold[:3], cached[:3])
            self.assertEqual(model.request.call_count, 1)

    def test_spans_count_overlap_once(self):
        self.assertEqual(_union([(0, 4), (2, 7), (3, 6), (10, 12)]), 9)
