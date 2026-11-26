import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.flexible_execution.acceleration_audit import _union
from experiments.flexible_execution.acceleration_results import _attempt_history, _wall_seconds
from experiments.flexible_execution.acceleration_revision import initialize_budget
from experiments.flexible_execution.adapters import v2_execution
from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.frontend_evidence import atomic_facts
from experiments.flexible_execution.timeline import Timeline
from experiments.task_attack.agent import _seed_records


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

    def test_frontend_seeds_only_released_matching_facts(self):
        self.assertEqual(atomic_facts("该设备的正面外壳是否存在裂纹？", "negative", {"cracked": "flag"}),
                         {"cracked": False})
        self.assertEqual(atomic_facts("该设备是否有裂纹或温度超过80？", "positive", {"cracked": "flag"}), {})
        self.assertEqual(atomic_facts("该设备的正面外壳是否存在裂纹？", "uncertain", {"cracked": "flag"}), {})
        record = {"object_id": "EQ-02", "evidence_image_id": "probe_0001", "facts": {"cracked": True}}
        source = {"object": "EQ-02", "photo": "probe_0001", "field": "cracked", "value": True,
                  "image_sha256": "actual", "available_monotonic": 12., "valid_until_monotonic": 20.}
        hashes = {"probe_0001": "actual"}
        self.assertEqual(_seed_records([record], [source], hashes, 11.), ([], []))
        self.assertEqual(_seed_records([record], [source], hashes, 12.)[0], [record])
        self.assertEqual(_seed_records([record], [source], hashes, 20.), ([], []))
        self.assertEqual(_seed_records([{**record, "object_id": "visible-1"}], [source], hashes, 12.), ([], []))
        self.assertEqual(_seed_records([record], [source], {"probe_0001": "other"}, 12.), ([], []))

    def test_migrated_budget_keeps_the_frozen_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, replacement = (Path(tmp) / name for name in ("original.jsonl", "replacement.jsonl"))
            saved = b'{"reservation":1}\n'
            original.write_bytes(saved)
            job = {"budget_class": "original_development", "budget_parent": {
                "original_ledger": str(original), "original_ledger_sha256": hashlib.sha256(saved).hexdigest()}}
            initialize_budget(job, replacement)
            self.assertEqual(replacement.read_bytes(), saved)
            replacement.write_bytes(saved + b'{"reservation":2}\n')
            initialize_budget(job, replacement)
            self.assertEqual(original.read_bytes(), saved)
            replacement.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "does not inherit"):
                initialize_budget(job, replacement)
