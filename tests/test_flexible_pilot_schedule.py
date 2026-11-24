import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.flexible_execution.pilot_schedule import PAIRED_MODES, prepare, status


class ScheduleTests(unittest.TestCase):
    defaults = {"data": {"split_seed": 20261124, "natural_latency_pilot_instances": 48}}

    def test_shared_schedule_is_reproducible_for_all_legacy_task_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare(root, self.defaults, 480)
            self.assertEqual(prepare(root, self.defaults, 480), manifest)
            with self.assertRaises(ValueError):
                prepare(root, self.defaults, 360)
            cases = manifest["cases"]
            self.assertEqual(Counter(case["domain"] for case in cases), {
                "equipment": 16, "warehouse": 16, "campus": 16,
            })
            self.assertEqual(len({case["seed"] for case in cases}), 48)
            self.assertTrue(all(set(case["order"]) == set(PAIRED_MODES) for case in cases))
            self.assertEqual(status(root)["recorded_pairs"], 0)

    def test_a_missing_or_mismatched_arm_cannot_count_as_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = prepare(root, self.defaults, 480)["cases"][0]
            for count, method in enumerate(case["order"], 1):
                folder = root / case["id"] / method
                folder.mkdir(parents=True)
                preflight = {**case, "mode": PAIRED_MODES[method], "model": "same-model", "reasoning_effort": "max"}
                (folder / "clean_preflight.json").write_text(json.dumps(preflight))
                self.assertEqual(status(root)["recorded_pairs"], 0)
                (folder / "result.json").write_text(json.dumps({"score": {"outcome": "correct"}}))
                (folder / "clean_coverage.json").write_text(json.dumps({"complete": True}))
                self.assertEqual(status(root)["recorded_arms"], count)
                self.assertEqual(status(root)["accepted_arms"], count)
                self.assertEqual(status(root)["recorded_pairs"], int(count == 2))
            self.assertEqual(status(root)["pairs_with_both_reports_and_coverage"], 1)
            (folder / "clean_preflight.json").write_text(json.dumps({**preflight, "seed": case["seed"] + 1}))
            with self.assertRaisesRegex(ValueError, "frozen case"):
                status(root)


if __name__ == "__main__":
    unittest.main()
