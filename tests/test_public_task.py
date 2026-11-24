import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.public_schedule import VALIDATION_ROLE, prepare_final
from experiments.flexible_execution.public_task import VALIDATION, parse_intent, verify
from experiments.task_attack.runtime import write_json


class PublicInspectionScoreTest(unittest.TestCase):
    def setUp(self):
        self.front = Frame("front", "room", 1, 1., b"front camera", "procthor_rgb")
        self.side = Frame("side", "room", 2, 2., b"side camera", "procthor_rgb")

    def test_positive_needs_new_photo_and_original_heading(self):
        report = {"front": "positive", "front_sha256": self.front.digest,
                  "side": "negative", "side_sha256": self.side.digest}
        args = (VALIDATION, VALIDATION.expected, report, self.front, self.side, "positive", "negative")
        self.assertTrue(verify(*args, True)["complete"])
        self.assertFalse(verify(*args, False)["complete"])
        self.assertFalse(verify(VALIDATION, VALIDATION.expected, report, self.front,
                                None, "positive", None, True)["complete"])

    def test_negative_branch_does_not_deliver_speculative_photo(self):
        report = {"front": "negative", "front_sha256": self.front.digest, "side": None, "side_sha256": None}
        args = (VALIDATION, VALIDATION.expected, report, self.front, self.side, "negative", "positive", True)
        self.assertTrue(verify(*args)["complete"])
        report.update(side="positive", side_sha256=self.side.digest)
        self.assertFalse(verify(*args)["complete"])

    def test_model_needs_to_parse_the_actual_conditional_report(self):
        reply = {"primary": "Chair", "direction": "right", "secondary": "Bed", "side_when": ["positive"],
                 "restore_heading": True, "unresolved_allowed": True}
        self.assertEqual(parse_intent(reply), VALIDATION.expected)
        reply["side_when"].append("uncertain")
        with self.assertRaises(ValueError):
            parse_intent(reply)

    def test_unfinished_validation_leaves_the_official_test_archive_sealed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            validation = root / "validation"
            write_json(validation / "pilot_manifest.json", {"role": VALIDATION_ROLE, "cases": [{
                "id": f"house-{index}", "order": ["conditional-a"]} for index in range(4)]})
            with patch("experiments.flexible_execution.public_schedule.prepare") as blind_assignment:
                with self.assertRaisesRegex(ValueError, "public_validation has 0/4 recorded cases"):
                    prepare_final(root / "sealed-test.jsonl.gz", root / "blind-plan", validation, root / "selection",
                                  root / "final", {"data": {"development_selection_instances": 48}})
            blind_assignment.assert_not_called()
            self.assertFalse((root / "final").exists())
