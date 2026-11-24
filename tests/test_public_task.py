import unittest

from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.public_task import VALIDATION, parse_intent, verify


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
