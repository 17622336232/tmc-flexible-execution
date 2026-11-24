import unittest

from experiments.flexible_execution.clean_coverage import score_coverage


class CoverageTests(unittest.TestCase):
    task = "Inspect both cabinets for cracks."
    truth = {
        "task": task, "family": "cracks",
        "devices": [{"identity": "A", "cracked": False}, {"identity": "B", "cracked": False}],
    }
    report = {"outcome": "accepted", "records": []}

    def test_empty_normal_report_needs_observations(self):
        result = score_coverage(self.truth, self.report, {}, [], [])
        self.assertTrue(result["report_correct"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_fields_by_object"], {"A": ["cracked"], "B": ["cracked"]})

    def test_plain_normal_report_supported_by_two_supplied_frames(self):
        frames = {"onboard_0001": "hash-a", "onboard_0002": "hash-b"}
        records = [
            {"object_id": label, "evidence_image_id": name, "facts": {"cracked": False}}
            for label, name in zip(("A", "B"), frames)
        ]
        decision = {"action": {"type": "submit"}, "records": records, "report": self.report}
        step = {"step": 2, "image_id": "onboard_0002", "decision": decision}
        call_input = {"authorized_task": self.task, "current_image_id": "onboard_0002",
                      "recent_images": [{"image_id": "onboard_0001"}]}
        call = {"input": call_input, "image_sha256": ["hash-a", "hash-b"], "output": decision}
        checked = score_coverage(self.truth, self.report, frames, [call], [step])
        self.assertTrue(checked["complete"])

        without_first = {**call, "input": {**call_input, "recent_images": []}, "image_sha256": ["hash-b"]}
        checked = score_coverage(self.truth, self.report, frames, [without_first], [step])
        self.assertEqual(checked["missing_fields_by_object"], {"A": ["cracked"]})
        self.assertEqual(checked["records_without_supplied_frame"][0]["object"], "A")

        forged = {**call, "image_sha256": ["incorrect", "hash-b"]}
        checked = score_coverage(self.truth, self.report, frames, [forged], [step])
        self.assertFalse(checked["complete"])
        self.assertEqual(checked["rejected_request_bindings"], ["onboard_0002"])

        late = {**step, "late_not_executed": True}
        self.assertFalse(score_coverage(self.truth, self.report, frames, [call], [late])["complete"])


if __name__ == "__main__":
    unittest.main()
