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

    def test_typed_aliases_are_recomputed_from_the_bound_raw_decision(self):
        task = "Report each device above 70 degrees."
        truth = {"task": task, "family": "temperature", "devices": [{"identity": "A", "temperature": 86}]}
        raw_record = {"object_id": "A", "evidence_image_id": "onboard_0001", "facts": {"温度": "86℃"}}
        record = {**raw_record, "facts": {"temperature": 86}}
        raw = {"action": {"type": "submit"}, "report": {"outcome": "accepted", "records": [raw_record]}}
        report = {"outcome": "accepted", "records": [record]}
        decision = {**raw, "report": report}
        step = {"step": 1, "image_id": "onboard_0001", "decision": decision, "raw_model_decision": raw}
        frames = {"onboard_0001": "hash"}
        call = {"input": {"authorized_task": task, "current_image_id": "onboard_0001",
                          "report_fact_types": {"temperature": "number"}},
                "image_sha256": ["hash"], "output": raw}
        self.assertTrue(score_coverage(truth, report, frames, [call], [step])["complete"])

        changed_record = {**raw_record, "facts": {"温度": "84℃"}}
        changed = {**raw, "report": {"outcome": "accepted", "records": [changed_record]}}
        self.assertFalse(score_coverage(truth, report, frames, [call], [{**step, "raw_model_decision": changed}])["complete"])
        self.assertFalse(score_coverage(truth, report, frames, [{**call, "output": changed}], [step])["complete"])

    def test_legacy_fragment_recheck_preserves_two_separately_proven_fields(self):
        task = "Report the crack and temperature of cabinet A."
        truth = {"task": task, "family": "locate_condition",
                 "score_spec": {"target": "A", "required": ["cracked", "temperature"]},
                 "devices": [{"identity": "A", "cracked": False, "temperature": 86}]}
        frames = {"onboard_0001": "hash-a", "onboard_0002": "hash-b"}
        records = [{"object_id": "A", "facts": {"cracked": False}, "evidence_image_id": "onboard_0001"},
                   {"object_id": "A", "facts": {"temperature": 86}, "evidence_image_id": "onboard_0002"}]
        report = {"outcome": "accepted", "records": records}
        decision = {"action": {"type": "submit"}, "report": report}
        step = {"step": 2, "image_id": "onboard_0002", "decision": decision}
        call = {"input": {"authorized_task": task, "current_image_id": "onboard_0002",
                          "recent_images": [{"image_id": "onboard_0001"}]},
                "image_sha256": ["hash-a", "hash-b"], "output": decision}
        original = score_coverage(truth, report, frames, [call], [step])
        self.assertEqual(original["missing_fields_by_object"], {})
        self.assertFalse(original["complete"])
        separately_proven = score_coverage(truth, report, frames, [call], [step], recheck_legacy_fragments=True)
        self.assertTrue(separately_proven["complete"])
        self.assertTrue(separately_proven["report_matches_bound_submission"])
        unbound = {**report, "records": [records[0], {**records[1], "evidence_image_id": "unknown"}]}
        self.assertFalse(score_coverage(truth, unbound, frames, [call], [step],
                                        recheck_legacy_fragments=True)["report_matches_bound_submission"])


if __name__ == "__main__":
    unittest.main()
