import unittest

from experiments.flexible_execution.clean_coverage import score_coverage
from experiments.flexible_execution.requirement_program import check_submission, parse


class RequirementProgramTests(unittest.TestCase):
    def test_custom_or_needs_evidence_for_rejected_subjects(self):
        instruction = "Report cracked devices or devices above 80 degrees."
        fields = {"cracked": "flag", "temperature": "number"}
        task = {"scope": {"mode": "all", "description": "all devices"}, "fields": fields,
                "filter": {"op": "or", "children": [
                    {"op": "eq", "field": "cracked", "right": {"literal": True}},
                    {"op": "gt", "field": "temperature", "right": {"literal": 80}}]},
                "extreme": None, "report": ["cracked", "temperature"]}
        actual = [{"identity": "EQ-01", "cracked": True, "temperature": 65},
                  {"identity": "EQ-02", "cracked": False, "temperature": 78}]
        truth = {"task": instruction, "devices": actual, "oracle_requirement_program": task}
        first = {"object_id": "EQ-01", "evidence_image_id": "view", "facts": {"cracked": True, "temperature": 65}}
        second = {"object_id": "EQ-02", "evidence_image_id": "view", "facts": {"cracked": False}}
        report = {"outcome": "accepted", "records": [first]}
        decision = {"action": {"type": "submit"}, "records": [first, second], "report": report}
        calls = [{"input": {"authorized_task": instruction, "current_image_id": "view"},
                  "image_sha256": ["registered"], "output": decision}]
        actions = [{"image_id": "view", "step": 1, "decision": decision}]
        missing = score_coverage(truth, report, {"view": "registered"}, calls, actions)
        self.assertEqual(missing["missing_fields_by_object"], {"EQ-02": ["temperature"]})
        self.assertFalse(missing["complete"])
        second["facts"]["temperature"] = 78
        self.assertTrue(score_coverage(truth, report, {"view": "registered"}, calls, actions)["complete"])

    def test_compound_boolean_and_threshold_do_not_treat_unknown_as_false(self):
        draft = {"scope": {"mode": "all", "description": "the row"},
                 "fields": {"cracked": "flag", "temperature": "number"},
                 "filter": {"op": "and", "children": [
                     {"op": "eq", "field": "cracked", "right": {"literal": True}},
                     {"op": "gt", "field": "temperature", "right": {"literal": 70}}]},
                 "extreme": None, "report": ["temperature"]}
        program = parse(draft, "Find devices with cracks and temperature above 70.")
        observed = {"EQ-01": {"cracked": True, "temperature": 71.5}, "EQ-02": {"cracked": True}}
        self.assertEqual(program.evaluate(observed, scope_confirmed=True)["unresolved"], ["EQ-02"])
        observed["EQ-02"]["temperature"] = 70
        self.assertEqual(program.evaluate(observed, scope_confirmed=False)["unresolved"], ["scope"])
        self.assertEqual(program.evaluate(observed, scope_confirmed=True)["findings"],
                         [{"object_id": "EQ-01", "facts": {"temperature": 71.5}}])

    def test_reports_all_tied_minima(self):
        draft = {"scope": {"mode": "all", "description": "the visible racks"}, "fields": {"count": "count"},
                 "filter": {"op": "keep"}, "extreme": {"op": "min", "field": "count"}, "report": ["count"]}
        program = parse(draft, "List all racks with the minimum count.")
        rows = program.evaluate({"A": {"count": 4}, "B": {"count": 2}, "C": {"count": 2}}, scope_confirmed=True)
        self.assertTrue(rows["complete"])
        self.assertEqual([item["object_id"] for item in rows["findings"]], ["B", "C"])
        observations = [{"object_id": key, "facts": {"count": value}} for key, value in (("A", 4), ("B", 2), ("C", 2))]
        self.assertIsNotNone(check_submission(program, {"records": observations[1:2]}, observations))
        self.assertIsNone(check_submission(program, {"records": observations[1:]}, observations))

    def test_reference_count_must_match_user_text(self):
        draft = {"scope": {"mode": "target", "description": "WH-02", "identity": "WH-02"},
                 "fields": {"box_count": "count"},
                 "filter": {"op": "lt", "field": "box_count", "right": {
                     "by_identity": {"WH-02": {"value": 3, "quote": "WH-02需要3箱"}}}},
                 "extreme": None, "report": ["box_count"]}
        instruction = "检查 WH-02。订单要求 WH-02需要3箱。报告短缺。"
        program = parse(draft, instruction)
        self.assertTrue(program.evaluate({"WH-02": {"box_count": 2}}, scope_confirmed=True)["findings"])
        draft["filter"]["right"]["by_identity"]["WH-02"]["value"] = 2
        with self.assertRaises(ValueError):
            parse(draft, instruction)
