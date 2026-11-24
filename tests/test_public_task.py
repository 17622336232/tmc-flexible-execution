import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.live_calibration import calibrate_public_visual
from experiments.flexible_execution.pilot_report import _method_summary
from experiments.flexible_execution.public_audit import audit_camera_permissions
from experiments.flexible_execution.public_schedule import VALIDATION_ROLE, prepare_final
from experiments.flexible_execution.public_task import VALIDATION, parse_intent, verify
from experiments.flexible_execution.timeline import Timeline
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

    def test_interrupted_public_turn_still_requires_a_prior_grant(self):
        grant = "采集规则：允许在判读结果返回前采集以下材料：side_photo。"
        with TemporaryDirectory() as temp:
            for allowed in (True, False):
                episode = Path(temp) / str(allowed)
                write_json(episode / "source.json", {"instruction": "Check both views.\n" + grant})
                trace = Timeline(episode / "timeline.jsonl")
                trace.event("query_submitted", "front", binding={"id": "front"})
                if allowed:
                    trace.event("approved_preparation", "side", permission=grant)
                trace.event("action_started", "RotateRight")
                if not allowed:
                    trace.event("approved_preparation", "side", permission=grant)
                self.assertEqual(audit_camera_permissions(episode, "right"),
                                 {"audited": 1, "violations": int(not allowed)})

    def test_structured_public_model_failures_still_enter_the_summary(self):
        row = {"state": "FINISHED_TASK_FAILURE", "method": "conditional-a", "success": False,
               "restricted_seconds": 480,
               "failure_reason": {"type": "TransportError", "code": "unsupported_output", "status": 200}}
        method = _method_summary([row])[row["method"]]
        self.assertEqual(method["recorded"], 1)
        self.assertEqual(method["successes"], 0)
        self.assertEqual(method["failure_types"], {"unsupported_output": 1})

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

    def test_public_calibration_weights_each_validation_house_once(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            cases = [{"id": "a", "group_id": "first-house", "order": ["one", "two"]},
                     {"id": "b", "group_id": "second-house", "order": ["one"]}]
            for case, method, front in (("a", "one", 10.), ("a", "two", 14.), ("b", "one", 20.)):
                arm = directory / case / method
                episode = arm / "episode"
                episode.mkdir(parents=True)
                write_json(arm / "job.json", {"id": case + "/" + method, "state": "FINISHED_VALID",
                                              "artifact_dir": str(episode)})
                clock = 0.
                rows = []
                for kind, step, duration in (("model_request", "front", front), ("action", "RotateRight", 1.),
                                             ("observation", "front", 0.5),
                                             ("model_request", "report", 4. if case == "a" else 8.)):
                    rows.append({"event": kind + "_started", "step": step, "monotonic_seconds": clock})
                    clock += duration
                    rows.append({"event": kind + "_finished", "step": step, "monotonic_seconds": clock, "ok": True})
                (episode / "timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            fitted = calibrate_public_visual(directory, {"role": VALIDATION_ROLE, "cases": cases})
            self.assertEqual(fitted["inference"], {"initial_prediction": 16., "initial_ceiling": 90.,
                                                   "remaining_judgments": 25.})
            self.assertEqual(fitted["motion"], {"turn": 1., "photo": 0.5})
            self.assertEqual(fitted["inherited_estimates"], [])
            self.assertEqual(len(fitted["samples_by_house"]), 2)
