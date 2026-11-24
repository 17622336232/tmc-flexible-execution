import unittest
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from PIL import Image

from experiments.flexible_execution.adapters import live_inspection
from experiments.flexible_execution.adapters.live_inspection import _verify
from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.executor import Snapshot
from experiments.flexible_execution.gpt_vision import JUDGE, LOCATE, Material, OUTCOMES, VisibleTarget
from experiments.flexible_execution.live_planning import Costs, permission_for, select
from experiments.flexible_execution.task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract
from experiments.flexible_execution.timeline import Timeline, summarize


GRANT = "采集规则：允许在判读结果返回前采集以下材料：identity。"
MATERIAL = Material("identity", ("positive",), "Which label is on this object?", GRANT)


def _task(rule):
    requirement = Requirement("device:identity", "device", "identity", Condition("check", ("positive",)))
    skill = Skill("view", "side", (requirement.id,), 1, rule.for_query("check"))
    return TaskContract("Inspect", "start", "handoff", 90, 12, (requirement,),
                        (QuerySpec("check", "device", OUTCOMES, 20, 35),), (skill,),
                        (("start", "side", 2.), ("side", "handoff", 2.),
                         ("start", "handoff", 0.), ("handoff", "side", 2.)))


class LiveInspectionTests(unittest.TestCase):
    def test_final_verifier_sees_scope_even_when_a_subject_has_no_claim(self):
        requests = []

        class Model:
            def request(self, _prompt, payload, photos):
                requests.append((payload, photos))
                return {"issues": ["The second visible device has no recorded finding"]}

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for image_id in ("onboard_0007", "probe_0001"):
                data = image_id.encode("ascii")
                (directory / (image_id + ".jpg")).write_bytes(data)
                (directory / (image_id + ".json")).write_text(json.dumps({"image_id": image_id,
                    "sha256": sha256(data).hexdigest()}))
            scope = {"initial_scene_image": "probe_0001", "visible_targets": [{"id": "visible-1"}, {"id": "visible-2"}]}
            report = {"outcome": "accepted", "records": [{"object_id": "EQ-02", "evidence_image_id": "onboard_0007"}]}
            guard = live_inspection.submission_guard("Inspect all visible devices", directory, Model(),
                                                     Timeline(directory / "time.jsonl"), scope)
            self.assertIn("second visible device", guard(report, report["records"], 7))
        payload, photos = requests[0]
        self.assertEqual(payload["initial_scope"]["visible_targets"], scope["visible_targets"])
        self.assertEqual([photo["data"] for photo in photos], [b"onboard_0007", b"probe_0001"])

    def test_complete_report_checks_all_five_photos_within_transport_limit(self):
        requested = []

        class Model:
            def request(self, prompt, payload, photos):
                requested.append((len(photos), len(payload.get("verified_observations", []))))
                if prompt == live_inspection.ACCEPT_EVIDENCE_GROUP:
                    return {"issues": [], "observations": payload["claims"]}
                return {"issues": []}

        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            records = []
            for number in range(1, 6):
                tag = f"probe_{number:04d}"
                data = ("camera " + tag).encode("utf8")
                (folder / (tag + ".jpg")).write_bytes(data)
                (folder / (tag + ".json")).write_text(json.dumps({"image_id": tag, "sha256": sha256(data).hexdigest()}))
                records.append({"object_id": f"EQ-{number:02d}", "evidence_image_id": tag, "facts": {"cracked": True}})
            guard = live_inspection.submission_guard("Inspect the row", folder, Model(), Timeline(folder / "time.jsonl"), {})
            self.assertIsNone(guard({"outcome": "accepted", "records": records}, [], 7))
        self.assertEqual(requested, [(4, 0), (1, 0), (0, 5)])

    def test_permission_needs_explicit_semantics_for_this_material(self):
        self.assertTrue(permission_for("If cracked, report ID.\n" + GRANT, MATERIAL).early)
        self.assertFalse(permission_for("Do not collect early, even when quoted: " + GRANT, MATERIAL).early)
        self.assertFalse(permission_for("最终只报告编号。", MATERIAL).early)
        late = "采集规则：只有判读结果为 positive 才允许采集以下材料：identity。"
        self.assertFalse(permission_for(GRANT + "\n" + late, MATERIAL).early)
        self.assertEqual(permission_for(GRANT + "\n" + late, MATERIAL).after, ("positive",))

    def test_all_comparisons_share_permission_and_check_uncertain(self):
        state = Snapshot(0., "start", {}, {}, {})
        for method in ("conditional-a", "conditional-prefetch", "conditional-planner", "conditional-async"):
            events = []
            log = lambda event, **data: events.append((event, data))
            self.assertIsNone(select(_task(permission_for("最终只报告编号。", MATERIAL)), state, method, log))
            choice = select(_task(permission_for(GRANT, MATERIAL)), state, method, log)
            if method == "conditional-async":
                self.assertIsNone(choice)
                continue
            self.assertEqual(choice.continuation.site, "handoff")
            checked = [row["branches"] for event, row in events if event == "candidate_checked" and row.get("common")]
            self.assertEqual({row["outcomes"]["check"] for row in checked[-1]}, set(OUTCOMES))

    def test_candidate_forecast_does_not_treat_projected_time_as_a_late_reply(self):
        task = _task(permission_for(GRANT, MATERIAL))
        task = replace(task, queries=(replace(task.queries[0], estimated_ready=5.),),
                       travel=(("start", "side", 5.), ("side", "handoff", 0.),
                               ("start", "handoff", 0.), ("handoff", "side", 5.)))
        state = Snapshot(0., "start", {}, {}, {})
        choice = select(task, state, "conditional-a", lambda *_args, **_fields: None)
        self.assertIsNotNone(choice)
        self.assertEqual(choice.expected_finish, 18.)

    def test_evidence_requires_same_readable_object_region_and_fresh_camera(self):
        def jpeg(color):
            buffer = BytesIO()
            Image.new("RGB", (640, 480), color).save(buffer, "JPEG")
            return buffer.getvalue()

        origin = Frame("initial", "scene", 1, 0., jpeg("white"), "simulated_onboard_rgb")
        captured = Frame("new", "device", 2, 1., jpeg("gray"), "simulated_onboard_rgb")
        target = VisibleTarget("device", "station with a label", (.1, .1, .5, .7), True)
        photo = (captured, {"observation_id": "new"}, {})
        requirement = Requirement("device:identity", "device", "identity")

        class Stub:
            draft = {"same_object": True, "readable": True, "field": "identity", "value": "EQ-02",
                     "object_bbox": [.1, .1, .5, .7], "material_bbox": [.2, .2, .3, .3],
                     "source_role": "body_identity"}

            def request(self, system, payload, images):
                assert images[0]["data"] == origin.payload and images[1]["data"] == captured.payload
                return self.draft

        reader = Stub()
        evidence, _ = _verify(reader, "Report ID", requirement, MATERIAL, origin, target, photo, 2., 9.)
        self.assertEqual(evidence.value, "EQ-02")
        self.assertTrue(evidence.usable(requirement, 8.))
        self.assertFalse(evidence.usable(requirement, 9.))
        for alteration in ({"same_object": False}, {"readable": False}, {"material_bbox": [.9, .9, .95, .95]}):
            with self.subTest(alteration=alteration):
                reader.draft = dict(Stub.draft, **alteration)
                rejected, _ = _verify(reader, "Report ID", requirement, MATERIAL, origin, target, photo, 2., 9.)
                self.assertIsNone(rejected)
        reader.draft = dict(Stub.draft, source_role="other")
        rejected, _ = _verify(reader, "Read the asset identity", requirement, MATERIAL, origin, target, photo, 2., 9.)
        self.assertIsNone(rejected)

    def test_conditional_photo_reaches_original_agent_and_complete_report_check(self):
        photographed = Event()
        initial = {"x": 0., "y": 0., "z": 1.2, "yaw": 0.}
        second = {**initial, "x": 2.}

        class Backend:
            scene = {"catalog": {}}
            pose = initial
            observations = 0

            def check_goal(self, pose):
                return True

            def move(self, pose):
                self.pose = pose
                return {"position": pose, "yaw": pose["yaw"]}

            def observe(self):
                self.observations += 1
                color = (self.observations * 20, 50, 100)
                buffer = BytesIO()
                Image.new("RGB", (640, 480), color).save(buffer, "JPEG")
                if self.pose == second:
                    photographed.set()
                return {"observation_id": f"photo-{self.observations}", "data": buffer.getvalue(),
                        "position": self.pose, "yaw": self.pose["yaw"], "ros_stamp": self.observations}, {}

        class Model:
            calls = 0
            wall_seconds = 0.
            external_error = None
            purposes = []

            def request(self, system, payload, images):
                self.calls += 1
                self.purposes.append(system)
                if system == live_inspection.LIVE_INTERPRET:
                    return {"status": "supported", "object_scope": "all local devices", "question": "Cracked?",
                            "materials": [{"field": "identity", "needed_when": ["positive"],
                                           "question": "Read ID", "permission_quote": GRANT}]}
                if system == LOCATE:
                    return {"targets": [{"description": "a marked device", "bbox": [.1, .1, .5, .7],
                                         "view_sufficient": True}]}
                if system == JUDGE:
                    if not photographed.wait(1):
                        raise AssertionError("No native motion/capture entered the pending window")
                    return {"outcome": "positive"}
                if system == live_inspection.READ_MATERIAL:
                    return {"same_object": True, "readable": True, "field": "identity", "value": "EQ-02",
                            "object_bbox": [.1, .1, .5, .7], "material_bbox": [.2, .2, .3, .3],
                            "source_role": "body_identity"}
                if system == live_inspection.ACCEPT_TASK:
                    return {"issues": []}
                selected = next(row["image_id"] for row in payload["recent_images"] if row["image_id"] == "probe_0004")
                rows = [{"object_id": "EQ-02", "facts": {"cracked": True}, "evidence_image_id": selected}]
                return {"action": {"type": "submit"}, "records": rows,
                        "report": {"outcome": "accepted", "summary": "EQ-02 is cracked", "records": rows}}

        costs = Costs(4., 7., .2, .5, .1, 5., "offline-diagnostic")
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            model = Model()
            with patch.object(live_inspection, "_candidates", return_value=iter([(second, {"method": "test"})])):
                report, metrics = live_inspection.run_live(Backend(), "Cracked devices need IDs.\n" + GRANT,
                    directory, lambda *_args, **_fields: None, 20., model=model,
                    trace=Timeline(directory / "events.jsonl"), method="conditional-a", costs=costs)
            self.assertTrue(metrics["conditional_query_bound"])
            self.assertEqual(metrics["conditional_materials"], 1)
            self.assertEqual(report["outcome"], "accepted")
            self.assertIn(live_inspection.ACCEPT_TASK, model.purposes)
            self.assertEqual(len(summarize(directory / "events.jsonl")["intervals"]["search"]), 1)
