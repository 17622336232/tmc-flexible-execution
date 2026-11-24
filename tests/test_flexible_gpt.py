import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from threading import Event

from experiments.flexible_execution.adapters.gpt_probe import run_probe
from experiments.flexible_execution.adapters.viewpoints import measured_views
from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.gpt_vision import VisibleTarget, frozen_prompt, interpret, judge, locate_targets
from experiments.flexible_execution.timeline import Timeline, summarize
from experiments.task_attack.agent import run_agent


INSTRUCTION = "Inspect each device. If a crack is visible, report its serial number."
DRAFT = {
    "status": "supported",
    "object_scope": "each device",
    "question": "Is a crack visible on the device?",
    "materials": [{"field": "serial", "needed_when": ["positive"],
                   "question": "What is its serial number?", "permission_quote": None}],
}
VISIBLE = {"targets": [{"description": "left device", "bbox": [0.1, 0.2, 0.3, 0.7], "view_sufficient": False},
                       {"description": "right device", "bbox": [0.55, 0.2, 0.75, 0.7], "view_sufficient": False}]}


class ModelDouble:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.wall_seconds = 0.0

    def request(self, instructions, data, images):
        self.calls += 1
        return self.response(self.calls, data, images)


class CameraDouble:
    def __init__(self, second_motion=None):
        self.adapter = self
        self.scene = {"catalog": {}}
        self.yaw = 0.0
        self.position = {"x": 0.0, "y": 0.0, "z": 1.2}
        self.images = 0
        self.movements = 0
        self.second_motion = second_motion

    def observe(self):
        self.images += 1
        return {"observation_id": "camera-%d" % self.images, "data": b"frame-%d" % self.images,
                "position": self.position, "yaw": self.yaw, "ros_stamp": self.images}, {}

    def resolve_visual_target(self, observation, bbox, standoff):
        return {"goal": {"x": bbox[0] * 10, "y": 0.0, "z": 1.2, "yaw": 0.0}}

    def move(self, goal):
        self.movements += 1
        self.yaw = goal["yaw"]
        self.position = {axis: goal[axis] for axis in ("x", "y", "z")}
        if self.movements == 2 and self.second_motion:
            self.second_motion.set()
        return {"position": self.position, "yaw": self.yaw}


class ExistingGPTTests(unittest.TestCase):
    def test_reading_required_on_positive_does_not_gain_early_permission(self):
        def request(_index, data, images):
            self.assertEqual(data, {"authorized_task": INSTRUCTION})
            self.assertFalse(images)
            return DRAFT

        task = interpret(ModelDouble(request), INSTRUCTION)
        requirement, = task.requirements_for("visible-1", "answer")
        self.assertTrue(requirement.applies({"answer": "positive"}))
        self.assertFalse(requirement.applies({"answer": "negative"}))
        self.assertIsNone(task.materials[0].permission_quote)

        invalid = deepcopy(DRAFT)
        invalid["materials"][0]["permission_quote"] = "You can photograph the label early."
        with self.assertRaisesRegex(ValueError, "original instruction"):
            interpret(ModelDouble(lambda *_: invalid), INSTRUCTION)

    def test_completed_vision_response_cannot_carry_an_action_from_old_pose(self):
        frame = Frame("initial", "scene", 1, 0, b"camera", "test")
        task = interpret(ModelDouble(lambda *_: DRAFT), INSTRUCTION)
        target = locate_targets(ModelDouble(lambda *_: VISIBLE), task, frame)[0]
        bound = Frame("view-1", target.id, 2, 1, b"view-1", "test")
        prompt = frozen_prompt(bound, target, "See a crack?")
        with self.assertRaisesRegex(ValueError, "only a declared outcome"):
            judge(ModelDouble(lambda *_: {"outcome": "positive", "action": "move"}), bound.payload, prompt)
        with self.assertRaisesRegex(ValueError, "does not match"):
            frozen_prompt(frame, target, "See a crack?")

    def test_same_baseline_response_runs_before_action(self):
        def response(index, data, images):
            self.assertIn("authorized_task", data)
            self.assertTrue(images[-1]["label"].startswith("CURRENT onboard RGB"))
            if index == 1:
                return {"action": {"type": "move", "goal": {"x": 1.0, "y": 0.0, "z": 1.2, "yaw": 0.0}}}
            return {"action": {"type": "submit"}, "report": {"outcome": "accepted", "records": []}}

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            timeline = Timeline(directory / "timeline.jsonl")
            report, _ = run_agent(CameraDouble(), INSTRUCTION, directory, lambda *_args, **_fields: None,
                                  10, model=ModelDouble(response), trace=timeline)
            result = summarize(timeline.path)
        self.assertEqual(report["outcome"], "accepted")
        self.assertEqual(result["model_action_overlap_seconds"], 0)
        self.assertEqual(len(result["intervals"]["model_request"]), 2)

    def test_remote_entry_sends_old_image_while_other_required_target_moves(self):
        moved = Event()

        def response(index, data, images):
            if index == 1:
                return DRAFT
            if index == 2:
                return VISIBLE
            self.assertEqual(data["frame_id"], "camera-2")
            self.assertEqual(images[0]["data"], b"frame-2")
            self.assertTrue(moved.wait(3), "The independent target was never dispatched")
            return {"outcome": "negative"}

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            timeline = Timeline(directory / "timeline.jsonl")
            report, metrics = run_probe(CameraDouble(moved), INSTRUCTION, directory,
                                        lambda *_args, **_fields: None, 25,
                                        model=ModelDouble(response), trace=timeline)
            probe = json.loads((directory / "gpt_probe.json").read_text())
            result = summarize(timeline.path)
        self.assertEqual(report["outcome"], "incomplete")
        self.assertTrue(metrics["probe_completed"])
        self.assertEqual(probe["query"]["frame_id"], "camera-2")
        self.assertEqual(probe["independent_photo_id"], "camera-3")
        self.assertGreater(result["model_action_overlap_seconds"], 0)

    def test_already_sufficient_views_reuse_the_first_photo_and_skip_moves(self):
        draft = deepcopy(VISIBLE)
        for target in draft["targets"]:
            target["view_sufficient"] = True

        def response(index, data, images):
            if index == 1:
                return DRAFT
            if index == 2:
                return draft
            self.assertEqual(images[0]["data"], b"frame-1")
            self.assertEqual(data["bbox_in_attached_photo"], draft["targets"][0]["bbox"])
            return {"outcome": "negative"}

        backend = CameraDouble()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            timeline = Timeline(directory / "timeline.jsonl")
            run_probe(backend, INSTRUCTION, directory, lambda *_args, **_fields: None, 10,
                      model=ModelDouble(response), trace=timeline)
            probe = json.loads((directory / "gpt_probe.json").read_text())
        self.assertEqual(backend.movements, 0)
        self.assertTrue(probe["probe_completed"])
        self.assertTrue(probe["original_photo_reused_from_initial_view"])
        self.assertIsNone(probe["independent_photo_id"])

    def test_ambiguous_target_uses_only_map_admitted_surface_views(self):
        class SurfaceCamera(CameraDouble):
            def resolve_visual_target(self, observation, bbox, standoff):
                raise RuntimeError("Cannot establish object depth")

            def check_goal(self, goal):
                self.checked.append(goal)
                if not self.admit:
                    raise RuntimeError("Map does not admit this goal")

        backend = SurfaceCamera()
        backend.checked = []
        backend.admit = False
        observation, _ = backend.observe()
        geometry = {"measured_surface_grid": {"columns": 8, "rows": 4, "cells": [
            {"pixel_cell": [1, 1], "world_xyz": [4.0, 0.0, 1.2], "measured_points": 12},
        ]}}
        target = VisibleTarget("visible-1", "left cabinet", (0.1, 0.2, 0.3, 0.7), False)
        with tempfile.TemporaryDirectory() as tmp:
            timeline = Timeline(Path(tmp) / "timeline.jsonl")
            self.assertEqual(measured_views(backend, observation, geometry, (target,), timeline), [])
            self.assertTrue(backend.checked)
            backend.admit = True
            admitted = measured_views(backend, observation, geometry, (target,), timeline)
            self.assertEqual(admitted[0][0], target)
            self.assertEqual(admitted[0][1], backend.checked[-1])
            self.assertEqual(backend.movements, 0)


if __name__ == "__main__":
    unittest.main()
