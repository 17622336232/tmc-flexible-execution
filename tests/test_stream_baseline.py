from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
import unittest

from experiments.flexible_execution.stream_baseline import decide
from experiments.flexible_execution.timeline import Timeline
from experiments.task_attack.agent import ModelDecisionInvalid


class StreamBaselineTest(unittest.TestCase):
    def test_segmented_continuation_uses_frozen_input_during_exactly_one_movement(self):
        action = {"type": "move", "goal": {"x": 2, "y": 1, "z": 1.2, "yaw": 0}}
        image = [{"label": "current", "data": b"frozen RGB"}]
        context = {"current_image_id": "onboard_0001", "authorized_task": "Inspect the equipment"}

        for stream in (True, False):
            with self.subTest(stream=stream), TemporaryDirectory() as temp:
                moved = Event()
                requests, movements = [], []

                class Model:
                    calls = 0
                    deadline_monotonic = monotonic() + 2

                    def request(self, _prompt, payload, images):
                        requests.append((payload, images))
                        self.calls += 1
                        if self.calls == 1:
                            return {"action": action}
                        if stream and not moved.wait(1):
                            raise AssertionError("Native movement did not start during continuation")
                        return {"memory": "read the faceplate", "records": [], "next_view_check": "check fresh RGB"}

                def execute(candidate):
                    movements.append(candidate)
                    moved.set()
                    return {"ok": True}

                decision, feedback = decide(Model(), context, image, execute=execute, step=1, image_sha256="frozen",
                                             trace=Timeline(Path(temp) / "timeline.jsonl"), stream=stream)
                self.assertEqual(movements, [action] if stream else [])
                self.assertEqual(feedback, {"ok": True} if stream else None)
                self.assertEqual(decision["action"], action)
                self.assertIn("fresh RGB", decision["memory"])
                self.assertEqual(requests[0], (context, image))
                self.assertEqual(requests[1], ({**context, "segmented_navigation": {"first_action": action,
                                                        "executed_pose_known": False}}, image))

    def test_rejected_geometry_cannot_reach_the_navigation_tool(self):
        class InvalidModel:
            def request(self, *_args):
                return {"action": {"type": "approach", "bbox": [0.9, 0.0, 0.1, 1.0]}}

        with TemporaryDirectory() as temp, self.assertRaises(ModelDecisionInvalid):
            decide(InvalidModel(), {"current_image_id": "front"}, [], step=1, image_sha256="frozen",
                   execute=lambda _action: self.fail("Invalid model action reached the simulator"),
                   trace=Timeline(Path(temp) / "timeline.jsonl"), stream=True)
