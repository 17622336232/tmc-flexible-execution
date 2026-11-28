from hashlib import sha256
from threading import Event
from time import monotonic
import unittest

from research.flexible_scheduling.inspection_travel import Capture, accepted_read, execute, schedule


STOPS = [{"target":"BAY-01","goal":{"x":1}}, {"target":"BAY-02","goal":{"x":2}}]


class CameraPlatform:
    def __init__(self):
        self.actions = []
        self.at_b = Event()
        self.at_home = Event()

    def move(self, key, goal):
        self.actions.append(("move",key))
        if key == "b":
            self.at_b.set()

    def take(self, key):
        self.actions.append(("capture",key))
        return Capture(key,key.encode())

    def home(self):
        self.actions.append(("move","home"))
        self.at_home.set()


class Model:
    def __init__(self, platform, first, concurrent, blocked_capture):
        self.calls = 0
        self.platform = platform
        self.first = first
        self.concurrent = concurrent
        self.blocked_capture = blocked_capture
        self.seen = []

    def request(self, prompt, payload, images, attempt_limit):
        self.calls += 1
        k = payload["image_id"]
        assert payload["sha256"] == sha256(images[0]["data"]).hexdigest()
        assert attempt_limit == 1
        self.seen.append(k)
        if k == "a" and self.concurrent:
            assert self.platform.at_b.wait(2)
            if self.blocked_capture:
                assert ("capture","b") not in self.platform.actions
            else:
                assert self.platform.at_home.wait(2)
        return {"visible_id":payload["target"],"label_box":[.41,.48,.48,.55],
                "value":self.first if k=="a" else 1}


class InspectionTravel(unittest.TestCase):
    def test_forbidden_photo_waits_for_a_positive_actual_return(self):
        for n in [0,3,None]:
            for method in ["serial","strong-prefetch","dependency-schedule"]:
                robot=CameraPlatform()
                model=Model(robot,n,method!="serial",True)
                out=execute({"field":"box_count","b_capture":"after-positive-only"},
                            STOPS,robot,model,method,monotonic()+5)
                self.assertEqual(model.seen,["a","b"] if n==3 else ["a"])
                self.assertIs(out["returned"],True)
                self.assertEqual(("capture","b") in robot.actions,n==3)
                self.assertEqual([k for verb,k in robot.actions if verb=="move"],["a","b","home"])

    def test_required_archive_can_return_while_the_first_question_is_unresolved(self):
        robot=CameraPlatform()
        m=Model(robot,0,True,False)
        out=execute({"field":"box_count","b_capture":"mandatory-archive"},STOPS,robot,m,
                    "dependency-schedule",monotonic()+5)
        self.assertEqual(set(out["archived"]),{"a","b"})
        self.assertEqual(m.seen,["a"])

    def test_controls_and_typed_evidence(self):
        for grant in ["mandatory-archive","after-positive-only"]:
            self.assertEqual(schedule("strong-prefetch",grant),schedule("dependency-schedule",grant))
            self.assertNotEqual(schedule("serial",grant),schedule("dependency-schedule",grant))
        base={"visible_id":"BAY-01","label_box":[.1,.2,.3,.4],"value":1}
        for patch in [{"value":True},{"value":-1},{"visible_id":"BAY-02"},{"label_box":[0,0,2,1]}]:
            self.assertIsNone(accepted_read({**base,**patch},STOPS[0],"box_count",Capture("a",b"x")))
