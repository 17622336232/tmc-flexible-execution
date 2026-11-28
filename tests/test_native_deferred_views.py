import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.flexible_execution.adapters.native_views import Offer, ViewScheduler
from experiments.flexible_execution.live_planning import Costs
from experiments.flexible_execution.timeline import Timeline


class Robot:
    def __init__(self):
        self.initial=dict(x=0.,y=0.,z=1.,yaw=0.)
        self.pose=dict(self.initial)
        self.camera=threading.Event()
        self.fail_return=False
    def check_goal(self,goal):
        return True
    def move(self,goal):
        if self.fail_return and goal==self.initial:
            raise RuntimeError("blocked return")
        self.pose=dict(goal)
        return dict(position=self.position(),yaw=goal["yaw"])
    def position(self):
        return {k:self.pose[k] for k in ("x","y","z")}
    def observe(self):
        self.camera.set()
        return dict(position=self.position(),yaw=self.pose["yaw"],ros_stamp=time.time(),data=b"actual-camera"),{}


class Model:
    def __init__(self,backend,need,side_in_flight):
        self.backend=backend; self.need=need; self.await_side=side_in_flight
        self.deadline_monotonic=time.monotonic()+30; self.calls=0
    def request(self,instructions,context,images):
        self.calls+=1
        assert len(images)==1
        if self.await_side:
            assert self.backend.camera.wait(2)
        return dict(action={"type":"observe"},use_view="v0" if self.need else None)


class NativeViewsTest(unittest.TestCase):
    def run_case(self,method,need,early,fail_return=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);robot=Robot();robot.fail_return=fail_return
            trace=Timeline(root/"timeline.jsonl")
            scheduler=ViewScheduler(robot,"Inspect.\n采集规则：允许在判读结果返回前采集以下材料：identity。",root,trace,
                                    method,Costs(5,8,1,1,.1,1,"unit"))
            scheduler.offers=[Offer("v0","identity","visible device",(0.,0.,.5,.8),"r","digest",
                                    dict(x=1.,y=0.,z=1.,yaw=0.),False,"grant")]
            model=Model(robot,need,early)
            obs=dict(position=robot.position(),yaw=0.,data=b"current")
            scheduler.decide(model,dict(current_image_id="r",recent_images=[],records=[]),[dict(data=b"current")],obs,{},1)
            photos=list(root.glob("onboard_side_*.jpg"))
            return [f.read_bytes() for f in photos], scheduler.release(),scheduler.unrestored,model.calls

    def test_positive_uses_one_return_and_one_capture(self):
        photos,released,failed,n=self.run_case("native-deferred",True,True)
        self.assertEqual((photos,len(released),failed,n),([b"actual-camera"],1,False,1))
        self.assertEqual(released[0]["source_reference"],"r")

    def test_negative_keeps_raw_but_never_exposes_to_following_request(self):
        photos,released,failed,n=self.run_case("native-deferred",False,True)
        self.assertEqual((photos,released,failed,n),([b"actual-camera"],[],False,1))

    def test_serial_and_async_have_same_after_reply_tool(self):
        for arm in ("native-off","native-async"):
            self.assertEqual(len(self.run_case(arm,True,False)[1]),1)
            self.assertEqual(self.run_case(arm,False,False)[0],[])

    def test_unrestored_anchor_invalidates_decision(self):
        _,ready,unrestored,n=self.run_case("native-deferred",True,True,True)
        self.assertTrue(unrestored)
        self.assertEqual((ready,n),([],1))
