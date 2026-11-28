import math
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from experiments.flexible_execution.timeline import Timeline,read_trace_rows
from research.flexible_scheduling.native_joint_execution import run_joint


TASK=("正面是起始朝向。原地向左转40度为侧面。看看正面照片里有没有红色货箱。"
      "也看看侧面照片里有没有蓝色货箱。如果正面有红色货箱，再检查侧面照片里有没有白色货箱。"
      "可以在等待时提前拍摄侧面照片。最后停在侧面。")
PRICES=dict(travel_s=[[0,1],[1,0]],photo_s=.1,single_s=12,same_s=14,cross_s=60,cross_enabled=True)


class Camera:
    scene={"catalog":{"scene_conditions":{}}}
    def __init__(self):
        self.position={"x":-4.,"y":-3.,"z":1.}
        self.yaw=.4
        self.side=Event()
        self.side_photo=Event()
        self.observations=0

    def observe(self):
        self.observations+=1
        if self.side.is_set():
            self.side_photo.set()
        geo=dict(position=self.position,yaw=self.yaw)
        frame=dict(**geo,ros_stamp=self.observations,data=("view-"+str(self.yaw)).encode())
        return frame,geo

    def move(self, goal):
        self.yaw=goal["yaw"]
        self.side.set()
        return dict(position=self.position,yaw=self.yaw)


class ReplyModel:
    def __init__(self,camera,*,first_wait):
        self.calls=0
        self.wall_seconds=0
        self.camera=camera
        self.first_wait=first_wait
        self.requests=[]

    def request(self, prompt,payload,images,attempt_limit=1):
        self.calls+=1
        scope=payload.get("this_request_scope",{})
        allowed=scope.get("open_for_reading",[])
        self.requests.append((allowed,images))
        rows=[]
        for spec in allowed:
            name=spec["field"]
            rows.append(dict(object_id=name,facts=dict(visible=name!="白色货箱"),
                             evidence_image_id=spec["evidence_image_id"],note="source bound"))
        if [q["field"] for q in allowed]==["红色货箱"]:
            if not self.first_wait.wait(2):
                raise RuntimeError("The request did not observe the legal concurrent move")
            # Model wanted repeat movement from its inference anchor. Runtime must not repeat it.
            return dict(action=dict(type="move",goal={**self.camera.position,"yaw":2.4}),records=rows)
        if len(allowed)==2:
            previous=scope["observed_facts"]
            return dict(action=dict(type="submit"),records=rows,
                        report=dict(outcome="accepted",records=previous+rows,summary="confirmed"))
        raise RuntimeError("Unexpected ordinary controller dispatch")


class NativeJointExecutionTest(unittest.TestCase):
    def test_real_control_loop_orders_motion_and_reuses_bound_fields(self):
        camera=Camera()
        model=ReplyModel(camera,first_wait=camera.side)
        with TemporaryDirectory() as folder:
            d=Path(folder)
            trace=Timeline(d/"timeline.jsonl")
            report,stats=run_joint(camera,TASK,d,lambda *a,**k:None,150.,model=model,trace=trace,mode="joint",prices=PRICES)
            self.assertEqual(report["outcome"],"accepted")
            self.assertEqual(stats["calls"],2)
            self.assertAlmostEqual(camera.yaw,.4+math.radians(40))
            self.assertEqual([len(q) for q,_ in model.requests],[1,2])
            actual=report["records"][-1]["evidence_image_id"]
            self.assertEqual((d/(actual+".jpg")).read_bytes(),model.requests[-1][1][-1]["data"])
            events=read_trace_rows(trace.path)[0]
            self.assertEqual(sum(e["event"]=="joint_motion_during_request" for e in events),1)

    def test_captures_before_reply_and_adopts_exact_frame(self):
        camera=Camera()
        model=ReplyModel(camera,first_wait=camera.side_photo)
        with TemporaryDirectory() as folder:
            d=Path(folder)
            trace=Timeline(d/"timeline.jsonl")
            report,stats=run_joint(camera,TASK,d,lambda *a,**k:None,150.,model=model,trace=trace,
                                   mode="joint",prices=PRICES,retain_pending_photo=True)
            self.assertEqual(report["outcome"],"accepted")
            self.assertEqual(stats["calls"],2)
            self.assertEqual(camera.observations,2)
            events=read_trace_rows(trace.path)[0]
            capture=next(e for e in events if e["event"]=="joint_pending_capture")
            front=next(e for e in events if e["event"]=="joint_fact")
            self.assertLess(capture["monotonic_seconds"],front["monotonic_seconds"])
            side=[e for e in events if e["event"]=="joint_fact" and e["step"]==2]
            self.assertEqual(len(side),2)
            self.assertTrue(all(e["image_sha256"]==capture["image_sha256"] for e in side))


if __name__=="__main__":
    unittest.main()
