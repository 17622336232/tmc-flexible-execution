from hashlib import sha256
from pathlib import Path
import tempfile
from time import monotonic
import unittest

from experiments.flexible_execution.adapters.native_views import Offer
from experiments.flexible_execution.live_planning import Costs
from experiments.flexible_execution.timeline import Timeline
from research.flexible_scheduling.coordinated_controller import CoordinatedScheduler,activation_matches


TASK=("检查这一排设备的正面外壳，报告存在裂纹的设备编号，并保留图像证据。编号指设备本体身份。 "
      "报告中以 object_id 表示对象编号，facts 使用字段：cracked（是否有裂纹）。\n"
      "采集规则：允许在判读结果返回前采集以下材料：identity。")
SOURCE=b"frozen observed frame"
ANCHOR=dict(x=0.,y=0.,z=1.2,yaw=0.)
OFFER=Offer("cell","identity","equipment face",(.1,.2,.4,.8),"ref",sha256(SOURCE).hexdigest(),
            {**ANCHOR,"x":1.},False,"explicit")
CONDITION={"op":"eq","field":"cracked","right":{"literal":True}}
JUDGMENT=dict(id="cell",source="ref",field="cracked",value=True)


class Camera:
    def __init__(self):
        self.pose=ANCHOR
        self.count=0
    def check_goal(self,goal):
        return True
    def move(self,goal):
        self.pose=goal
        return self.current()
    def current(self):
        return dict(position={k:self.pose[k] for k in ("x","y","z")},yaw=self.pose["yaw"])
    def observe(self):
        self.count+=1
        return {**self.current(),"ros_stamp":1.,"data":b"side"},dict(camera={})


class Model:
    def __init__(self,decision):
        self.decision=decision
        self.calls=0
        self.deadline_monotonic=monotonic()+10
    def request(self,prompt,context,images):
        self.calls+=1
        self.context=context
        return self.decision


def movement(**changes):
    return dict(dict(action=dict(type="move",goal={**ANCHOR,"y":1.}),use_views=["cell"],
                     view_judgments=[JUDGMENT],motion_independent_of_view=True),**changes)


class CoordinatedHandoffTests(unittest.TestCase):
    def execute(self,decision,mode="coordinated-off",rgb=SOURCE):
        with tempfile.TemporaryDirectory() as tmp:
            backend=Camera()
            model=Model(decision)
            scheduler=CoordinatedScheduler(backend,TASK,Path(tmp),Timeline(Path(tmp)/"events"),mode,
                                           Costs(2,8,.5,.1,.04,10,"test"))
            scheduler.offers=[OFFER]
            context=dict(current_image_id="ref",recent_images=[],records=[])
            actual=scheduler.decide(model,context,[dict(data=rgb)],{**backend.current(),"data":rgb},{},1)
            return actual,scheduler.release(),backend.count,backend.pose,model.calls

    def test_moves_and_carries_raw_photo_without_altering_the_model_action(self):
        command=movement()
        actual,photos,n,pose,calls=self.execute(command)
        self.assertEqual(actual,command)
        self.assertEqual(([p["data"] for p in photos],n,pose,calls),([b"side"],1,ANCHOR,1))

    def test_unsupported_branch_or_motion_never_releases_a_photo(self):
        for action in (movement(motion_independent_of_view=False),movement(view_judgments=[]),
                       movement(view_judgments=[dict(JUDGMENT,value=False)]),
                       movement(view_judgments=[dict(JUDGMENT,value=1)]),movement(view_judgments=[JUDGMENT,JUDGMENT]),
                       movement(action={"type":"submit"})):
            self.assertEqual(self.execute(action)[1:3],([],0))
        self.assertEqual(self.execute(movement(),rgb=b"mismatched pixels")[1:3],([],0))

    def test_observe_only_and_supply_checks(self):
        self.assertFalse(activation_matches(OFFER,movement(),CONDITION,set()))
        self.assertEqual(self.execute(movement(),mode="coordinated-observe-only")[1],[])
        self.assertEqual(len(self.execute(movement(action={"type":"observe"}))[1]),1)
