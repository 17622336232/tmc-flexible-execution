from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import unittest

import numpy as np
from PIL import Image

from experiments.flexible_execution.live_planning import Costs
from experiments.flexible_execution.timeline import Timeline
from research.flexible_scheduling.scoped_execution import execute


class TestSensor:
    def __init__(self,flat=False):
        output=BytesIO()
        data=np.random.RandomState(173).randint(0,255,(60,90,3)).astype("uint8")
        if flat:
            data[:]=128
        Image.fromarray(data).save(output,format="PNG")
        self.rgb=output.getvalue()
        self.pose={"x":0.,"y":0.,"z":1.}
        self.samples=0

    def observe(self):
        self.samples+=1
        geometry=dict(position=self.pose,yaw=0.,measured_surface_grid={"cells":[]},
            camera=dict(width=90,height=60,intrinsics=dict(cx=45,cy=30,fx=40,fy=40),world_T_camera=np.eye(4).tolist()))
        return dict(data=self.rgb,ros_stamp=float(self.samples),position=self.pose,yaw=0.),geometry

    def move(self,*args):
        raise AssertionError("An unmeasured or unauthorized view must never create motion")


class TestReply:
    def __init__(self,response):
        self.response=response
        self.calls=0

    def request(self,prompt,payload,images,attempt_limit):
        self.calls+=1
        self.payload=payload
        self.hashes=[sha256(p["data"]).hexdigest() for p in images]
        if attempt_limit!=1:
            raise AssertionError("Multiple sends for a grouped verdict")
        return self.response


class ClosedExecutionTests(unittest.TestCase):
    def run_fixture(self,rows,flat=False,wrong_reference=False):
        sensor=TestSensor(flat)
        source=sensor.rgb
        client=TestReply({"answers":rows})
        mission=dict(id="fixture",static=True,reference_sha256=sha256(source).hexdigest(),regions=[
            dict(id="left",bbox=[.1,.1,.4,.9],may_photograph_identity_before_result=False),
            dict(id="right",bbox=[.6,.1,.9,.9],may_photograph_identity_before_result=False)])
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=execute(sensor,client,mission,b"unrelated" if wrong_reference else source,root,Timeline(root/"events"),
                           Costs(4.,9.,.4,1.,.02,12.,"fixture"),"evidence-deferred",monotonic()+20.)
        return result,client

    def test_genuine_interface_one_batch_and_source_bound_report(self):
        result,client=self.run_fixture([{"id":"left","outcome":"negative"},{"id":"right","outcome":"negative"}])
        self.assertEqual(client.calls,1)
        self.assertEqual(client.hashes,[result["source_sha256"]])
        self.assertEqual(len(client.payload["regions"]),2)
        self.assertEqual((result["anchor_checked"],result["report"]["findings"],result["gaps"]),(True,[],[]))

    def test_missing_positive_identity_and_unanswered_region_remain_open(self):
        result,_=self.run_fixture([{"id":"left","outcome":"positive"}])
        self.assertIsNone(result["report"])
        self.assertEqual(set(result["gaps"]),{("left","source-bound-identity"),("right","resolved-crack-judgment")})

    def test_ambiguous_overview_stops_before_dispatch(self):
        result,client=self.run_fixture([],flat=True)
        self.assertEqual((result["report"],client.calls),(None,0))
        with self.assertRaises(ValueError):
            self.run_fixture([],wrong_reference=True)
