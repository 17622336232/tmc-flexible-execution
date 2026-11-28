import tempfile
import unittest
from pathlib import Path

from experiments.flexible_execution.adapters.native_views import Offer
from experiments.flexible_execution.live_planning import Costs
from experiments.flexible_execution.timeline import Timeline
from research.flexible_scheduling.native_controller import EvidenceScheduler
from research.flexible_scheduling.workflow import Workflow,View,DecisionProblem


class Map:
    def check_goal(self,goal):
        if goal["x"]>4:
            raise ValueError("Outside test map")


class EvidenceWorkflows(unittest.TestCase):
    def test_multiple_legal_return_trips_and_common_budget(self):
        anchor=dict(x=0.,y=0.,z=1.,yaw=0.)
        offers=[Offer("v"+str(i),"identity","label",(0.,0.,1.,1.),"source",str(i),
                    dict(anchor,x=float(i)),False,"explicit") for i in (1,2,8)]
        costs=Costs(10,16,1,.1,.2,12,"test")
        with tempfile.TemporaryDirectory() as tmp:
            for method,expected in (("evidence-deferred",2),("evidence-planner",2),("evidence-prefetch",2),
                                    ("evidence-off",0),("evidence-async",0)):
                s=EvidenceScheduler(Map(),"检查这一排设备",Path(tmp),Timeline(Path(tmp)/(method+".jsonl")),method,costs)
                picked=s.select_many(offers,anchor,10,35,1)
                self.assertEqual(len(picked),expected)
                self.assertNotIn(offers[2],picked)
                self.assertEqual(s.requested_views({"use_views":["v2","v1"]},offers),offers[:2])
                self.assertEqual(s.requested_views({"use_views":"v2"},offers),[])

    def test_budget_applies_to_the_total_physical_plan(self):
        task=Workflow((((1,1.),),((2,1.),)),(View("a",1,3.,True,False),View("b",2,3.,True,False)),0,
                      (9.,),(0.,0.),0.,max_early_motion=4.)
        for method in ("strong-prefetch","generic","factored"):
            result=DecisionProblem(task).solve(method)
            self.assertEqual(len(result["overlap"]),1)
            self.assertEqual(result["expected_seconds"],12.)

