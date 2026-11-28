from functools import lru_cache
import unittest

from research.flexible_scheduling.deadline_workflow import DeadlineWorkflow
from research.flexible_scheduling.event_workflow import EventModel,Mission,Query


def model(times,slots=1):
    return EventModel(Mission(("here",),(Query("first",(0,)),Query("second",(0,))),
        ((0,),),(1,),(True,),0,1,times,0,0,slots))


class DeadlineTests(unittest.TestCase):
    def test_a_larger_group_can_improve_delivery_despite_more_success_service_time(self):
        m=model((0,3,7))
        none=DeadlineWorkflow(m,(0,0,0),100)
        base,action=none.solve(m.initial())
        self.assertEqual(base.restricted,6)
        scoped=DeadlineWorkflow(m,(0,.20,.20),100)
        val,chosen=scoped.solve(m.initial())
        self.assertEqual(chosen.reads,((0,3),))
        self.assertAlmostEqual(val.delivered,.8)
        self.assertAlmostEqual(val.restricted,.8*7+.2*100)
        independent=scoped.fixed(m.initial(),lambda s,t:none.solve(s,t)[1])
        self.assertAlmostEqual(independent.delivered,.8*.8)
        self.assertLess(val.restricted,independent.restricted)

    def test_expiry_and_call_counts(self):
        m=model((0,3,7))
        missed=DeadlineWorkflow(m,(0,0,0),5)
        v,_=missed.solve(m.initial())
        self.assertEqual(v.delivered,0.)
        self.assertEqual(v.restricted,5)
        certain=DeadlineWorkflow(m,(0,1,1),100)
        v,_=certain.solve(m.initial())
        self.assertEqual(v.delivered,0.)
        self.assertEqual(v.sends,1)
        with self.assertRaises(ValueError):
            DeadlineWorkflow(model((0,3,7),slots=2),(0,0,0),100)

    def test_gate_probability_cannot_be_read_before_the_request_returns(self):
        m=Mission(("here",),(Query("first",(0,),prior=.25),Query("conditional",(0,),(((0,True),),))),
                  ((0,),),(1,),(True,),0,1,(0,3,5))
        p=DeadlineWorkflow(EventModel(m),(0,.1,.4),100,quotient=False)
        score,first=p.solve(p.model.initial())
        self.assertEqual(first.reads,((0,1),))
        self.assertAlmostEqual(score.delivered,.9*(.75+.25*.9))
        self.assertAlmostEqual(score.sends,1+.9*.25)
