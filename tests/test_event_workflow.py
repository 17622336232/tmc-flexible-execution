from functools import lru_cache
import unittest

from research.flexible_scheduling.event_workflow import ALWAYS,UNKNOWN,Dispatch,EventModel,Mission,Query,State
from research.flexible_scheduling.event_policy import Planner


def mission(q,slots=1,grant=True,batch=(0,5,6),finish=0):
    return Mission(("front","side"),tuple(q),((0,1),(2,0)),(1,1),(True,grant),0,1,batch,0,finish,slots)


class EventWorkflow(unittest.TestCase):
    def test_a_real_pending_result_does_not_license_an_early_capture(self):
        q=[Query("first",(0,)),Query("dependent",(1,),(((0,True),),))]
        e=EventModel(mission(q,grant=False))
        s=e.initial()
        self.assertNotIn(Dispatch((1,True),((0,1),)),e.dispatches(s))
        transit=Dispatch((1,False),((0,1),))
        self.assertIn(transit,e.dispatches(s))
        dt,rows=e.successors(s,transit)
        still=rows[0][1]
        self.assertEqual((dt,still.site,still.raw,still.values),(1,1,1,(UNKNOWN,UNKNOWN)))
        self.assertFalse(any(a.motion==(1,True) for a in e.dispatches(still)))
        _,branches=e.successors(still,Dispatch())
        for _,state in branches:
            self.assertEqual(any(a.motion==(1,True) for a in e.dispatches(state)),state.values[0]==1)

    def test_two_outstanding_requests_reveal_only_the_one_that_finished(self):
        e=EventModel(mission([Query(str(i),(0,)) for i in range(3)],slots=2,batch=(0,3,7)))
        a=Dispatch((),((0,1),(0,6)))
        self.assertIn(a,e.dispatches(e.initial()))
        elapsed,states=e.successors(e.initial(),a)
        self.assertEqual(elapsed,3)
        self.assertEqual({s.values for _,s in states},{(0,UNKNOWN,UNKNOWN),(1,UNKNOWN,UNKNOWN)})
        self.assertTrue(all(s.jobs==((0,6,4),) for _,s in states))

    def test_jointly_required_fields_can_share_one_read_and_a_source(self):
        e=EventModel(mission([Query("x",(0,)),Query("y",(0,))]))
        grouped=Planner(e).solve(e.initial())
        separate=Planner(e,ablation="single-read").solve(e.initial())
        self.assertEqual((grouped["value_ticks"],separate["value_ticks"]),(6,10))

    def test_future_equivalence_and_complete_report_ledger_are_separate(self):
        q=[Query("x",(0,)),Query("y",(0,)),Query("z",(1,),(((0,True),),((1,True),)))]
        e=EventModel(mission(q))
        origin=e.initial()
        a=State((0,1,UNKNOWN),origin.raw,origin.site)
        b=State((1,0,UNKNOWN),origin.raw,origin.site)
        c=State((0,0,UNKNOWN),origin.raw,origin.site)
        self.assertEqual(e.signature(a),e.signature(b))
        self.assertNotEqual(e.signature(a),e.signature(c))
        self.assertNotEqual(a.values,b.values)

    def test_certificates_match_unpruned_outcome_tree_with_and_without_early_read(self):
        for early in (True,False):
            for slots in (1,2):
                e=EventModel(mission([Query("x",(0,),prior=.3),Query("y",(1,),(((0,True),),),early_read=early),
                                      Query("z",(0,),(((1,False),),),early_read=early)],slots=slots,finish=1))
                @lru_cache(None)
                def unpruned(s):
                    if e.complete(s):return 0.
                    values=[]
                    for a in e.dispatches(s):
                        t,sub=e.successors(s,a)
                        values.append(t+sum(p*unpruned(y) for p,y in sub))
                    return min(values)
                expected=unpruned(e.initial())
                for quotient in (False,True):
                    got=Planner(e,quotient=quotient).solve(e.initial(),budget=100000)
                    self.assertTrue(got["proved"])
                    self.assertAlmostEqual(got["value_ticks"],expected)
