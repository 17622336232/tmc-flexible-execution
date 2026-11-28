"""Small exact controls for permission, shared evidence and double-counting of overlap."""
import unittest

from research.flexible_scheduling.workflow import DecisionProblem, View, Workflow, merged_demands


def problem(*,trip=3.,early=True,semantic=True,existing=False,p=1.,front=5.,joint=6.,follow=8.):
    return Workflow((((0,1.-p),(1,p)),),(View("side",1,trip,early,semantic,existing),),
                    0,(front,joint),(0.,follow),2.)


class RemainingTaskTests(unittest.TestCase):
    def test_semantic_and_capture_grants_are_distinct(self):
        # Compare three independently hand-computed policies.
        for task,cost,joint,over in (
            (problem(),11.,(0,),()), (problem(semantic=False),15.,(),(0,)),
            (problem(early=False,semantic=False),18.,(),()),
            (problem(existing=True,early=False),8.,(0,),()),
            (problem(trip=25.,p=.01),7.33,(),())):
            for mode in ("generic","factored"):
                decision=DecisionProblem(task).solve(mode)
                self.assertAlmostEqual(decision["expected_seconds"],cost)
                self.assertEqual((decision["joint"],decision["overlap"]),(joint,over))

    def test_only_real_uncovered_future_fields_remain(self):
        factors=(((1,.5),(0,.5)),((1,.5),(0,.5)),((2,1.),))
        self.assertEqual(merged_demands(factors,0),{3:.75,2:.25})
        self.assertEqual(merged_demands(factors,1),{2:1.})
        common=View("both",3,3.,True,True)
        distinct=(View("first",1,5.,True,True), View("second",2,5.,True,False))
        task=Workflow(factors,(common,*distinct),0,(5.,5.),(0.,9.,11.),1.)
        for mode in ("factored","generic"):
            output=DecisionProblem(task).solve(mode)
            self.assertEqual(output["joint"],(0,))
            self.assertEqual(output["overlap"],())
            self.assertEqual(output["expected_seconds"],9.)

    def test_unreadable_creates_a_different_followup(self):
        task=Workflow((((0,.6),(1,.3),(2,.1)),),(View("label",1,8.,False,False),
                      View("recheck",2,6.,False,False)),0,(4.,),(0.,7.),2.)
        self.assertAlmostEqual(DecisionProblem(task).solve("factored")["expected_seconds"],6+.3*15+.1*13)
