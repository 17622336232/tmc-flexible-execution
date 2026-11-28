import unittest
from research.flexible_scheduling.observed_return import ObservedTrail
from scripts.continuing_travel_reference import analytic, reference


def waypoint(x):
    return dict(x=x,y=0.,z=1.,yaw=0.)


class Denied(ValueError):
    code="goal_unobserved"
    def __init__(self,reason):
        self.evidence={"map_query_reason":reason}


class ObservedReturn(unittest.TestCase):
    def test_unknown_home_requires_admitted_older_measured_point(self):
        trail=ObservedTrail(waypoint(0))
        trail.record(waypoint(1))
        trail.record(waypoint(2))
        trace=[];moved=[]
        def check(g):
            if g["x"]==0 and not moved:
                raise Denied("not_known_free")
        def move(i,g):
            moved.append(i)
            return g
        trail.return_home(waypoint(3),check,move,lambda kind,**v:trace.append(v))
        self.assertEqual(moved,[1,0])
        self.assertEqual([r["accepted"] for r in trace],[False,True,True])

    def test_unavailable_map_and_motion_failure_never_switch_waypoints(self):
        trail=ObservedTrail(waypoint(0))
        trail.record(waypoint(1))
        def stale(g):
            raise Denied("scan_stale")
        def failed(i,g):
            self.assertEqual(i,0)
            raise RuntimeError("motion already started")
        with self.assertRaises(Denied):
            trail.return_home(waypoint(2),stale,lambda *a:self.fail("unverified move"),lambda *a,**k:None)
        with self.assertRaisesRegex(RuntimeError,"already started"):
            trail.return_home(waypoint(2),lambda g:True,failed,lambda *a,**k:None)

    def test_exact_nonanticipating_schedules_match_both_legal_outcomes(self):
        for duration in (.3,9,80):
            t={"to-a":2,"photo-a":.2,"to-b":6,"photo-b":.2,"home":4,"read-a":duration,"read-b":.7}
            for active in (True,False):
                for grant in ("mandatory-archive","after-positive-only"):
                    best,n=reference(t,active,grant)
                    self.assertGreater(n,1)
                    self.assertAlmostEqual(best,analytic(t,active,grant))
