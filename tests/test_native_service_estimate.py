import unittest

from research.flexible_scheduling.native_service_estimate import RequestObservation, restricted_request_estimate


class NativeServiceEstimateTest(unittest.TestCase):
    def test_pending_expired_and_incorrect_observations_have_distinct_effects(self):
        rows=[
            RequestObservation(4,5,True,True),
            RequestObservation(4,20,True,False),
            RequestObservation(4,120,False),
            RequestObservation(8,12,True,True),
        ]
        result=restricted_request_estimate(rows,now=6,elapsed_s=10,limit_s=90,prior_s=15)
        self.assertEqual(result["samples"],2)
        self.assertEqual(result["incorrect"],1)
        self.assertEqual(result["nondelivery"],1)
        self.assertEqual(result["estimated_restricted_s"],80)
        updated=restricted_request_estimate(rows,now=9,elapsed_s=10,limit_s=90,prior_s=15)
        self.assertEqual(updated["successful"],1)
        self.assertEqual(updated["median_until_usable_s"],2)

    def test_unknown_quality_informs_delivery_only(self):
        result=restricted_request_estimate([RequestObservation(1,5,True)],now=2,elapsed_s=0,limit_s=80,prior_s=15)
        self.assertEqual(result["successful"],1)
        self.assertEqual(result["independently_checked"],0)
        self.assertEqual(result["estimated_restricted_s"],10)
