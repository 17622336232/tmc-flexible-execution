import unittest

from research.flexible_scheduling.native_condition_resolution import resolve_grouped_conditions
from research.flexible_scheduling.natural_frontier import released


CLAUSES = [
    "正面是起始朝向",
    "原地向左转40度为侧面",
    "看看正面照片里有没有红色货箱",
    "如果正面有红色货箱，再检查正面照片里有没有蓝色货箱，也看看侧面照片里有没有绿色货箱",
    "如果正面有蓝色货箱，再检查侧面照片里有没有白色货箱",
    "可以在等待时提前拍摄侧面照片",
    "最后停在侧面",
]


class NativeConditionResolutionTest(unittest.TestCase):
    def test_bound_true_observation_makes_both_plans_equivalent(self):
        value={"records":[{"object_id":"红色货箱","facts":{"visible":True},"evidence_image_id":"actual"}]}
        plan=resolve_grouped_conditions(CLAUSES,value,{"actual":"front"})
        self.assertIsNotNone(plan)
        self.assertEqual(released(plan,{0:True}),{1,2})
        self.assertEqual(released(plan,{0:True,1:True,2:False}),{3})

    def test_unknown_negative_and_wrong_photo_cannot_waive_ambiguity(self):
        for value,images in ((False,{"actual":"front"}),(True,{"wrong":"front"}),(None,{"actual":"front"})):
            decision={"records":[{"object_id":"红色货箱","facts":{"visible":value},"evidence_image_id":"actual"}]}
            self.assertIsNone(resolve_grouped_conditions(CLAUSES,decision,images))
