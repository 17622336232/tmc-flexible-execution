import unittest

from research.flexible_scheduling.native_continuation import proposal_plan
from research.flexible_scheduling.natural_frontier import discover, released


CLAUSES = [
    "正面是起始朝向",
    "原地向左转40度为侧面",
    "看看正面照片里有没有红色货箱",
    "也看看侧面照片里有没有蓝色货箱",
    "如果正面有红色货箱，再检查侧面照片里有没有白色货箱",
    "可以在等待时提前拍摄侧面照片",
    "最后停在侧面",
]


class NativeContinuationTest(unittest.TestCase):
    def test_model_can_decompose_continuous_prose_without_rewriting_gates(self):
        prose = "。".join([*CLAUSES[:2],"，".join(CLAUSES[2:4]),*CLAUSES[4:]]) + "。"
        self.assertIsNone(discover(prose))
        plan = proposal_plan(prose, {"continuation_work":{"clauses":CLAUSES}})
        self.assertIsNotNone(plan)
        self.assertEqual(released(plan, {}), {0, 1})
        self.assertEqual(released(plan, {0:False, 1:True}), set())
        self.assertEqual(released(plan, {0:True, 1:True}), {2})

    def test_omission_paraphrase_and_negation_fail_closed(self):
        normal = "，".join(CLAUSES)
        forbidden = normal.replace("可以在等待时", "不可以在等待时")
        rewritten = { "continuation_work":{"clauses":CLAUSES} }
        self.assertIsNone(proposal_plan(forbidden,rewritten))
        omitted = {"continuation_work":{"clauses":CLAUSES[:4]+CLAUSES[5:]}}
        self.assertIsNone(proposal_plan(normal,omitted))
        condition = {"continuation_work":{"clauses":[*CLAUSES[:4],
            "如果正面有红色货箱","再检查侧面照片里有没有白色货箱",*CLAUSES[5:]]}}
        self.assertIsNone(proposal_plan(normal,condition))

    def test_model_cannot_remove_conditional_scope_by_splitting_exact_quotes(self):
        original=[*CLAUSES[:3],
                  CLAUSES[4]+"，"+CLAUSES[3],*CLAUSES[5:]]
        text="。".join(original)+"。"
        quotes=[*CLAUSES[:3],CLAUSES[4],CLAUSES[3],*CLAUSES[5:]]
        annotation=dict(continuation_work=dict(clauses=quotes))
        self.assertIsNone(proposal_plan(text,annotation,{"front":"front"}))
        red=dict(object_id="红色货箱",facts=dict(visible=True),evidence_image_id="front")
        self.assertIsNone(proposal_plan(text,{**annotation,"records":[red]},{"wrong":"front"}))
        plan=proposal_plan(text,{**annotation,"records":[red]},{"front":"front"})
        self.assertIsNotNone(plan)
        self.assertEqual(released(plan,{}),{0})
        self.assertEqual(released(plan,{0:True}),{1,2})


if __name__ == "__main__":
    unittest.main()
