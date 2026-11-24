import unittest

from experiments.flexible_execution.candidate_ranking import LinearRanker, select_ranked
from experiments.flexible_execution.executor import Snapshot
from experiments.flexible_execution.task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract


class CandidateOrderingTest(unittest.TestCase):
    def test_model_ranks_before_any_common_continuation_is_checked(self):
        query = QuerySpec("judge", "asset", ("positive", "negative", "uncertain"), 10., 15.)
        requirement = Requirement("part", "asset", "identity", Condition("judge", ("positive",)))
        skills = (Skill("cheap", "near", ("part",), 1.), Skill("costly", "far", ("part",), 2.))
        routes = (("start", "near", 1.), ("start", "far", 2.), ("near", "end", 1.),
                  ("far", "end", 1.), ("start", "end", 1.))
        task = TaskContract("inspect", "start", "end", 50., 1., (requirement,), (query,), skills, routes)
        state = Snapshot(0., "start", {}, {}, {})
        ranker = LinearRanker("cost", (0.,) * 10, (1.,) * 10, (0., 1.) + (0.,) * 9, "fixture")

        for model, expected in ((None, "cheap"), (ranker, "costly")):
            observed = []

            def check(_task, _state, prefix):
                observed.append(prefix[0].skill.id)
                return None, []

            select_ranked(task, state, check, lambda *_args, **_fields: None, ranker=model,
                          check_cap=1, budget_ms=1000, max_prepare=1)
            self.assertEqual(observed, [expected])
