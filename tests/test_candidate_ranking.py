import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.flexible_execution.candidate_ranking import (LinearRanker, ONLINE_RANKED_METHODS, RetrievalRanker,
                                                             select_ranked)
from experiments.flexible_execution.executor import Snapshot
from experiments.flexible_execution.learning_schedule import select_live_configuration
from experiments.flexible_execution.pilot_report import _paired_ranking
from experiments.flexible_execution.runner import _hash
from experiments.flexible_execution.task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract
from experiments.task_attack.runtime import write_json


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
        with TemporaryDirectory() as tmp:
            index = Path(tmp) / "retrieval.json"
            index.write_text(json.dumps({"objective": "retrieval", "offset": [0.] * 10, "scale": [1.] * 10,
                                         "points": [[1., d, 1., 0., 1., .2, .3, 0., 1., .02]
                                                    for d in (.04, .08)],
                                         "gains": [0., 1.], "neighbors": 1}))
            retrieval = RetrievalRanker.load(index)

        for model, expected in ((None, "cheap"), (ranker, "costly"), (retrieval, "costly")):
            observed = []

            def check(_task, _state, prefix):
                observed.append(prefix[0].skill.id)
                return None, []

            select_ranked(task, state, check, lambda *_args, **_fields: None, ranker=model,
                          check_cap=1, budget_ms=1000, max_prepare=1)
            self.assertEqual(observed, [expected])

    def test_live_configuration_uses_matched_success_then_restricted_time(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = {"ranker_selection_plan": {"training_rows": 1000, "caps": {}},
                       "cases": [{"id": "matched", "episode_timeout_seconds": 480, "order": []}]}
            for cap in (4, 8):
                config = root / f"cap-{cap}.json"
                write_json(config, {"cap": cap})
                catalog["ranker_selection_plan"]["caps"][str(cap)] = {
                    "path": str(config), "config_sha256": _hash({"cap": cap})}
                for index, method in enumerate(ONLINE_RANKED_METHODS):
                    label = f"{method}-cap{cap}"
                    catalog["cases"][0]["order"].append(label)
                    valid = cap == 8 or index > 0
                    write_json(root / "matched" / label / "job.json", {
                        "state": "FINISHED_VALID" if valid else "TIMEOUT", "ranker_check_cap": cap,
                        "charged_seconds": 30 if cap == 8 else 20})
            config, evidence = select_live_configuration(root, catalog)
            self.assertEqual(config, {"cap": 8})
            self.assertEqual(evidence["candidate_configurations"]["4"]["successes"], 4)
            self.assertEqual(evidence["candidate_configurations"]["8"]["successes"], 5)
            first = f"{ONLINE_RANKED_METHODS[0]}-cap4"
            write_json(root / "matched" / first / "job.json",
                       {"state": "FINISHED_VALID", "ranker_check_cap": 4, "charged_seconds": 20})
            self.assertEqual(select_live_configuration(root, catalog)[0], {"cap": 4})

    def test_multiple_training_seeds_do_not_increase_independent_case_counts(self):
        rows = []
        for case in ("full", "partial"):
            for seed in (17, 29, 43):
                for method, seconds in (("conditional-rank-b1", 100), ("conditional-rank-ordinary", 115)):
                    state = "PENDING" if case == "partial" and seed == 43 else "FINISHED_VALID"
                    rows.append({"case": case, "group": "same-scene-source", "execution_method": method,
                                 "ranker_check_cap": 4, "training_seed": seed, "state": state,
                                 "success": True, "restricted_seconds": seconds})
        comparison, = _paired_ranking(rows)
        self.assertEqual(comparison["seed_pairs_per_case"], 3)
        self.assertEqual(comparison["matched_cases"], 1)
        self.assertEqual(comparison["independent_groups"], 1)
        self.assertEqual(comparison["restricted_seconds_difference"]["mean"], -15)
        self.assertIsNone(comparison["restricted_seconds_difference"]["cluster_bootstrap_95"])
