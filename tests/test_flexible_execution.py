import asyncio
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import yaml

from experiments.flexible_execution.cli import CHECKOUT, DEFAULTS
from experiments.flexible_execution.evidence import Frame, Observation, accept
from experiments.flexible_execution.evaluate import score
from experiments.flexible_execution.executor import permitted_observation, project, terminal_action
from experiments.flexible_execution.fixtures import diagnostics
from experiments.flexible_execution.queries import AsyncQueryWorker, QueryBinding, QueryBook, QueryReply
from experiments.flexible_execution.runner import manifest, reproduce, run
from experiments.flexible_execution.search import Search, witness
from experiments.flexible_execution.simulation import METHODS, Episode, visible_state_hash
from experiments.flexible_execution.task_ir import Condition


class DiagnosticTests(unittest.TestCase):
    def test_empty_report_needs_actual_check_coverage(self):
        case = diagnostics()[-1]
        events = Episode(case, "ASYNC").run()
        events = [event for event in events
                  if event["event"] not in {"observation_captured", "evidence_accepted"}]
        result = score(case, events)
        self.assertTrue(result["wrong_accepted"])
        self.assertEqual(result["missing_requirements"], ["primary"])

    def test_material_rejects_wrong_object_and_expired_capture(self):
        requirement = diagnostics()[0].task.requirements[0]
        frame = Frame("frame", "B", 0, 1, b"frozen pixels", "test")
        self.assertIsNone(accept(requirement, Observation(frame, requirement.field, "seen", True), 1))
        frame = replace(frame, object_id=requirement.object_id)
        self.assertIsNone(accept(requirement, Observation(frame, requirement.field, "seen", True, 2), 2))
        self.assertIsNone(accept(requirement, Observation(frame, requirement.field, "seen", False), 1))
        self.assertIsNotNone(accept(requirement, Observation(frame, requirement.field, "seen", True, 2), 1))

    def test_stale_reply_cannot_fill_replacement(self):
        spec = diagnostics()[0].task.queries[0]
        raw = bytearray(b"first")
        old_frame = Frame("frame1", spec.object_id, 0, 0, raw, "test")
        raw[0] = ord("X")
        self.assertEqual(old_frame.payload, b"first")
        old = QueryBinding.bind(spec, old_frame, "query", 0)
        new = QueryBinding.bind(spec, replace(old_frame, id="frame2", version=1), "query", 1)
        book = QueryBook()
        book.register(old)
        book.register(new)
        self.assertFalse(book.accept(QueryReply(old, "normal"), 2))
        self.assertFalse(book.accept(QueryReply(replace(new, input_hash="bad"), "normal"), 2))
        self.assertTrue(book.accept(QueryReply(new, "normal"), 2))
        self.assertFalse(book.accept(QueryReply(new, "normal"), 2))

    def test_permission_only_counterfactual(self):
        allowed = diagnostics()[2]
        episode = Episode(allowed, "A")
        episode.initialize()
        self.assertIsNotNone(episode.search.propose(episode.snapshot()).choice)
        permission = Condition("query-A", ("exception", "inconclusive"))
        restricted = replace(allowed, task=replace(
            allowed.task, skills=tuple(replace(skill, permission=permission) for skill in allowed.task.skills),
        ))
        events = Episode(restricted, "A").run()
        starts = [row for row in events if row["event"] == "action_started" and row["skill_id"]]
        self.assertTrue(starts)
        self.assertTrue(all(not row["queries_pending"] for row in starts))
        self.assertTrue(score(restricted, events)["success"])

    def test_unknown_branch_needs_review_even_with_a_valid_id(self):
        case = diagnostics()[2]
        episode = Episode(case, "A")
        episode.initialize()
        state = episode.snapshot()
        get_detail = permitted_observation(case.task, state, case.task.skills[0])
        self.assertIsNone(witness(case.task, state, (get_detail,)))
        self.assertIsNone(terminal_action(case.task, project(case.task, state, get_detail)))

    def test_expected_evidence_cannot_override_capture_window(self):
        case = diagnostics()[2]
        requirements = tuple(replace(item, due=0.25) if item.id == "review" else item
                             for item in case.task.requirements)
        task = replace(case.task, requirements=requirements)
        episode = Episode(replace(case, task=task), "A")
        episode.initialize()
        state = episode.snapshot()
        shared = permitted_observation(task, state, task.skills[2])
        self.assertIsNone(witness(task, state, (shared,)))

    def test_hidden_outcome_does_not_change_policy_before_the_answer(self):
        first = diagnostics()[2]
        second = replace(first, query_truth=(replace(first.query_truth[0], outcome="normal"),))
        initial = []
        decisions = []
        for case in (first, second):
            episode = Episode(case, "A")
            episode.initialize()
            initial.append(visible_state_hash(case.task, episode.snapshot()))
            events = Episode(case, "A").run()
            decisions.append([row["skill_id"] for row in events
                              if row["event"] == "action_started" and row["queries_pending"]])
        self.assertEqual(initial[0], initial[1])
        self.assertTrue(decisions[0])
        self.assertEqual(decisions[0], decisions[1])

    def test_declined_proposal_pays_actual_computation_once(self):
        case = diagnostics()[4]
        events = Episode(case, "A").run()
        planned_at = next(row["at"] for row in events if row["event"] == "proposal_started")
        paid = next(row for row in events if row["event"] == "proposal_finished")
        wait = next(row for row in events if row["event"] == "blocked_wait")
        self.assertEqual(paid["proposed"], [])
        self.assertGreater(paid["spent_seconds"], 0)
        self.assertAlmostEqual(paid["at"], planned_at + paid["spent_seconds"], places=8)
        self.assertEqual(wait["at"], paid["at"])
        self.assertEqual(score(case, events)["proposal_wall_seconds"], paid["spent_seconds"])

    def test_ordinary_independent_action_starts_before_pending_query_returns(self):
        case = diagnostics()[0]
        events = Episode(case, "ASYNC").run()
        start = next(row for row in events if row["event"] == "action_started")
        reply = next(row for row in events if row["event"] == "query_returned")
        self.assertEqual(start["skill_id"], "other_view")
        self.assertLess(start["at"], reply["at"])

    def test_witness_matches_independent_small_enumeration(self):
        case = diagnostics()[2]
        episode = Episode(case, "A")
        episode.initialize()
        initial = episode.snapshot()
        for size in range(1, len(case.task.skills) + 1):
            for skills in permutations(case.task.skills, size):
                state = initial
                prefix = []
                for skill in skills:
                    observation = permitted_observation(case.task, state, skill)
                    prefix.append(observation)
                    state = project(case.task, state, observation)
                expected = {"detail", "review"} <= {key for skill in skills for key in skill.covers}
                self.assertEqual(witness(case.task, initial, tuple(prefix)) is not None, expected)

    def test_static_skill_names_do_not_change_choice(self):
        case = diagnostics()[2]
        original = Episode(case, "A")
        original.initialize()
        renumbered = replace(case.task, skills=tuple(replace(skill, id=f"option-{i}")
                                                     for i, skill in enumerate(case.task.skills)))
        first = Search(case.task).propose(original.snapshot()).choice
        second = Search(renumbered).propose(original.snapshot()).choice
        self.assertEqual([item.skill.covers for item in first.prepare],
                         [item.skill.covers for item in second.prepare])

    def test_all_planned_diagnostics_score_from_events(self):
        for case in diagnostics():
            for method in METHODS:
                with self.subTest(case=case.id, method=method):
                    events = Episode(case, method).run()
                    result = score(case, events)
                    self.assertTrue(result["success"], result)
                    self.assertEqual(result["trace_errors"], [])
        case = diagnostics()[9]
        outcomes = score(case, Episode(case, "A").run())
        self.assertGreater(outcomes["rejected_materials"], 0)

    def test_interrupted_batch_retains_completed_episode(self):
        defaults = yaml.safe_load(DEFAULTS.read_text(encoding="utf8"))
        real_run = Episode.run
        invoked = 0

        def interrupt_on_second(episode):
            nonlocal invoked
            invoked += 1
            if invoked == 2:
                raise KeyboardInterrupt
            return real_run(episode)

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch.object(Episode, "run", interrupt_on_second):
                with self.assertRaises(KeyboardInterrupt):
                    run(directory, CHECKOUT, defaults, False)
            first = directory / "episodes/mechanism-01-ASYNC.json"
            saved = first.read_bytes()
            written_at = first.stat().st_mtime_ns
            resumed = run(directory, CHECKOUT, defaults, True)
            self.assertEqual(first.read_bytes(), saved)
            self.assertEqual(first.stat().st_mtime_ns, written_at)
            self.assertEqual(resumed["completed_episodes"], 36)
            self.assertEqual(reproduce(directory, manifest(CHECKOUT, defaults)), resumed)


class AsyncWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_worker_does_not_block_independent_action(self):
        entered = threading.Event()
        release = threading.Event()

        def blocking_inference(payload, prompt):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Independent action did not run")
            return "normal"

        spec = diagnostics()[0].task.queries[0]
        started = time.monotonic()
        frame = Frame("current-frame", spec.object_id, 0, started, b"frozen", "test")
        worker = AsyncQueryWorker(blocking_inference)
        book = QueryBook()
        pending = worker.submit(spec, frame, "read visible state", time.monotonic(), book)
        try:
            self.assertTrue(await asyncio.get_running_loop().run_in_executor(None, entered.wait, 1))
            self.assertFalse(pending.done())
            self.assertIn(spec.id, book.pending)
            release.set()
            reply = await asyncio.wait_for(pending, timeout=2)
            self.assertTrue(book.accept(reply, time.monotonic()))
        finally:
            release.set()
            worker.close()
