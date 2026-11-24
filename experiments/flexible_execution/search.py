"""Budgeted requirement-to-skill search for a witnessed common return."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from time import monotonic
from typing import Optional, Tuple

from .executor import (Action, Snapshot, forecast_async, permitted_observation,
                       project, ready_times, report_time, terminal_action)
from .task_ir import TaskContract


@dataclass(frozen=True)
class Candidate:
    prepare: Tuple[Action, ...]
    continuation: Optional[Action]
    required_evidence: Tuple[str, ...]
    expected_finish: float


@dataclass(frozen=True)
class SearchResult:
    choice: Optional[Candidate]
    baseline_finish: float
    expanded: int
    witnessed: int
    exhausted_budget: bool


def witness(task: TaskContract, state: Snapshot, prepare: Tuple[Action, ...], *,
            anchor_readiness=False) -> Optional[Candidate]:
    """Prove that one common terminal move works after the expected evidence."""
    initial = state
    for proposed in prepare:
        if proposed.skill is None:
            return None
        action = permitted_observation(task, state, proposed.skill)
        if action is None or action != proposed:
            return None
        state = project(task, state, action)
    continuation = terminal_action(task, state)
    if continuation is None:
        return None
    worst_finish = report_time(task, state, state.at + continuation.seconds, worst_case=True)
    worlds = tuple(task.worlds(state.results))
    required = tuple(
        requirement.id for requirement in task.requirements
        if not initial.supports(requirement, worst_finish)
        and any(requirement.applies(world) for world in worlds)
    )
    clock = initial if anchor_readiness else state
    return Candidate(prepare, continuation, required,
                     report_time(task, clock, state.at + continuation.seconds))


class Search:
    def __init__(self, task: TaskContract, *, max_candidates=32, max_prepare=3, budget_ms=1000):
        self.task = task
        self.max_candidates = max_candidates
        self.max_prepare = max_prepare
        self.budget_ms = budget_ms
        self.skill_order = {skill.id: index for index, skill in enumerate(task.skills)}
        self.skill_index = {
            requirement.id: tuple(skill for skill in task.skills if requirement.id in skill.covers)
            for requirement in task.requirements
        }

    def propose(self, state: Snapshot, *, shared_prefetch=False, anchor_readiness=False) -> SearchResult:
        end = monotonic() + self.budget_ms / 1000
        worlds = tuple(self.task.worlds(state.results))
        estimate = ready_times(self.task, state)
        worst = ready_times(self.task, state, worst_case=True)
        baseline = sum(forecast_async(self.task, state, world, estimate)
                       for world in worlds) / len(worlds)
        relevant = tuple(
            item for item in self.task.requirements
            if any(item.applies(world) for world in worlds)
            and not state.supports(item, report_time(self.task, state, state.at, worst_case=True))
        )
        if shared_prefetch:
            # Shared branch prefetch uses the union of required materials.
            desired = {item.id for item in relevant}
            skills = [skill for skill in self.task.skills if desired.intersection(skill.covers)]
        else:
            pool = {skill for item in relevant for skill in self.skill_index[item.id]}
            skills = sorted(pool, key=lambda skill: self.skill_order[skill.id])
        serial = count()
        queue = [(state.at, next(serial), state, ())]
        expanded = witnessed = 0
        best = None
        while queue and expanded < self.max_candidates and monotonic() < end:
            _, _, current, prefix = heappop(queue)
            if prefix:
                expanded += 1
                if shared_prefetch and all(
                    forecast_async(self.task, current, world, worst) < self.task.deadline
                    for world in worlds
                ):
                    finish = sum(forecast_async(self.task, current, world, estimate)
                                 for world in worlds) / len(worlds)
                    candidate = Candidate(prefix, None, (), finish)
                else:
                    candidate = None if shared_prefetch else witness(self.task, state, prefix,
                                                                      anchor_readiness=anchor_readiness)
                if candidate is not None:
                    witnessed += 1
                    if (candidate.expected_finish < baseline
                            and (best is None or candidate.expected_finish < best.expected_finish)):
                        best = candidate
            if len(prefix) == self.max_prepare:
                continue
            for skill in skills:
                if any(action.skill == skill for action in prefix):
                    continue
                proof_time = report_time(self.task, current, current.at, worst_case=True)
                if not any(item.id in skill.covers and not current.supports(item, proof_time)
                           for item in relevant):
                    continue
                action = permitted_observation(self.task, current, skill)
                if action is not None:
                    after = project(self.task, current, action)
                    heappush(queue, (after.at, next(serial), after, prefix + (action,)))
        return SearchResult(best, baseline, expanded, witnessed, bool(queue))
