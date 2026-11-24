"""Shared task scheduler for the first bounded observation experiments."""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from .evidence import Evidence
from .task_ir import Requirement, Skill, TaskContract


@dataclass(frozen=True)
class Action:
    site: str
    seconds: float
    skill: Optional[Skill] = None


@dataclass(frozen=True)
class Snapshot:
    at: float
    site: str
    results: Dict[str, str]
    evidence: Dict[str, Evidence]
    attempts: Dict[str, int]

    def supports(self, requirement: Requirement, at: Optional[float] = None) -> bool:
        material = self.evidence.get(requirement.id)
        return material is not None and material.usable(requirement, self.at if at is None else at)


def permitted_observation(task: TaskContract, state: Snapshot, skill: Skill) -> Optional[Action]:
    trip = task.travel_seconds(state.site, skill.site)
    if (trip is None or not skill.permitted(state.results)
            or state.attempts.get(skill.id, 0) >= skill.attempts
            or state.at + trip + skill.seconds + task.report_seconds > task.deadline):
        return None
    return Action(skill.site, trip + skill.seconds, skill)


def terminal_action(task: TaskContract, state: Snapshot) -> Optional[Action]:
    if state.site == task.terminal:
        return None
    trip = task.travel_seconds(state.site, task.terminal)
    if trip is None:
        return None
    finish = report_time(task, state, state.at + trip, worst_case=True)
    if finish > task.deadline:
        return None
    worlds = tuple(task.worlds(state.results))
    for requirement in task.requirements:
        if any(requirement.applies(world) for world in worlds) and not state.supports(requirement, finish):
            return None
    return Action(task.terminal, trip)


def report_time(task: TaskContract, state: Snapshot, arrival: float, *, worst_case=False) -> float:
    outstanding = [query for query in task.queries if query.id not in state.results]
    release = [
        query.latest_ready if worst_case else expected_ready(query.estimated_ready, query.latest_ready, state.at)
        for query in outstanding
    ]
    return max([arrival] + release) + task.report_seconds


def expected_ready(estimate: float, latest: float, at: float) -> float:
    if estimate > at:
        return estimate
    return max(at, (at + latest) / 2)


def async_action(task: TaskContract, state: Snapshot) -> Optional[Action]:
    if state.site == task.terminal:
        return None
    worlds = tuple(task.worlds(state.results))
    missing = {
        requirement.id: requirement for requirement in task.requirements
        if not state.supports(requirement) and all(requirement.applies(world) for world in worlds)
    }
    options = []
    for skill in task.skills:
        action = permitted_observation(task, state, skill)
        if action is None:
            continue
        due = [
            missing[key].due if missing[key].due is not None else task.deadline
            for key in skill.covers if key in missing
            and (missing[key].due is None or state.at + action.seconds <= missing[key].due)
        ]
        if due:
            options.append((min(due), action.seconds, -len(due), action))
    if options:
        return min(options, key=lambda item: item[:3])[3]
    return terminal_action(task, state)


def project(task: TaskContract, state: Snapshot, action: Action) -> Snapshot:
    """An optimistic forecast. The executor never commits this projected evidence."""
    at = state.at + action.seconds
    evidence = dict(state.evidence)
    attempts = dict(state.attempts)
    if action.skill:
        skill = action.skill
        attempts[skill.id] = attempts.get(skill.id, 0) + 1
        expiry = at + skill.fresh_for if skill.fresh_for is not None else None
        for requirement in task.requirements:
            if requirement.id in skill.covers and (requirement.due is None or at <= requirement.due):
                evidence[requirement.id] = Evidence(
                    requirement.id, requirement.object_id, requirement.field, "forecast",
                    "", at, "", "predicted_support", expiry,
                )
    return replace(state, at=at, site=action.site, evidence=evidence, attempts=attempts)


def ready_times(task: TaskContract, state: Snapshot, *, worst_case=False) -> Dict[str, float]:
    return {
        query.id: (query.latest_ready if worst_case else
                   expected_ready(query.estimated_ready, query.latest_ready, state.at))
        for query in task.queries if query.id not in state.results
    }


def forecast_async(task: TaskContract, start: Snapshot, outcome: Dict[str, str],
                   ready: Dict[str, float]) -> float:
    """Follow the same ordinary scheduler on one estimated future, revealing it over time."""
    state = start
    while state.at <= task.deadline:
        arrived = {key: outcome[key] for key, when in ready.items() if when <= state.at}
        state = replace(state, results={**state.results, **arrived})
        if state.site == task.terminal and len(state.results) == len(task.queries):
            finish = state.at + task.report_seconds
            if all(state.supports(requirement, finish) for requirement in task.requirements
                   if requirement.applies(state.results)):
                return min(finish, task.deadline)
            return task.deadline
        action = async_action(task, state)
        if action is not None:
            state = project(task, state, action)
            continue
        future = [when for key, when in ready.items() if key not in state.results]
        if not future:
            return task.deadline
        state = replace(state, at=min(future))
    return task.deadline
