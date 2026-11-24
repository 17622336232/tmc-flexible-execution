"""Permission, measured costs and comparable choices for the live executor."""

import re
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import dist, isfinite
from time import monotonic

from .executor import forecast_async, permitted_observation, project, ready_times, report_time
from .gpt_vision import OUTCOMES
from .search import Search, witness
from .task_ir import Condition


EARLY_GRANT = re.compile(r"采集规则：允许在判读结果返回前采集以下材料：([a-z][a-z0-9_]*(?:、[a-z][a-z0-9_]*)*)。")
LATE_GRANT = re.compile(
    r"采集规则：只有判读结果为 (positive|negative|uncertain) 才允许采集以下材料："
    r"([a-z][a-z0-9_]*(?:、[a-z][a-z0-9_]*)*)。")
RESTRICTION = re.compile(r"不得|禁止|仅在|才可|才允许|之后才能|之后方可|do not|must not|only after", re.I)
BODY_IDENTITY = re.compile(r"(?:编号|身份)[^。；\n]{0,24}(?:本体|本身)|(?:body|asset) identity", re.I)
NEW_METHODS = ("conditional-serial", "conditional-async", "conditional-prefetch", "conditional-planner", "conditional-a")
CONDITIONAL_MODES = {name: "ACTUAL_GPT_" + name.upper().replace("-", "_") for name in NEW_METHODS}


def needs_body_identity(instruction):
    return bool(BODY_IDENTITY.search(instruction))


@dataclass(frozen=True)
class Permission:
    early: bool
    after: tuple
    source: str

    def for_query(self, query_id):
        return None if self.early else Condition(query_id, self.after)


def permission_for(instruction, material):
    """Accept a standalone controlled-language grant and fail closed on restrictions."""
    lines = [line.strip() for line in instruction.strip().splitlines() if line.strip()]
    early = [line for line in lines if (EARLY_GRANT.fullmatch(line) and
             material.field in EARLY_GRANT.fullmatch(line)[1].split("、"))]
    late = [(line, LATE_GRANT.fullmatch(line)) for line in lines if LATE_GRANT.fullmatch(line)]
    explicit_late = [(line, match[1]) for line, match in late if material.field in match[2].split("、")]
    if explicit_late:
        outcomes = tuple(outcome for outcome in material.needed_when
                         if all(outcome == permitted for _, permitted in explicit_late))
        return Permission(False, outcomes, " ".join(line for line, _ in explicit_late))
    if len(early) == 1 and not any(RESTRICTION.search(line) for line in lines if line != early[0]):
        return Permission(True, tuple(OUTCOMES), early[0])
    return Permission(False, material.needed_when, "No unambiguous early acquisition grant")


@dataclass(frozen=True)
class Costs:
    frozen_query_seconds: float
    frozen_query_ceiling_seconds: float
    move_start_seconds: float
    move_seconds_per_meter: float
    observation_seconds: float
    remaining_controller_seconds: float
    calibration_id: str

    def __post_init__(self):
        terms = (self.frozen_query_seconds, self.frozen_query_ceiling_seconds, self.move_start_seconds,
                 self.move_seconds_per_meter, self.observation_seconds, self.remaining_controller_seconds)
        if not self.calibration_id or any(not isfinite(value) or value <= 0 for value in terms):
            raise ValueError("Live costs must have positive finite measurements and recorded provenance")
        if self.frozen_query_ceiling_seconds < self.frozen_query_seconds:
            raise ValueError("Query time ceiling cannot be below its prediction")

    def move_seconds(self, origin, destination):
        def xyz(pose):
            return tuple(pose[axis] for axis in ("x", "y", "z"))
        return self.move_start_seconds + self.move_seconds_per_meter * dist(xyz(origin), xyz(destination))


def _prefix_id(prefix):
    return "|".join(action.skill.id for action in prefix)


def checked(task, state, prefix):
    """Validate all possible outcomes at a common approved handoff location."""
    proof = witness(task, state, tuple(prefix), anchor_readiness=True)
    if proof is None:
        return None, []
    predicted = state
    for action in prefix:
        predicted = project(task, predicted, action)
    arrival = predicted.at + proof.continuation.seconds
    finish = report_time(task, predicted, arrival, worst_case=True)
    branches = []
    for world in task.worlds(state.results):
        requirements = [requirement for requirement in task.requirements if requirement.applies(world)]
        if finish > task.deadline or any(not predicted.supports(requirement, finish) for requirement in requirements):
            return None, []
        branches.append({"outcomes": world, "required_materials": [r.id for r in requirements],
                         "common_site": task.terminal, "latest_finish": finish,
                         "remaining_controller_seconds": task.report_seconds})
    return proof, branches


def _planner(task, state, *, budget_ms, max_candidates, max_prepare, log):
    """Best-first search of allowed action sequences for the general planner arm."""
    start = monotonic()
    end = start + budget_ms / 1000
    worlds = tuple(task.worlds(state.results))
    expected = ready_times(task, state)
    baseline = sum(forecast_async(task, state, world, expected) for world in worlds) / len(worlds)
    frontier = [(state.at, 0, (), state)]
    order = count(1)
    examined = 0
    best = None
    while frontier and monotonic() < end and examined < max_candidates:
        _, _, prefix, current = heappop(frontier)
        if prefix:
            examined += 1
            proof, branches = checked(task, state, prefix)
            log("candidate_checked", candidate=_prefix_id(prefix), branches=branches, common=bool(proof))
            if proof and proof.expected_finish < (best.expected_finish if best else baseline):
                best = proof
        if len(prefix) < max_prepare:
            for skill in task.skills:
                if skill.id in {action.skill.id for action in prefix}:
                    continue
                action = permitted_observation(task, current, skill)
                if action:
                    after = project(task, current, action)
                    lower_bound = max([after.at] + list(expected.values())) + task.report_seconds
                    if lower_bound < (best.expected_finish if best else baseline):
                        heappush(frontier, (lower_bound, next(order), prefix + (action,), after))
    log("candidate_search_summary", baseline_finish=baseline, expanded=examined, exhausted=bool(frontier),
        wall_seconds=monotonic() - start)
    return best


def select(task, state, method, log, *, budget_ms=1000, max_candidates=32, max_prepare=3):
    for skill in task.skills:
        action = permitted_observation(task, state, skill)
        log("candidate_generated", candidate=skill.id, materials=list(skill.covers), site=skill.site,
            permission="all_pending_outcomes" if skill.permission is None else
            {"query": skill.permission.query_id, "after": list(skill.permission.outcomes)})
        log("candidate_checked", candidate=skill.id, admission=bool(action), check="permission_and_resources")
    if method in ("conditional-serial", "conditional-async"):
        log("candidate_rejected", reason="ordinary policy does not advance branch-specific material")
        return None
    if method == "conditional-planner":
        choice = _planner(task, state, budget_ms=budget_ms, max_candidates=max_candidates,
                          max_prepare=max_prepare, log=log)
    else:
        search = Search(task, max_candidates=max_candidates, max_prepare=max_prepare, budget_ms=budget_ms)
        started = monotonic()
        result = search.propose(state, shared_prefetch=method == "conditional-prefetch", anchor_readiness=True)
        choice = result.choice
        log("candidate_search_summary", baseline_finish=result.baseline_finish, expanded=result.expanded,
            witnessed=result.witnessed, exhausted=result.exhausted_budget,
            wall_seconds=monotonic() - started)
    if not choice:
        log("candidate_rejected", reason="no witnessed cost improvement")
        return None
    proof, branches = checked(task, state, choice.prepare)
    key = _prefix_id(choice.prepare)
    log("candidate_checked", candidate=key, branches=branches, common=bool(proof))
    if not proof:
        log("candidate_rejected", candidate=key, reason="common handoff, branch materials or time unproved")
        return None
    log("candidate_selected", candidate=key, expected_finish=proof.expected_finish,
        continuation=proof.continuation.site, required_materials=list(proof.required_evidence))
    return proof
