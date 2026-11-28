"""Permission, measured costs and comparable choices for the live executor."""

import re
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import dist, isfinite
from time import monotonic

from .candidate_ranking import ONLINE_RANKED_METHODS, select_ranked
from .executor import forecast_async, permitted_observation, project, ready_times, report_time
from .gpt_vision import OUTCOMES
from .search import Candidate, Search, witness
from .task_ir import Condition


EARLY_GRANT = re.compile(r"采集规则：允许在判读结果返回前采集以下材料：([a-z][a-z0-9_]*(?:、[a-z][a-z0-9_]*)*)。")
LATE_GRANT = re.compile(
    r"采集规则：只有判读结果为 (positive|negative|uncertain) 才允许采集以下材料："
    r"([a-z][a-z0-9_]*(?:、[a-z][a-z0-9_]*)*)。")
RESTRICTION = re.compile(r"不得|禁止|仅在|才可|才允许|之后才能|之后方可|do not|must not|only after", re.I)
_SCOPE_PHOTO = re.compile(
    r"(?:本次任务)?(?:授权|允许)(?:机器人)?(?:对|在)(?:(?:本次任务|本次巡检|指定巡检)范围内的"
    r"(?:全部|所有|每个|各个)?(?:目标设备|巡检对象|设备)|(?:本次)?指定(?:设备|巡检对象))"
    r"(?:进行|开展)非接触(?:式)?巡检(?:并|和|及)(?:拍摄|拍照)(?:取证)?")
_DENIAL = re.compile(r"不允许|不准|不可|不能|不许|不得|禁止|请勿|别拍|不要|暂缓|除外|除非|未经|限制|"
                     r"只有|仅限|仅可|才(?:能|可|允许)|确认后|判定后|之后|之前不要|仅拍|只拍|勿拍|only after|must not|do not", re.I)
_REPORT_ONLY = re.compile(r"(?:只|仅)(?:需)?报告(?:异常|阳性|有裂纹)(?:的)?(?:设备|对象)(?:的)?编号")
_POSITIVE_ONLY_PHOTO = re.compile(
    r"(?:只有|仅在)?(?:确认|判定)(?:设备|对象)?(?:为)?(?:异常|阳性|有裂纹)(?:之)?后"
    r"(?:才(?:能|可|允许)|方可)?(?:拍摄|拍照|采集)(?:设备|对象)?(?:的)?(?:编号|本体编号)")
BODY_IDENTITY = re.compile(
    r"(?:编号|身份)[^。；\n]{0,24}(?:本体|本身)|(?:本体|本身)[^。；\n]{0,12}(?:编号|身份)|(?:body|asset) identity",
    re.I)
NEW_METHODS = ("conditional-serial", "conditional-async", "conditional-prefetch", "conditional-planner", "conditional-a")
STREAM_METHODS = ("conditional-stream", "conditional-stream-buffered")
CONDITIONAL_MODES = {name: "ACTUAL_GPT_" + name.upper().replace("-", "_")
                     for name in NEW_METHODS + ONLINE_RANKED_METHODS + STREAM_METHODS}


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
    """A report filter cannot grant observation. Require affirmative scoped authority."""
    lines = [line.strip() for line in instruction.strip().splitlines() if line.strip()]
    early = [line for line in lines if (EARLY_GRANT.fullmatch(line) and
             material.field in EARLY_GRANT.fullmatch(line)[1].split("、"))]
    late = [(line, LATE_GRANT.fullmatch(line)) for line in lines if LATE_GRANT.fullmatch(line)]
    explicit_late = [(line, match[1]) for line, match in late if material.field in match[2].split("、")]
    clauses = [s.strip() for s in re.split(r"[。；;，,\n]+", instruction) if s.strip()]
    denials = [s for s in clauses if re.search(r"禁止|不得|不许|不可|不允许|不要|do not|must not", s, re.I)]
    if denials:
        return Permission(False, (), denials[0])
    if explicit_late:
        outcomes = tuple(outcome for outcome in material.needed_when
                         if all(outcome == permitted for _, permitted in explicit_late))
        return Permission(False, outcomes, " ".join(line for line, _ in explicit_late))
    if material.field == "identity":
        late_photo = next((s for s in clauses if _POSITIVE_ONLY_PHOTO.fullmatch(s)), None)
        if late_photo:
            return Permission(False, tuple(o for o in material.needed_when if o == "positive"), late_photo)
    other = [s for s in clauses if not _REPORT_ONLY.fullmatch(s) and not EARLY_GRANT.fullmatch(s + "。")]
    restrictions = [s for s in other if _DENIAL.search(s) or RESTRICTION.search(s)]
    if restrictions:
        return Permission(False, (), "Unresolved collection restriction: " + restrictions[0])
    if len(early) == 1:
        return Permission(True, tuple(OUTCOMES), early[0])
    scoped = next((s for s in clauses if _SCOPE_PHOTO.fullmatch(s)), None)
    if scoped and material.field == "identity" and any(_REPORT_ONLY.fullmatch(s) for s in clauses):
        return Permission(True, tuple(OUTCOMES), scoped)
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


def check_branchwise(task, state, prefix):
    """Require feasible continuations with the same terminal and permissions in every branch."""
    projected = state
    for action in prefix:
        if action.skill is None or permitted_observation(task, projected, action.skill) != action:
            return None, []
        projected = project(task, projected, action)
    expected = ready_times(task, state)
    worst = ready_times(task, state, worst_case=True)
    branches = []
    for world in task.worlds(state.results):
        finish = forecast_async(task, projected, world, worst)
        if finish >= task.deadline:
            return None, []
        branches.append({"outcomes": world, "required_materials": [r.id for r in task.requirements if r.applies(world)],
                         "terminal": task.terminal, "latest_finish": finish,
                         "expected_finish": forecast_async(task, projected, world, expected)})
    finish = sum(branch["expected_finish"] for branch in branches) / len(branches)
    return Candidate(tuple(prefix), None, (), finish), branches


def continuation_check(method):
    return check_branchwise if method in ("conditional-planner", "conditional-prefetch", "adaptive-progress") else checked


def progress_choice(task, state, log, *, budget_ms, max_candidates):
    """Take one useful step; each possible reply must still admit a full continuation."""
    started = monotonic()
    worlds = tuple(task.worlds(state.results))
    ready = ready_times(task, state)
    baseline = sum(forecast_async(task, state, world, ready) for world in worlds) / len(worlds)
    missing = {r.id for r in task.requirements if not state.supports(r)
               and any(r.applies(world) for world in worlds)}
    actions = [permitted_observation(task, state, skill) for skill in task.skills if missing.intersection(skill.covers)]
    actions = sorted((a for a in actions if a), key=lambda a: (a.seconds, -len(missing.intersection(a.skill.covers)), a.skill.id))
    best = None
    checked_count = 0
    for action in actions[:max_candidates]:
        if (monotonic() - started) * 1000 >= budget_ms:
            break
        proof, branches = check_branchwise(task, state, (action,))
        checked_count += 1
        log("candidate_checked", candidate=action.skill.id, branches=branches, feasible=bool(proof))
        if proof and proof.expected_finish < (best.expected_finish if best else baseline):
            best = proof
    overhead = monotonic() - started
    log("candidate_search_summary", baseline_finish=baseline, expanded=checked_count, wall_seconds=overhead)
    return best if best and best.expected_finish + overhead < baseline else None


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
            proof, branches = check_branchwise(task, state, prefix)
            log("candidate_checked", candidate=_prefix_id(prefix), branches=branches, feasible=bool(proof))
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


def select(task, state, method, log, *, budget_ms=1000, max_candidates=32, max_prepare=3, ranked_policy=None):
    for skill in task.skills:
        action = permitted_observation(task, state, skill)
        log("candidate_generated", candidate=skill.id, materials=list(skill.covers), site=skill.site,
            permission="all_pending_outcomes" if skill.permission is None else
            {"query": skill.permission.query_id, "after": list(skill.permission.outcomes)})
        log("candidate_checked", candidate=skill.id, admission=bool(action), check="permission_and_resources")
    if method in ("conditional-serial", "conditional-async", *STREAM_METHODS):
        log("candidate_rejected", reason="ordinary policy does not advance branch-specific material")
        return None
    if method == "adaptive-progress":
        choice = progress_choice(task, state, log, budget_ms=budget_ms, max_candidates=max_candidates)
    elif method in ONLINE_RANKED_METHODS:
        if ranked_policy is None:
            raise ValueError("The learned comparison needs its frozen ranking profile")
        choice = select_ranked(task, state, checked, log, ranker=ranked_policy.ranker,
                               check_cap=min(max_candidates, ranked_policy.check_cap), budget_ms=budget_ms,
                               max_prepare=max_prepare, minimum_net_gain=ranked_policy.minimum_net_gain)
    elif method == "conditional-planner":
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
    proof, branches = continuation_check(method)(task, state, choice.prepare)
    key = _prefix_id(choice.prepare)
    log("candidate_checked", candidate=key, branches=branches, feasible=bool(proof),
        common=bool(proof and proof.continuation))
    if not proof:
        log("candidate_rejected", candidate=key, reason="permitted continuation, branch materials or time unproved")
        return None
    log("candidate_selected", candidate=key, expected_finish=proof.expected_finish,
        continuation=proof.continuation.site if proof.continuation else None,
        terminal=task.terminal, required_materials=list(proof.required_evidence))
    return proof
