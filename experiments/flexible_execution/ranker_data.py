"""Generate grouped, explicitly symbolic counterfactual ranking data."""

from dataclasses import asdict, replace
from hashlib import sha256
from itertools import permutations
from math import dist
from random import Random

from .candidate_ranking import features
from .evaluate import score
from .executor import forecast_async, permitted_observation, project, ready_times
from .fixtures import Diagnostic, QueryFixture
from .runner import _hash
from .search import witness
from .simulation import Episode, visible_state_hash
from .task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract


LABEL_MODE = "SYMBOLIC_CLOSED_LOOP_COUNTERFACTUAL"
BRANCHES = ("positive", "negative", "uncertain")
STATES_PER_GROUP = 4
SPLIT_GROUPS = {"train": (0, 2500), "calibration": (2500, 2548), "selection": (2548, 2596)}


def _rng(seed, group, state=None):
    material = f"{seed}/{group}/{state}".encode()
    return Random(int.from_bytes(sha256(material).digest()[:8], "big"))


def _instance(group, state, split_seed):
    scene = _rng(split_seed, group)
    conditions = _rng(split_seed, group, state)
    permissions = scene.choice(("early", "branch-specific", "early-with-expiry"))
    shared = scene.choice((True, False))
    query_id = "inspection"
    requirements = (
        Requirement("original_view", "observed-object", "primary_rgb"),
        Requirement("identity", "observed-object", "identity", Condition(query_id, ("positive",)), report=True),
        Requirement("review", "observed-object", "review_rgb", Condition(query_id, ("uncertain",)), report=True),
    )

    def permit(branch):
        return Condition(query_id, branch) if permissions == "branch-specific" else None

    expiry = float(scene.randint(3, 20)) if permissions == "early-with-expiry" else None
    skills = [
        Skill("identifier", "side", ("identity",), scene.uniform(.5, 3.), permit(("positive",)), expiry),
        Skill("review", "alternate", ("review",), scene.uniform(1., 5.), permit(("uncertain",)), expiry),
    ]
    if shared:
        skills.append(Skill("wide", "wide", ("identity", "review"), scene.uniform(2., 6.),
                            permit(("positive", "uncertain")), expiry))
    corners = {site: (scene.uniform(-12, 12), scene.uniform(-8, 8)) for site in
               ("initial", "side", "alternate", "wide", "terminal")}
    speed = scene.uniform(.85, 2.5)
    travel = tuple((a, b, dist(corners[a], corners[b]) / speed)
                   for a, b in permutations(corners, 2))
    predicted = conditions.uniform(2., 28.)
    max_ready = predicted + conditions.uniform(2., 12.)
    deadline = max_ready + conditions.uniform(9., 42.)
    query = QuerySpec(query_id, "observed-object", BRANCHES, predicted, max_ready)
    task = TaskContract("Inspect observed objects under the registered permission and finish at terminal.",
                        "initial", "terminal", deadline, 1., requirements, (query,), tuple(skills), travel)
    initial = [("original_view", None)]
    if state == 1:
        initial.append(("identity", None))
    elif state == 2:
        initial.append(("identity", conditions.uniform(2., 12.)))
    duration = conditions.uniform(max(.01, predicted - 4), max_ready)
    fault = ("wide",) if shared and state == 3 else ()
    shape = {"operator": "conditional_identity_and_review", "permission": permissions,
             "shared_observation": shared, "skill_sites": [skill.site for skill in skills]}
    scene_data = {"layout": corners, "structure": shape, "speed": speed}
    values = {requirement.id: requirement.field + "/" + str(group) for requirement in requirements}
    record = Diagnostic(f"synthetic-{group:04d}-{state}", LABEL_MODE, None, task,
                        (QueryFixture(query_id, duration, "uncertain"),), tuple(initial), values, fault)
    return record, _hash(scene_data), _hash(shape)


class _CandidateRollout(Episode):
    def __init__(self, case, prefix):
        super().__init__(case, "ASYNC")
        self.prefix = prefix

    def initialize(self):
        super().initialize()
        for skill in self.prefix:
            if not self.book.pending:
                break
            action = permitted_observation(self.task, self.snapshot(), skill)
            if action is None:
                break
            self.act(action, "candidate_before_pending_fallback")


def _rollouts(case, prefix):
    results = []
    for branch in BRANCHES:
        matched = replace(case, query_truth=(replace(case.query_truth[0], outcome=branch),))
        ledger = _CandidateRollout(matched, prefix).run()
        judged = score(matched, ledger)
        results.append({"branch": branch, "success": judged["success"],
                        "restricted_seconds": judged["restricted_completion_seconds"],
                        "wrong_accepted": judged["wrong_accepted"]})
    return results


def label(case, group_hash, structure_hash):
    environment = Episode(case, "ASYNC")
    environment.initialize()
    state = environment.snapshot()
    baseline = _rollouts(case, ())
    baseline_seconds = sum(row["restricted_seconds"] for row in baseline) / len(baseline)
    worlds = tuple(case.task.worlds(state.results))
    ready = ready_times(case.task, state)
    forecast = sum(forecast_async(case.task, state, result, ready) for result in worlds) / len(worlds)
    candidates = []
    for length in (1, 2, 3):
        for skills in permutations(case.task.skills, length):
            here = state
            actions = []
            for skill in skills:
                action = permitted_observation(case.task, here, skill)
                if action is None:
                    break
                actions.append(action)
                here = project(case.task, here, action)
            if len(actions) != length:
                continue
            proof = witness(case.task, state, tuple(actions))
            rollouts = _rollouts(case, skills) if proof else []
            successful = bool(rollouts and all(row["success"] and not row["wrong_accepted"] for row in rollouts))
            result_seconds = (sum(row["restricted_seconds"] for row in rollouts) / len(rollouts)
                              if rollouts else case.task.deadline)
            candidates.append({"candidate": [skill.id for skill in skills],
                               "features": features(case.task, state, actions),
                               "verified_common_continuation": bool(proof),
                               "success_label": int(successful),
                               "cost_gain_label": (baseline_seconds - result_seconds) / case.task.deadline,
                               "baseline_restricted_seconds": baseline_seconds,
                               "counterfactual_restricted_seconds": result_seconds,
                               "estimated_safe_gain": forecast - proof.expected_finish if proof else None,
                               "branches": rollouts})
    return {"id": case.id, "group_sha256": group_hash, "structure_sha256": structure_hash,
            "mode": LABEL_MODE, "visible_state_sha256": visible_state_hash(case.task, state),
            "protocol_input_sha256": _hash({"contract": asdict(case.task), "truth": asdict(case.query_truth[0])}),
            "baseline": baseline, "candidates": candidates}


def generate(split, count, split_seed):
    start, end = SPLIT_GROUPS[split]
    per_group = STATES_PER_GROUP if split == "train" else 1
    if type(count) is not int or count < 1 or count > (end - start) * per_group:
        raise ValueError("Requested state count exceeds independently assigned groups")
    for index in range(count):
        group = start + index // per_group
        state = index % per_group if split == "train" else group % STATES_PER_GROUP
        record, source, structure = _instance(group, state, split_seed)
        yield {"split": split, **label(record, source, structure)}
