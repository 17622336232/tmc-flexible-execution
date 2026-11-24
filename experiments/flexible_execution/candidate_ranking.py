"""Rank action prefixes before calling the expensive common continuation checker."""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Optional

from .executor import forecast_async, permitted_observation, project, ready_times


RANKED_METHODS = ("conditional-rank-heuristic", "conditional-rank-ordinary", "conditional-rank-b1",
                  "conditional-rank-c")


def features(task, snapshot, prefix):
    current = snapshot
    requested = set()
    duration = 0.
    for action in prefix:
        requested.update(action.skill.covers)
        duration += action.seconds
        current = project(task, current, action)
    query = next((query for query in task.queries if query.id not in snapshot.results), task.queries[0])
    distance = task.travel_seconds(current.site, task.terminal)
    return [
        float(len(prefix)), duration / task.deadline, len(requested) / len(task.requirements),
        len(snapshot.evidence) / len(task.requirements), (task.deadline - snapshot.at) / task.deadline,
        (query.estimated_ready - snapshot.at) / task.deadline, (query.latest_ready - snapshot.at) / task.deadline,
        sum(action.skill.fresh_for is not None for action in prefix) / len(prefix),
        sum(action.skill.permission is None for action in prefix) / len(prefix),
        (task.deadline if distance is None else distance) / task.deadline,
    ]


@dataclass(frozen=True)
class LinearRanker:
    objective: str
    offset: tuple
    scale: tuple
    weights: tuple
    digest: str

    @classmethod
    def load(cls, path, objective):
        raw = path.read_bytes()
        row = json.loads(raw)
        offset, scale, weights = (tuple(row[key]) for key in ("offset", "scale", "weights"))
        if (row["objective"] != objective or objective not in ("ordinary", "cost")
                or len(offset) != 10 or len(scale) != 10 or len(weights) != 11
                or any(type(x) not in (int, float) or not isfinite(x) for x in (*offset, *scale, *weights))
                or any(x <= 0 for x in scale)):
            raise ValueError("The checkpoint does not match the candidate feature format")
        return cls(objective, offset, scale, weights, sha256(raw).hexdigest())

    def score(self, row):
        return self.weights[-1] + sum((value - offset) * weight / scale
            for value, offset, scale, weight in zip(row, self.offset, self.scale, self.weights[:-1]))


@dataclass(frozen=True)
class RankedPolicy:
    ranker: Optional[LinearRanker]
    check_cap: int
    minimum_net_gain: Optional[float]

    @classmethod
    def load(cls, row, method):
        objective = {RANKED_METHODS[0]: None, RANKED_METHODS[1]: "ordinary",
                     RANKED_METHODS[2]: "cost", RANKED_METHODS[3]: "cost"}[method]
        cap = row["check_cap"]
        minimum = row["minimum_net_gain"]
        if (type(cap) is not int or not 1 <= cap <= 32
                or (minimum is not None and (type(minimum) not in (float, int)
                                             or not isfinite(minimum) or minimum < 0))
                or (minimum is not None) != (method == RANKED_METHODS[-1])):
            raise ValueError("The ranker check budget does not match the registered method")
        checkpoint = row["checkpoint"]
        if bool(checkpoint) != bool(objective):
            raise ValueError("The registered ranking method needs its corresponding checkpoint")
        ranker = LinearRanker.load(Path(checkpoint), objective) if checkpoint else None
        if (None if ranker is None else ranker.digest) != row["checkpoint_sha256"]:
            raise ValueError("The registered ranking checkpoint digest changed")
        return cls(ranker, cap, minimum)


def _prefixes(task, state, skills, maximum_length, maximum_generated, deadline):
    frontier = [(state, ())]
    candidates = []
    for current, prefix in frontier:
        if len(prefix) >= maximum_length:
            continue
        for skill in skills:
            if len(candidates) >= maximum_generated or monotonic() >= deadline:
                return candidates
            if skill in {action.skill for action in prefix}:
                continue
            action = permitted_observation(task, current, skill)
            if action is not None:
                extended = prefix + (action,)
                candidates.append(extended)
                frontier.append((project(task, current, action), extended))
    return candidates


def select_ranked(task, state, check, log, *, ranker, check_cap, budget_ms, max_prepare,
                  minimum_net_gain=None, generated_cap=256):
    start = monotonic()
    deadline = start + budget_ms / 1000
    worlds = tuple(task.worlds(state.results))
    needed = {requirement.id for requirement in task.requirements
              if not state.supports(requirement) and any(requirement.applies(world) for world in worlds)}
    eligible = tuple(skill for skill in task.skills if needed.intersection(skill.covers))
    ranked = []
    for prefix in _prefixes(task, state, eligible, max_prepare, generated_cap, deadline):
        row = features(task, state, prefix)
        score = -row[1] if ranker is None else ranker.score(row)
        key = tuple(action.skill.id for action in prefix)
        ranked.append((score, key, prefix))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    readiness = ready_times(task, state)
    baseline = sum(forecast_async(task, state, world, readiness) for world in worlds) / len(worlds)
    best = None
    checked = 0
    check_seconds = 0.
    stop = "candidate_pool_exhausted"
    for score, key, prefix in ranked:
        if checked == check_cap or monotonic() >= deadline:
            stop = "check_or_wall_budget"
            break
        if minimum_net_gain is not None and checked:
            known_gain = 0. if best is None else (baseline - best.expected_finish) / task.deadline
            if score - known_gain <= minimum_net_gain + check_seconds / checked / task.deadline:
                stop = "estimated_increment_le_cost"
                break
        started_check = monotonic()
        proof, branches = check(task, state, prefix)
        checked += 1
        check_seconds += monotonic() - started_check
        log("candidate_checked", candidate="|".join(key), search_order=checked, rank_score=score,
            branches=branches, common=bool(proof))
        if proof is not None and proof.expected_finish < (baseline if best is None else best.expected_finish):
            best = proof
    log("candidate_search_summary", baseline_finish=baseline, generated=len(ranked), checked=checked,
        check_seconds=check_seconds, wall_seconds=monotonic() - start, stop_reason=stop,
        ranker="travel_cost" if ranker is None else ranker.objective,
        checkpoint=None if ranker is None else ranker.digest)
    return best
