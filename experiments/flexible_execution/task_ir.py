"""Visible task requirements and the bounded skills advertised by a platform."""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Condition:
    query_id: str
    outcomes: Tuple[str, ...]

    def holds(self, results: Dict[str, str]) -> bool:
        return results.get(self.query_id) in self.outcomes


@dataclass(frozen=True)
class Requirement:
    id: str
    object_id: str
    field: str
    when: Optional[Condition] = None
    due: Optional[float] = None
    report: bool = False

    def applies(self, results: Dict[str, str]) -> bool:
        return self.when is None or self.when.holds(results)


@dataclass(frozen=True)
class QuerySpec:
    id: str
    object_id: str
    outcomes: Tuple[str, ...]
    estimated_ready: float
    latest_ready: float


@dataclass(frozen=True)
class Skill:
    id: str
    site: str
    covers: Tuple[str, ...]
    seconds: float
    permission: Optional[Condition] = None
    fresh_for: Optional[float] = None
    attempts: int = 2

    def permitted(self, results: Dict[str, str]) -> bool:
        return self.permission is None or self.permission.holds(results)


@dataclass(frozen=True)
class TaskContract:
    instruction: str
    start: str
    terminal: str
    deadline: float
    report_seconds: float
    requirements: Tuple[Requirement, ...]
    queries: Tuple[QuerySpec, ...]
    skills: Tuple[Skill, ...]
    travel: Tuple[Tuple[str, str, float], ...]

    def __post_init__(self):
        requirements = {item.id: item for item in self.requirements}
        queries = {item.id: item for item in self.queries}
        if (len(requirements) != len(self.requirements) or len(queries) != len(self.queries)
                or len({skill.id for skill in self.skills}) != len(self.skills)):
            raise ValueError("Requirement, query and skill IDs must be unique")
        if not isfinite(self.deadline) or self.deadline <= 0:
            raise ValueError("The episode must have a finite positive deadline")
        if not isfinite(self.report_seconds) or self.report_seconds < 0:
            raise ValueError("Reporting duration must be finite and nonnegative")
        worlds = 1
        for query in self.queries:
            if (not query.outcomes or len(set(query.outcomes)) != len(query.outcomes)
                    or not 0 <= query.estimated_ready <= query.latest_ready <= self.deadline):
                raise ValueError("Invalid outcome or timing contract for " + query.id)
            worlds *= len(query.outcomes)
        if worlds > 64:
            raise ValueError("The diagnostic supports at most 64 compatible outcome combinations")
        for requirement in self.requirements:
            if requirement.due is not None and not 0 <= requirement.due <= self.deadline:
                raise ValueError("Invalid acquisition deadline for " + requirement.id)
        rules = [item.when for item in self.requirements]
        for skill in self.skills:
            rules.append(skill.permission)
            if (not skill.covers or not set(skill.covers) <= requirements.keys()
                    or not isfinite(skill.seconds) or skill.seconds <= 0 or skill.attempts < 1
                    or (skill.fresh_for is not None and
                        (not isfinite(skill.fresh_for) or skill.fresh_for <= 0))):
                raise ValueError("Invalid observation skill " + skill.id)
        for rule in rules:
            if rule and (rule.query_id not in queries
                         or not set(rule.outcomes) <= set(queries[rule.query_id].outcomes)):
                raise ValueError("Condition refers to an undeclared outcome")
        routes = {(origin, target): seconds for origin, target, seconds in self.travel}
        if len(routes) != len(self.travel):
            raise ValueError("Travel routes must be unique")
        for origin, target, seconds in self.travel:
            if origin == target or not isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid travel duration")

    def travel_seconds(self, origin: str, target: str) -> Optional[float]:
        if origin == target:
            return 0.0
        return next((seconds for start, end, seconds in self.travel
                     if (start, end) == (origin, target)), None)

    def worlds(self, results: Dict[str, str]) -> Iterable[Dict[str, str]]:
        pending = [query for query in self.queries if query.id not in results]
        for outcomes in product(*(query.outcomes for query in pending)):
            yield {**results, **dict(zip((query.id for query in pending), outcomes))}
