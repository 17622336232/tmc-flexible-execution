"""Small controlled worlds for development, with truth kept in the environment."""

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Optional, Tuple

from .task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract


MODE = "SYMBOLIC_MECHANISM_DIAGNOSTIC"
OUTCOMES = ("normal", "exception", "inconclusive")


@dataclass(frozen=True)
class QueryFixture:
    id: str
    ready_at: float
    outcome: str


@dataclass(frozen=True)
class Diagnostic:
    id: str
    mechanism: str
    legacy_family: str
    task: TaskContract
    query_truth: Tuple[QueryFixture, ...]
    initial_material: Tuple[Tuple[str, Optional[float]], ...]
    material_values: Dict[str, str]
    reject_first: Tuple[str, ...] = ()


def diagnostics() -> Tuple[Diagnostic, ...]:
    # The cases share the same scheduler. Their conditions, layout and
    # outcomes are explicit development fixtures rather than measured data.
    definitions = (
        ("independent_work", "crack_and_heat", "normal", 4, 1),
        ("independent_work", "large_inventory", "exception", 4, 1),
        ("legal_preparation", "crack_and_heat", "exception", 14, 1),
        ("legal_preparation", "blocked_exit", "inconclusive", 14, 2),
        ("cost_exceeds_gain", "large_inventory", "normal", 1, -9),
        ("cost_exceeds_gain", "blocked_exit", "exception", 2, -10),
        ("permission_requires_result", "crack_and_heat", "exception", 10, 1),
        ("permission_requires_result", "blocked_exit", "inconclusive", 10, 1),
        ("evidence_changes", "crack_and_heat", "exception", 13, 1),
        ("evidence_changes", "large_inventory", "inconclusive", 13, 1),
        ("no_opportunity", "inventory", "exception", 8, 1),
        ("no_opportunity", "locate_access", "normal", 0, 1),
    )
    return tuple(_case(number, *definition) for number, definition in enumerate(definitions, 1))


def _case(number, mechanism, family, outcome, delay, side):
    name, field = {
        "crack_and_heat": ("设备", "温度显示"),
        "blocked_exit": ("通道", "通道编号"),
        "locate_access": ("通道", "通道编号"),
        "large_inventory": ("货位", "货位编号"),
        "inventory": ("货位", "货位编号"),
    }[family]
    requirement = [
        Requirement("primary", "A", "inspection_image"),
        Requirement("detail", "A", field, Condition("query-A", ("exception",)), report=True),
        Requirement("review", "A", "review_image", Condition("query-A", ("inconclusive",)), report=True),
    ]
    sensitivity = None
    instruction = f"检查{name} A；异常时记录{field}，看不清时保留斜视图。完成现场检查后返回。"
    if mechanism == "permission_requires_result":
        sensitivity = Condition("query-A", ("exception", "inconclusive"))
        instruction += "等判读确认需要相应材料后再拍。"
    else:
        instruction += "允许提前拍所需材料。"
    skills = [
        Skill("detail_view", "side", ("detail",), 1.2, sensitivity),
        Skill("review_view", "front", ("review",), 3.0, sensitivity),
        Skill("shared_view", "side", ("detail", "review"), 2.0, sensitivity),
    ]
    if mechanism == "independent_work":
        requirement.append(Requirement("other", "B", "inspection_image"))
        skills.append(Skill("other_view", "other", ("other",), 3.0))
        instruction += f"同时检查{name} B。"
    initial = [("primary", None)]
    if mechanism == "evidence_changes" and number == 9:
        initial.append(("detail", 1.0))
    if mechanism == "no_opportunity" and number == 11:
        initial.extend((("detail", None), ("review", None)))
    sites = {"front": 0.0, "side": float(side), "other": 4.0, "finish": 12.0}
    travel = tuple((a, b, abs(sites[a] - sites[b])) for a, b in permutations(sites, 2))
    query = QuerySpec("query-A", "A", OUTCOMES, float(delay), float(delay + 1))
    task = TaskContract(instruction, "front", "finish", 60.0, 0.5,
                        tuple(requirement), (query,), tuple(skills), travel)
    return Diagnostic(
        f"mechanism-{number:02}", mechanism, family, task,
        (QueryFixture(query.id, float(delay), outcome),), tuple(initial),
        {item.id: f"{item.object_id}/{item.field}/fixture" for item in requirement},
        ("shared_view",) if mechanism == "evidence_changes" and number == 10 else (),
    )
