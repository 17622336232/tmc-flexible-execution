"""Discover a bounded visual-presence workflow from prose on the original GPT path.

The user owns the permission and goal. Text from the model never adds either.
Instructions outside the supported grammar fall back to the original controller.
"""
from dataclasses import dataclass
from math import radians
import re

from .ordinary_frontier import camera_goals, pose_matches


TURN = re.compile(r"原地向(?P<direction>[左右])转(?P<angle>[1-9][0-9]{0,2})度(?:后|的朝向)?(?:是|作为|为)侧面")
SUBJECT = re.compile(
    r"(?P<view>正面|侧面)(?:照片)?(?:里|中|上)?(?:是否(?:能)?(?:看见|看到|有)|能否(?:看见|看到)|"
    r"有没有|有无|(?:能)?(?:看见|看到)|有)(?P<name>[^，。,；、：？！?\s]{1,28})"
)
ASK = re.compile(r"^(?:请)?(?:也|再|分别)?(?:看看|检查|确认|判断|报告|核实)")
EARLY = re.compile(r"(?P<denial>不)?(?:可以|允许|准许)(?:在等待结果时|在等待时|提前|先)(?:提前)?(?:采集|拍摄|保存)侧面(?:原图|照片)")
REPORT = re.compile(r"(?:请)?(?:报告每项的实际照片来源|每项附实拍照片|看不清就说无法判定|分别报告各项结果|用物品名称作记录标识，布尔字段用 visible)")


@dataclass(frozen=True)
class VisionField:
    name: str
    view: str
    any_of: tuple  # A tuple of AND-clause tuples of previous field indices.


@dataclass(frozen=True)
class NaturalFrontier:
    direction: str
    angle_degrees: int
    finish: str
    early_capture: bool
    fields: tuple


def _field(text):
    value = SUBJECT.fullmatch(ASK.sub("", text, count=1))
    if not value or any(mark in value["name"] for mark in ("或者", "并且", "以及", "同时", "否则", "不见", "没有")):
        return None
    return ("front" if value["view"] == "正面" else "side", value["name"])


def discover(instruction):
    """Accept explicit visual conditions in supported wording, otherwise keep native behavior."""
    if not isinstance(instruction, str) or len(instruction)>2200:
        return None
    clauses = [c.strip() for c in re.split(r"[。；！!?？\n]", instruction) if c.strip()]
    finish = turn = early = None
    fields, indices = [], {}
    for c in clauses:
        if c in ("正面是起始朝向", "正面指起始朝向"):
            continue
        if TURN.fullmatch(c):
            if turn is not None:
                return None
            turn = TURN.fullmatch(c).groupdict()
            continue
        if c in ("最后恢复起始朝向", "结束时恢复起始朝向", "最后停在侧面", "结束时停在侧面"):
            if finish is not None:
                return None
            finish = "front" if "恢复" in c else "side"
            continue
        grant=EARLY.fullmatch(c)
        if grant:
            if early is not None:
                return None
            early = grant["denial"] is None
            continue
        if REPORT.fullmatch(c):
            continue
        gate = ((),)
        if c.startswith(("若","如果")):
            condition, comma, c = (c[2:] if c.startswith("如果") else c[1:]).partition("，")
            if not comma:
                return None
            c = c[1:] if c.startswith(("就","则","才")) else c
            rows = []
            for branch in condition.split("或者"):
                fields_in_branch = [_field(term) for term in branch.split("并且")]
                if any(f not in indices for f in fields_in_branch):
                    return None
                rows.append(tuple(sorted({indices[f] for f in fields_in_branch})))
            gate = tuple(sorted(set(rows)))
        target = _field(c) if ASK.match(c) else None
        if target is None or target in indices:
            return None
        indices[target] = len(fields)
        fields.append(VisionField(target[1], target[0], gate))
    if turn is None or not 0<int(turn["angle"])<=180 or finish is None or len(fields)<2 or not any(f.view=="side" for f in fields):
        return None
    return NaturalFrontier("left" if turn["direction"]=="左" else "right",int(turn["angle"]),finish,early is True,tuple(fields))


def released(plan, previously_observed):
    return {i for i,f in enumerate(plan.fields) if i not in previously_observed
            and any(all(previously_observed.get(parent) is True for parent in clause) for clause in f.any_of)}


def source_map(plan, anchor, supplied):
    goals = camera_goals(dict(direction=plan.direction,angle_degrees=plan.angle_degrees),anchor["position"],anchor["yaw"])
    result = {}
    for source in supplied:
        pose = {**source["position"],"yaw":source["yaw"]}
        matches = [view for view in goals if pose_matches(pose,goals[view],metres=.10,radians_tol=.10)]
        if len(matches)==1:
            result[source["image_id"]] = matches[0]
    return result


def read_ordinary_records(plan, reply, prior, sources):
    """Accept evidence only from the attached camera bound to that named natural-language field."""
    new, rejected = {}, set()
    legal = released(plan, prior)
    rows = reply.get("records")
    if not isinstance(rows,list):
        return {}
    for rec in rows:
        if not isinstance(rec,dict) or not isinstance(rec.get("facts"),dict):
            continue
        name, image_id = rec.get("object_id"), rec.get("evidence_image_id")
        if not isinstance(image_id,str):
            continue
        view = sources.get(image_id)
        value = rec["facts"].get("visible")
        if type(value) is not bool:
            continue
        for i, f in enumerate(plan.fields):
            if f.name!=name or f.view!=view or i not in legal:
                continue
            if i in new and new[i]!=value:
                rejected.add(i)
            new[i]=value
    return {i:v for i,v in new.items() if i not in rejected}
