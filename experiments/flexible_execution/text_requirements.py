"""Cold compilation of a small explicit inspection language.

Only the complete authorized string is input. Unknown clauses, fields or scopes
are handed back to the model. This module never supplies readings or actions.
"""
import re

from .gpt_vision import InspectionTask, Material
from .live_planning import EARLY_GRANT, LATE_GRANT
from .material_schema import INSPECTION_FIELDS
from .requirement_program import parse


FIELDS = {
    "cracked": ("是否有裂纹", "裂纹"),
    "blocked": ("是否被实物堵住", "通行情况"),
    "extinguisher_present": ("是否配备灭火器", "是否配置灭火器"),
    "temperature": ("摄氏温度", "温度"),
    "box_count": ("箱数", "实际箱数", "实际货箱数量", "实物箱数"),
    "order_count": ("订单箱数",),
    "color": ("货箱颜色", "英文颜色名"),
}
NOUN = {"设备": "设备", "仓储货位": "货位", "校园通道": "通道", "校园消防点": "点位", "校园点位": "点位"}
PREDICATES = {
    "存在裂纹": ("cracked", True, "该设备的正面外壳是否存在裂纹？"),
    "被实物堵住": ("blocked", True, "该通道是否被实物堵住？"),
    "没有被实物堵住": ("blocked", False, "该通道是否没有被实物堵住？"),
    "没有配置灭火器": ("extinguisher_present", False, "该点位是否没有配置灭火器？"),
}
FIELD_OF = {token: field for field, names in FIELDS.items() for token in names}
ENTITY_FIELDS = {"设备": {"cracked","temperature"}, "货位": {"box_count","color","order_count"},
                 "通道": {"blocked"}, "点位": {"extinguisher_present","blocked"}}
DECLARATION = re.compile(r"(.+?)\s*报告中以 object_id 表示对象编号，facts 使用字段：([^。]+)。")
ROW = re.compile(r"检查(?:眼前)?这一排(设备|校园通道|校园消防点|校园点位)(的正面外壳)?，报告([^，]+)的(设备|通道|点位)编号(?:和([^，]+))?，(?:并)?保留图像证据。")
TARGET = re.compile(r"找到编号为([A-Za-z0-9][A-Za-z0-9_.-]{0,63})的(设备|仓储货位|校园通道|校园点位)，"
                    r"(?:仅检查并报告|仅报告|报告|读取并报告)(.+?)，(?:并)?保留图像证据。")


def fields_in(phrase):
    return tuple(FIELD_OF.get(v) for v in re.split(r"和|、", phrase))


def compile_text(instruction):
    lines = [s.strip() for s in instruction.splitlines() if s.strip()]
    if not lines or any(not (EARLY_GRANT.fullmatch(s) or LATE_GRANT.fullmatch(s)) for s in lines[1:]):
        return None
    match = DECLARATION.fullmatch(lines[0])
    if not match:
        return None
    clause, declaration = match.groups()
    declarations = [re.fullmatch(r"([a-z][a-z0-9_]*)（([^（）]+)）", item) for item in declaration.split("、")]
    if not all(m and FIELD_OF.get(m[2]) == m[1] for m in declarations):
        return None
    fields = {m[1]: INSPECTION_FIELDS[m[1]] for m in declarations}
    if len(fields) != len(declarations):
        return None
    body_scope = "编号指设备本体身份。"
    text = clause[:-len(body_scope)] if clause.endswith(body_scope) else clause
    target, row = TARGET.fullmatch(text), ROW.fullmatch(text)
    query = None
    if target:
        ident, group, report_phrase = target.groups()
        entity = NOUN[group]
        ref = next((prefix for prefix in ("该"+entity+"的", "该"+entity, "它的", "该处")
                    if report_phrase.startswith(prefix)), None)
        if ref is None:
            return None
        report = fields_in(report_phrase[len(ref):])
        scope = {"mode":"target", "description":group, "identity":ident}
        filt = {"op":"keep"}
    elif row:
        group, surface, condition, entity, extra = row.groups()
        if entity != NOUN[group] or surface and entity != "设备":
            return None
        scope = {"mode":"all","description":"这一排"+group+(surface or "")}
        if condition in PREDICATES:
            field, value, question = PREDICATES[condition]
            if field not in ENTITY_FIELDS[entity] or (surface and field != "cracked"):
                return None
            filt = {"op":"eq","field":field,"right":{"literal":value}}
            report = (field,) if extra is None else fields_in(extra)
            materials = [Material("identity", ("positive",), "读取任务所要求的对象编号。", None)]
        else:
            threshold = re.fullmatch(r"(温度)严格高于([0-9]{1,3}(?:\.[0-9]+)?)摄氏度", condition)
            if not threshold or extra is None or surface:
                return None
            field = FIELD_OF[threshold[1]]
            filt = {"op":"gt","field":field,"right":{"literal":float(threshold[2])}}
            question = "该设备的温度是否严格高于"+threshold[2]+"摄氏度？"
            report = fields_in(extra)
            materials = [Material("identity",("positive",),"读取任务所要求的对象编号。",None),
                         Material(field,("positive",),"读取可见的实际摄氏温度数值。",None)]
        query = InspectionTask(scope["description"],question,tuple(materials))
    else:
        return None
    if ((clause.endswith(body_scope) and entity != "设备") or not set(fields) <= ENTITY_FIELDS[entity]
            or None in report or set(report) != set(fields) or len(report) != len(fields)
            or (filt.get("field") and filt["field"] not in fields)):
        return None
    return parse(dict(scope=scope, fields=fields, filter=filt, extreme=None, report=list(report)), instruction), query
