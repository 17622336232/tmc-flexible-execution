"""Shared value types for simulator inspection evidence and completion."""

from math import isfinite
import re

from experiments.task_attack.records import facts


INSPECTION_FIELDS = {"cracked": "flag", "temperature": "number", "box_count": "count", "order_count": "count",
                     "color": "label", "vehicle_present": "flag", "restricted": "flag", "blocked": "flag",
                     "extinguisher_present": "flag"}
FIELD_ALIASES = {"有无裂纹": "cracked", "是否有裂纹": "cracked", "温度": "temperature", "摄氏温度": "temperature",
                 "箱数": "box_count", "实际箱数": "box_count", "实物箱数": "box_count", "订单箱数": "order_count",
                 "颜色": "color", "车辆颜色": "color", "是否有车": "vehicle_present", "是否禁停": "restricted",
                 "是否堵塞": "blocked", "有无灭火器": "extinguisher_present", "是否有灭火器": "extinguisher_present"}
FLAG_LITERALS = {"true": True, "有": True, "是": True, "false": False, "无": False, "否": False}
COLOR_LITERALS = dict(zip("红蓝绿黄白黑灰橙紫", ("red", "blue", "green", "yellow", "white", "black", "gray", "orange", "purple")))


def value_matches(kind, value):
    if kind == "flag":
        return type(value) is bool
    if kind == "count":
        return type(value) is int and value >= 0
    if kind == "number":
        return type(value) is int or (type(value) is float and isfinite(value))
    return kind == "label" and isinstance(value, str) and bool(value.strip())


def _explicit_value(kind, value):
    if not isinstance(value, str):
        return value
    text = value.strip().casefold()
    if kind == "flag":
        return FLAG_LITERALS.get(text)
    if kind == "label":
        return COLOR_LITERALS.get(text[:-1] if text.endswith("色") else text, text)
    if kind == "count":
        found = re.fullmatch(r"([0-9]{1,12})\s*(?:箱)?", text)
    else:
        found = re.fullmatch(r"([+-]?[0-9]{1,12}(?:\.[0-9]{1,8})?)\s*(?:℃|°c|摄氏度)?", text)
    if not found:
        return None
    number = found[1]
    return float(number) if "." in number else int(number)


def canonical_record(schema, record):
    if not isinstance(record, dict) or not isinstance(record.get("facts", {}), dict):
        return record
    canonical, conflicting = {}, set()
    for raw, value in facts(record).items():
        field = FIELD_ALIASES.get(raw, raw)
        if field not in schema:
            continue
        kind = schema[field]
        value = _explicit_value(kind, value)
        if value_matches(kind, value):
            if field in canonical and canonical[field] != value:
                conflicting.add(field)
            canonical[field] = value
    return {**{key: value for key, value in record.items() if key not in ("facts", "cracked", "temperature")},
            "facts": {field: value for field, value in canonical.items() if field not in conflicting}}
