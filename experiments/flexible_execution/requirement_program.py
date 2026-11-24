"""Typed completion rules for counts, comparisons, conjunctions and extrema."""

import operator
import re
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite

from experiments.task_attack.records import facts, object_id


COMPARISONS = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
               "gt": operator.gt, "ge": operator.ge}
TYPES = {"flag", "count", "number", "label"}
FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
NUMBER_TEXT = re.compile(r"(?<![0-9.])[-+]?[0-9]+(?:\.[0-9]+)?(?![0-9]|\.[0-9])")
INSPECTION_FIELDS = {"cracked": "flag", "temperature": "number", "box_count": "count", "order_count": "count",
                     "color": "label", "vehicle_present": "flag", "restricted": "flag", "blocked": "flag",
                     "extinguisher_present": "flag"}

COMPILE = """Compile only the authorized inspection requirements. Return compact
JSON with exactly these keys:
{"scope":{"mode":"all","description":"visible material scope"},
 "fields":{"field_name":"flag|count|number|label"},
 "filter":{"op":"keep"},
 "extreme":null,
 "report":["field_name"]}.
Use {"mode":"target","description":"scope","identity":"literal from task"} for
a single named object. For a conjunction or disjunction, use {"op":"and|or",
"children":[filters]}. For a comparison, use {"op":"eq|ne|lt|le|gt|ge",
"field":"left_field","right":{"literal":70}} or {"field":"other_field"}.
If different objects have different values explicitly prescribed by the task,
use {"by_identity":{"literal_id":{"value":2,"quote":"exact clause from task"}}}.
For a minimum or maximum, use {"op":"min|max","field":"field_name"}; keep all
ties. Every referenced field must be declared. Report identity is implicit.
For the simulator, use the provided `supported_fields` names and their types.
Do not invent observations, permission, object identities or task outcomes.
"""


def _value(kind, value):
    if kind == "flag":
        return type(value) is bool
    if kind == "count":
        return type(value) is int and value >= 0
    if kind == "number":
        return type(value) is int or (type(value) is float and isfinite(value))
    return isinstance(value, str) and bool(value.strip())


def _mentions_number(text, number):
    return Decimal(str(number)) in (Decimal(match.group()) for match in NUMBER_TEXT.finditer(text))


def _check_filter(rule, fields, instruction, depth=0):
    if not isinstance(rule, dict) or depth > 4:
        raise ValueError("The filter exceeds the supported task expression")
    operation = rule.get("op")
    if operation == "keep" and set(rule) == {"op"}:
        return
    if operation in ("and", "or") and set(rule) == {"op", "children"}:
        children = rule["children"]
        if not isinstance(children, list) or not 2 <= len(children) <= 6:
            raise ValueError("Expected a bounded group of conditions")
        for child in children:
            _check_filter(child, fields, instruction, depth + 1)
        return
    if (not isinstance(operation, str) or operation not in COMPARISONS
            or set(rule) != {"op", "field", "right"} or not isinstance(rule["field"], str)
            or rule["field"] not in fields):
        raise ValueError("Invalid comparison or unregistered observation field")
    right, left_type = rule["right"], fields[rule["field"]]
    if operation in {"lt", "le", "gt", "ge"} and left_type not in {"count", "number"}:
        raise ValueError("Ordered comparisons require numeric observation types")
    if not isinstance(right, dict) or len(right) != 1:
        raise ValueError("One comparison operand is required")
    if "field" in right:
        other_type = fields.get(right["field"]) if isinstance(right["field"], str) else None
        if other_type is None or (other_type != left_type and {left_type, other_type} != {"number", "count"}):
            raise ValueError("The two observed fields cannot be compared")
    elif "literal" in right:
        literal = right["literal"]
        if not _value(left_type, literal):
            raise ValueError("The comparison literal has an incompatible type")
        if left_type in {"count", "number"} and not _mentions_number(instruction, literal):
            raise ValueError("The numeric comparison literal was not in the authorized task")
    elif "by_identity" in right:
        references = right["by_identity"]
        if not isinstance(references, dict) or not references or len(references) > 64:
            raise ValueError("Expected a bounded table of user-provided values")
        for identity, item in references.items():
            if not isinstance(item, dict) or set(item) != {"value", "quote"} or not _value(left_type, item["value"]):
                raise ValueError("Invalid value in the reference table")
            quote = item["quote"]
            # A reference row must cite its own literal identity and value.
            if not isinstance(quote, str) or quote not in instruction or identity not in quote:
                raise ValueError("A reference comparison value was not grounded in the original wording")
            suffix = quote.split(identity, 1)[1]
            value = item["value"]
            grounded = _mentions_number(suffix, value) if left_type in {"count", "number"} else str(value) in suffix
            if not grounded:
                raise ValueError("A reference comparison value was not grounded in the original wording")
    else:
        raise ValueError("Unknown comparison operand")


def _matches(rule, identity, facts):
    operation = rule["op"]
    if operation == "keep":
        return True
    if operation in ("and", "or"):
        children = [_matches(child, identity, facts) for child in rule["children"]]
        if operation == "and":
            return False if False in children else None if None in children else True
        return True if True in children else None if None in children else False
    left = facts.get(rule["field"])
    source = rule["right"]
    if "literal" in source:
        right = source["literal"]
    elif "field" in source:
        right = facts.get(source["field"])
    else:
        right = source["by_identity"].get(identity, {}).get("value")
    return None if left is None or right is None else COMPARISONS[operation](left, right)


@dataclass(frozen=True)
class Program:
    scope: dict
    fields: dict
    filter: dict
    extreme: dict
    report: tuple

    def evaluate(self, records, *, scope_confirmed):
        """Use only caller-verified observations. No hidden truth is required."""
        if not scope_confirmed:
            return {"complete": False, "findings": [], "unresolved": ["scope"]}
        if self.scope["mode"] == "target":
            identity = self.scope["identity"]
            records = {identity: records.get(identity, {})}
        unresolved = []
        findings = []
        for identity, observed in records.items():
            facts = {key: value for key, value in observed.items() if key in self.fields and _value(self.fields[key], value)}
            matches = _matches(self.filter, identity, facts)
            needed = set(self.report)
            if self.extreme:
                needed.add(self.extreme["field"])
            if matches is None or (matches and not needed <= facts.keys()):
                unresolved.append(identity)
            elif matches:
                findings.append({"object_id": identity, "facts": facts})
        if unresolved:
            return {"complete": False, "findings": [], "unresolved": sorted(unresolved)}
        if self.extreme and findings:
            field = self.extreme["field"]
            select = min if self.extreme["op"] == "min" else max
            value = select(row["facts"][field] for row in findings)
            findings = [row for row in findings if row["facts"][field] == value]
        return {"complete": True, "unresolved": [], "findings": [
            {"object_id": row["object_id"], "facts": {field: row["facts"][field] for field in self.report}}
            for row in sorted(findings, key=lambda row: row["object_id"])]}


def parse(draft, instruction):
    if not isinstance(draft, dict) or set(draft) != {"scope", "fields", "filter", "extreme", "report"}:
        raise ValueError("Expected one bounded task program")
    scope, fields = draft["scope"], draft["fields"]
    if (not isinstance(scope, dict) or scope.get("mode") not in ("all", "target")
            or not isinstance(scope.get("description"), str) or not scope["description"].strip()):
        raise ValueError("The inspection scope is missing")
    if scope["mode"] == "target":
        identity = scope.get("identity")
        if not isinstance(identity, str) or not identity or identity not in instruction:
            raise ValueError("Target identity must come from the authorized task")
    if (not isinstance(fields, dict) or not 1 <= len(fields) <= 12
            or any(not isinstance(name, str) or not FIELD_NAME.fullmatch(name)
                   or not isinstance(kind, str) or kind not in TYPES for name, kind in fields.items())):
        raise ValueError("Observation fields and types are invalid")
    _check_filter(draft["filter"], fields, instruction)
    extreme = draft["extreme"]
    if extreme is not None and (not isinstance(extreme, dict) or set(extreme) != {"op", "field"}
                                or extreme["op"] not in ("min", "max")
                                or not isinstance(extreme["field"], str)
                                or fields.get(extreme["field"]) not in ("count", "number")):
        raise ValueError("Extrema require a common numeric field")
    report = draft["report"]
    if not isinstance(report, list) or not all(isinstance(field, str) and field in fields for field in report):
        raise ValueError("The report refers to an undeclared field")
    if len(set(report)) != len(report):
        raise ValueError("A report field was declared twice")
    return Program(scope, fields, draft["filter"], extreme, tuple(report))


def compile_task(client, instruction):
    program = parse(client.request(COMPILE, {"authorized_task": instruction, "supported_fields": INSPECTION_FIELDS}, []),
                    instruction)
    if any(INSPECTION_FIELDS.get(field) != kind for field, kind in program.fields.items()):
        raise ValueError("The program declared a field that the inspection platform does not observe")
    return program


def check_submission(program, report, collected):
    """Check the report after the shared visual verifier has confirmed its scope."""
    rows = {}
    for record in collected + report["records"]:
        identity = object_id(record)
        observed = rows.setdefault(identity, {})
        for field, value in facts(record).items():
            if field in program.fields:
                if field in observed and observed[field] != value:
                    return f"现有证据对 {identity} 的 {field} 记录不一致"
                observed[field] = value
    result = program.evaluate(rows, scope_confirmed=True)
    if not result["complete"]:
        return "证据还不足以判定这些对象：" + "、".join(result["unresolved"])
    expected = {row["object_id"]: row["facts"] for row in result["findings"]}
    submitted = {object_id(row): {key: value for key, value in facts(row).items() if key in program.report}
                 for row in report["records"]}
    if len(submitted) != len(report["records"]) or submitted != expected:
        return "报告和由现有证据算出的任务结果不一致，请重新检查条件、范围和并列结果"
    return None
