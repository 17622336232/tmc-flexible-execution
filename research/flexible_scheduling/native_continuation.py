"""Bound candidate dependencies proposed in the ordinary controller reply."""
import re

from .native_condition_resolution import resolve_grouped_conditions
from .natural_frontier import REPORT, discover


REPLY_EXTENSION = {
    "optional_reply_field": "continuation_work",
    "rules": (
        "Keep the normal action, memory, records and report behavior. If the user asked for a "
        "visual inspection with several checks, you may also return continuation_work with "
        "clauses, an array of exact contiguous quotations of the user's instruction, in original "
        "order. Cover the whole request, including every restriction and final reporting rule. "
        "Keep each condition with the check it governs. Separate independent clauses only. "
        "Do not add checks, rewrite permission, change the action, or send a separate reply. "
        "Keep a dependent visual reading unresolved until the prerequisite was already "
        "confirmed in an earlier reply. Apply the normal source requirements to any new record."
    ),
}
NATIVE_REPLY_CONTRACT = (
    "\nFor this one ordinary decision, also include the top-level key continuation_work with an object "
    "containing clauses, an array of exact work clauses copied from authorized_task. "
    "Segment the whole user request into individual work clauses, without rewording or dropping any "
    "permission, prohibition, or report requirement. Keep each conditional sentence together. "
    "For example, {\"continuation_work\":{\"clauses\":[\"first exact clause\",\"second exact clause\"]}}. "
    "Use an empty array if you cannot do this reliably. This field is in addition to the ordinary "
    "action, memory and actual source-bound records. For presence checks, use the exact object names "
    "asked for and Boolean facts.visible. Do not judge a conditional field before its required earlier "
    "judgment appears as a completed record. Do not change the ordinary action to extract clauses."
)
_BETWEEN = re.compile(r"[\s，,。；！!?？]*")
_HARD_BREAK = re.compile(r"[。；！!?？\n]")
_SECTION = re.compile(r"[^。；！!?？\n]+")


def proposal_plan(task, reply, sources=None):
    """Use model-proposed clause boundaries, with authorization derived from all original text.

    The supported semantic vocabulary is still the native visual-presence compiler's.
    A quote can propose the grouping of clauses. A paraphrase, omission, quoted instruction,
    or unverified free-form dependency cannot change the robot's permission.
    """
    proposal = reply.get("continuation_work") if isinstance(reply, dict) else None
    clauses = proposal.get("clauses") if isinstance(proposal, dict) else None
    if (not isinstance(clauses, list) or not 2 <= len(clauses) <= 26 or len(task) > 2200
            or any(not isinstance(clause, str) or not clause.strip() or len(clause)>420 for clause in clauses)):
        return None
    cursor = 0
    sections = {}
    for clause in clauses:
        cursor = _BETWEEN.match(task, cursor).end()
        literal=clause.strip()
        if not task.startswith(literal,cursor):
            return None
        for match in _SECTION.finditer(literal):
            offset=cursor+match.start()
            number=sum(1 for _ in _HARD_BREAK.finditer(task,0,offset))
            part=match.group(0).strip(" ，,\t")
            if part:
                reports=part.split("，")
                sections.setdefault(number,[]).extend(
                    reports if len(reports)>1 and all(REPORT.fullmatch(s) for s in reports) else [part])
        cursor += len(literal)
    if _BETWEEN.fullmatch(task[cursor:]) is None:
        return None
    parts=[]
    ambiguous=False
    for group in sections.values():
        conditions=[i for i,s in enumerate(group) if s.startswith(("若","如果"))]
        if len(conditions)>1:
            return None
        if conditions:
            i=conditions[0]
            protected="，".join(group[i:])
            ambiguous |= protected.count("，")>1
            parts.extend([*group[:i],protected])
        else:
            parts.extend(group)
    if ambiguous:
        return resolve_grouped_conditions(parts,reply,sources or {})
    return discover("。".join(parts))
