"""Optional work discovery within an ordinary original-GPT decision. The task owner verifies grants."""
from .event_workflow import ALWAYS,Query

VOCABULARY={"red_carton":"红色货箱","blue_carton":"蓝色货箱","white_carton":"白色货箱",
            "vehicle":"停放车辆","extinguisher":"灭火器","obstruction_crate":"通道上的货箱障碍物"}
SECTION_PROMPT="""You retain the ordinary robot tools, memory, record and submit contract.
For a bounded inspection described by the user at two headings named front and side, you may
delegate the remaining evidence work as an optional inspection_section in this SAME reply.
If choosing to delegate, use the ordinary action observe or record. Include:
"inspection_section":{"from":"exact current image id","finish":"front or side",
"items":[{"id":"Q0","view":"front or side","check":"one allowed predicate","if_any":[]},...]}.
An empty if_any denotes an unconditional check; other entries are question ids, any one of
which must return positive to open this check. Derive every subject, view and prerequisite from
the user's actual instruction. The permitted identifiers for visual predicates are red_carton
(red cargo box), blue_carton, white_carton, vehicle (parked vehicle), extinguisher, and
obstruction_crate (cargo crate obstructing the passage). Names alone are not visual evidence.
For unconditional checks at the supplied current heading you can record an actually observed
visible boolean with its exact evidence_image_id now. Do not answer a gated check early.
The tool will validate the scope and can schedule the requested actions and reads. If you prefer
to continue yourself or there is an unresolved scope, use the ordinary control interface."""


def expected(case):
    return {f"Q{i}":dict(view=site,check=cat,parents=frozenset(gate)) for i,(site,cat,gate)
            in enumerate(zip(case["sites"],case["categories"],case["any_of"]))}


def proposal(reply,case,frame):
    """Return the model-proposed graph only after matching the user's independent scope and permissions."""
    if not isinstance(reply,dict) or (reply.get("action") or {}).get("type") not in ("record","observe"):
        return None,"ordinary-action"
    raw=reply.get("inspection_section")
    if not isinstance(raw,dict):
        return None,"not-proposed"
    if set(raw)!={"from","finish","items"} or raw["from"]!=frame or raw["finish"]!=case["finish"] or not isinstance(raw["items"],list):
        return None,"section-scope-mismatch"
    declared={}
    for row in raw["items"]:
        if (not isinstance(row,dict) or set(row)!={"id","view","check","if_any"} or type(row["id"]) is not str
                or row["id"] in declared or row["view"] not in ("front","side") or type(row["check"]) is not str
                or row["check"] not in VOCABULARY or not isinstance(row["if_any"],list)
                or not all(type(v) is str for v in row["if_any"]) or len(set(row["if_any"]))!=len(row["if_any"])):
            return None,"malformed-or-duplicate-check"
        declared[row["id"]]=dict(view=row["view"],check=row["check"],parents=frozenset(row["if_any"]))
    if declared!=expected(case):
        return None,"mismatched-user-fields-or-gates"
    # Build the operative query graph from the actual proposal, with stable public question IDs.
    idx={f"Q{i}":i for i in range(len(declared))}
    qs=[]
    for key in idx:
        row=declared[key]
        guard=tuple(((idx[parent],True),) for parent in sorted(row["parents"])) if row["parents"] else ALWAYS
        qs.append(Query(row["check"]+"@"+key,(int(row["view"]=="side"),),guard))
    return tuple(qs),"delegated"


def initial_records(reply,case,frame):
    contract=expected(case);found={};conflict=set();violations=[]
    for row in (reply or {}).get("records",[]) or []:
        if not isinstance(row,dict):continue
        key=row.get("object_id")
        val=(row.get("facts") or {}).get("visible")
        if type(key) is not str or key not in contract or type(val) is not bool:continue
        if row.get("evidence_image_id")!=frame:continue
        if contract[key]["parents"] or contract[key]["view"]!="front":
            violations.append(key);continue
        if key in found and found[key]!=val:conflict.add(key)
        found[key]=val
    return {k:v for k,v in found.items() if k not in conflict},sorted(set(violations))


def describe(reply,case,frame):
    plan,reason=proposal(reply,case,frame)
    facts,invalid=initial_records(reply,case,frame)
    return dict(action=(reply.get("action") or {}).get("type"),offered=isinstance(reply.get("inspection_section"),dict),
                accepted=plan is not None and not invalid,reason="premature-claim" if invalid else reason,
                seeded=facts,premature_fields=invalid)
