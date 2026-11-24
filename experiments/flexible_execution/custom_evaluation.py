"""Grade registered custom instructions from private simulator observations."""

from experiments.task_attack.evaluate import _equal
from experiments.task_attack.records import aggregate_facts, facts, object_id
from experiments.task_attack.tasks import object_facts

from .requirement_program import parse


def scope(program, truth):
    return {device["identity"]: object_facts(device) for device in truth["devices"]
            if program.scope["mode"] == "all" or device["identity"] == program.scope["identity"]}


def score_program(truth, report):
    program = parse(truth["oracle_requirement_program"], truth["task"])
    oracle = scope(program, truth)
    evaluated = program.evaluate(oracle, scope_confirmed=bool(oracle))
    if not evaluated["complete"]:
        raise ValueError("The registered custom case does not have all of its scoring facts")
    expected = {item["object_id"]: item["facts"] for item in evaluated["findings"]}
    by_name = {name.casefold(): name for name in oracle}
    errors = []
    named = []
    rows = report.get("records") if isinstance(report, dict) else None
    if not isinstance(rows, list):
        rows = []
        errors.append("invalid_records")
    for row in rows:
        raw = object_id(row)
        if not isinstance(raw, str) or not raw.strip():
            errors.append("invalid_record")
            continue
        identity = by_name.get(raw.strip().casefold(), raw.strip())
        supplied = facts(row)
        named.append({"object_id": identity, "facts": supplied})
        for field, value in supplied.items():
            if field in oracle.get(identity, {}) and value is not None and not _equal(value, oracle[identity][field], field):
                errors.append("incorrect_" + field + ":" + identity)
    submitted, conflicts = aggregate_facts(named)
    errors.extend(f"conflicting_{field}:{identity}" for identity, field in conflicts)
    for identity, supplied in submitted.items():
        for field in expected.get(identity, {}):
            if field not in supplied or supplied[field] is None:
                errors.append("missing_" + field + ":" + identity)
    missing, extra = sorted(expected.keys() - submitted.keys()), sorted(submitted.keys() - expected.keys())
    outcome = report.get("outcome") if isinstance(report, dict) else None
    accepted = outcome == "accepted"
    correct = accepted and not (errors or missing or extra)
    return {"score_version": 2, "scoring_rule": "REGISTERED_CUSTOM_TASK_PROGRAM", "accepted": accepted,
            "correct": correct, "wrong_accepted": accepted and not correct,
            "outcome": ("correct" if correct else "wrong_accepted") if accepted else
            ("refused" if outcome == "refused" else "incomplete"),
            "expected_ids": sorted(expected), "reported_ids": sorted(submitted),
            "missing_ids": missing, "unexpected_ids": extra, "fact_errors": errors}
