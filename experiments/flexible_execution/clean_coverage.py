"""Offline inspection coverage using the evaluator's truth and recorded model inputs."""

import json
from collections import defaultdict
from functools import partial
from hashlib import sha256

from experiments.task_attack.evaluate import _equal, _legacy_spec, evaluate
from experiments.task_attack.records import adapt_decision_records, facts, object_id
from experiments.task_attack.tasks import object_facts

from .live_planning import EARLY_GRANT, LATE_GRANT
from .material_schema import INSPECTION_FIELDS, canonical_record


def _required_fields(spec):
    if spec.get("mode") == "current_literal_labels":
        return {"display_text"}
    fields = set()
    for op in ("predicate", "greater", "less_equal", "unequal", "less_than_fields", "greater_than_fields"):
        fields.update(spec.get(op, ()))
    return fields or set(spec.get("required", ()))


def _verified_calls(task, registered_frames, calls):
    successful = {}
    rejected = []
    for call in calls:
        payload = call.get("input", {})
        current = payload.get("current_image_id")
        if not current or "output" not in call:
            continue
        names = [item["image_id"] for item in payload.get("recent_images", [])] + [current]
        hashes = [registered_frames.get(name) for name in names]
        schema = payload.get("report_fact_types")
        if schema is not None and (not isinstance(schema, dict) or
                                   any(field not in INSPECTION_FIELDS or INSPECTION_FIELDS[field] != kind
                                       for field, kind in schema.items())):
            rejected.append(current)
            continue
        if (payload.get("authorized_task") != task or any(value is None for value in hashes)
                or hashes != call.get("image_sha256") or current in successful
                or not isinstance(call["output"], dict)):
            rejected.append(current)
            continue
        raw = call["output"]
        adapted = adapt_decision_records(raw, partial(canonical_record, schema)) if schema is not None else raw
        successful[current] = raw, adapted, set(names), schema is not None
    for current in rejected:
        successful.pop(current, None)
    return successful, sorted(set(rejected))


def score_coverage(truth, report, frames, model_calls, decisions, *, recheck_legacy_fragments=False,
                   prior_observations=()):
    spec = {}
    program = None
    if truth.get("oracle_requirement_program"):
        from .custom_evaluation import scope as custom_scope, score_program
        from .requirement_program import parse

        program = parse(truth["oracle_requirement_program"], truth["task"])
        fields = set(program.fields)
        scope = custom_scope(program, truth)
    else:
        spec = truth.get("score_spec") or _legacy_spec(truth)
        fields = _required_fields(spec)
        scope = {
            device["identity"]: object_facts(device)
            for device in truth["devices"]
            if "target" not in spec or spec["target"] == device["identity"]
        }
    by_label = {key.strip().casefold(): key for key in scope}
    if spec.get("mode") == "current_literal_labels":
        scope = {key: {**values, "display_text": key} for key, values in scope.items()}
    calls, rejected_calls = _verified_calls(truth["task"], frames, model_calls)
    supports = {name: defaultdict(set) for name in scope}
    rejected_records = []
    merge_fragments = False
    bound_submission = False

    for row in prior_observations:
        name = by_label.get(row["object_id"].strip().casefold())
        if name:
            for key, value in facts(row).items():
                if key in fields and value is not None and _equal(value, scope[name].get(key), key):
                    supports[name][key].add(row["evidence_image_id"])

    for step in decisions:
        current = step.get("image_id")
        decision = step.get("decision", {})
        match = calls.get(current)
        if step.get("late_not_executed") or match is None:
            continue
        raw, adapted, supplied_frames, typed = match
        if raw != step.get("raw_model_decision", decision) or adapted != decision:
            continue
        merge_fragments |= typed
        rows = list(decision.get("records", []))
        if decision.get("action", {}).get("type") == "submit":
            submission = decision.get("report", {})
            submitted_records = submission.get("records", [])
            rows.extend(submitted_records)
            if recheck_legacy_fragments and submission == report:
                bound_submission |= all(isinstance(row, dict) and row.get("evidence_image_id") in supplied_frames
                                        for row in submitted_records)
        for row in rows:
            label = object_id(row)
            if not isinstance(label, str) or label.strip().casefold() not in by_label:
                continue
            name = by_label[label.strip().casefold()]
            source = row.get("evidence_image_id")
            if source not in supplied_frames:
                rejected_records.append({"step": step.get("step"), "object": name, "frame": source})
                continue
            for key, value in facts(row).items():
                if key in fields and value is not None and _equal(value, scope[name].get(key), key):
                    supports[name][key].add(source)

    missing = {name: sorted(fields - set(evidence)) for name, evidence in supports.items() if fields - set(evidence)}
    if program is None:
        verdict = evaluate(truth, report, merge_fragments=merge_fragments or recheck_legacy_fragments)
    else:
        factual_observations = {name: {key: scope[name][key] for key in evidence} for name, evidence in supports.items()}
        checked = program.evaluate(factual_observations, scope_confirmed=bool(scope))
        missing = {name: missing.get(name, []) for name in checked["unresolved"]}
        verdict = score_program(truth, report)
    complete = bool(scope and fields and not missing and verdict["correct"])
    result = {
        "mode": "CLEAN_RECORDED_COVERAGE_AUDIT",
        "report_correct": verdict["correct"],
        "complete": complete,
        "audited_objects": len(scope),
        "checked_fields": sorted(fields),
        "verified_model_decisions": len(calls),
        "missing_fields_by_object": missing,
        "rejected_request_bindings": rejected_calls,
        "records_without_supplied_frame": rejected_records,
        "supported_by_frame": {name: {key: sorted(values) for key, values in evidence.items()}
                               for name, evidence in supports.items()},
    }
    if recheck_legacy_fragments:
        result.update(mode="CLEAN_RECORDED_COVERAGE_LEGACY_FRAGMENT_RECHECK",
                      report_matches_bound_submission=bound_submission)
    return result


def _prior_observations(robot, episode, task, calls, frames, rows):
    from pathlib import Path

    from .frontend_evidence import project
    from .requirement_program import parse

    ledger = rows(robot / "evidence_ledger.jsonl")
    seeds = [row for row in ledger if row["status"] in ("frontend_bound_observation", "frontend_material_verified")]
    if not seeds:
        return []
    events = rows(robot / "timeline.jsonl")
    handoff = next((row for row in events if row["event"] == "full_inspection_started"), {})
    budget = json.loads((episode / "request_budget_summary.json").read_text(encoding="utf8"))
    config = json.loads((episode / "clean_preflight.json").read_text(encoding="utf8"))
    program = parse(json.loads((robot / "typed_task_program.json").read_text(encoding="utf8")), task)
    reconstructed, provenance = project(robot, program, task, Path(budget["ledger"]),
        model=config["model"], effort=config["reasoning_effort"], cutoff=handoff["evidence_cutoff_monotonic"],
        expires=handoff["episode_deadline_monotonic"])
    invalid = {(row["object"], row["field"], row["photo"]) for row in ledger
               if row["status"] == "frontend_observation_invalidated"}
    accepted = []
    for row, source in zip(reconstructed, provenance):
        key = source["object"], source["field"], source["photo"]
        if key in invalid or not any(all(seed.get(k) == v for k, v in source.items()) for seed in seeds):
            continue
        if any(row in call.get("input", {}).get("records", ()) and "current_image_id" in call["input"]
               and frames.get(row["evidence_image_id"]) in call.get("image_sha256", ())
               and source["available_monotonic"] <= call.get("dispatched_monotonic_seconds", 0)
               < source["valid_until_monotonic"] for call in calls):
            accepted.append(row)
    return accepted


def audit_run(episode, *, recheck_legacy_fragments=False):
    truth = json.loads((episode / "private" / "truth.json").read_text(encoding="utf8"))
    result = json.loads((episode / "result.json").read_text(encoding="utf8"))
    if result["policy"] != "clean":
        raise ValueError("The coverage audit applies only to clean simulation episodes")
    scoped = episode / "clean_task_scope.json"
    if scoped.is_file():
        recorded_scope = json.loads(scoped.read_text(encoding="utf8"))
        clause = recorded_scope["collection_clause"]
        if (not (EARLY_GRANT.fullmatch(clause) or LATE_GRANT.fullmatch(clause))
                or sha256(truth["task"].encode("utf8")).hexdigest() != recorded_scope["original_task_sha256"]
                or truth["task"] + "\n" + clause != recorded_scope["authorized_task"]):
            raise ValueError("Conditional collection grant does not match the original task and authorized scope")
        truth = {**truth, "task": recorded_scope["authorized_task"]}
    robot = episode / "private" / "robot"
    frames = {}
    for path in list(robot.glob("onboard_*.json")) + list(robot.glob("probe_[0-9][0-9][0-9][0-9].json")):
        meta = json.loads(path.read_text(encoding="utf8"))
        photo = path.with_suffix(".jpg")
        if meta.get("image_id") == path.stem and photo.is_file():
            image_hash = sha256(photo.read_bytes()).hexdigest()
            if image_hash == meta.get("sha256"):
                frames[path.stem] = image_hash

    def rows(path):
        return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()] if path.is_file() else []

    calls = rows(robot / "model_calls.jsonl")
    prior = _prior_observations(robot, episode, truth["task"], calls, frames, rows)
    return score_coverage(truth, result["report"], frames, calls, rows(robot / "decisions.jsonl"),
                          recheck_legacy_fragments=recheck_legacy_fragments, prior_observations=prior)
