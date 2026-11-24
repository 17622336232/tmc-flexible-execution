"""Offline evaluation from the event ledger and the held-back fixture truth."""

from math import isclose, isfinite

from .evidence import Evidence
from .fixtures import MODE
from .simulation import PROVENANCE


def score(case, events):
    task = case.task
    truth = {reply.id: reply.outcome for reply in case.query_truth}
    requirements = {item.id: item for item in task.requirements}
    skills = {item.id: item for item in task.skills}
    starting_material = dict(case.initial_material)
    captures = {}
    accepted = {}
    submitted = {}
    resolved = {}
    errors = []
    previous_at = 0.0
    site = task.start
    running = None
    last_capture_window = None
    terminal_at = None
    delivered = None
    wait = 0.0
    cost = 0.0
    observations = 0
    action_overlap = 0
    rejections = 0
    for event in events:
        at = event["at"]
        kind = event["event"]
        if not isfinite(at) or not previous_at <= at <= task.deadline:
            errors.append("event_clock")
        previous_at = at
        if delivered is not None:
            errors.append("event_after_delivery")
        if kind == "query_submitted":
            binding = event["query"]
            key = binding["id"]
            if key not in truth or key in submitted:
                errors.append("query_identity")
            submitted[key] = binding["input_hash"]
        elif kind == "query_returned" and event["accepted"]:
            key = event["query_id"]
            if (submitted.get(key) != event["input_hash"] or key in resolved
                    or truth.get(key) != event["outcome"]):
                errors.append("query_binding")
            resolved[key] = event["outcome"]
        elif kind == "action_started":
            skill_id = event["skill_id"]
            skill = skills.get(skill_id)
            trip = task.travel_seconds(site, event["target"])
            if running is not None or terminal_at is not None or event["origin"] != site or trip is None:
                errors.append("invalid_motion_start")
            if skill_id is not None:
                if skill is None or skill.site != event["target"] or not skill.permitted(resolved):
                    errors.append("observation_permission")
                observations += 1
            elif event["target"] != task.terminal:
                errors.append("unexpected_terminal_target")
            if set(submitted) - set(resolved):
                action_overlap += 1
            last_capture_window = None
            running = (at, skill_id, event["target"], (trip or 0) + (skill.seconds if skill else 0))
        elif kind == "action_finished":
            if (running is None or (event["skill_id"], event["site"]) != running[1:3]
                    or not isclose(at, running[0] + running[3], abs_tol=1e-8)):
                errors.append("invalid_motion_end")
            site = event["site"]
            running = None
            last_capture_window = (at, skills.get(event["skill_id"]))
        elif kind == "terminal_reached":
            if terminal_at is not None or event["site"] != task.terminal or site != task.terminal:
                errors.append("terminal_state")
            terminal_at = at
        elif kind == "observation_captured":
            key = event["requirement_id"]
            initial = at == 0 and key in starting_material and event["frame_id"] == "initial-" + key
            observed = (
                last_capture_window is not None and last_capture_window[0] == at
                and last_capture_window[1] is not None and key in last_capture_window[1].covers
            )
            if not (initial or observed):
                errors.append("observation_without_capture")
            captures[(event["frame_id"], key)] = event
        elif kind == "evidence_accepted":
            material = Evidence(**event["evidence"])
            item = requirements.get(material.requirement_id)
            captured = captures.get((material.frame_id, material.requirement_id))
            matches_capture = (
                captured is not None and captured["readable"]
                and (captured["at"], captured["frame_hash"], captured["object_id"],
                     captured["field"], captured["value"], captured["valid_until"], captured["provenance"])
                == (material.captured_at, material.frame_hash, material.object_id, material.field,
                    material.value, material.valid_until, material.provenance)
            )
            if (item is None or material.provenance != PROVENANCE
                    or not matches_capture or not material.usable(item, at)
                    or material.value != case.material_values.get(material.requirement_id)):
                errors.append("invalid_accepted_evidence")
            else:
                accepted[item.id] = material
        elif kind == "evidence_rejected":
            rejections += 1
        elif kind == "blocked_wait":
            until = event["until"]
            if not at <= until <= task.deadline or not (set(submitted) - set(resolved)):
                errors.append("invalid_wait")
            else:
                wait += until - at
        elif kind == "proposal_finished":
            duration = event["spent_seconds"]
            if not isfinite(duration) or duration < 0:
                errors.append("invalid_proposal_cost")
            else:
                cost += duration
        elif kind == "report_delivered":
            delivered = (at, event["findings"])
        elif kind == "execution_error":
            errors.append("execution_error:" + event["type"])

    finish = None
    missing = []
    if delivered is not None:
        at, findings = delivered
        required = [item for item in task.requirements if item.applies(truth)]
        missing = [item.id for item in required
                   if item.id not in accepted or not accepted[item.id].usable(item, at)]
        expected = {item.id: case.material_values[item.id] for item in required if item.report}
        if findings != expected:
            errors.append("report_contents")
        if resolved != truth:
            errors.append("missing_authoritative_result")
        if terminal_at is None or running is not None:
            errors.append("missing_terminal")
        if not errors and not missing:
            finish = max(at, terminal_at)
    return {
        "mode": MODE,
        "success": finish is not None,
        "wrong_accepted": delivered is not None and finish is None,
        "completion_seconds": finish,
        "restricted_completion_seconds": finish if finish is not None else task.deadline,
        "deadline_seconds": task.deadline,
        "terminal_seconds": terminal_at,
        "blocked_seconds": wait,
        "proposal_wall_seconds": cost,
        "observations": observations,
        "actions_started_with_pending_query": action_overlap,
        "rejected_materials": rejections,
        "missing_requirements": missing,
        "trace_errors": sorted(set(errors)),
    }
