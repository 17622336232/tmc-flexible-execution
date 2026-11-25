"""Common v2 inspection front end and repeated, bound visual scheduling."""

import asyncio
import json
from dataclasses import asdict
from functools import partial
from hashlib import sha256
from math import dist
from time import monotonic

from experiments.task_attack.agent import run_agent
from experiments.task_attack.runtime import write_json

from .. import shared_frontend
from ..evidence import Frame
from ..gpt_vision import LOCATE, locate_targets, parse_intent
from ..live_planning import STREAM_METHODS
from ..material_schema import INSPECTION_FIELDS, canonical_record
from ..requirement_program import COMPILE, check_submission, parse
from .gpt_probe import _take_photo
from .live_inspection import (LIVE_INTERPRET, MAX_VIEWPOINTS, _archive, _inspect, _safe_pose,
                              _worlds, submission_guard)
from .viewpoints import _candidates, measured_views, xyz


COMPILER = """Compile the authorized task just once. Return exactly one JSON object
with keys program and conditional. Put the typed task object in program. Put the
structured one-object inspection in conditional or null when it cannot be
represented by a yes/no visual predicate. Keep the complete requirement in the
typed part even when only one predicate can be scheduled early.

Rules for program:
""" + COMPILE + """
Rules for conditional:
""" + LIVE_INTERPRET


def compile_requirements(model, instruction):
    draft = model.request(COMPILER, {"authorized_task": instruction, "supported_fields": INSPECTION_FIELDS}, [])
    if not isinstance(draft, dict) or set(draft) != {"program", "conditional"}:
        raise ValueError("The combined compilation must return program and conditional")
    program = parse(draft["program"], instruction)
    if any(INSPECTION_FIELDS.get(field) != kind for field, kind in program.fields.items()):
        raise ValueError("The combined compiler declared an unsupported observation field")
    conditional = draft["conditional"]
    return program, parse_intent(conditional, instruction) if conditional is not None else None


def _front_end(backend, directory, trace, model, started, instruction, reference, scene):
    source, observation, geometry = _take_photo(backend, directory, trace, started, 1, "scene")
    status, shared, reference_sha = "disabled", None, None
    if reference:
        signature = shared_frontend.identity(instruction, scene, source, True)
        signature["compiler_sha256"] = sha256(COMPILER.encode("utf8")).hexdigest()
        signature["locator_sha256"] = sha256(LOCATE.encode("utf8")).hexdigest()
        status, shared = shared_frontend.get(reference, signature, instruction)
    if shared:
        program, intent, targets, reference_sha = shared
    else:
        with trace.measure("model_request", 0, purpose="compile_joint_task"):
            program, intent = compile_requirements(model, instruction)
        targets = ()
        if intent:
            with trace.measure("model_request", 2, purpose="locate_visible_targets", image_sha256=source.digest):
                targets = locate_targets(model, intent, source)
        if reference and status == "unregistered":
            reference_sha = shared_frontend.register(reference, signature, program, intent, targets, directory, source)
            status = "registered"
    trace.event("frontend_alignment", 2, alignment=status, reference_sha256=reference_sha,
                overview_sha256=source.digest, fresh_targets=None if shared else len(targets),
                used_targets=len(targets), cache_hits=int(bool(shared)))
    write_json(directory / "typed_task_program.json", asdict(program))
    if intent is not None:
        write_json(directory / "task_requirements.json", asdict(intent))
    return program, intent, targets, source, observation, geometry


def _goals(backend, observation, geometry, target, current, trace):
    goals = []
    for goal, _ in _candidates(backend, observation, geometry, target, trace):
        if dist(xyz(current), xyz(goal)) < .35 or any(dist(xyz(item), xyz(goal)) < .35 for item in goals):
            continue
        if _safe_pose(backend, goal):
            goals.append(goal)
        if len(goals) >= MAX_VIEWPOINTS:
            break
    return goals


def run(backend, instruction, directory, event, timeout, *, model, trace, method, costs, hold_at_view,
        ranked_policy, frontend_reference, scene_sha):
    started = monotonic()
    model.deadline_monotonic = backend.execution_deadline_monotonic = started + timeout
    program, intent, targets, source, overview, geometry = _front_end(
        backend, directory, trace, model, started, instruction, frontend_reference, scene_sha)
    step = 2
    summary = {"initial_scene_image": "probe_0001", "visible_targets": [asdict(t) for t in targets],
               "accepted_materials": {}}
    pending_results, captures = [], {}
    available = measured_views(backend, overview, geometry, (t for t in targets if not t.view_sufficient), trace)
    measured = {item.id: pose for item, pose in available}
    observation = overview
    for index, target in enumerate(targets):
        if monotonic() - started >= timeout:
            break
        budget = getattr(model, "request_budget", None)
        if budget is not None and budget.summary()["remaining"] <= 2:
            trace.event("pending_query_deferred", step, reason="reserve_for_complete_task", object_id=target.id)
            break
        current_pose = {**observation["position"], "yaw": observation["yaw"]}
        expected_view = measured.get(target.id)
        prior_capture = captures.pop(target.id, None)
        if prior_capture:
            first = prior_capture[0]
        elif target.view_sufficient:
            first = Frame(source.id, target.id, source.version, source.captured_at, source.payload, source.provenance)
        elif expected_view and _safe_pose(backend, expected_view):
            step += 1
            try:
                with trace.measure("action", step, action="capture_initial_judgment_view", target_id=target.id):
                    backend.move(expected_view)
                first, observation, _ = _take_photo(backend, directory, trace, started, step, target.id)
                current_pose = {**observation["position"], "yaw": observation["yaw"]}
            except (ValueError, RuntimeError):
                trace.event("pending_query_deferred", step, reason="native_view_unavailable", object_id=target.id)
                continue
        else:
            trace.event("pending_query_deferred", step, reason="no_readable_or_admitted_view", object_id=target.id)
            continue

        step += 1
        log = lambda name, **fields: trace.event(name, step, **fields)
        goals = _goals(backend, overview, geometry, target, current_pose, trace)
        task, query, poses, materials = _worlds(instruction, intent, target, current_pose, goals, started,
                                                timeout, costs, log, hold_at_view, query_id="visual-" + target.id)
        write_json(directory / ("conditional_contract_%04d.json" % step), {
            "task": asdict(task), "poses": poses, "method": method, "calibration_id": costs.calibration_id})
        independent = [(other, measured[other.id]) for other in targets[index + 1:]
                       if other.id in measured and other.id not in captures]
        outcome = asyncio.run(_inspect(backend, directory, trace, model, started, task, query, poses,
                                       materials, source, first, target, first.id == source.id, intent.question,
                                       independent, method, hold_at_view, ranked_policy,
                                       step_base=step, continuous=True))
        captures.update(outcome["captured"])
        step = outcome["step"]
        pending_results.append({"query": query.id, "target": target.id, "frame_sha256": first.digest,
                                "frame_id": first.id, "outcome": outcome["outcome"],
                                "accepted_binding": outcome["query_bound"]})
        photos = {entry["frame_id"]: entry["image_id"] for entry in
                  (json.loads(path.read_text(encoding="utf8")) for path in directory.glob("probe_[0-9][0-9][0-9][0-9].json"))}
        for key, evidence in outcome["evidence"].items():
            if outcome["query_bound"]:
                summary["accepted_materials"][key] = {
                    **asdict(evidence), "image_id": photos.get(evidence.frame_id), "query_id": query.id}
        if index + 1 < len(targets) and monotonic() - started < timeout:
            step += 1
            _, observation, _ = _take_photo(backend, directory, trace, started, step, "current-pose")

    summary["visual_judgments"] = pending_results
    write_json(directory / "conditional_result.json", summary)
    trace.event("full_inspection_started", step + 1, seeded_frames=len(_archive(directory)),
                accepted_active_materials=len(summary["accepted_materials"]))
    visual_guard = submission_guard(instruction, directory, model, trace, summary, cache_verified=True)

    def guard(report, collected, report_step):
        if report["outcome"] == "accepted":
            issue = check_submission(program, report, collected)
            trace.event("typed_task_acceptance", report_step, accepted=issue is None, issue=issue)
            if issue:
                return issue
        return visual_guard(report, collected, report_step)

    pipeline = None
    if method in STREAM_METHODS:
        from ..stream_baseline import decide

        pipeline = partial(decide, stream=method == "conditional-stream")
    report, metrics = run_agent(backend, instruction, directory, event, timeout, model=model, trace=trace,
                                prior_views=_archive(directory), prior_memory=json.dumps(summary, ensure_ascii=False),
                                episode_started=started, step_offset=step, submit_guard=guard,
                                decision_pipeline=pipeline, fact_types=program.fields,
                                record_adapter=partial(canonical_record, program.fields), evidence_journal=True)
    return report, {**metrics, "typed_task_program_compiled": True, "protocol_version": "protocol-v2",
                    "conditional_query_bound": any(row["accepted_binding"] for row in pending_results),
                    "scheduled_visual_queries": len(pending_results),
                    "conditional_materials": len(summary["accepted_materials"])}
