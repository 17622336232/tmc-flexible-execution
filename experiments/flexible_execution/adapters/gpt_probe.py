"""Measure one live vision request against independent simulated inspection work."""

import asyncio
from dataclasses import asdict, replace
from math import dist
from threading import Event
from time import monotonic

from experiments.task_attack.runtime import write_json

from ..evidence import Frame
from ..executor import Snapshot, async_action
from ..gpt_vision import OUTCOMES, frozen_prompt, interpret, judge, locate_targets
from ..queries import AsyncQueryWorker, QueryBook
from ..task_ir import QuerySpec, Requirement, Skill, TaskContract


def _take_photo(backend, directory, trace, started, step, target_id):
    with trace.measure("observation", step, target_id=target_id):
        observation, _ = backend.observe()
    frame = Frame(observation["observation_id"], target_id, step, monotonic() - started,
                  observation["data"], "simulated_onboard_rgb")
    (directory / ("probe_%04d.jpg" % step)).write_bytes(frame.payload)
    trace.event("frozen_photo", step, target_id=target_id, frame_id=frame.id,
                image_sha256=frame.digest, pose=observation["position"], ros_stamp=observation["ros_stamp"])
    return frame, observation


def _xyz(goal):
    return tuple(goal[key] for key in ("x", "y", "z"))


def _measured_views(backend, observation, targets, trace):
    views = []
    for target in targets:
        for standoff in (3.0, 2.0, 1.5):
            try:
                measured = backend.adapter.resolve_visual_target(observation, target.bbox, standoff)
            except (RuntimeError, ValueError) as error:
                trace.event("viewpoint_rejected", 1, target_id=target.id, standoff=standoff,
                            detail=str(error)[:300])
                continue
            goal = measured["goal"]
            if any(dist(_xyz(goal), _xyz(known)) < 0.35 for _, known in views):
                trace.event("viewpoint_rejected", 1, target_id=target.id, standoff=standoff,
                            detail="Not a distinct measured view")
                continue
            views.append((target, goal))
            trace.event("viewpoint_measured", 1, target_id=target.id, goal=goal,
                        source_frame=observation["observation_id"])
            break
    return sorted(views, key=lambda pair: dist(_xyz(observation["position"]), _xyz(pair[1])))


async def _parallel(backend, directory, trace, client, started, task, snapshot, spec,
                    first_frame, first_target, question, views, reused):
    loop = asyncio.get_running_loop()
    invoked = Event()

    def infer(payload, prompt):
        with trace.measure("model_request", 3, purpose="judge_frozen_photo",
                           frame_id=first_frame.id, image_sha256=first_frame.digest):
            invoked.set()
            return judge(client, payload, prompt)

    book = QueryBook()
    worker = AsyncQueryWorker(infer)
    prompt = frozen_prompt(first_frame, first_target, question, use_initial_region=reused)
    submitted = monotonic() - started
    future = worker.submit(spec, first_frame, prompt, submitted, book)
    trace.event("query_submitted", 3, binding=asdict(book.pending[spec.id]))
    next_frame = None
    try:
        if not await loop.run_in_executor(None, invoked.wait, max(0.0, task.deadline - submitted)):
            raise RuntimeError("Frozen vision worker did not start before the episode deadline")
        state = replace(snapshot, at=monotonic() - started)
        action = async_action(task, state)
        if action is not None:
            target, goal = views[action.site]
            try:
                with trace.measure("action", 2, action="inspect_other_required_target", target_id=target.id):
                    await loop.run_in_executor(None, backend.move, goal)
                after, _ = _take_photo(backend, directory, trace, started, 3, target.id)
                if after.id == first_frame.id or after.digest == first_frame.digest:
                    raise RuntimeError("Independent work did not capture a distinct onboard photo")
                next_frame = after
            except (RuntimeError, ValueError) as error:
                trace.event("independent_view_failed", 2, target_id=target.id, detail=str(error)[:300])

        reply = await future
        returned = monotonic() - started
        if returned >= task.deadline or not book.accept(reply, returned):
            raise RuntimeError("Frozen vision answer was late or did not match its query binding")
        trace.event("query_applied", 3, binding=asdict(reply.binding), outcome=reply.outcome)
        return {
            "query": asdict(reply.binding),
            "outcome": reply.outcome,
            "independent_target": next_frame.object_id if next_frame else None,
            "independent_photo_id": next_frame.id if next_frame else None,
            "independent_photo_sha256": next_frame.digest if next_frame else None,
            "original_photo_sha256": first_frame.digest,
            "original_photo_bound": reply.binding.frame_id == first_frame.id,
            "original_photo_reused_from_initial_view": reused,
            "admitted_independent_view_count": len(views),
        }
    finally:
        if not future.done():
            future.cancel()
        worker.close()


def run_probe(backend, instruction, directory, event, timeout, *, model, trace):
    started = monotonic()
    model_deadline = started + timeout
    model.deadline_monotonic = model_deadline
    backend.execution_deadline_monotonic = model_deadline
    result = {"mode": "ACTUAL_GPT_FROZEN_QUERY_INTEGRATION_PROBE", "probe_completed": False}
    try:
        with trace.measure("model_request", 1, purpose="compile_task_requirements"):
            intent = interpret(model, instruction)
        write_json(directory / "task_requirements.json", asdict(intent))
        initial, observation = _take_photo(backend, directory, trace, started, 1, "scene")
        with trace.measure("model_request", 2, purpose="locate_visible_targets",
                           frame_id=initial.id, image_sha256=initial.digest):
            targets = locate_targets(model, intent, initial)
        write_json(directory / "observed_task.json", {**asdict(intent),
                   "targets": [asdict(target) for target in targets]})
        reusable = [target for target in targets if target.view_sufficient]
        views = _measured_views(backend, observation,
                                (target for target in targets if not target.view_sufficient), trace)
        reused = bool(reusable)
        if reused:
            first = reusable[0]
            first_goal = observation["position"]
            first_frame = Frame(initial.id, first.id, initial.version, initial.captured_at,
                                initial.payload, initial.provenance)
            trace.event("initial_photo_reused", 1, target_id=first.id, frame_id=first_frame.id)
        else:
            if not views:
                raise RuntimeError("No adequate initial photos or map-admitted replacement viewpoints")
            first, first_goal = views.pop(0)
            with trace.measure("action", 1, action="inspect_first_required_target", target_id=first.id):
                backend.move(first_goal)
            first_frame, _ = _take_photo(backend, directory, trace, started, 2, first.id)
        spec = QuerySpec("first-visual-answer", first.id, OUTCOMES,
                         min(first_frame.captured_at + 15, timeout), timeout)
        requirements = list(intent.requirements_for(first.id, spec.id))
        skills = []
        travel = []
        for target, goal in views:
            requirement = Requirement(target.id + ":inspection_view", target.id, "inspection_view")
            requirements.append(requirement)
            skills.append(Skill("view:" + target.id, target.id, (requirement.id,), 1.0))
            # A conservative local estimate only for this integration probe.
            travel.append((first.id, target.id, 2.0 + 2.0 * dist(_xyz(first_goal), _xyz(goal))))
        task = TaskContract(instruction, first.id, "probe-end", timeout, 0,
                            tuple(requirements), (spec,), tuple(skills), tuple(travel))
        snapshot = Snapshot(first_frame.captured_at, first.id, {}, {}, {})
        result.update(asyncio.run(_parallel(
            backend, directory, trace, model, started, task, snapshot, spec,
            first_frame, first, intent.question, {target.id: (target, goal) for target, goal in views}, reused,
        )))
        result["probe_completed"] = True
    except (RuntimeError, ValueError) as error:
        result["error"] = {"type": type(error).__name__, "detail": str(error)[:350]}
    finally:
        write_json(directory / "gpt_probe.json", result)

    event("gpt_query_probe_finished", probe_completed=result["probe_completed"])
    report = {"outcome": "incomplete", "records": [], "summary": "Frozen query integration probe"}
    write_json(directory / "report.json", report)
    trace.event("report_written", 3, outcome=report["outcome"])
    return report, {"calls": model.calls, "model_wall_seconds": model.wall_seconds,
                    "episode_wall_seconds": monotonic() - started, "probe_completed": result["probe_completed"]}
