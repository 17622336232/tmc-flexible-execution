"""Measure one live vision request against independent simulated inspection work."""

import asyncio
import json
from dataclasses import asdict, replace
from hashlib import sha256
from math import dist
from threading import Event
from time import monotonic

from experiments.task_attack.agent import run_agent
from experiments.task_attack.runtime import write_json

from ..evidence import Frame
from ..executor import Snapshot, async_action
from ..gpt_vision import OUTCOMES, frozen_prompt, interpret, judge, locate_targets
from ..queries import AsyncQueryWorker, QueryBook
from ..task_ir import QuerySpec, Requirement, Skill, TaskContract
from .viewpoints import measured_views, xyz


def _take_photo(backend, directory, trace, started, step, target_id):
    with trace.measure("observation", step, target_id=target_id):
        observation, geometry = backend.observe()
    frame = Frame(observation["observation_id"], target_id, step, monotonic() - started,
                  observation["data"], "simulated_onboard_rgb")
    tag = "probe_%04d" % step
    (directory / (tag + ".jpg")).write_bytes(frame.payload)
    write_json(directory / (tag + ".json"), {
        "image_id": tag, "sha256": frame.digest, "position": observation["position"],
        "yaw": observation["yaw"], "capture_ros": observation["ros_stamp"], "frame_id": frame.id,
    })
    write_json(directory / ("probe_%04d_geometry.json" % step), geometry)
    trace.event("frozen_photo", step, target_id=target_id, frame_id=frame.id,
                image_sha256=frame.digest, pose=observation["position"], ros_stamp=observation["ros_stamp"])
    return frame, observation, geometry


async def _observe_while_pending(backend, directory, trace, client, started, task, snapshot, spec,
                                 first_frame, first_target, question, views, reused, during_pending):
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

    async def receive():
        reply = await future
        returned = monotonic() - started
        if returned >= task.deadline or not book.accept(reply, returned):
            raise RuntimeError("Frozen vision answer was late or did not match its query binding")
        trace.event("query_applied", 3, binding=asdict(reply.binding), outcome=reply.outcome)
        return reply

    try:
        if not await loop.run_in_executor(None, invoked.wait, max(0.0, task.deadline - submitted)):
            raise RuntimeError("Frozen vision worker did not start before the episode deadline")
        reply = None if during_pending else await receive()
        state = replace(snapshot, at=monotonic() - started, results=dict(book.results))
        action = async_action(task, state)
        if action is not None:
            target, goal = views[action.site]
            try:
                with trace.measure("action", 2, action="inspect_other_required_target", target_id=target.id):
                    await loop.run_in_executor(None, backend.move, goal)
                after, _, _ = _take_photo(backend, directory, trace, started, 3, target.id)
                if after.id == first_frame.id or after.digest == first_frame.digest:
                    raise RuntimeError("Independent work did not capture a distinct onboard photo")
                next_frame = after
            except (RuntimeError, ValueError) as error:
                trace.event("independent_view_failed", 2, target_id=target.id, detail=str(error)[:300])

        if reply is None:
            reply = await receive()
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


def _prepare_views(backend, instruction, directory, timeout, model, trace, started, during_pending):
    model_deadline = started + timeout
    model.deadline_monotonic = model_deadline
    backend.execution_deadline_monotonic = model_deadline
    result = {"mode": "FROZEN_QUERY_OBSERVATION_PREFIX", "probe_completed": False,
              "may_dispatch_while_pending": during_pending}
    try:
        with trace.measure("model_request", 1, purpose="compile_task_requirements"):
            intent = interpret(model, instruction)
        write_json(directory / "task_requirements.json", asdict(intent))
        initial, observation, geometry = _take_photo(backend, directory, trace, started, 1, "scene")
        with trace.measure("model_request", 2, purpose="locate_visible_targets",
                           frame_id=initial.id, image_sha256=initial.digest):
            targets = locate_targets(model, intent, initial)
        write_json(directory / "observed_task.json", {**asdict(intent),
                   "targets": [asdict(target) for target in targets]})
        reusable = [target for target in targets if target.view_sufficient]
        views = measured_views(backend, observation, geometry,
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
            first_frame, _, _ = _take_photo(backend, directory, trace, started, 2, first.id)
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
            travel.append((first.id, target.id, 2.0 + 2.0 * dist(xyz(first_goal), xyz(goal))))
        task = TaskContract(instruction, first.id, "probe-end", timeout, 0,
                            tuple(requirements), (spec,), tuple(skills), tuple(travel))
        snapshot = Snapshot(first_frame.captured_at, first.id, {}, {}, {})
        result.update(asyncio.run(_observe_while_pending(
            backend, directory, trace, model, started, task, snapshot, spec,
            first_frame, first, intent.question, {target.id: (target, goal) for target, goal in views}, reused,
            during_pending,
        )))
        result["question"] = intent.question
        result["probe_completed"] = True
    except (RuntimeError, ValueError) as error:
        result["error"] = {"type": type(error).__name__, "detail": str(error)[:350]}
    finally:
        write_json(directory / "gpt_probe.json", result)
    return result


def run_probe(backend, instruction, directory, event, timeout, *, model, trace):
    started = monotonic()
    result = _prepare_views(backend, instruction, directory, timeout, model, trace, started, True)

    event("gpt_query_probe_finished", probe_completed=result["probe_completed"])
    report = {"outcome": "incomplete", "records": [], "summary": "Frozen query integration probe"}
    write_json(directory / "report.json", report)
    trace.event("report_written", 3, outcome=report["outcome"])
    return report, {"calls": model.calls, "model_wall_seconds": model.wall_seconds,
                    "episode_wall_seconds": monotonic() - started, "probe_completed": result["probe_completed"]}


def run_paired(backend, instruction, directory, event, timeout, *, model, trace, during_pending):
    started = monotonic()
    prefix = _prepare_views(backend, instruction, directory, timeout, model, trace, started, during_pending)
    archive = []
    original_id = None
    for path in sorted(directory.glob("probe_[0-9][0-9][0-9][0-9].json")):
        meta = json.loads(path.read_text(encoding="utf8"))
        payload = path.with_suffix(".jpg").read_bytes()
        if path.stem != meta["image_id"] or sha256(payload).hexdigest() != meta["sha256"]:
            raise ValueError("Cannot hand an unverified observation to the report controller")
        archive.append({"image_id": meta["image_id"], "position": meta["position"], "yaw": meta["yaw"],
                        "capture_ros": meta["capture_ros"], "data": payload})
        if prefix.get("query", {}).get("frame_id") == meta["frame_id"]:
            original_id = meta["image_id"]
    memory = ''
    if prefix["probe_completed"] and original_id is not None:
        memory = json.dumps({"frozen_photo": original_id, "question": prefix["question"],
                             "observation": prefix["outcome"], "coverage": "one queried target"}, ensure_ascii=False)
        archive.sort(key=lambda view: view["image_id"] != original_id)
    trace.event("full_inspection_started", 4, seeded_frames=len(archive), known_visual_result=bool(memory))
    report, metrics = run_agent(backend, instruction, directory, event, timeout, model=model, trace=trace,
                                prior_views=archive, prior_memory=memory, episode_started=started, step_offset=3)
    metrics.update(probe_completed=prefix["probe_completed"], shared_input_frames=len(archive))
    return report, metrics
