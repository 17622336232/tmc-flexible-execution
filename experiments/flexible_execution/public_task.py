"""Run a derived language-and-RGB inspection in an official validation house."""

import argparse
import asyncio
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from math import dist
from pathlib import Path
from time import monotonic

from experiments.task_attack.model import ModelClient
from experiments.task_attack.runtime import write_json

from .evidence import Frame, Observation, accept
from .executor import Snapshot, permitted_observation
from .gpt_vision import OUTCOMES, Material
from .live_planning import NEW_METHODS, permission_for, select
from .public_audit import save_segmentation
from .public_visual import _house, camera_jpeg, open_controller, source_record
from .queries import AsyncQueryWorker, QueryBook
from .request_budget import RequestBudget
from .task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract
from .timeline import Timeline, summarize


CLASSES = ("ArmChair", "Bed", "Chair", "CoffeeTable", "Desk", "DiningTable", "Fridge",
           "HousePlant", "Microwave", "Sofa", "Television", "Toilet", "Window")
TURNS = {"right": ("RotateRight", "RotateLeft", 90), "left": ("RotateLeft", "RotateRight", -90)}
MIN_OBJECT_PIXELS = 36
MODEL_TIMES = {"initial_prediction": 12.0, "initial_ceiling": 90.0, "remaining_judgments": 80.0}
PLATFORM_TIMES = {"turn": 2.0, "photo": 0.1}
PARSE = """把用户的房间巡检指令转换为一个 JSON 对象。只整理目标和分支。采集规则适用于执行器，
不属于这份输出。primary 是初始照片中要找的类别，secondary 是侧向新照片中要找的类别。
对象类别使用给定词表的英文名称。direction 使用 left 或 right。如果用户要求只在初始结果为
“看到了”时检查侧面，side_when 就是 ["positive"]。按用户文字保留恢复朝向和看不清时报告的要求。
示例格式（下面的类别和方向仅作示例）：
{"primary":"Sofa","direction":"left","secondary":"Desk","side_when":["positive"],
"restore_heading":true,"unresolved_allowed":true}
仅当目标或分支无法用这些字段表达时，返回 {"unsupported":true}。"""
JUDGE = """Use only the attached RGB photo. Is an instance of the requested class visibly
present in this exact field of view? Answer JSON only: {"outcome":"positive"},
{"outcome":"negative"}, or {"outcome":"uncertain"} when the photo does not suffice.
For absence, answer negative when no identifiable instance appears in the photo."""
REPORT = """Submit the completed room inspection from the verified judgments and the exact
attached photos, following the user instruction. Return JSON only with front, front_sha256,
side, side_sha256. The front and side values are the supplied outcomes. When the side
material is not required, both side fields must be null. Copy the supplied digests exactly."""


@dataclass(frozen=True)
class RoomIntent:
    primary: str
    direction: str
    secondary: str


@dataclass(frozen=True)
class ValidationCase:
    house_index: int
    instruction: str
    expected: RoomIntent


VALIDATION = ValidationCase(
    0, "先用初始照片判断是否能看到椅子。若看到了椅子，向右转90度，用一张新照片判断是否能看到床，"
    "恢复原来的朝向后报告两项结论。若没看到椅子，只报告初始照片的结论。看不清时报告无法判定。"
    "报告中标明作为依据的照片。\n"
    "采集规则：允许在判读结果返回前采集以下材料：side_photo。",
    RoomIntent("Chair", "right", "Bed"),
)


def parse_intent(value):
    if (not isinstance(value, dict) or set(value) != {"primary", "direction", "secondary", "side_when",
                                                      "restore_heading", "unresolved_allowed"}
            or value["primary"] not in CLASSES or value["secondary"] not in CLASSES
            or value["direction"] not in tuple(TURNS) or value["side_when"] != ["positive"]
            or value["restore_heading"] is not True or value["unresolved_allowed"] is not True):
        raise ValueError("The shared room inspector did not return a supported instruction")
    return RoomIntent(value["primary"], value["direction"], value["secondary"])


def _same_pose(first, last, turn=0):
    xyz = ("x", "y", "z")
    return (dist([first["position"][axis] for axis in xyz], [last["position"][axis] for axis in xyz]) < 0.01
            and abs((last["rotation"]["y"] - first["rotation"]["y"] - turn + 180) % 360 - 180) < 0.1
            and abs(last["cameraHorizon"] - first["cameraHorizon"]) < 0.1)


def _truth(event, target):
    identifiers = {item["objectId"] for item in event.metadata["objects"] if item["objectType"] == target}
    sizes = [int(mask.sum()) for key, mask in event.instance_masks.items() if key in identifiers]
    return "positive" if any(count >= MIN_OBJECT_PIXELS for count in sizes) else "negative"


def verify(case, intent, report, front, side, front_truth, side_truth, restored):
    required_side = front_truth == "positive"
    fresh_side = (side is not None and side.digest != front.digest and side.captured_at >= front.captured_at
                  and side_truth in ("positive", "negative"))
    expected = {"front": front_truth, "front_sha256": front.digest,
                "side": side_truth if required_side else None,
                "side_sha256": side.digest if required_side and side is not None else None}
    return {"complete": restored and intent == case.expected and (not required_side or fresh_side)
            and report == expected, "parse_correct": intent == case.expected, "restored_heading": restored,
            "expected": expected, "side_photo_used": required_side and fresh_side,
            "truth_origin": "AI2-THOR instance segmentation after execution",
            "minimum_visible_instance_pixels": MIN_OBJECT_PIXELS}


class Room:
    def __init__(self, controller, output, trace, started):
        self.controller = controller
        self.output = output
        self.trace = trace
        self.started = started
        self.events = {}
        self.initial_pose = controller.last_event.metadata["agent"]

    def now(self):
        return monotonic() - self.started

    def take(self, name, event):
        with self.trace.measure("observation", name):
            frame = Frame(name, "room", len(self.events) + 1, self.now(), camera_jpeg(event), "procthor_rgb")
            (self.output / (name + ".jpg")).write_bytes(frame.payload)
            self.events[name] = event
        self.trace.event("camera_observation", name, image_sha256=frame.digest, captured_at=frame.captured_at,
                         pose=event.metadata["agent"])
        return frame

    def turn(self, action, expected):
        with self.trace.measure("action", action):
            event = self.controller.step(action=action)
            if not event.metadata["lastActionSuccess"] or not _same_pose(self.initial_pose, event.metadata["agent"], expected):
                raise RuntimeError("An approved camera turn did not reach its contracted pose")
        self.trace.event("camera_heading_checked", action, pose=event.metadata["agent"])
        return event


def _request(client, trace, key, prompt, data, frames=()):
    with trace.measure("model_request", key, image_sha256=[frame.digest for frame in frames]):
        return client.request(prompt, data, [{"label": frame.id + " camera RGB", "data": frame.payload}
                                             for frame in frames])


def _judge(client, trace, key, payload, prompt):
    with trace.measure("model_request", key, image_sha256=prompt["sha256"]):
        value = client.request(JUDGE, prompt, [{"label": key + " camera RGB", "data": payload}])
    if not isinstance(value, dict) or set(value) != {"outcome"} or value["outcome"] not in OUTCOMES:
        raise ValueError("Room image judgment did not contain a declared outcome")
    return value["outcome"]


async def _inspect(case, room, client, method, deadline):
    trace = room.trace
    front = room.take("front", room.controller.last_event)
    intent = parse_intent(_request(client, trace, "parse", PARSE, {"task": case.instruction, "classes": CLASSES}))
    material = Material("side_photo", ("positive",), intent.secondary, None)
    grant = permission_for(case.instruction, material)
    trace.event("material_permission", "side", **asdict(grant))
    start = room.now()
    latest = min(start + MODEL_TIMES["initial_ceiling"], deadline)
    query = QuerySpec("front_presence", "room", OUTCOMES, min(start + MODEL_TIMES["initial_prediction"], latest), latest)
    required = Requirement("side_photo", "room", material.field, Condition(query.id, ("positive",)), report=True)
    skill = Skill("photograph_side", "side", (required.id,), PLATFORM_TIMES["photo"], grant.for_query(query.id))
    contract = TaskContract(case.instruction, "front", "terminal", deadline, MODEL_TIMES["remaining_judgments"],
                            (required,), (query,), (skill,), (("front", "terminal", 0),
                            ("front", "side", PLATFORM_TIMES["turn"]), ("side", "terminal", PLATFORM_TIMES["turn"])))
    turn, restore, angle = TURNS[intent.direction]
    side = None
    early_dispatched = False
    calls_before = client.calls
    book = QueryBook()
    worker = AsyncQueryWorker(lambda payload, prompt: _judge(client, trace, "front", payload, json.loads(prompt)))
    prompt = {"target_class": intent.primary, "frame": front.id, "sha256": front.digest}
    pending = worker.submit(query, front, json.dumps(prompt, sort_keys=True), room.now(), book)
    trace.event("query_submitted", "front", binding=asdict(book.pending[query.id]))

    def log(event, **fields):
        trace.event(event, "side-search", **fields)

    try:
        while not pending.done() and client.calls == calls_before:
            await asyncio.sleep(0.025)
        if not pending.done():
            snapshot = Snapshot(room.now(), "front", {}, {}, {})
            with trace.measure("search", "side-search"):
                choice = select(contract, snapshot, method, log)
            if choice and not pending.done():
                first = permitted_observation(contract, Snapshot(room.now(), "front", {}, {}, {}), skill)
                if first and choice.prepare[0].skill == skill:
                    trace.event("approved_preparation", "side", permission=grant.source)
                    early_dispatched = True
                    side = room.take("side", room.turn(turn, angle))
        reply = await pending
        if not book.accept(reply, room.now()):
            raise RuntimeError("Reply did not match the immutable starting image")
        trace.event("query_applied", "front", binding=asdict(reply.binding), outcome=reply.outcome)
        if reply.outcome == "positive" and side is None:
            state = Snapshot(room.now(), "front", book.results, {}, {})
            if not permitted_observation(contract, state, skill):
                raise RuntimeError("The required camera turn was not permitted in the remaining budget")
            side = room.take("side", room.turn(turn, angle))
        if side is not None:
            room.turn(restore, 0)
    finally:
        await asyncio.gather(pending, return_exceptions=True)
        worker.close()

    supplied = [front]
    side_outcome = None
    if reply.outcome == "positive":
        side_outcome = _judge(client, trace, "side", side.payload,
                              {"target_class": intent.secondary, "frame": side.id, "sha256": side.digest})
        evidence = accept(required, Observation(side, material.field, side_outcome, side.digest != front.digest), room.now())
        if evidence is None:
            raise RuntimeError("The side judgment lacks newly acquired camera evidence")
        trace.event("verified_material", "side", evidence=asdict(evidence))
        supplied.append(side)
    report = _request(client, trace, "report", REPORT, {
        "task": case.instruction, "front": reply.outcome, "front_sha256": front.digest,
        "side": side_outcome, "side_sha256": side.digest if len(supplied) == 2 else None}, supplied)
    write_json(room.output / "report.json", report)
    front_truth = _truth(room.events[front.id], case.expected.primary)
    side_truth = None if side is None else _truth(room.events[side.id], case.expected.secondary)
    save_segmentation(room.output, front.id, room.events[front.id], case.expected.primary, MIN_OBJECT_PIXELS)
    if side is not None:
        save_segmentation(room.output, side.id, room.events[side.id], case.expected.secondary, MIN_OBJECT_PIXELS)
    result = verify(case, intent, report, front, side, front_truth, side_truth,
                    _same_pose(room.initial_pose, room.controller.last_event.metadata["agent"]))
    result.update(early_preparation_selected=early_dispatched, parsed=asdict(intent))
    trace.event("report_written", "report", outcome="complete" if result["complete"] else "failure")
    return result


def run(archive, output, build_dir, display, method, auth_project, *, deadline=260, request_limit=8):
    from uav_bridge.chatgpt_auth import CredentialStore

    from .control import _proc

    case = VALIDATION
    house, digest = _house(archive, case.house_index)
    output.mkdir(parents=True, exist_ok=False)
    source = source_record(case.house_index, digest, "opengl", "PUBLIC_VISUAL_VALIDATION_CLOSED_LOOP")
    write_json(output / "source.json", {**source, "instruction": case.instruction, "method": method,
        "motion_estimates": PLATFORM_TIMES, "inference_estimates": MODEL_TIMES, "episode_timeout": deadline,
        "request_limit": request_limit, "role": "ENGINEERING_VALIDATION", "ground_truth_input_to_model": False,
        "case_version": "public-room-presence-v2", "previous_validation": "closed-loop-attempt-001",
        "version_reason": "clarify the positive report and separate permit text from the parsing format"})
    write_json(output / "process.json", {"process": _proc(os.getpid()), "method": method, "episode_timeout": deadline})
    trace = Timeline(output / "timeline.jsonl")
    budget = RequestBudget(output / "requests.jsonl", request_limit)
    client = ModelClient(output / "model_calls.jsonl", credential_store=CredentialStore(
        directory=auth_project / ".runtime" / "chatgpt"), request_budget=budget)
    controller = None
    started = monotonic()
    client.deadline_monotonic = started + deadline
    try:
        controller = open_controller(house, build_dir, "opengl", display, segmentation=True, timeout=min(90, deadline))
        result = asyncio.run(_inspect(case, Room(controller, output, trace, started), client, method, deadline))
    except Exception as error:
        result = {"complete": False, "error": {"type": type(error).__name__, "code": getattr(error, "code", None),
                                               "status": getattr(error, "status", None)}}
    finished = monotonic()
    if controller is not None:
        try:
            controller.stop()
        except Exception as error:
            result["cleanup_error"] = type(error).__name__
    timing = summarize(trace.path)
    replies = timing["intervals"].get("model_request", [])
    actions = timing["intervals"].get("action", [])
    front_intervals = [item for item in replies if item["step"] == "front" and item["ok"]]
    raw_calls = output / "model_calls.jsonl"
    if raw_calls.is_file():
        calls = [json.loads(line) for line in raw_calls.read_text(encoding="utf8").splitlines()]
        dispatched = next((row for row in calls if row["input"].get("frame") == "front"
                           and row.get("output", {}).get("outcome") in OUTCOMES
                           and "dispatched_monotonic_seconds" in row), None)
        if dispatched:
            front_intervals = [{"start": dispatched["dispatched_monotonic_seconds"],
                                "finish": dispatched["return_monotonic_seconds"]}]
    result["first_reply_after_motion_started"] = any(
        req["start"] < action["start"] < req["finish"] for req in front_intervals for action in actions)
    write_json(output / "timing.json", timing)
    result.update(episode_seconds=finished - started, cleanup_seconds=monotonic() - finished,
                  requests=budget.summary(), model_action_overlap_seconds=timing["model_action_overlap_seconds"])
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("validation", "output", "build-dir", "auth-project", "simulator-lock"):
        parser.add_argument("--" + path, type=Path, required=True)
    parser.add_argument("--display", default=os.environ.get("DISPLAY"))
    parser.add_argument("--method", choices=NEW_METHODS, default="conditional-a")
    args = parser.parse_args()
    if not args.display:
        parser.error("Provide the existing isolated X display")
    with args.simulator_lock.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another simulator job holds the shared lock")
        result = run(args.validation, args.output, args.build_dir, args.display, args.method, args.auth_project)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
