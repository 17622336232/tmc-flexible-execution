"""Bind matched simulator arms to the same genuinely observed front-end output."""

from dataclasses import asdict
from hashlib import sha256

from experiments.task_attack import EFFORT, MODEL
from experiments.task_attack.runtime import write_json

from .gpt_vision import parse_intent, parse_targets
from .requirement_program import parse
from .runner import _hash, read_json


def identity(instruction, scene, frame, uses_program):
    return {"scene_sha256": scene, "overview_sha256": frame.digest,
            "instruction_sha256": sha256(instruction.encode("utf8")).hexdigest(),
            "model": MODEL, "reasoning_effort": EFFORT, "typed": uses_program}


def get(path, key, instruction):
    if not path.is_file():
        return "unregistered", None
    record = read_json(path)
    content = record["reference"]
    if content["schema"] != 1 or record["sha256"] != _hash(content):
        raise ValueError("The matched front-end reference changed")
    original = content["input"]
    if any(value != original.get(name) for name, value in key.items() if name != "overview_sha256"):
        raise ValueError("The matched front-end belongs to different authorized inputs")
    if key["overview_sha256"] != original["overview_sha256"]:
        return "different_overview", None
    program = parse(content["program"], instruction) if key["typed"] else None
    intent = parse_intent(content["intent"], instruction)
    targets = parse_targets({"targets": content["targets"]})
    return "matched", (program, intent, targets, record["sha256"])


def register(path, key, program, intent, targets, episode, photo):
    if path.exists():
        raise ValueError("The matched front-end reference is already registered")
    content = {"schema": 1, "input": key, "program": asdict(program) if program else None,
               "intent": {"status": "supported", **asdict(intent)},
               "targets": [{name: value for name, value in asdict(target).items() if name != "id"} for target in targets],
               "source": {"episode": str(episode), "overview_frame_id": photo.id}}
    digest = _hash(content)
    write_json(path, {"sha256": digest, "reference": content})
    return digest
