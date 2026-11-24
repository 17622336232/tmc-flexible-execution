"""Recheck saved symbolic labels on a reproducible, stratified sample."""

from collections import Counter, defaultdict, deque
import json
from math import isclose
import re
from statistics import mean

from experiments.task_attack.runtime import write_json

from .ranker_data import LABEL_MODE, _instance, label
from .runner import _hash, read_json


def _rows(path):
    with path.open(encoding="utf8") as handle:
        return [json.loads(line) for line in handle]


def _group_record(rows, split):
    if any(row["mode"] != LABEL_MODE or row["split"] != split for row in rows):
        raise ValueError("The archived examples contain a wrong data mode or split")
    groups = Counter(row["group_sha256"] for row in rows)
    return set(groups), {"states": len(rows), "layout_groups": len(groups),
                         "structures": len({row["structure_sha256"] for row in rows}),
                         "states_per_group_maximum": max(groups.values()), "visual_sources": 0}


def _category(candidate):
    if not candidate["verified_common_continuation"]:
        return "unverified_continuation"
    return "positive_gain" if candidate["cost_gain_label"] > 0 else "nonpositive_gain"


def _sample(rows, count):
    buckets = defaultdict(list)
    for row in rows:
        by_category = {}
        for entry in row["candidates"]:
            kind = _category(entry)
            key = _hash([row["id"], entry["candidate"]])
            if kind not in by_category or key < by_category[kind][0]:
                by_category[kind] = key, entry
        for kind, (key, entry) in by_category.items():
            buckets[(row["structure_sha256"], kind)].append((key, row, entry))
    queues = [deque(sorted(buckets[key], key=lambda pair: pair[0])) for key in sorted(buckets)]
    used = set()
    while queues and len(used) < count:
        remaining = []
        for queue in queues:
            while queue and queue[0][1]["id"] in used:
                queue.popleft()
            if queue and len(used) < count:
                _, row, candidate = queue.popleft()
                used.add(row["id"])
                yield row, candidate
            if queue:
                remaining.append(queue)
        queues = remaining


def audit(directory, output, *, sample_count=100):
    protocol = read_json(directory / "protocol.json")
    seed = protocol["split_seed"]
    data = {}
    for name, size in (("train", 10000), ("calibration", 48), ("selection", 48)):
        data[name] = _rows(directory / f"{name}-{size}.jsonl")
        if len(data[name]) != size:
            raise ValueError("An archived data split has missing records")
    groups = {name: _group_record(rows, name) for name, rows in data.items()}
    if any(groups[a][0] & groups[b][0] for a, b in (("train", "calibration"), ("train", "selection"),
                                                 ("calibration", "selection"))):
        raise ValueError("A layout group crossed the training, calibration or selection boundary")
    training_scales = {str(size): _group_record(data["train"][:size], "train")[1]
                       for size in (1000, 3000, 10000)}
    if any(record["states_per_group_maximum"] > 4 for record in training_scales.values()):
        raise ValueError("A training layout contributed more states than allowed")

    examples = []
    for saved, candidate in _sample(data["train"], sample_count):
        match = re.fullmatch(r"synthetic-(\d+)-(\d+)", saved["id"])
        if match is None:
            raise ValueError("Cannot trace the example to its generation seed and decision state")
        case, group, structure = _instance(*(int(i) for i in match.groups()), seed)
        replayed = {"split": "train", **label(case, group, structure)}
        if saved != replayed:
            raise ValueError("A saved candidate label changed when replayed: " + saved["id"])
        branches = candidate["branches"]
        baseline = mean(item["restricted_seconds"] for item in saved["baseline"])
        restricted = mean(item["restricted_seconds"] for item in branches) if branches else case.task.deadline
        valid = bool(branches and all(item["success"] and not item["wrong_accepted"] for item in branches))
        if not isclose((baseline - restricted) / case.task.deadline, candidate["cost_gain_label"], abs_tol=1e-12) \
                or candidate["success_label"] != int(valid):
            raise ValueError("Saved labels disagree with their branch scores: " + saved["id"])
        worlds = [{"outcomes": world, "required_materials": [requirement.field for requirement in case.task.requirements
                   if requirement.applies(world)]} for world in case.task.worlds({})]
        examples.append({"decision_state_id": case.id, "layout_group": group, "program_structure": structure,
                         "canonical_instruction": case.task.instruction, "classification": _category(candidate),
                         "requirements_by_branch": worlds, "candidate": candidate, "baseline_branches": saved["baseline"]})
    if len(examples) != sample_count:
        raise ValueError("Insufficient different states for the registered audit sample")
    report = {"mode": LABEL_MODE, "source": str(directory), "training_scales": training_scales,
              "calibration": groups["calibration"][1], "selection": groups["selection"][1],
              "sampled_states_replayed": len(examples), "sample_categories": dict(Counter(row["classification"]
                                                                                         for row in examples)),
              "human_reviewed": False, "examples": examples}
    write_json(output, report)
    return {key: value for key, value in report.items() if key != "examples"}
