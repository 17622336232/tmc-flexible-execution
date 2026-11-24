"""Assign public test houses and tasks without loading their layouts or images."""

from hashlib import sha256
from pathlib import Path
from random import Random

from experiments.task_attack.runtime import write_json

from .public_visual import PROCTHOR_REVISION, THOR_COMMIT, VALIDATION_SOURCE
from .runner import _hash, read_json


TEST_BYTES = 5_022_093
TEST_SHA256 = "9a9fa6f134e76fe87f3fd92c00883651cf9fadf4e9ad4072d6d73be229f001dc"
TEST_HOUSES = 1000
TEST_SOURCE = {"official_split": "test", "archive_bytes": TEST_BYTES, "archive_sha256": TEST_SHA256}
TARGETS = (("Chair", "椅子", "right", "Bed", "床"),
           ("Sofa", "沙发", "left", "CoffeeTable", "茶几"),
           ("Window", "窗户", "right", "HousePlant", "室内盆栽"),
           ("DiningTable", "餐桌", "left", "Fridge", "冰箱"))
METHODS = ("conditional-async", "conditional-prefetch", "conditional-planner", "conditional-a")
VALIDATION_ROLE = "PUBLIC_VISUAL_MATCHED_DEVELOPMENT"
FINAL_ROLE = "PUBLIC_VISUAL_MATCHED_HELDOUT"


def _tasks(house_index, split_seed, split="test"):
    house = f"procthor-{split}-house-{house_index:04d}"
    for number, (front, front_text, direction, side, side_text) in enumerate(TARGETS, 1):
        base = f"{house}-view-{number}"
        rotation = "向右" if direction == "right" else "向左"
        instruction = (f"先用初始照片判断是否能看到{front_text}。若看到了{front_text}，{rotation}转90度，"
                       f"用一张新照片判断是否能看到{side_text}，恢复原来的朝向后报告两项结论。"
                       "若没看到，只报告初始照片的结论。看不清时报告无法判定。报告中标明作为依据的照片。\n"
                       "采集规则：允许在判读结果返回前采集以下材料：side_photo。")
        yield {"id": base, "group_id": house, "official_house_index": house_index, "task_number": number,
               "instruction": instruction, "expected_language_contract": {"primary": front,
               "direction": direction, "secondary": side},
               "order": Random(_hash([split_seed, base])).sample(METHODS, len(METHODS))}


def prepare(archive, output, defaults):
    data = defaults["data"]
    count, houses, seed = data["public_procthor_instances"], data["public_procthor_test_houses"], data["split_seed"]
    if count != houses * len(TARGETS) or not 1 <= houses <= TEST_HOUSES or data["public_methods_cap"] != len(METHODS):
        raise ValueError("Public test slots do not match the registered target and method schedule")
    if archive.stat().st_size != TEST_BYTES or sha256(archive.read_bytes()).hexdigest() != TEST_SHA256:
        raise ValueError("This is not the pinned official ProcTHOR test archive")
    order = sorted(range(TEST_HOUSES), key=lambda index: _hash([PROCTHOR_REVISION, "test", seed, index]))
    cases = [case for index in order[:houses] for case in _tasks(index, seed)]
    source = {"dataset": "ProcTHOR-10K", **TEST_SOURCE, "revision": PROCTHOR_REVISION, "split_houses": TEST_HOUSES}
    catalog = {"schema": 1, "role": "PUBLIC_SCENES_DERIVED_TASKS", "state": "PLAN_ONLY",
               "split_seed": seed, "source": source, "cases": cases, "primary_house_indices": order[:houses],
               "reserve_house_indices": order[houses:], "house_replacement_reasons":
               ["engine_cannot_load", "registered_task_uninstantiable"],
               "asset_contents_examined": False, "placed_in_official_benchmark": False}
    fingerprint = _hash(catalog)
    jobs = [{"id": "public-" + _hash([fingerprint, case["id"], method])[:20], "case": case["id"],
             "group_id": case["group_id"], "method": method, "state": "PENDING_FINAL_PROTOCOL"}
            for case in cases for method in case["order"]]
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("split_manifest.json", catalog), ("run_manifest.json", {
            "catalog_sha256": fingerprint, "mode": data["main_evaluation_mode"], "planned_jobs": len(jobs),
            "recorded_jobs": 0, "jobs": jobs})):
        path = output / name
        if path.is_file() and read_json(path) != value:
            raise ValueError("The existing public schedule differs from the registered split")
        if not path.is_file():
            write_json(path, value)
    return {"cases": len(cases), "houses": houses, "planned_jobs": len(jobs), "recorded_jobs": 0,
            "catalog_sha256": fingerprint, "state": "PLAN_ONLY"}


def prepare_validation(archive, output, defaults, *, build_dir, packages, xvfb_run):
    if (archive.stat().st_size != VALIDATION_SOURCE["archive_bytes"]
            or sha256(archive.read_bytes()).hexdigest() != VALIDATION_SOURCE["archive_sha256"]):
        raise ValueError("Use the pinned official ProcTHOR validation archive")
    build = "thor-Linux64-" + THOR_COMMIT
    executable = build_dir / "releases" / build / build
    if (not executable.is_file() or not (packages / "ai2thor/controller.py").is_file()
            or not Path(xvfb_run).is_file()):
        raise ValueError("Prepare the existing isolated AI2-THOR OpenGL runtime before registration")
    runtime = {"archive": str(archive.resolve()), "build_dir": str(build_dir.resolve()),
               "unity_executable": str(executable.resolve()), "packages": str(packages.resolve()),
               "xvfb_run": str(xvfb_run.resolve()), "source": VALIDATION_SOURCE}
    seed = defaults["data"]["split_seed"]
    # One authored prompt per house, distinct from validation house 0 used for the connection test.
    indices = sorted(range(1, 100), key=lambda index: _hash([PROCTHOR_REVISION, "val", seed, index]))[:len(TARGETS)]
    cases = []
    for number, index in enumerate(indices):
        row = tuple(_tasks(index, seed, "val"))[number]
        cases.append({**row, "family": "conditional_camera_presence", "domain": "public-room", "seed": index,
                      "episode_timeout_seconds": 480})
    catalog = {"schema": 1, "role": VALIDATION_ROLE, "intended_pairs": len(cases),
               "runtime": runtime, "cases": cases}
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "pilot_manifest.json"
    if marker.is_file() and read_json(marker) != catalog:
        raise ValueError("An existing validation schedule cannot be changed")
    if not marker.is_file():
        write_json(marker, catalog)
    return {"cases": len(cases), "jobs": sum(len(case["order"]) for case in cases), "output": str(output)}


def prepare_final(archive, blind_plan, validation, selection, output, defaults):
    """Bind the test schedule after validation and configuration selection, without unsealing houses."""
    from .control import REQUEST_LIMIT, verified_cohort
    from .pilot_schedule import VISUAL_ROLES
    from experiments.task_attack import EFFORT, MODEL

    evidence = {}
    for name, directory, role, count in (
        ("public_validation", validation, VALIDATION_ROLE, len(TARGETS)),
        ("configuration_selection", selection, VISUAL_ROLES["selection"],
         defaults["data"]["development_selection_instances"]),
    ):
        _, evidence[name] = verified_cohort(name, directory, role, count)

    prepare(archive, blind_plan, defaults)
    assigned = read_json(blind_plan / "split_manifest.json")
    runtime = {**read_json(validation / "pilot_manifest.json")["runtime"], "archive": str(archive.resolve()),
               "source": TEST_SOURCE}
    output = Path(output).resolve()
    freeze = {"schema": 1, "role": FINAL_ROLE, "blind_assignment_sha256": _hash(assigned), "evidence": evidence,
              "model": MODEL, "reasoning_effort": EFFORT, "request_limit": REQUEST_LIMIT,
              "episode_timeout_seconds": 480, "methods": list(METHODS), "metrics": defaults["evaluation"],
              "runtime": runtime, "mode": defaults["data"]["main_evaluation_mode"]}
    cases = [{**row, "family": "conditional_camera_presence", "domain": "public-room",
              "seed": row["official_house_index"], "episode_timeout_seconds": freeze["episode_timeout_seconds"]}
             for row in assigned["cases"]]
    catalog = {"schema": 1, "role": FINAL_ROLE, "intended_pairs": len(cases), "runtime": runtime, "cases": cases,
               "frozen_evaluation_sha256": _hash(freeze), "frozen_evaluation": str(output / "frozen_evaluation.json")}
    output.mkdir(parents=True, exist_ok=True)
    for path, value in ((output / "frozen_evaluation.json", freeze), (output / "pilot_manifest.json", catalog)):
        if path.is_file() and read_json(path) != value:
            raise ValueError("The held-out public evaluation is already registered")
        if not path.is_file():
            write_json(path, value)
    return {"cases": len(cases), "jobs": sum(len(case["order"]) for case in cases), "output": str(output),
            "frozen_evaluation_sha256": catalog["frozen_evaluation_sha256"]}
