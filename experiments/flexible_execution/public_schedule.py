"""Assign public test houses and tasks without loading their layouts or images."""

from hashlib import sha256
from random import Random

from experiments.task_attack.runtime import write_json

from .public_visual import PROCTHOR_REVISION
from .runner import _hash, read_json


TEST_BYTES = 5_022_093
TEST_SHA256 = "9a9fa6f134e76fe87f3fd92c00883651cf9fadf4e9ad4072d6d73be229f001dc"
TEST_HOUSES = 1000
TARGETS = (("Chair", "椅子", "right", "Bed", "床"),
           ("Sofa", "沙发", "left", "CoffeeTable", "茶几"),
           ("Window", "窗户", "right", "HousePlant", "室内盆栽"),
           ("DiningTable", "餐桌", "left", "Fridge", "冰箱"))
METHODS = ("conditional-async", "conditional-prefetch", "conditional-planner", "conditional-a")


def _tasks(house_index, split_seed):
    house = f"procthor-test-house-{house_index:04d}"
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
    source = {"dataset": "ProcTHOR-10K", "official_split": "test", "revision": PROCTHOR_REVISION,
              "archive_bytes": TEST_BYTES, "archive_sha256": TEST_SHA256, "split_houses": TEST_HOUSES}
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
