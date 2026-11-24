"""Assign independent custom evaluation inputs before rendering any test scene."""

from collections import Counter
from hashlib import sha256
from random import Random

from experiments.task_attack.runtime import write_json
from experiments.task_attack.tasks import SCENES

from .requirement_program import parse
from .runner import _hash, read_json


def _comparison(op, field, value):
    return {"op": op, "field": field, "right": {"literal": value}}


def _predicate(op, *children):
    return {"op": op, "children": list(children)}


def _structures():
    yield "crack_and_heat", (
        "检查每台设备，留下有裂纹且温度严格高于70摄氏度的候选，只报告这些候选中温度最低的所有设备。"
        "报告编号、裂纹和温度，附各自的相机证据。没有符合条件的设备时报告无结果。"
    ), {"scope": {"mode": "all", "description": "设备"}, "fields": {"cracked": "flag", "temperature": "number"},
        "filter": _predicate("and", _comparison("eq", "cracked", True), _comparison("gt", "temperature", 70)),
        "extreme": {"op": "min", "field": "temperature"}, "report": ["cracked", "temperature"]}
    yield "inventory", (
        "检查所有货位。报告红箱超过2箱或蓝箱达到4箱的货位编号、实际箱数和颜色，"
        "给出每个货位的相机证据。"
    ), {"scope": {"mode": "all", "description": "仓储货位"}, "fields": {"box_count": "count", "color": "label"},
        "filter": _predicate("or",
                             _predicate("and", _comparison("eq", "color", "red"), _comparison("gt", "box_count", 2)),
                             _predicate("and", _comparison("eq", "color", "blue"), _comparison("ge", "box_count", 4))),
        "extreme": None, "report": ["box_count", "color"]}
    yield "fire_and_access", (
        "查看各校园点位。只报告两种点位，第一种是通道堵住且没有灭火器，第二种是通道畅通且有灭火器。"
        "报告编号、通道情况和灭火器情况，附相机证据。"
    ), {"scope": {"mode": "all", "description": "校园点位"},
        "fields": {"blocked": "flag", "extinguisher_present": "flag"},
        "filter": _predicate("or", *(_predicate("and", _comparison("eq", "blocked", value),
                                                _comparison("eq", "extinguisher_present", not value))
                                      for value in (True, False))),
        "extreme": None, "report": ["blocked", "extinguisher_present"]}


def prepare(output, defaults, development):
    data = defaults["data"]
    strata = data["custom_test_strata"]
    if set(strata) != {"structure", "scene", "execution_condition"} or sum(strata.values()) != data["final_custom_instances"]:
        raise ValueError("The three custom test strata do not match the requested total")
    seen = sorted({row["family"] for row in development["cases"] if row["split"] == "calibration"})
    structures = tuple(_structures())
    for _, instruction, reference in structures:
        parse(reference, instruction)
    cases = []
    for stratum, count in strata.items():
        rng = Random(_hash([data["split_seed"], "custom-test", stratum]))
        start = {"structure": 4_000_000, "scene": 5_000_000, "execution_condition": 6_000_000}[stratum]
        layouts = rng.sample(range(start, start + 100_000), count)
        states = rng.sample(range(start + 100_000, start + 200_000), count)
        for index, (layout, state) in enumerate(zip(layouts, states), 1):
            family, instruction, expected = None, None, None
            if stratum == "structure":
                family, instruction, expected = structures[(index - 1) % len(structures)]
            else:
                family = seen[(index - 1) % len(seen)]
            base = f"custom-{stratum.replace('_', '-')}-{index:03d}"
            count_objects = (3, 5)[index % 2] if stratum == "scene" else (2, 4)[index % 2]
            duration = (360, 600)[(index - 1) % 2] if stratum == "execution_condition" else 480
            cases.append({"id": base, "base_group_id": base, "stratum": stratum, "family": family,
                          "domain": SCENES[family], "seed": state, "layout_seed": layout,
                          "equipment_count": count_objects, "episode_timeout_seconds": duration,
                          "instruction": instruction, "oracle_requirement_program": expected,
                          "independent_scene_materialized": False})
    layout_seeds = [case["layout_seed"] for case in cases]
    if len(set(layout_seeds)) != len(cases) or set(layout_seeds) & {row["layout_seed"] for row in development["cases"]}:
        raise ValueError("A held-out layout seed overlaps another dataset role")
    manifest = {"schema": 1, "role": "FINAL_CUSTOM_EVALUATION", "state": "PLAN_ONLY", "split_seed": data["split_seed"],
                "development_manifest_sha256": _hash(development), "counts": dict(Counter(row["stratum"] for row in cases)),
                "mode": data["main_evaluation_mode"], "cases": cases, "recorded_episode_count": 0,
                "pending": ["renderer verification", "frozen baseline selection", "final protocol lock"]}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "split_manifest.json"
    if path.is_file() and read_json(path) != manifest:
        raise ValueError("The reserved custom evaluation assignment is already saved")
    if not path.is_file():
        write_json(path, manifest)
    return {"state": manifest["state"], "strata": manifest["counts"], "cases": len(cases),
            "recorded_episodes": 0, "manifest_sha256": _hash(manifest)}


def prepare_development(output, calibration):
    from .development_inputs import PILLOW_VERSION, _reuse_or_build, scene_fingerprint
    from .pilot_schedule import TYPED_ROLE

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    instruction = ("查看这一排设备。报告有裂纹或者温度严格高于80摄氏度的设备，"
                   "写出设备本体编号、是否有裂纹及温度，附每台的相机证据。没有符合条件的设备时报告无结果。")
    program = {"scope": {"mode": "all", "description": "这排设备"},
               "fields": {"cracked": "flag", "temperature": "number"},
               "filter": _predicate("or", _comparison("eq", "cracked", True), _comparison("gt", "temperature", 80)),
               "extreme": None, "report": ["cracked", "temperature"]}
    parse(program, instruction)
    task = {"instruction": instruction, "oracle_requirement_program": program}
    task_file = output / "custom-development-task.json"
    case = {"id": "custom-connection-001", "family": "cracks", "domain": "equipment", "seed": 3_700_101,
            "layout_seed": 3_700_102, "equipment_count": 2, "source_profile": "independent", "backend": "cpu",
            "episode_timeout_seconds": 480, "typed_task_program": True, "order": ["conditional-serial"],
            "study_task_file": str(task_file), "study_task_sha256": _hash(task), "collection_clause": None}
    layout = _reuse_or_build(case, output / "layout-development-only")
    case["scene_fingerprint"] = scene_fingerprint({**layout, "instruction_sha256":
                                                    sha256(instruction.encode("utf8")).hexdigest()})
    catalog = {"schema": 1, "role": TYPED_ROLE, "intended_pairs": 1, "cost_config_sha256": _hash(calibration),
               "asset_renderer": {"package": "Pillow", "version": PILLOW_VERSION},
               "cases": [case]}
    for path, value in ((task_file, task), (output / "calibrated_costs.json", calibration),
                        (output / "pilot_manifest.json", catalog)):
        if path.is_file() and read_json(path) != value:
            raise ValueError("The registered custom development case has changed")
        if not path.is_file():
            write_json(path, value)
    return catalog
