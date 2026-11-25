"""Assign independent custom evaluation inputs before rendering any test scene."""

from collections import Counter
from hashlib import sha256
from random import Random
from shutil import disk_usage
import subprocess

from experiments.task_attack.runtime import write_json
from experiments.task_attack.tasks import SCENES

from .requirement_program import parse
from .runner import _hash, read_json


FINAL_ROLE = "CUSTOM_VISUAL_MATCHED_HELDOUT"


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


def _save_locked(path, data):
    if path.is_file():
        if read_json(path) != data:
            raise ValueError("Saved evaluation input has changed: " + str(path))
    else:
        write_json(path, data)


def _heldout_layout(row, output, development_assets, used_seeds, split_seed, *, shared_frontend=True):
    from .development_inputs import _reuse_or_build, scene_fingerprint

    case = {key: row[key] for key in ("id", "seed", "layout_seed", "family", "domain", "equipment_count",
                                    "episode_timeout_seconds", "stratum")}
    case.update(source_profile="independent" if row["family"] == "cracks" else "single", backend="cpu",
                collection_clause=None, typed_task_program=True, handoff_at_observation=True,
                archive_external_rgb=False, shared_frontend=shared_frontend)
    if row["stratum"] == "execution_condition" and row["family"] == "cracks":
        case["collection_clause"] = ("采集规则：允许在判读结果返回前采集以下材料：identity。" if row["episode_timeout_seconds"] == 600
                                    else "采集规则：只有判读结果为 positive 才允许采集以下材料：identity。")
    alternatives = Random(_hash([split_seed, row["id"], "blind-asset-replacement"])).sample(range(7_000_000, 8_000_000), 16)
    for seed in (row["seed"], *alternatives):
        if seed in used_seeds:
            continue
        case["seed"] = seed
        directory = output / "reference-scenes" / row["id"] / str(seed)
        material = _reuse_or_build(case, directory)
        if material["asset_group_sha256"] not in development_assets:
            break
    else:
        raise ValueError("The sealed assignment could not isolate its visual source: " + row["id"])
    used_seeds.add(seed)
    if row["instruction"] is not None:
        task = {"instruction": row["instruction"], "oracle_requirement_program": row["oracle_requirement_program"]}
        task_file = output / "study-tasks" / (row["id"] + ".json")
        _save_locked(task_file, task)
        case.update(study_task_file=str(task_file), study_task_sha256=_hash(task))
        given = task["instruction"] + ("\n" + case["collection_clause"] if case["collection_clause"] else "")
        material["instruction_sha256"] = sha256(given.encode("utf8")).hexdigest()
    case.update(collection_rule=case.pop("collection_clause"), scene_fingerprint=scene_fingerprint(material),
                group_id="visual-source-" + material["asset_group_sha256"], original_seed=row["seed"],
                physical_layout_sha256=material["layout_group_sha256"])
    return case


def prepare_final(blind, development, calibration, selection, native_training, diagnostic, output, checkout, defaults,
                  auth_project):
    """Freeze the custom protocol before generating any held-out scene or starting inference."""
    from .candidate_ranking import ONLINE_RANKED_METHODS, RANKED_METHODS, RETRIEVAL_METHOD
    from .control import _runtime_inputs, verified_cohort
    from .learning_schedule import ROLE as RANKER_ROLE, build_config, registered_config, select_live_configuration
    from .pilot_schedule import VISUAL_ROLES

    data = defaults["data"]
    chosen, selection_evidence = verified_cohort("configuration_selection", selection, VISUAL_ROLES["selection"],
                                                 data["development_selection_instances"])
    calibrated, calibration_evidence = verified_cohort("timing_calibration", calibration, VISUAL_ROLES["calibration"],
                                                        data["development_calibration_instances"])
    native_catalog, native_evidence = verified_cohort("native_training_check", native_training, RANKER_ROLE, 2)
    if (native_catalog.get("data_role") != "INDEPENDENT_NATIVE_TRAINING_CHECK"
            or native_catalog["development_manifest_sha256"] != _hash(development)
            or any(catalog["layout_manifest_sha256"] != _hash(development) for catalog in (chosen, calibrated))):
        raise ValueError("The development evidence was generated from different visual groups")
    prepare(blind, defaults, development)
    assigned = read_json(blind / "split_manifest.json")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    minimum_free = defaults["resources"]["minimum_free_disk_gb"] * 1024**3
    if disk_usage(output).free < minimum_free:
        raise OSError("Disk reserve is below the registered minimum")
    cost = read_json(selection / "calibrated_costs.json")
    ranking, rank_selection = select_live_configuration(selection, chosen)
    native_rankers = registered_config(native_training, native_catalog)
    if (_hash(cost) != chosen["cost_config_sha256"] or native_rankers["training_rows"] != ranking["training_rows"]
            or set(native_rankers["policies"]) != set(ONLINE_RANKED_METHODS)):
        raise ValueError("The selected controller configuration no longer matches development")
    cap = rank_selection["selected_check_cap"]
    bind = {}
    for seed_index, seed in enumerate(data["training_seeds"]):
        config = build_config(diagnostic, ranking["training_rows"], seed, cap, include_retrieval=True)
        config_file = output / "rankers" / f"seed-{seed}.json"
        _save_locked(config_file, config)
        for method in ONLINE_RANKED_METHODS:
            seedless = method in (RANKED_METHODS[0], RETRIEVAL_METHOD)
            if seedless and seed_index:
                continue
            label = method if seedless else f"{method}-seed-{seed}"
            bind[label] = {"method": method, "ranker_file": str(config_file), "sha256": _hash(config),
                           "seed": None if seedless else seed, "check_cap": cap}
    methods = ("conditional-async", "conditional-prefetch", "conditional-planner", "conditional-stream", "conditional-a", *bind)
    template = {"schema": 1, "role": FINAL_ROLE, "intended_pairs": len(assigned["cases"]),
                "ranker_bindings": bind, "cost_config_sha256": _hash(cost)}
    signature_cases = [{"layout_seed": row["layout_seed"], "order": methods,
                        "study_task_sha256": row["instruction"] is not None,
                        "archive_external_rgb": False, "shared_frontend": True} for row in assigned["cases"]]
    inputs = _runtime_inputs(checkout, {**template, "cases": signature_cases}, auth_project.resolve(), (11674, 11675))
    lock = {key: inputs[key] for key in ("model", "reasoning_effort", "request_limit", "source_sha256", "environment_sha256")}
    frozen = {**lock, "schema": 1, "role": FINAL_ROLE, "blind_assignment_sha256": _hash(assigned),
              "selection": selection_evidence, "ranking_selection": rank_selection,
              "calibration": calibration_evidence, "native_training": native_evidence,
              "methods": list(methods),
              "ranker_bindings": bind, "cost_config_sha256": _hash(cost), "metrics": defaults["evaluation"],
              "archive_external_rgb": False, "shared_frontend": True,
              "asset_replacement": "ordered independent seeds; reject exact training or development texture-group overlap",
              "mode": data["main_evaluation_mode"]}
    lock_path = output / "frozen_evaluation.json"
    if lock_path.is_file():
        code = read_json(lock_path)["code_git_commit"]
    else:
        code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    frozen["code_git_commit"] = code
    _save_locked(lock_path, frozen)
    _save_locked(output / "calibrated_costs.json", cost)

    cases = []
    known = [*development["cases"], *native_catalog["cases"]]
    forbidden = {row["asset_group_sha256"] for row in known}
    known_layouts = {row["layout_group_sha256"] for row in known}
    used = set()
    for row in assigned["cases"]:
        if disk_usage(output).free < minimum_free:
            raise OSError("Disk reserve reached while preparing the sealed evaluation")
        case = _heldout_layout(row, output, forbidden, used, data["split_seed"])
        case["order"] = Random(_hash([row["id"], data["split_seed"], "method-order"])).sample(methods, len(methods))
        cases.append(case)
    layouts = {case["physical_layout_sha256"] for case in cases}
    if len(layouts) != len(cases) or layouts & known_layouts:
        raise ValueError("The custom test shares a physical layout across base cases")
    catalog = {**template, "cases": cases, "frozen_evaluation": str(lock_path), "frozen_evaluation_sha256": _hash(frozen)}
    _save_locked(output / "pilot_manifest.json", catalog)
    return {"cases": len(cases), "jobs": sum(len(row["order"]) for row in cases), "output": str(output),
            "frozen_evaluation_sha256": catalog["frozen_evaluation_sha256"]}
