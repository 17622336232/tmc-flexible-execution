"""Register a bounded live check of matched candidate ordering policies."""

from random import Random

from experiments.task_attack.runtime import write_json

from .candidate_ranking import LinearRanker, RANKED_METHODS, RankedPolicy
from .runner import _hash, read_json


ROLE = "CONDITIONAL_RANKER_DEVELOPMENT"


def registered_config(directory, catalog):
    config = read_json(directory / "calibrated_rankers.json")
    if _hash(config) != catalog["ranker_config_sha256"] or set(config["policies"]) != set(RANKED_METHODS):
        raise ValueError("The ranking comparison configuration does not match its manifest")
    for method, row in config["policies"].items():
        RankedPolicy.load(row, method)
    return config


def build_config(diagnostic, size, training_seed, cap):
    diagnostic = diagnostic.resolve()
    summary = read_json(diagnostic / "latest_summary.json")
    selected = next(row for row in summary["run_rows"] if row["objective"] == "cost"
                    and row["seed"] == training_seed and row["training_states"] == size)
    policies = {}
    for method, objective in zip(RANKED_METHODS, (None, "ordinary", "cost", "cost")):
        checkpoint = diagnostic / "models" / f"{size}-{training_seed}-{objective}.json" if objective else None
        ranker = LinearRanker.load(checkpoint, objective) if checkpoint else None
        threshold = selected["adaptive"]["threshold"] if method == RANKED_METHODS[-1] else None
        policies[method] = {"checkpoint": str(checkpoint) if checkpoint else None,
                            "checkpoint_sha256": ranker.digest if ranker else None,
                            "check_cap": cap, "minimum_net_gain": threshold}
    return {"policies": policies, "role": "RANKER_ENGINEERING_CHECK", "training_rows": size,
            "training_seed": training_seed, "training_label_mode": summary["mode"],
            "threshold_origin": "symbolic calibration; fixed before live validation"}


def prepare_ranker_pilot(directory, base, calibration, diagnostic):
    config = build_config(diagnostic, 3000, 17, 4)
    seed = base["split_seed"] + 2
    order = Random(seed).sample(RANKED_METHODS, len(RANKED_METHODS))
    catalog = {"schema": 3, "role": ROLE, "parent_schedule_sha256": _hash(base), "split_seed": seed,
               "cost_config_sha256": _hash(calibration), "ranker_config_sha256": _hash(config),
               "intended_pairs": 1, "cases": [{"id": "learner-001", "family": "cracks", "domain": "equipment",
               "seed": 2000043, "equipment_count": 2, "source_profile": "independent", "backend": "cpu",
               "episode_timeout_seconds": 480, "early_fields": ["identity"], "handoff_at_observation": True,
               "group_id": "ranker-visual-development-2000043", "order": order}]}
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("pilot_manifest.json", catalog), ("calibrated_costs.json", calibration),
                        ("calibrated_rankers.json", config)):
        target = directory / name
        if target.is_file() and read_json(target) != value:
            raise ValueError("The learning pilot is frozen at a different configuration")
        if not target.is_file():
            write_json(target, value)
    registered_config(directory, catalog)
    return catalog


def prepare_native_training_check(directory, development, calibration, rankers, split_seed):
    """Reserve separate real layouts for checking symbolic ranking labels and costs."""
    from .development_inputs import _reuse_or_build, scene_fingerprint

    directory = directory.resolve()
    forbidden = {row["asset_group_sha256"] for row in development["cases"]}
    layouts = {row["layout_group_sha256"] for row in development["cases"]}
    methods = ("conditional-async", "conditional-a", *RANKED_METHODS)
    cases = []
    for number, equipment in enumerate((2, 4), 1):
        identifier = f"native-training-{number:03d}"
        root = directory / "registered-inputs" / identifier
        rng = Random(_hash([split_seed, identifier]))
        layout = 2_510_000 + number
        for seed in rng.sample(range(2_530_000, 2_540_000), 12):
            case = {"id": identifier, "family": "cracks", "domain": "equipment", "seed": seed,
                    "layout_seed": layout, "equipment_count": equipment, "source_profile": "independent",
                    "backend": "cpu", "episode_timeout_seconds": 480,
                    "collection_clause": "采集规则：允许在判读结果返回前采集以下材料：identity。"}
            inputs = _reuse_or_build(case, root / str(seed))
            if inputs["asset_group_sha256"] not in forbidden and inputs["layout_group_sha256"] not in layouts:
                break
        else:
            raise ValueError("No independent native training check could be assigned")
        forbidden.add(inputs["asset_group_sha256"])
        layouts.add(inputs["layout_group_sha256"])
        cases.append({**case, "group_id": identifier, "collection_rule": case["collection_clause"],
                      "asset_group_sha256": inputs["asset_group_sha256"], "layout_group_sha256": inputs["layout_group_sha256"],
                      "scene_fingerprint": scene_fingerprint(inputs), "handoff_at_observation": True,
                      "order": rng.sample(methods, len(methods))})
    manifest = {"schema": 1, "role": ROLE, "data_role": "INDEPENDENT_NATIVE_TRAINING_CHECK", "intended_pairs": len(cases),
                "development_manifest_sha256": _hash(development), "ranker_config_sha256": _hash(rankers),
                "cost_config_sha256": _hash(calibration), "cases": cases}
    for filename, value in (("calibrated_costs.json", calibration), ("calibrated_rankers.json", rankers),
                            ("pilot_manifest.json", manifest)):
        path = directory / filename
        if path.is_file() and read_json(path) != value:
            raise ValueError("The native training verification cohort is already frozen")
        if not path.is_file():
            write_json(path, value)
    registered_config(directory, manifest)
    return {"output": str(directory), "cases": len(cases), "jobs": sum(len(row["order"]) for row in cases),
            "data_role": manifest["data_role"]}
