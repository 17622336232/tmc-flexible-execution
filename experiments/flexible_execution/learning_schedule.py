"""Register a bounded live check of matched candidate ordering policies."""

from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean

from experiments.task_attack.runtime import write_json

from .candidate_ranking import (LinearRanker, ONLINE_RANKED_METHODS, RANKED_METHODS, RETRIEVAL_METHOD,
                                RankedPolicy, RetrievalRanker)
from .runner import _hash, read_json


ROLE = "CONDITIONAL_RANKER_DEVELOPMENT"
LIVE_CHECK_CAPS = (4, 8)


def registered_config(directory, catalog):
    config = read_json(directory / "calibrated_rankers.json")
    if (_hash(config) != catalog["ranker_config_sha256"]
            or set(config["policies"]) not in (set(RANKED_METHODS), set(ONLINE_RANKED_METHODS))):
        raise ValueError("The ranking comparison configuration does not match its manifest")
    for method, row in config["policies"].items():
        RankedPolicy.load(row, method)
    return config


def _retrieval_index(diagnostic, size, checkpoint):
    import numpy as np

    from .ranker_train import _load_jsonl

    path = diagnostic / "retrieval" / f"{size}-cost.json"
    fingerprint = read_json(checkpoint)["training_sha256"]
    metadata = {"objective": "retrieval", "training_states": size, "training_sha256": fingerprint, "neighbors": 7}
    if path.is_file():
        saved = read_json(path)
        if any(saved[key] != value for key, value in metadata.items()):
            raise ValueError("The retrieval index was built with different training data")
    else:
        sources = sorted((int(item.stem.split("-")[1]), item) for item in diagnostic.glob("train-*.jsonl"))
        source = next((item for count, item in sources if count >= size), None)
        if source is None:
            raise ValueError("The labelled candidate data for retrieval is missing")
        states = _load_jsonl(source)[:size]
        if _hash(states) != fingerprint or len(states) != size or any(row["split"] != "train" for row in states):
            raise ValueError("Retrieval must use the same frozen training subset as B1")
        rows = [candidate for state in states for candidate in state["candidates"]]
        data = np.asarray([row["features"] for row in rows], dtype=np.float64)
        offset = data.mean(axis=0)
        scale = np.maximum(data.std(axis=0), 1e-6)
        write_json(path, {**metadata, "offset": offset.tolist(), "scale": scale.tolist(),
                         "points": ((data - offset) / scale).tolist(),
                         "gains": [row["cost_gain_label"] for row in rows]})
    return path, RetrievalRanker.load(path)


def build_config(diagnostic, size, training_seed, cap, *, include_retrieval=False):
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
    if include_retrieval:
        cost_checkpoint = diagnostic / "models" / f"{size}-{training_seed}-cost.json"
        path, index = _retrieval_index(diagnostic, size, cost_checkpoint)
        policies[RETRIEVAL_METHOD] = {"checkpoint": str(path), "checkpoint_sha256": index.digest,
                                      "check_cap": cap, "minimum_net_gain": None}
    return {"policies": policies, "role": "RANKER_ENGINEERING_CHECK", "training_rows": size,
            "training_seed": training_seed, "training_label_mode": summary["mode"],
            "threshold_origin": "symbolic calibration; fixed before live validation"}


def prepare_configurations(directory, diagnostic):
    from .ranker_train import SEEDS, SIZES

    summary = read_json(diagnostic / "latest_summary.json")
    groups = {}
    for row in summary["run_rows"]:
        if row["objective"] != "cost":
            continue
        cost = next(item["mean_restricted_seconds"] for item in row["fixed"] if item["cap"] == LIVE_CHECK_CAPS[0])
        groups.setdefault(row["training_states"], {})[row["seed"]] = cost
    if set(groups) != set(SIZES) or any(set(by_seed) != set(SEEDS) for by_seed in groups.values()):
        raise ValueError("Configuration selection needs the complete grouped symbolic scaling study")
    results = {size: mean(by_seed.values()) for size, by_seed in groups.items()}
    chosen = min(results, key=lambda size: (results[size], size))
    directory = directory.resolve()
    caps = {}
    for cap in LIVE_CHECK_CAPS:
        config = build_config(diagnostic, chosen, SEEDS[0], cap, include_retrieval=True)
        path = directory / f"cap-{cap}.json"
        _save_definition(path, config)
        caps[str(cap)] = {"path": str(path), "config_sha256": _hash(config)}
    plan = {"schema": 1, "symbolic_model_selection_sha256": _hash(summary), "training_rows": chosen,
            "mean_symbolic_restricted_seconds": {str(key): value for key, value in results.items()},
            "development_seed": SEEDS[0], "caps": caps,
            "selection_rule": ["more complete successes", "less restricted time", "smaller check cap"]}
    _save_definition(directory / "selection_plan.json", plan)
    return plan


def _save_definition(path, definition):
    if path.is_file():
        if read_json(path) != definition:
            raise ValueError("The ranking development configuration is already frozen")
    else:
        write_json(path, definition)


def select_live_configuration(directory, catalog):
    from .control import TASK_TERMINAL

    plan = catalog["ranker_selection_plan"]
    if set(plan["caps"]) != {str(cap) for cap in LIVE_CHECK_CAPS}:
        raise ValueError("The frozen ranking selection grid is incomplete")
    results = {}
    for cap in LIVE_CHECK_CAPS:
        expected = {f"{method}-cap{cap}" for method in ONLINE_RANKED_METHODS}
        successes, times = 0, []
        for case in catalog["cases"]:
            if not expected.issubset(case["order"]):
                raise ValueError("A ranking configuration is missing one of the matched controls")
            for method in sorted(expected):
                job = read_json(directory / case["id"] / method / "job.json")
                if job["state"] not in TASK_TERMINAL or job["ranker_check_cap"] != cap:
                    raise ValueError("Ranking selection contains an unresolved or mismatched job")
                deadline = case["episode_timeout_seconds"]
                valid = job["state"] == "FINISHED_VALID"
                seconds = job.get("charged_seconds") if valid else deadline
                if not isinstance(seconds, (int, float)) or not isfinite(seconds) or not 0 < seconds <= deadline:
                    raise ValueError("A saved ranking outcome has an invalid task time")
                successes += valid
                times.append(seconds)
        results[cap] = {"episodes": len(times), "successes": successes, "mean_restricted_seconds": mean(times)}
    winner = min(results, key=lambda cap: (-results[cap]["successes"], results[cap]["mean_restricted_seconds"], cap))
    source = plan["caps"][str(winner)]
    config = read_json(Path(source["path"]))
    if _hash(config) != source["config_sha256"]:
        raise ValueError("The selected ranking configuration changed")
    evidence = {"plan_sha256": _hash(plan), "selected_check_cap": winner, "training_rows": plan["training_rows"],
                "candidate_configurations": {str(cap): value for cap, value in results.items()},
                "selected_config_sha256": source["config_sha256"]}
    return config, evidence


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
    methods = ("conditional-async", "conditional-a",
               *(method for method in ONLINE_RANKED_METHODS if method in rankers["policies"]))
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
