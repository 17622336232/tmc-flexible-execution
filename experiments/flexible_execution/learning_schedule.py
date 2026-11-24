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


def prepare_ranker_pilot(directory, base, calibration, diagnostic):
    diagnostic = diagnostic.resolve()
    summary = read_json(diagnostic / "latest_summary.json")
    size, training_seed, cap = 3000, 17, 4
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
    config = {"policies": policies, "role": "RANKER_ENGINEERING_CHECK", "training_rows": size,
              "training_seed": training_seed, "training_label_mode": summary["mode"],
              "threshold_origin": "symbolic calibration; fixed before live validation"}
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
