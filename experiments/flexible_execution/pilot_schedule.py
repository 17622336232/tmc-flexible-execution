"""Freeze and audit the small natural-latency development pairing schedule."""

import json
from math import isfinite
from random import Random

from experiments.task_attack import FAMILIES
from experiments.task_attack.runtime import write_json
from experiments.task_attack.tasks import SCENES

from .live_planning import CONDITIONAL_MODES, NEW_METHODS
from .runner import _hash


PAIRED_MODES = {"paired-serial": "ACTUAL_GPT_SHARED_SERIAL", "paired-async": "ACTUAL_GPT_SHARED_ASYNC"}


def prepare(directory, defaults, episode_timeout):
    if not isfinite(episode_timeout) or episode_timeout <= 0:
        raise ValueError("The pilot timeout must be finite and positive")
    data = defaults["data"]
    families = [name for name in FAMILIES if not name.endswith("_labels")]
    if len(families) * 2 != data["natural_latency_pilot_instances"]:
        raise ValueError("The configured pilot size does not match two instances per legacy task")
    rng = Random(data["split_seed"])
    designs = [(family, count) for family in families for count in (2, 4)]
    rng.shuffle(designs)
    seeds = rng.sample(range(1_000_000, 2_000_000), len(designs))
    cases = [{"id": "pair-%03d" % index, "family": family, "domain": SCENES[family],
              "seed": seed, "equipment_count": count, "source_profile": "single", "backend": "cpu",
              "episode_timeout_seconds": episode_timeout, "order": rng.sample(list(PAIRED_MODES), len(PAIRED_MODES))}
             for index, ((family, count), seed) in enumerate(zip(designs, seeds), 1)]
    manifest = {"schema": 1, "role": "LEGACY_TASK_DEVELOPMENT_PAIR_SCHEDULE",
                "split_seed": data["split_seed"], "intended_pairs": len(cases), "cases": cases}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pilot_manifest.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf8")) != manifest:
            raise ValueError("The existing pilot schedule differs from these settings")
    else:
        with path.open("x", encoding="utf8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return manifest


def _read(path):
    return json.loads(path.read_text(encoding="utf8")) if path.is_file() else None


def prepare_conditional(directory, base, calibration, *, connection_only=False, hold_at_view=False,
                        source_profile="single"):
    role = "CONDITIONAL_CONNECTION_DEVELOPMENT"
    seed = base["split_seed"] + 1
    if connection_only:
        order = ["conditional-a"]
        instance = "connection-001"
    else:
        order = Random(seed).sample(NEW_METHODS, len(NEW_METHODS))
        instance = "conditional-001"
    catalog = {"schema": 2, "role": role, "connection_only": connection_only,
               "parent_schedule_sha256": _hash(base), "split_seed": seed,
               "cost_config_sha256": _hash(calibration), "intended_pairs": 1,
               "cases": [{"id": instance, "family": "cracks", "domain": "equipment", "seed": 1,
                          "equipment_count": 2, "source_profile": source_profile, "backend": "cpu",
                          "episode_timeout_seconds": 480, "early_fields": ["identity"], "order": order}]}
    if hold_at_view:
        catalog["cases"][0]["handoff_at_observation"] = True
    directory.mkdir(parents=True, exist_ok=True)
    for filename, value in (("calibrated_costs.json", calibration), ("pilot_manifest.json", catalog)):
        path = directory / filename
        existing = _read(path)
        if existing is not None and existing != value:
            raise ValueError("The existing conditional schedule or calibration is already frozen")
        if existing is None:
            write_json(path, value)
    return catalog


def _arm(directory, case, method):
    base = directory / case["id"] / method
    job = _read(base / "job.json")
    if job:
        if job["state"] not in {"FINISHED_VALID", "FINISHED_TASK_FAILURE", "TIMEOUT"}:
            return None
        return {"model": job["model"], "reasoning_effort": job["reasoning_effort"],
                "completed_coverage": job["state"] == "FINISHED_VALID",
                "outcome": job["result"].get("score", job["state"]),
                "state": job["state"], "output": job.get("artifact_dir")}
    preflight = _read(base / "clean_preflight.json")
    result = _read(base / "result.json")
    coverage = _read(base / "clean_coverage.json")
    if preflight is None:
        return None
    _valid_preflight(preflight, case, method, base)
    if result is None:
        return None
    return {"model": preflight["model"], "reasoning_effort": preflight["reasoning_effort"],
            "completed_coverage": bool(coverage and coverage["complete"]),
            "outcome": result["score"]["outcome"]}


def _valid_preflight(preflight, case, method, directory):
    keys = ("family", "seed", "backend", "equipment_count", "source_profile")
    planned = preflight.get("planned_episode_timeout_seconds", preflight.get("episode_timeout_seconds"))
    budget = preflight.get("episode_timeout_seconds")
    mode = {**PAIRED_MODES, **CONDITIONAL_MODES}[method]
    if (preflight.get("mode") != mode or any(preflight.get(key) != case[key] for key in keys)
            or preflight.get("early_fields", []) != case.get("early_fields", [])
            or preflight.get("handoff_at_observation", False) != case.get("handoff_at_observation", False)
            or planned != case["episode_timeout_seconds"] or not isinstance(budget, (int, float))
            or not 0 < budget <= planned):
        raise ValueError("Run does not match the frozen case and method: " + str(directory))


def status(directory):
    manifest = _read(directory / "pilot_manifest.json")
    if manifest is None:
        raise ValueError("Prepare the development pairing schedule before auditing its runs")
    finished = successful = 0
    incomplete = []
    for case in manifest["cases"]:
        arms = {method: _arm(directory, case, method) for method in case["order"]}
        waiting = [method for method, outcome in arms.items() if outcome is None]
        if waiting:
            incomplete.append({"case": case["id"], "pending": waiting})
            continue
        if len({(row["model"], row["reasoning_effort"]) for row in arms.values()}) != 1:
            raise ValueError("A pilot pair used different model settings: " + case["id"])
        finished += 1
        successful += all(row["completed_coverage"] for row in arms.values())
    return {"role": manifest["role"], "planned_pairs": len(manifest["cases"]), "recorded_pairs": finished,
            "pairs_with_both_reports_and_coverage": successful, "pending": incomplete}
