"""Versioned acceleration development schedule and independent test registration."""

from pathlib import Path
from random import Random
from subprocess import PIPE, run

from experiments.task_attack.runtime import write_json

from .control import REQUEST_LIMIT, control_status, pilot_run
from .pilot_schedule import status as pilot_status
from .runner import _hash, read_json


PROTOCOL = "protocol-v2"
METHODS = ("conditional-async", "conditional-prefetch", "conditional-planner", "conditional-stream", "conditional-a")


def lock(path, value):
    if path.is_file():
        if read_json(path) != value:
            raise ValueError("Acceleration input changed after registration: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)


def prepare_development(output_root):
    legacy = output_root / "natural-pilot"
    root = output_root / "acceleration-v2"
    tournament = legacy / "conditional-comparison-v4"
    old_calibration = legacy / "visual-calibration-v3"
    early = read_json(tournament / "pilot_manifest.json")["cases"][0]
    available = {row["id"]: row for row in read_json(old_calibration / "pilot_manifest.json")["cases"]}
    sources = [(early, "repeat_bound_judgment_and_early_permission", METHODS),
               (available["visual-calibration-005"], "after_positive_permission", METHODS),
               (available["visual-calibration-008"], "numeric_and_uncertain_sight", ("conditional-async", "conditional-a"))]
    cases = []
    for number, (case, reason, methods) in enumerate(sources, 1):
        order = list(methods)
        Random(20261125 + number).shuffle(order)
        if number == 1:
            order.remove("conditional-a")
            order.insert(0, "conditional-a")
        cases.append({**case, "id": f"accel-dev-{number:03d}", "legacy_case_id": case["id"],
                      "development_purpose": reason, "order": order, "archive_external_rgb": False,
                      "typed_task_program": True, "shared_frontend": False})
    development = root / "development"
    costs = read_json(tournament / "calibrated_costs.json")
    lock(development / "calibrated_costs.json", costs)
    lock(development / "pilot_manifest.json", {
        "schema": 2, "protocol_version": PROTOCOL, "role": "ACCELERATION_V2_DEVELOPMENT",
        "intended_pairs": 3, "evidence_level": "NATURAL_ONLINE", "cost_config_sha256": _hash(costs),
        "cases": cases})
    lock(root / "run_manifest.json", {
        "protocol_version": PROTOCOL, "worker_limit": 1, "existing_plan_only": True,
        "incremental_paid_budget": 0, "legacy_policy": "PRESERVE_UNSCHEDULED",
        "first_development": {"evidence_level": "NATURAL_ONLINE", "cases": len(cases),
                              "jobs": sum(len(case["order"]) for case in cases), "request_ceiling": 288,
                              "episode_deadline_seconds": 480, "request_ceiling_per_job": REQUEST_LIMIT,
                              "catalog_sha256": _hash(read_json(development / "pilot_manifest.json"))},
        "repair_ceiling": {"rounds": 2, "jobs_per_round": 6, "requests_per_round": 144},
        "main": {"custom_cases": 72, "custom_arms": 360, "public_cases": 24, "public_arms": 96,
                 "jobs": 456, "primary_reference": "conditional-planner", "alpha": 0.05, "seed": 17},
        "optional_online_ceiling": 72, "formal_online_ceiling": 528, "formal_request_ceiling": 12672})
    return development


def status(output_root):
    root = output_root / "acceleration-v2"
    directory = root / "development"
    dev = pilot_status(directory) if (directory / "pilot_manifest.json").is_file() else None
    control = control_status(root / "control")
    formal = {}
    for domain in ("custom", "public"):
        pilot = root / (domain + "-final")
        if (pilot / "pilot_manifest.json").is_file():
            formal[domain] = pilot_status(pilot)
    return {"protocol_version": PROTOCOL, "execution_status": control["execution_status"],
            "active_jobs": control["active_jobs"], "development": dev, "formal": formal, "control": control,
            "legacy_execution": control_status(output_root / "control")["execution_status"],
            "next_command": "python3 -m experiments.flexible_execution.cli continue --profile local5060 "
                            "--protocol protocol-v2 --resume --max-jobs 12"}


def advance_stage(output_root, checkout, defaults, args, launch_limit):
    from .acceleration_results import update

    root = output_root / "acceleration-v2"
    directory = prepare_development(output_root)
    development = pilot_status(directory)
    legacy = output_root / "control"

    def batch(pilot):
        result = pilot_run(pilot, root / "control", checkout, resume=args.resume, max_jobs=launch_limit,
                           auth_project=args.chatgpt_project, unlock_external=args.retry_external,
                           minimum_free_disk_gb=defaults["resources"]["minimum_free_disk_gb"], lease_control=legacy)
        used = 0 if result.get("worker_owned_by_another_batch") else result.get("last_batch", {}).get("launched", 0)
        return result, used

    execution = control_status(root / "control")
    if execution["active_jobs"]:
        if args.resume and execution.get("pilot_directory") and any(j.get("arm") for j in execution["active_jobs"]):
            return batch(Path(execution["pilot_directory"]))
        return execution, 0
    if execution["execution_status"] == "BLOCKED_EXTERNAL" and not args.retry_external:
        return execution, 0
    if development["recorded_arms"] < development["planned_arms"]:
        batch_result, used = batch(directory)
        if (directory / "run_protocol.json").is_file() and not batch_result.get("active_jobs"):
            update(output_root, directory)
        return batch_result, used
    from .cli import _scene_python
    from .pilot_report import rebuild

    update(output_root, directory)
    estimate = read_json(root / "measured_estimate.json")
    if estimate["main_hours_q1_median_q3"][2] > 7 * 24:
        return {**status(output_root), "execution_status": "BLOCKED_DURATION", "estimate": estimate}, 0
    custom, public = root / "custom-final", root / "public-final"
    if not all((path / "pilot_manifest.json").is_file() for path in (custom, public)):
        prepared = run([str(_scene_python()), "-m", "experiments.flexible_execution.cli", "acceleration-final-prepare",
                        "--output-root", str(output_root), "--chatgpt-project", str(args.chatgpt_project.resolve())],
                       cwd=checkout, stdout=PIPE, stderr=PIPE, text=True)
        if prepared.returncode:
            raise ValueError(prepared.stderr.strip() or "The independent acceleration protocol could not be locked")
    finished = {}
    for cohort in (custom, public):
        counters = pilot_status(cohort)
        if counters["recorded_arms"] < counters["planned_arms"]:
            return batch(cohort)
        result = rebuild(cohort, root / "reports" / cohort.name)
        finished[cohort.name] = result["recorded_episodes"]
    return {**status(output_root), "execution_status": "EVALUATED", "completed_reports": finished}, 0
