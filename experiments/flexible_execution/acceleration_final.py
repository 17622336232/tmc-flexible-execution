"""Blind selection and runtime locks for the smaller natural-online comparison."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from random import Random
from shutil import disk_usage
from subprocess import check_output

from . import custom_schedule, public_schedule
from .acceleration import METHODS, PROTOCOL, lock
from .control import _runtime_inputs, verified_cohort
from .live_calibration import calibrate_public_visual, calibrate_visual
from .runner import _hash, read_json


SEED = 20261125
OURS = "conditional-a"
PRIMARY = "conditional-planner"


def register_subset(output_root):
    root = output_root / "acceleration-v2"
    custom = read_json(output_root / "custom-visual/test-plan-v2/split_manifest.json")
    public = read_json(output_root / "public-visual/test-plan/split_manifest.json")
    strata = defaultdict(lambda: defaultdict(list))
    for row in custom["cases"]:
        key = row["family"], row["equipment_count"], row["episode_timeout_seconds"]
        strata[row["stratum"]][key].append(row["id"])
    selected, optional = {}, {}
    for stratum, groups in sorted(strata.items()):
        buckets = [sorted(groups[key], key=lambda item: _hash([SEED, item])) for key in sorted(groups)]
        balance = [group[index] for index in range(max(map(len, buckets))) for group in buckets if index < len(group)]
        selected[stratum] = balance[:24]
        optional[stratum] = sorted(balance[:24], key=lambda item: _hash([SEED, "ablation", item]))[:4]
    if len(selected) != 3 or any(len(ids) != 24 for ids in selected.values()):
        raise ValueError("The sealed custom assignment cannot provide the three registered strata")
    houses = defaultdict(dict)
    for row in public["cases"]:
        houses[row["group_id"]][row["task_number"]] = row["id"]
    chosen_houses = sorted(houses, key=lambda group: _hash([SEED, "house", group]))[:24]
    if len(chosen_houses) != 24:
        raise ValueError("The sealed public assignment has fewer than 24 registered houses")
    public_ids = [houses[group][1 + index % 4] for index, group in enumerate(chosen_houses)]
    result = {"protocol_version": PROTOCOL, "state": "REGISTERED_BLIND_IDS", "sampling_seed": SEED,
              "custom_parent_sha256": _hash(custom), "public_parent_sha256": _hash(public),
              "custom_balance_keys": ["family", "equipment_count", "episode_timeout_seconds"],
              "custom": selected, "public": public_ids, "public_houses": chosen_houses,
              "optional_cases": optional, "registered_online_jobs": 456,
              "matching": {"custom": list(METHODS[:-1]) + ["OURS"], "public": list(public_schedule.METHODS[:-1]) + ["OURS"]},
              "primary_reference": PRIMARY, "primary_metric": "deadline_restricted_complete_task_seconds",
              "primary_two_sided_alpha": .05, "cluster": "visual_source_or_public_house", "seeds": [17, 29, 43],
              "public_method_capabilities": {"OURS": {"A": True, "B1": False, "C": False, "STREAM": False,
                  "scope": "single conditional side view with the common camera interface"}},
              "heldout_layouts_or_results_loaded": False}
    lock(root / "blind_online_subset.json", result)
    return result


def _runtime_lock(checkout, auth_project, catalog, path, **data):
    inputs = _runtime_inputs(checkout, catalog, auth_project.resolve(), (11674, 11675))
    freeze = {key: inputs[key] for key in ("model", "reasoning_effort", "request_limit",
                                           "environment_sha256", "source_sha256")}
    revision = (read_json(path)["code_git_commit"] if path.is_file() else
                check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip())
    freeze.update(data, schema=2, role=catalog["role"], protocol_version=PROTOCOL,
                  code_git_commit=revision, runtime=catalog.get("runtime"))
    if catalog.get("execution_revision"):
        freeze["execution_revision"] = catalog["execution_revision"]
    lock(path, freeze)
    return {**catalog, "frozen_evaluation": str(path), "frozen_evaluation_sha256": _hash(freeze)}


def prepare_online(output_root, checkout, defaults, auth_project):
    from .acceleration_revision import selected_root

    root = selected_root(output_root)
    original = output_root / "acceleration-v2"
    registry = read_json(root / "execution_revision.json") if root != original else None
    development = root / "development"
    verified, dev_evidence = verified_cohort("v2_development", development, "ACCELERATION_V2_DEVELOPMENT", 3)
    estimate = read_json(root / "measured_estimate.json")
    if estimate["sampled_v2_episodes"] != dev_evidence["recorded_jobs"] or estimate["main_hours_q1_median_q3"][2] > 168:
        raise ValueError("The bounded online estimate is incomplete or exceeds the seven-day guard")
    offline = read_json(original / "offline_selection.json")
    if (offline["evidence_level"] != "DECISION_REPLAY" or offline["actual_new_model_requests"] != 0
            or offline["settings"] != 18 or offline["independent_sources"] < 2):
        raise ValueError("The registered offline selection has not been independently rebuilt")
    from .acceleration_results import development_summary

    evidence = development_summary(development)
    revision = registry["execution_revision"] if registry else None
    if revision and (any(row["execution_revision"] != revision for row in evidence["rows"])
                     or registry["development_catalog_sha256"] != _hash(verified)):
        raise ValueError("Formal evaluation requires one complete revised worker cohort")
    violations = [row["job"] for row in evidence["rows"]
                  if (row["permit_audit"] or {}).get("violations") or row["prior_attempt_permission_violations"]]
    if violations:
        raise ValueError("Development camera permission violations prevent the test freeze: " + ", ".join(violations))
    if not revision and not evidence["ours_complete_with_multiple_verified_queries"]:
        raise ValueError("A has no independently accepted v2 task with multiple bound judgments")
    if revision:
        observed = evidence["rows"]
        checks = {
            "structured_handoff": any(row["seeded_records"] and row["first_controller_records"] for row in observed),
            "complete_shared_report": any(row["state"] == "FINISHED_VALID" for row in observed),
            "multiple_bound_ours_queries": any(row["method"] == OURS and
                sum(query["binding_verified"] for query in row["queries"]) >= 2 for row in observed),
            "late_permission": any(row["audited_late_permission"] and row["queries"] for row in observed),
            "permitted_early_action": any(row["permitted_actions_started_before_return"] for row in observed),
            "waiting_or_uncertain": any(row["uncertain_or_evidence_rejected"] or
                (row["method"] == OURS and row["ordinary_wait_or_no_gain"]) for row in observed),
        }
        missing = [name for name, present in checks.items() if not present]
        if missing:
            raise ValueError("Missing revised development mechanism evidence: " + ", ".join(missing))
    # Learning has no v2 natural development contrast in the first budgeted cohort.
    # The small same-pool CPU difference alone does not qualify it for the primary arm.
    selection = {"method": OURS, "training_seed_if_applicable": 17, "natural_development": dev_evidence,
                 "offline_selection_sha256": _hash(offline),
                 "rule": "retain A unless a matched v2 online learning contrast supports retained task coverage and improved cost",
                 "learned_online_arm": "not registered in first development batch",
                 "completed_mechanism_cases": evidence["ours_complete_with_multiple_verified_queries"],
                 "optional_online_registered": 0}
    if revision:
        selection.update(execution_revision=revision, source_worker_sha256=registry["source_sha256"],
                         pre_holdout_power=read_json(root / "development_power.json"),
                         verified_development_mechanisms=checks,
                         analysis_roles={"custom": "confirmatory", "public": "external_validation"},
                         external_interruptions="failed restricted cost after task begins, preserve attempt and halt batch",
                         sensitivity_block_seconds=3600)
    lock(root / "method_selection.json", selection)
    subset = register_subset(output_root)
    cost = calibrate_visual(development, read_json(development / "calibrated_costs.json"),
                            allowed_role="ACCELERATION_V2_DEVELOPMENT")
    lock(root / "v2_calibration.json", cost)

    validation = output_root / "public-visual/validation-pilot-v1"
    public_dev, public_evidence = verified_cohort("public_validation", validation, public_schedule.VALIDATION_ROLE, 4)
    blind_public = read_json(output_root / "public-visual/test-plan/split_manifest.json")
    archive = output_root / "public-visual/procthor/test.jsonl.gz"
    if archive.stat().st_size != public_schedule.TEST_BYTES or sha256(archive.read_bytes()).hexdigest() != public_schedule.TEST_SHA256:
        raise ValueError("The public archive differs from its registered official test source")
    runtime = {**public_dev["runtime"], "archive": str(archive), "source": public_schedule.TEST_SOURCE,
               "cost_calibration": calibrate_public_visual(validation, public_dev)}
    if revision:
        validated_runtime = read_json(validation / "run_protocol.json")
        executables = [path for path in validated_runtime["environment_sha256"] if Path(path).name.startswith("python")]
        if len(executables) != 1:
            raise ValueError("The public validation did not identify one frozen Python interpreter")
        runtime["python_executable"] = executables[0]
    public_cases = [{**row, "family": "conditional_camera_presence", "domain": "public-room",
                     "seed": row["official_house_index"], "episode_timeout_seconds": 480}
                    for row in blind_public["cases"] if row["id"] in subset["public"]]
    public_dir = root / "public-final"
    public_dir.mkdir(parents=True, exist_ok=True)
    public_catalog = {"schema": 2, "protocol_version": PROTOCOL, "evidence_level": "NATURAL_ONLINE",
                      "role": public_schedule.FINAL_ROLE, "intended_pairs": 24, "runtime": runtime,
                      "primary_reference": PRIMARY, "ours_method": OURS, "cases": public_cases}
    if revision:
        public_catalog["execution_revision"] = revision
    public_catalog = _runtime_lock(checkout, auth_project, public_catalog, public_dir / "frozen_evaluation.json",
                                   source_subset_sha256=_hash(subset), evidence=public_evidence,
                                   selection=selection, metrics=defaults["evaluation"], episode_timeout_seconds=480)
    lock(public_dir / "pilot_manifest.json", public_catalog)

    custom = read_json(output_root / "custom-visual/test-plan-v2/split_manifest.json")
    identities = {item for group in subset["custom"].values() for item in group}
    assigned = [row for row in custom["cases"] if row["id"] in identities]
    custom_dir = root / "custom-final"
    custom_dir.mkdir(parents=True, exist_ok=True)
    guard = defaults["resources"]["minimum_free_disk_gb"] * 1024 ** 3
    if disk_usage(custom_dir).free < guard:
        raise OSError("Disk reserve reached before rendering sealed v2 layouts")
    old_layouts = read_json(output_root / "development-visual-v2/split_manifest_v2.json")["cases"]
    native = read_json(output_root / "natural-pilot/native-training-check-v2/pilot_manifest.json")["cases"]
    assets = {row["asset_group_sha256"] for row in [*old_layouts, *native]}
    layouts = {row["layout_group_sha256"] for row in [*old_layouts, *native]}
    template = {"schema": 2, "protocol_version": PROTOCOL, "evidence_level": "NATURAL_ONLINE",
                "role": custom_schedule.FINAL_ROLE, "intended_pairs": 72, "primary_reference": PRIMARY,
                "ours_method": OURS, "ranker_bindings": {}, "cost_config_sha256": _hash(cost)}
    if revision:
        template["execution_revision"] = revision
    signature_cases = [{"layout_seed": row["layout_seed"], "order": METHODS,
                        "study_task_sha256": row["instruction"] is not None,
                        "archive_external_rgb": False, "shared_frontend": False} for row in assigned]
    if revision:
        worker = read_json(development / "run_protocol.json")["source_sha256"]
        future_worker = _runtime_inputs(checkout, {**template, "cases": signature_cases}, auth_project.resolve(),
                                        (11674, 11675))["source_sha256"]
        if any(future_worker.get(path) != digest for path, digest in worker.items()):
            raise ValueError("The custom formal worker differs from the revised validated development worker")
    catalog = _runtime_lock(checkout, auth_project, {**template, "cases": signature_cases},
                            custom_dir / "frozen_evaluation.json", source_subset_sha256=_hash(subset),
                            selection=selection, metrics=defaults["evaluation"], ranker_bindings={},
                            cost_config_sha256=_hash(cost), shared_frontend=False, archive_external_rgb=False)
    lock(custom_dir / "calibrated_costs.json", cost)
    cases = []
    used_seeds = set()
    for row in assigned:
        if disk_usage(custom_dir).free < guard:
            raise OSError("Disk reserve reached while rendering sealed v2 layouts")
        case = custom_schedule._heldout_layout(row, custom_dir, assets, used_seeds, custom["split_seed"],
                                               shared_frontend=False)
        if case["physical_layout_sha256"] in layouts:
            raise ValueError("A registered v2 custom physical layout overlaps another scene")
        layouts.add(case["physical_layout_sha256"])
        case["order"] = Random(_hash([SEED, row["id"], "main-order"])).sample(METHODS, len(METHODS))
        cases.append(case)
    lock(custom_dir / "pilot_manifest.json", {**catalog, "cases": cases})
    return custom_dir, public_dir
