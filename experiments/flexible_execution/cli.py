"""Local audit, diagnostic execution, status and event-ledger reproduction."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from .audit import capture
from .clean_coverage import audit_run
from .control import DEFAULT_AUTH_PROJECT, control_status, is_live, pilot_run
from .learning_schedule import prepare_ranker_pilot
from .live_audit import audit_live_connection
from .live_calibration import calibrate, calibrate_visual
from .pilot_schedule import prepare as prepare_pilot, prepare_conditional, prepare_typed, prepare_visual, status as pilot_status
from .runner import manifest, read_json, reproduce, run


CHECKOUT = Path(__file__).resolve().parents[2]
DEFAULTS = Path(__file__).with_name("experiment_defaults.yaml")
CUSTOM_CONNECTION = "custom-vision-connection-v2"
VISUAL_INPUTS = "development-visual-v2"
VISUAL_PILOTS = {role: f"visual-{role}-v2" for role in ("calibration", "selection")}


def _scene_python():
    legacy = Path(os.environ.get("VLA_LEGACY_ROOT", str(Path.home() / "uav" / "Embodied-intelligent-behavior")))
    environment = legacy / ".venv"
    if (environment / "bin/activate").is_file() and (environment / "bin/python3").is_file():
        return environment / "bin/python3"
    return Path(sys.executable)


def _prepare_scenes(arguments):
    # The online simulator activates this same virtual environment. Pillow
    # versions can produce different text pixels from identical fixtures.
    subprocess.run([str(_scene_python()), "-m", "experiments.flexible_execution.cli", *map(str, arguments)],
                   cwd=CHECKOUT, check=True)


def _sensor_result(directory):
    result = directory / "result.json"
    if not result.exists():
        return None
    record = read_json(result)
    return {key: record.get(key) for key in ("backend", "execution_completed", "sensor_smoke_only")}


def _gpt_results(output_root):
    trials = []
    for preflight_path in sorted((output_root / "existing-gpt-pilot").rglob("clean_preflight.json")):
        directory = preflight_path.parent
        try:
            preflight = read_json(preflight_path)
            result_path = directory / "result.json"
            result = read_json(result_path) if result_path.exists() else None
            timing_path = directory / "clean_timing.json"
            timing = read_json(timing_path) if timing_path.exists() else None
            interruption_path = directory / "interruption.json"
            interruption = read_json(interruption_path) if interruption_path.exists() else None
        except json.JSONDecodeError:
            continue
        trials.append({
            "run": str(directory.relative_to(output_root / "existing-gpt-pilot")),
            "mode": preflight["mode"], "model": preflight["model"],
            "family": preflight["family"], "seed": preflight["seed"],
            "ready_to_attempt_model_call": preflight["ready_to_attempt_model_call"],
            "runner_finished": result is not None,
            "interruption": interruption,
            "legacy_score": result.get("score", {}).get("outcome") if result else None,
            "probe_completed": result.get("robot_metrics", {}).get("probe_completed") if result else None,
            "model_request_seconds": timing.get("model_request_seconds") if timing else None,
            "model_action_overlap_seconds": timing.get("model_action_overlap_seconds") if timing else None,
        })
    return trials


def _conditional_results(pilot):
    summaries = []
    for path in sorted(pilot.glob("conditional-*/pilot_manifest.json")):
        directory = path.parent
        summary = pilot_status(directory)
        row = {"name": directory.name, "recorded": summary["recorded_pairs"],
               "coverage_complete": summary["pairs_with_both_reports_and_coverage"]}
        arm = directory / "connection-001/conditional-a/job.json"
        if arm.is_file():
            job = read_json(arm)
            row["state"] = job["state"]
            if job.get("artifact_dir"):
                row["audit"] = audit_live_connection(Path(job["artifact_dir"]))
        summaries.append(row)
    return summaries


def _ranker_diagnostic(output_root):
    path = output_root / "ranker-diagnostic-v1/latest_summary.json"
    if not path.is_file():
        return {"models": 0, "mode": None}
    rows = read_json(path)
    return {"models": len(rows["run_rows"]), "mode": rows["mode"], "official_evaluation": False,
            "training_sizes": sorted({row["training_states"] for row in rows["run_rows"]})}


def _public_visual(output_root):
    attempts = []
    for directory in sorted((output_root / "public-visual").glob("closed-loop-attempt-*")):
        result = directory / "result.json"
        process = directory / "process.json"
        identity = read_json(process).get("process") if process.is_file() else None
        if result.is_file():
            data = read_json(result)
            attempts.append({"directory": str(directory), "state": "RECORDED", "complete": data["complete"],
                             "requests": data["requests"]["reserved"], "seconds": data["episode_seconds"]})
        elif is_live(identity):
            attempts.append({"directory": str(directory), "state": "RUNNING", "process": identity})
        else:
            attempts.append({"directory": str(directory), "state": "EXITED_NEEDS_RECOVERY"})
    return attempts


def status(directory, defaults):
    path = directory / "protocol.json"
    stats_path = directory / "statistics.json"
    audit_path = directory / "audit.json"
    stats = read_json(stats_path) if stats_path.exists() else None
    protocol = read_json(path) if path.exists() else None
    if protocol:
        completed = sum((directory / "episodes" / f"{case}-{method}.json").exists()
                        for case in protocol["case_ids"] for method in protocol["methods"])
        planned = len(protocol["case_ids"]) * len(protocol["methods"])
    else:
        completed = 0
        planned = None
    gpt = _gpt_results(directory.parent)
    ready = any(trial["legacy_score"] == "correct" and trial["mode"] == "ACTUAL_GPT_CLEAN_SERIAL"
                for trial in gpt)
    live = control_status(directory.parent / "control")
    pilot_manifest = directory.parent / "natural-pilot" / "pilot_manifest.json"
    ranker_pilot = pilot_manifest.parent / "learned-order-v1"
    typed_pilot = pilot_manifest.parent / "typed-vision-v1"
    custom_pilot = pilot_manifest.parent / CUSTOM_CONNECTION
    visual_root = directory.parent / VISUAL_INPUTS / "split_manifest_v2.json"
    visual_inputs = read_json(visual_root) if visual_root.is_file() else None
    public = _public_visual(directory.parent)
    online_public = [job for job in public if job["state"] == "RUNNING"]
    return {
        "profile": defaults["execution_profile"],
        "execution_status": "RUNNING" if online_public else live["execution_status"],
        "scientific_conclusion": live["scientific_conclusion"],
        "control": live,
        "development": pilot_status(pilot_manifest.parent) if pilot_manifest.is_file() else None,
        "conditional_development": _conditional_results(pilot_manifest.parent),
        "ranker_diagnostic": _ranker_diagnostic(directory.parent),
        "ranker_live_development": pilot_status(ranker_pilot) if (ranker_pilot / "pilot_manifest.json").is_file() else None,
        "typed_vision_development": pilot_status(typed_pilot) if (typed_pilot / "pilot_manifest.json").is_file() else None,
        "custom_vision_connection": pilot_status(custom_pilot) if (custom_pilot / "pilot_manifest.json").is_file() else None,
        "visual_layouts": None if visual_inputs is None else {key: visual_inputs[key] for key in (
            "state", "layout_count", "distinct_asset_groups", "split_roles_share_exact_assets")},
        "public_visual_validation": public,
        "run_directory": str(directory),
        "local_diagnostics": {"completed": completed, "planned": planned, "summary": stats},
        "legacy_cpu_sensor": _sensor_result(directory / "legacy_cpu_sensor"),
        "native_async_cpu": _sensor_result(directory / "native_async_cpu"),
        "existing_chatgpt_trials": gpt,
        "resources": read_json(audit_path)["resources"] if audit_path.exists() else None,
        "next": live.get("next_command") or ("pilot-run for the next bounded matched batch"
                if ready else "Audit the original clean baseline evidence"),
    }


def _online_args(parser):
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=2, help="Maximum new launches in this batch")
    parser.add_argument("--chatgpt-project", type=Path, default=DEFAULT_AUTH_PROJECT)
    parser.add_argument("--retry-external", action="store_true",
                        help="Retry a blocked job only after its external condition has changed")


def _continue_stage(output_root, directory, defaults, args, launch_limit):
    control = output_root / "control"
    state = control_status(control)

    def run_batch(path):
        result = pilot_run(path, control, CHECKOUT, resume=args.resume, max_jobs=launch_limit,
                           auth_project=args.chatgpt_project, unlock_external=args.retry_external)
        used = 0 if result.get("worker_owned_by_another_batch") else result.get("last_batch", {}).get("launched", 0)
        return result, used

    active = state["active_jobs"]
    if active:
        if args.resume and state.get("pilot_directory") and any(job.get("arm") for job in active):
            return run_batch(Path(state["pilot_directory"]))
        return state, 0
    if state["execution_status"] == "BLOCKED_EXTERNAL" and not args.retry_external:
        return state, 0
    active_public = [row for row in _public_visual(output_root) if row["state"] == "RUNNING"]
    if active_public:
        return {"execution_status": "RUNNING", "stage": "public-visual-validation", "active_jobs": active_public}, 0
    pilots = output_root / "natural-pilot"
    if not (pilots / "pilot_manifest.json").is_file():
        prepare_pilot(pilots, defaults, 480)
    warmup = output_root / "existing-gpt-pilot/temperature-gpt-budgeted-seed1/attempt-001"
    audit = warmup / "warmup_verification.json"
    if (not audit.is_file() or
            not all(read_json(audit).get(key) for key in ("frozen_query_binding_verified",
                    "observed_safe_action_entirely_while_pending", "controller_request_attempt_using_new_frame",
                    "all_attempts_match_ledger"))):
        return {"execution_status": "READY", "stage": "warmup",
                "next_action": "Record and verify the bounded live temperature integration job first"}, 0

    development = pilot_status(pilots)
    if development["recorded_pairs"] < 2:
        return run_batch(pilots)
    connection = pilots / "conditional-connection-v1"
    if not (connection / "pilot_manifest.json").is_file():
        crack = output_root / "existing-gpt-pilot/cracks-serial-seed1-recheck"
        pair = pilots / "pair-001"
        finished = [Path(read_json(pair / method / "job.json")["artifact_dir"])
                    for method in ("paired-async", "paired-serial")]
        calibration = calibrate(warmup, [crack, *finished])
        prepare_conditional(connection, read_json(pilots / "pilot_manifest.json"), calibration, connection_only=True)
    if not pilot_status(connection)["recorded_pairs"]:
        return run_batch(connection)
    arm = read_json(connection / "connection-001/conditional-a/job.json")
    verified = audit_live_connection(Path(arm["artifact_dir"]))
    comparison_version = 1
    for number in (2, 3, 4):
        if verified["early_motion_with_usable_material"]:
            break
        version = pilots / f"conditional-connection-v{number}"
        if not (version / "pilot_manifest.json").is_file():
            prepare_conditional(version, read_json(pilots / "pilot_manifest.json"),
                                read_json(connection / "calibrated_costs.json"),
                                connection_only=True, hold_at_view=True,
                                source_profile="independent" if number == 4 else "single")
        if not pilot_status(version)["recorded_pairs"]:
            return run_batch(version)
        arm = read_json(version / "connection-001/conditional-a/job.json")
        verified = audit_live_connection(Path(arm["artifact_dir"]))
        comparison_version = number
    if not verified["early_motion_with_usable_material"]:
        return {"execution_status": "NEEDS_DIAGNOSIS", "stage": "conditional-development",
                "scientific_conclusion": "INCONCLUSIVE", "development": development,
                "connection_job": arm["id"], "verified_connection": verified,
                "next_action": "Inspect the recorded permission, viewpoint, timing and material acceptance"}, 0
    comparison = pilots / f"conditional-comparison-v{comparison_version}"
    if not (comparison / "pilot_manifest.json").is_file():
        calibration = read_json(connection / "calibrated_costs.json")
        prepare_conditional(comparison, read_json(pilots / "pilot_manifest.json"), calibration,
                            hold_at_view=comparison_version > 1,
                            source_profile="independent" if comparison_version == 4 else "single")
    if not pilot_status(comparison)["recorded_pairs"]:
        return run_batch(comparison)
    if development["recorded_pairs"] < development["planned_pairs"]:
        return run_batch(pilots)
    diagnostic = output_root / "ranker-diagnostic-v1"
    if not (diagnostic / "latest_summary.json").is_file():
        from .ranker_train import experiments

        experiments(diagnostic, defaults["data"]["training_state_counts"], defaults["data"]["split_seed"])
    ranked = pilots / "learned-order-v1"
    if not (ranked / "pilot_manifest.json").is_file():
        prepare_ranker_pilot(ranked, read_json(pilots / "pilot_manifest.json"),
                             read_json(comparison / "calibrated_costs.json"), diagnostic)
    ranked_status = pilot_status(ranked)
    if ranked_status["recorded_pairs"] < ranked_status["planned_pairs"]:
        return run_batch(ranked)
    typed = pilots / "typed-vision-v1"
    if not (typed / "pilot_manifest.json").is_file():
        prepare_typed(typed, read_json(pilots / "pilot_manifest.json"), read_json(comparison / "calibrated_costs.json"))
    typed_status = pilot_status(typed)
    if typed_status["recorded_pairs"] < typed_status["planned_pairs"]:
        return run_batch(typed)
    custom = pilots / CUSTOM_CONNECTION
    prior = read_json(comparison / "calibrated_costs.json")
    if not (custom / "pilot_manifest.json").is_file():
        _prepare_scenes(("custom-development-prepare", "--output", custom, "--cost-config",
                         comparison / "calibrated_costs.json"))
    if not pilot_status(custom)["recorded_pairs"]:
        return run_batch(custom)
    layouts = output_root / VISUAL_INPUTS
    layout_manifest = layouts / "split_manifest_v2.json"
    if not layout_manifest.is_file():
        _prepare_scenes(("visual-development-prepare", "--output", layouts))
    calibration = pilots / VISUAL_PILOTS["calibration"]
    if not (calibration / "pilot_manifest.json").is_file():
        prepare_visual(calibration, read_json(layout_manifest), prior, "calibration")
    calibrated_status = pilot_status(calibration)
    if calibrated_status["recorded_pairs"] < calibrated_status["planned_pairs"]:
        return run_batch(calibration)
    selection = pilots / VISUAL_PILOTS["selection"]
    if not (selection / "pilot_manifest.json").is_file():
        prepare_visual(selection, read_json(layout_manifest), calibrate_visual(calibration, prior), "selection")
    selected_status = pilot_status(selection)
    if selected_status["recorded_pairs"] < selected_status["planned_pairs"]:
        return run_batch(selection)
    return {"execution_status": "READY", "stage": "evaluation-development",
            "scientific_conclusion": "INCONCLUSIVE", "run_directory": str(directory), "development": development,
            "ranker_development": ranked_status, "typed_vision_development": typed_status,
            "public_visual_validation": _public_visual(output_root),
            "visual_calibration": calibrated_status, "visual_selection": selected_status,
            "next_action": "Finish baseline and protocol checks before the held-out test"}, 0


def _continue(output_root, directory, defaults, args):
    if args.max_jobs < 1:
        raise ValueError("Choose a positive maximum batch size")
    remaining = args.max_jobs
    batches = []
    while remaining:
        result, launched = _continue_stage(output_root, directory, defaults, args, remaining)
        if launched:
            batches.append({"stage": result["stage"], **result["last_batch"]})
            remaining -= launched
        current = result.get("current_job") or {}
        if (not launched or result.get("active_jobs") or current.get("state") in
                {"INFRA_ERROR", "BLOCKED_EXTERNAL", "CANCELLED_BY_USER"}):
            break
    return {**result, "project_batches": batches, "remaining_launch_allowance": remaining}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    clean = commands.add_parser("audit-clean", help="Recheck complete-object coverage and actual model image bindings")
    clean.add_argument("--episode-dir", required=True, type=Path)
    live = commands.add_parser("audit-live", help="Recheck a real conditional action, photo and report handoff")
    live.add_argument("--episode-dir", required=True, type=Path)
    public = commands.add_parser("audit-public", help="Recheck public-camera evidence and actual model requests")
    public.add_argument("--episode-dir", required=True, type=Path)
    public_plan = commands.add_parser("public-plan", help="Assign blinded ProcTHOR test houses and episode slots")
    public_plan.add_argument("--test-archive", required=True, type=Path)
    public_plan.add_argument("--output", required=True, type=Path)
    custom_plan = commands.add_parser("custom-plan", help="Assign the three blind custom evaluation strata")
    custom_plan.add_argument("--development-manifest", required=True, type=Path)
    custom_plan.add_argument("--output", required=True, type=Path)
    custom_dev = commands.add_parser("custom-development-prepare", help="Register a custom-task connection check")
    custom_dev.add_argument("--cost-config", required=True, type=Path)
    custom_dev.add_argument("--output", required=True, type=Path)
    visual_inputs = commands.add_parser("visual-development-prepare", help="Materialize the 96 development layouts")
    visual_inputs.add_argument("--output", required=True, type=Path)
    visual_inputs.add_argument("--pilot-dir", type=Path, help="Register the initial calibration run")
    visual_inputs.add_argument("--cost-config", type=Path, help="Measured timing priors for the calibration run")
    learn = commands.add_parser("ranker-diagnostic", help="Run the grouped symbolic CPU ranking diagnostic")
    learn.add_argument("--data-dir", required=True, type=Path)
    learn.add_argument("--sizes", type=int, nargs="+", choices=(1000, 3000, 10000), default=[1000, 3000, 10000])
    audit_labels = commands.add_parser("ranker-audit", help="Replay a stratified sample from the saved ranking data")
    audit_labels.add_argument("--data-dir", required=True, type=Path)
    audit_labels.add_argument("--output", required=True, type=Path)
    plan = commands.add_parser("pilot-prepare", help="Write the frozen 48-case development schedule locally")
    plan.add_argument("--pilot-dir", required=True, type=Path)
    plan.add_argument("--episode-timeout", type=float, default=480)
    pilot = commands.add_parser("pilot-status", help="Audit recorded pairs against their scheduled cases")
    pilot.add_argument("--pilot-dir", required=True, type=Path)
    report = commands.add_parser("pilot-report", help="Rebuild episode and paired results from raw files")
    report.add_argument("--pilot-dir", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    batch = commands.add_parser("pilot-run", help="Run the next bounded batch from the locked pilot schedule")
    batch.add_argument("--pilot-dir", required=True, type=Path)
    batch.add_argument("--control-dir", type=Path, default=CHECKOUT / "outputs/control")
    _online_args(batch)
    for name in ("audit", "run", "status", "reproduce", "continue"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=("local5060",), default="local5060")
        command.add_argument("--run-id", default="local5060-start")
        command.add_argument("--output-root", type=Path, default=CHECKOUT / "outputs")
        if name == "run":
            command.add_argument("--resume", action="store_true")
        if name == "continue":
            _online_args(command)
    args = parser.parse_args(argv)
    if args.command == "audit-public":
        from .public_audit import audit

        print(json.dumps(audit(args.episode_dir), ensure_ascii=False, indent=2))
        return
    if args.command == "public-plan":
        from .public_schedule import prepare

        defaults = yaml.safe_load(DEFAULTS.read_text(encoding="utf8"))
        print(json.dumps(prepare(args.test_archive, args.output, defaults), ensure_ascii=False, indent=2))
        return
    if args.command == "custom-plan":
        from .custom_schedule import prepare

        defaults = yaml.safe_load(DEFAULTS.read_text(encoding="utf8"))
        print(json.dumps(prepare(args.output, defaults, read_json(args.development_manifest)), ensure_ascii=False, indent=2))
        return
    if args.command == "custom-development-prepare":
        if Path(sys.executable).parent != _scene_python().parent:
            _prepare_scenes((args.command, "--output", args.output.resolve(), "--cost-config", args.cost_config.resolve()))
            return
        from .custom_schedule import prepare_development

        catalog = prepare_development(args.output, read_json(args.cost_config))
        print(json.dumps({"case": catalog["cases"][0]["id"], "output": str(args.output)}, ensure_ascii=False))
        return
    if args.command == "visual-development-prepare":
        if bool(args.pilot_dir) != bool(args.cost_config):
            parser.error("Supply both --pilot-dir and --cost-config to register calibration")
        if Path(sys.executable).parent != _scene_python().parent:
            command = [args.command, "--output", args.output.resolve()]
            if args.pilot_dir:
                command.extend(("--pilot-dir", args.pilot_dir.resolve(), "--cost-config", args.cost_config.resolve()))
            _prepare_scenes(command)
            return
        from .development_inputs import prepare

        defaults = yaml.safe_load(DEFAULTS.read_text(encoding="utf8"))
        result = prepare(args.output, defaults)
        if args.pilot_dir:
            catalog = prepare_visual(args.pilot_dir, read_json(args.output / "split_manifest_v2.json"),
                                     read_json(args.cost_config), "calibration")
            result["calibration_job_count"] = sum(len(case["order"]) for case in catalog["cases"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "ranker-diagnostic":
        from .ranker_train import experiments
        defaults = yaml.safe_load(DEFAULTS.read_text(encoding="utf8"))
        result = experiments(args.data_dir, args.sizes, defaults["data"]["split_seed"])
        print(json.dumps({"mode": result["mode"], "status": result["status"],
                          "runs": len(result["run_rows"]), "output": str(args.data_dir)}, ensure_ascii=False))
        return
    if args.command == "ranker-audit":
        from .ranker_audit import audit

        print(json.dumps(audit(args.data_dir, args.output), ensure_ascii=False, indent=2))
        return
    if args.command == "pilot-report":
        from .pilot_report import rebuild
        try:
            result = rebuild(args.pilot_dir, args.output)
        except (OSError, ValueError, KeyError) as error:
            parser.exit(1, f"{error}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.command in ("pilot-prepare", "pilot-status", "pilot-run"):
        defaults = yaml.safe_load(DEFAULTS.read_text(encoding="utf8"))
        try:
            if args.command == "pilot-prepare":
                prepare_pilot(args.pilot_dir, defaults, args.episode_timeout)
            if args.command == "pilot-run":
                result = pilot_run(args.pilot_dir, args.control_dir, CHECKOUT, resume=args.resume,
                                   max_jobs=args.max_jobs, auth_project=args.chatgpt_project,
                                   unlock_external=args.retry_external)
            else:
                result = pilot_status(args.pilot_dir)
        except (ValueError, OSError, KeyError) as error:
            parser.exit(1, f"{error}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.command in ("audit-clean", "audit-live"):
        try:
            scorer = audit_run if args.command == "audit-clean" else audit_live_connection
            result = scorer(args.episode_dir)
        except (ValueError, OSError, KeyError) as error:
            parser.exit(1, f"{error}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.run_id):
        parser.error("Invalid run ID")
    directory = args.output_root.resolve() / args.run_id
    defaults = yaml.safe_load(DEFAULTS.read_text(encoding="utf8"))
    try:
        if args.command == "audit":
            result = capture(directory, CHECKOUT, defaults)
        elif args.command == "run":
            result = run(directory, CHECKOUT, defaults, args.resume)
        elif args.command == "reproduce":
            result = reproduce(directory, manifest(CHECKOUT, defaults))
        elif args.command == "continue":
            result = _continue(args.output_root.resolve(), directory, defaults, args)
        else:
            result = status(directory, defaults)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
