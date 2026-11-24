"""Local audit, diagnostic execution, status and event-ledger reproduction."""

import argparse
import json
import re
from pathlib import Path

import yaml

from .audit import capture
from .runner import manifest, read_json, reproduce, run


CHECKOUT = Path(__file__).resolve().parents[2]
DEFAULTS = Path(__file__).with_name("experiment_defaults.yaml")


def _sensor_result(directory):
    result = directory / "result.json"
    if not result.exists():
        return None
    record = read_json(result)
    return {key: record.get(key) for key in ("backend", "execution_completed", "sensor_smoke_only")}


def _gpt_results(output_root):
    trials = []
    for preflight_path in sorted((output_root / "existing-gpt-pilot").glob("*/clean_preflight.json")):
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
            "run": directory.name, "mode": preflight["mode"], "model": preflight["model"],
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
    return {
        "profile": defaults["execution_profile"],
        "execution_status": "IN_PROGRESS",
        "scientific_conclusion": "INCONCLUSIVE",
        "run_directory": str(directory),
        "local_diagnostics": {"completed": completed, "planned": planned, "summary": stats},
        "legacy_cpu_sensor": _sensor_result(directory / "legacy_cpu_sensor"),
        "native_async_cpu": _sensor_result(directory / "native_async_cpu"),
        "existing_chatgpt_trials": gpt,
        "resources": read_json(audit_path)["resources"] if audit_path.exists() else None,
        "next": ("Use the existing GPT front end for the complete task, then run the 48 paired development cases"
                 if ready else "Measure the clean baseline on the original project vision service"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "run", "status", "reproduce"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=("local5060",), default="local5060")
        command.add_argument("--run-id", default="local5060-start")
        command.add_argument("--output-root", type=Path, default=CHECKOUT / "outputs")
        if name == "run":
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
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
        else:
            result = status(directory, defaults)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
