"""Run and time the repository's existing clean GPT vision simulation."""

import argparse
import json
import os
import signal
from functools import partial
from math import isfinite
from pathlib import Path

from experiments.task_attack import EFFORT, FAMILIES, MODEL, SOURCE_PROFILES
from experiments.task_attack.agent import run_agent
from experiments.task_attack.model import ModelClient
from experiments.task_attack.runtime import write_json
from experiments.task_attack_integrity.run import main as run_simulation
from uav_bridge.chatgpt_auth import CredentialStore

from ..clean_coverage import audit_run
from ..timeline import Timeline, summarize
from .gpt_probe import run_paired, run_probe


METHOD_MODES = {"serial": "ACTUAL_GPT_CLEAN_SERIAL", "async-probe": "ACTUAL_GPT_FROZEN_QUERY_INTEGRATION_PROBE",
                "paired-serial": "ACTUAL_GPT_SHARED_SERIAL", "paired-async": "ACTUAL_GPT_SHARED_ASYNC"}


def status(store):
    current = store.status()
    return {
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "signed_in": current["signed_in"],
        "plan_usage_enabled": current["plan_usage_enabled"],
        "access_token_valid": current["access_token_valid"],
        "refresh_available": current["refresh_available"],
        "model_in_catalog": store.is_model_available(MODEL),
        "sign_in_error": current.get("error"),
    }


def _interrupt(_signal_number, _frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chatgpt-project", type=Path, help="Project with the existing ChatGPT sign-in")
    parser.add_argument("--method", choices=METHOD_MODES, default="serial")
    parser.add_argument("--family", choices=FAMILIES, default="cracks")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-profile", choices=SOURCE_PROFILES, default="single")
    parser.add_argument("--equipment-count", type=int, choices=range(2, 7), default=2)
    parser.add_argument("--backend", choices=("cpu", "gazebo"), default="cpu")
    parser.add_argument("--ros-port", type=int, default=11674)
    parser.add_argument("--gazebo-port", type=int, default=11675)
    parser.add_argument("--episode-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    if not isfinite(args.episode_timeout) or args.episode_timeout <= 0:
        parser.error("Choose a positive finite episode timeout")

    root = args.output.resolve()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Choose a new episode directory")
    credentials = CredentialStore(directory=args.chatgpt_project / ".runtime" / "chatgpt"
                                  if args.chatgpt_project else None)
    preflight = status(credentials)
    mode = METHOD_MODES[args.method]
    preflight.update(mode=mode, backend=args.backend, family=args.family, seed=args.seed,
                     equipment_count=args.equipment_count, source_profile=args.source_profile,
                     episode_timeout_seconds=args.episode_timeout)
    ready = (preflight["signed_in"] and preflight["plan_usage_enabled"] and preflight["model_in_catalog"]
             and (preflight["access_token_valid"] or preflight["refresh_available"]))
    preflight["ready_to_attempt_model_call"] = ready
    write_json(root / "clean_preflight.json", preflight)
    if not ready:
        print(json.dumps(preflight, ensure_ascii=False), flush=True)
        return 2

    def instrumented_agent(backend, task, directory, event, timeout):
        trace = Timeline(directory / "timeline.jsonl")
        client = ModelClient(directory / "model_calls.jsonl", credential_store=credentials)
        execution = {"serial": run_agent, "async-probe": run_probe}.get(args.method)
        if execution is None:
            execution = partial(run_paired, during_pending=args.method == "paired-async")
        try:
            return execution(backend, task, directory, event, timeout, model=client, trace=trace)
        finally:
            write_json(root / "clean_timing.json", summarize(trace.path))

    os.environ.setdefault("ROS_HOME", str(root / "ros_home"))
    code = run_simulation([
        "--backend", args.backend, "--headless", "--no-rviz", "--policy", "clean",
        "--family", args.family, "--source-profile", args.source_profile,
        "--equipment-count", str(args.equipment_count), "--seed", str(args.seed), "--output", str(root),
        "--ros-port", str(args.ros_port), "--gazebo-port", str(args.gazebo_port),
        "--episode-timeout", str(args.episode_timeout),
    ], agent_runner=instrumented_agent)
    if code == 0 and args.method != "async-probe":
        write_json(root / "clean_coverage.json", audit_run(root))
    return code


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())
