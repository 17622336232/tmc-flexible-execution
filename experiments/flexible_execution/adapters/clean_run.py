"""Run and time the repository's existing clean GPT vision simulation."""

import argparse
import json
import os
import signal
import time
from hashlib import sha256
from functools import partial
from math import isfinite
from pathlib import Path

from experiments.task_attack import EFFORT, FAMILIES, MODEL, SOURCE_PROFILES
from experiments.task_attack.agent import run_agent
from experiments.task_attack.model import ModelClient
from experiments.task_attack.runtime import write_json
from experiments.task_attack_integrity.run import main as run_simulation
from uav_bridge.chatgpt_auth import CredentialStore

from ..candidate_ranking import RANKED_METHODS, RankedPolicy
from ..clean_coverage import audit_run
from ..live_planning import CONDITIONAL_MODES, Costs, EARLY_GRANT
from ..pilot_schedule import PAIRED_MODES
from ..request_budget import RequestBudget
from ..runner import _hash
from ..timeline import Timeline, summarize
from .gpt_probe import run_paired, run_probe


METHOD_MODES = {"serial": "ACTUAL_GPT_CLEAN_SERIAL", "async-probe": "ACTUAL_GPT_FROZEN_QUERY_INTEGRATION_PROBE",
                **PAIRED_MODES, **CONDITIONAL_MODES}


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
    parser.add_argument("--planned-episode-timeout", type=float, help="Logical job time limit before partial attempts")
    parser.add_argument("--max-request-attempts", type=int, default=24)
    parser.add_argument("--request-ledger", type=Path, help="Persistent request ledger for this logical job")
    parser.add_argument("--cost-config", type=Path, help="Calibrated design input for the conditional methods")
    parser.add_argument("--ranker-config", type=Path, help="Frozen candidate scoring and check budget")
    parser.add_argument("--early-fields", help="Shared controlled-language grant for this development instance")
    parser.add_argument("--handoff-at-observation", action="store_true",
                        help="Allow a shared handoff at the acquired view with time reserved to revisit")
    args = parser.parse_args(argv)
    if not isfinite(args.episode_timeout) or args.episode_timeout <= 0:
        parser.error("Choose a positive finite episode timeout")
    if args.max_request_attempts < 1:
        parser.error("Choose a positive transport attempt limit")
    planned = args.episode_timeout if args.planned_episode_timeout is None else args.planned_episode_timeout
    if not isfinite(planned) or planned < args.episode_timeout:
        parser.error("The logical job duration must cover the attempt duration")
    conditional = args.method in CONDITIONAL_MODES
    if conditional != bool(args.cost_config) or ((args.early_fields or args.handoff_at_observation) and not conditional):
        parser.error("Conditional methods require a cost file and own the optional development grant")
    if (args.method in RANKED_METHODS) != bool(args.ranker_config):
        parser.error("Ranking methods require the registered ranking profile")
    ranking = json.loads(args.ranker_config.read_text(encoding="utf8")) if args.ranker_config else None
    profile = RankedPolicy.load(ranking["policies"][args.method], args.method) if ranking else None
    clause = ""
    early_fields = args.early_fields.split(",") if args.early_fields else []
    if early_fields:
        clause = "采集规则：允许在判读结果返回前采集以下材料：" + "、".join(early_fields) + "。"
        if not EARLY_GRANT.fullmatch(clause) or len(set(early_fields)) != len(early_fields):
            parser.error("Invalid or repeated field in the development grant")
    costs = None
    if conditional:
        from .live_inspection import SUPPORTED_MATERIALS, run_live
        if not set(early_fields) <= SUPPORTED_MATERIALS:
            parser.error("The live material checker does not support one of the requested fields")
        cost_document = json.loads(args.cost_config.read_text(encoding="utf8"))
        costs = Costs(**cost_document["costs"])

    root = args.output.resolve()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Choose a new episode directory")
    credentials = CredentialStore(directory=args.chatgpt_project / ".runtime" / "chatgpt"
                                  if args.chatgpt_project else None)
    budget = RequestBudget((args.request_ledger or root / "request_attempts.jsonl").resolve(),
                           args.max_request_attempts)
    preflight = status(credentials)
    mode = METHOD_MODES[args.method]
    preflight.update(mode=mode, backend=args.backend, family=args.family, seed=args.seed,
                     equipment_count=args.equipment_count, source_profile=args.source_profile,
                     episode_timeout_seconds=args.episode_timeout, planned_episode_timeout_seconds=planned,
                     request_budget=budget.summary())
    if conditional:
        preflight.update(early_fields=early_fields, calibration_id=costs.calibration_id,
                         handoff_at_observation=args.handoff_at_observation)
    if ranking:
        preflight.update(ranker_config_sha256=_hash(ranking), ranker_policy=ranking["policies"][args.method])
    ready = (preflight["signed_in"] and preflight["plan_usage_enabled"] and preflight["model_in_catalog"]
             and (preflight["access_token_valid"] or preflight["refresh_available"]))
    preflight["ready_to_attempt_model_call"] = ready
    write_json(root / "clean_preflight.json", preflight)
    if not ready:
        print(json.dumps(preflight, ensure_ascii=False), flush=True)
        return 2

    def instrumented_agent(backend, task, directory, event, timeout):
        trace = Timeline(directory / "timeline.jsonl")
        client = ModelClient(directory / "model_calls.jsonl", credential_store=credentials, request_budget=budget)
        execution = {"serial": run_agent, "async-probe": run_probe}.get(args.method)
        if conditional:
            execution = partial(run_live, method=args.method, costs=costs, hold_at_view=args.handoff_at_observation,
                                ranked_policy=profile)
            if clause:
                original_hash = sha256(task.encode("utf8")).hexdigest()
                task = task + "\n" + clause
                write_json(root / "clean_task_scope.json", {"original_task_sha256": original_hash,
                           "collection_clause": clause, "authorized_task": task})
        elif execution is None:
            execution = partial(run_paired, during_pending=args.method == "paired-async")
        started = time.monotonic()
        clock = {"started_wall": time.time(), "allowance_seconds": timeout}
        write_json(root / "job_clock.json", clock)
        try:
            return execution(backend, task, directory, event, timeout, model=client, trace=trace)
        finally:
            write_json(root / "job_clock.json", {**clock, "elapsed_seconds": time.monotonic() - started})
            write_json(root / "request_budget_summary.json", budget.summary())
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
