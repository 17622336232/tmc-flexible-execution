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
from experiments.task_attack.tasks import SCENES
from experiments.task_attack_integrity.run import main as run_simulation
from uav_bridge.chatgpt_auth import CredentialStore

from ..candidate_ranking import ONLINE_RANKED_METHODS, RankedPolicy
from ..clean_coverage import audit_run
from ..live_planning import CONDITIONAL_MODES, Costs, EARLY_GRANT, LATE_GRANT
from ..pilot_schedule import PAIRED_MODES
from ..request_budget import RequestBudget
from ..runner import _hash
from ..timeline import Timeline, summarize
from .gpt_probe import run_paired, run_probe
from .native_views import MODES, EVIDENCE_MODES, COORDINATED_MODES
NATIVE_VIEW_MODES = MODES + EVIDENCE_MODES + COORDINATED_MODES


METHOD_MODES = {"original-gpt": "UPSTREAM_GPT_HASH_PINNED", "serial": "ACTUAL_GPT_CLEAN_SERIAL", "async-probe": "ACTUAL_GPT_FROZEN_QUERY_INTEGRATION_PROBE",
                **PAIRED_MODES, **CONDITIONAL_MODES,
                **{name:"ORIGINAL_LOOP_"+name.upper().replace("-","_") for name in NATIVE_VIEW_MODES}}


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
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--scene-fingerprint", help="Expected geometry, materials and task digest")
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
    parser.add_argument("--compile-task-program", action="store_true", help="Check a typed task program at submission")
    parser.add_argument("--protocol-v2", action="store_true", help="Use the common v2 inspection loop")
    parser.add_argument("--bounded-text-compiler", action="store_true", help="Compile exactly recognized task text locally")
    parser.add_argument("--execution-revision", help="Registered execution implementation")
    parser.add_argument("--study-task", type=Path, help="Registered custom task with a private scoring contract")
    parser.add_argument("--frontend-reference", type=Path, help="Shared front-end reference for this registered case")
    parser.add_argument("--onboard-evidence-only", action="store_true",
                        help="Archive onboard task evidence and leave the spectator video out of the recording")
    rules = parser.add_mutually_exclusive_group()
    rules.add_argument("--early-fields", help="Shared controlled-language grant for this development instance")
    rules.add_argument("--collection-rule", help="Recorded controlled-language collection clause")
    parser.add_argument("--handoff-at-observation", action="store_true",
                        help="Allow a shared handoff at the acquired view with time reserved to revisit")
    args = parser.parse_args(argv)
    if args.bounded_text_compiler and (not args.protocol_v2 or args.frontend_reference):
        parser.error("The bounded compiler uses a fresh, uncached v2 front end")
    if not isfinite(args.episode_timeout) or args.episode_timeout <= 0:
        parser.error("Choose a positive finite episode timeout")
    if args.max_request_attempts < 1:
        parser.error("Choose a positive transport attempt limit")
    planned = args.episode_timeout if args.planned_episode_timeout is None else args.planned_episode_timeout
    if not isfinite(planned) or planned < args.episode_timeout:
        parser.error("The logical job duration must cover the attempt duration")
    conditional = args.method in CONDITIONAL_MODES
    native = args.method in NATIVE_VIEW_MODES
    if native and args.backend != "cpu":
        parser.error("Native view scheduling is currently registered only for CPU simulation")
    if (conditional or native) != bool(args.cost_config) or (
            (args.handoff_at_observation or args.compile_task_program
             or args.protocol_v2)
            and not conditional):
        parser.error("Conditional methods require a cost file and own the optional development grant")
    if (args.early_fields or args.collection_rule) and not (conditional or native or args.method == "original-gpt"):
        parser.error("Collection clauses require a scoped inspection arm")
    if args.scene_fingerprint and (args.layout_seed is None or len(args.scene_fingerprint) != 64
                                   or any(char not in "0123456789abcdef" for char in args.scene_fingerprint)):
        parser.error("A scene fingerprint requires a registered layout and SHA-256 digest")
    if args.frontend_reference and not (conditional and args.scene_fingerprint):
        parser.error("A front-end reference needs a registered conditional scene")
    study_task = None
    if args.study_task:
        from ..requirement_program import parse

        if not args.compile_task_program and args.method != "original-gpt" and not native:
            parser.error("Custom tasks use the shared language-to-program front end")
        study_task = json.loads(args.study_task.read_text(encoding="utf8"))
        if set(study_task) != {"instruction", "oracle_requirement_program"} or not isinstance(study_task["instruction"], str):
            parser.error("Registered custom tasks need task text and a private scoring contract")
        parse(study_task["oracle_requirement_program"], study_task["instruction"])
    if (args.method in ONLINE_RANKED_METHODS) != bool(args.ranker_config):
        parser.error("Ranking methods require the registered ranking profile")
    ranking = json.loads(args.ranker_config.read_text(encoding="utf8")) if args.ranker_config else None
    profile = RankedPolicy.load(ranking["policies"][args.method], args.method) if ranking else None
    clause = ""
    early_fields = args.early_fields.split(",") if args.early_fields else []
    collection_fields = early_fields
    if early_fields:
        clause = "采集规则：允许在判读结果返回前采集以下材料：" + "、".join(early_fields) + "。"
        if not EARLY_GRANT.fullmatch(clause) or len(set(early_fields)) != len(early_fields):
            parser.error("Invalid or repeated field in the development grant")
    if args.collection_rule:
        clause = args.collection_rule
        early = EARLY_GRANT.fullmatch(clause)
        late = LATE_GRANT.fullmatch(clause)
        if not early and not late:
            parser.error("Collection rule must match the registered permission syntax")
        collection_fields = (early[1] if early else late[2]).split("、")
    costs = None
    if conditional or native:
        from .live_inspection import SUPPORTED_MATERIALS, run_live
        if not set(collection_fields) <= SUPPORTED_MATERIALS or len(set(collection_fields)) != len(collection_fields):
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
    if args.onboard_evidence_only:
        preflight["archive_external_rgb"] = False
    if args.frontend_reference:
        preflight["shared_frontend"] = True
    if args.layout_seed is not None:
        preflight["layout_seed"] = args.layout_seed
    if args.collection_rule:
        preflight["collection_rule"] = args.collection_rule
    if args.scene_fingerprint:
        preflight["scene_fingerprint"] = args.scene_fingerprint
    if study_task is not None:
        preflight["study_task_sha256"] = _hash(study_task)
    if conditional:
        preflight.update(early_fields=early_fields, calibration_id=costs.calibration_id,
                         handoff_at_observation=args.handoff_at_observation)
    if args.compile_task_program:
        preflight["typed_task_program"] = True
    if args.protocol_v2:
        preflight.update(protocol_version="protocol-v2", evidence_level=(
            "TRACE_SUPPORTED" if args.frontend_reference else "NATURAL_ONLINE"))
        preflight["bounded_text_compiler"] = args.bounded_text_compiler
    if args.execution_revision:
        preflight["execution_revision"] = args.execution_revision
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
        if args.scene_fingerprint:
            from ..development_inputs import _verify_scene, scene_fingerprint

            expected_case = {"id": root.name, "domain": SCENES[args.family], "seed": args.seed,
                             "layout_seed": args.layout_seed, "family": args.family,
                             "source_profile": args.source_profile, "equipment_count": args.equipment_count,
                             "collection_clause": clause}
            actual_input = _verify_scene(expected_case, root)
            if scene_fingerprint(actual_input) != args.scene_fingerprint:
                raise ValueError("The generated simulator input differs from the registered case")
            write_json(root / "clean_verified_scene.json", {key: actual_input[key] for key in (
                "layout_group_sha256", "asset_group_sha256", "instruction_sha256", "world_sha256")})
        trace = Timeline(directory / "timeline.jsonl")
        client = ModelClient(directory / "model_calls.jsonl", credential_store=credentials, request_budget=budget)
        execution = {"serial": run_agent, "async-probe": run_probe}.get(args.method)
        if args.method == "original-gpt":
            from .original_controller import run as execution
        if native:
            from .native_views import run as run_native
            execution = partial(run_native, method=args.method, costs=costs)
        if conditional:
            execution = partial(run_live, method=args.method, costs=costs, hold_at_view=args.handoff_at_observation,
                                ranked_policy=profile, use_program=args.compile_task_program,
                                frontend_reference=args.frontend_reference, scene_sha=args.scene_fingerprint,
                                protocol_v2=args.protocol_v2, bounded_text=args.bounded_text_compiler)
        elif execution is None:
            execution = partial(run_paired, during_pending=args.method == "paired-async")
        if clause:
            original_hash = sha256(task.encode("utf8")).hexdigest()
            task = task + "\n" + clause
            write_json(root / "clean_task_scope.json", {"original_task_sha256": original_hash,
                       "collection_clause": clause, "authorized_task": task})
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
    simulation_args = [
        "--backend", args.backend, "--headless", "--no-rviz", "--policy", "clean",
        "--family", args.family, "--source-profile", args.source_profile,
        "--equipment-count", str(args.equipment_count), "--seed", str(args.seed), "--output", str(root),
        "--ros-port", str(args.ros_port), "--gazebo-port", str(args.gazebo_port),
        "--episode-timeout", str(args.episode_timeout),
    ]
    if args.layout_seed is not None:
        simulation_args.extend(("--layout-seed", str(args.layout_seed)))
    code = run_simulation(simulation_args, agent_runner=instrumented_agent, clean_study_task=study_task,
                          merge_report_fragments=args.compile_task_program,
                          archive_external_rgb=not args.onboard_evidence_only)
    if args.onboard_evidence_only:
        from ..storage import archive_sensor_logs

        archive_sensor_logs(root)
    if code == 0 and args.method != "async-probe":
        scoring_started = time.monotonic()
        write_json(root / "clean_coverage.json", audit_run(root))
        if args.execution_revision:
            write_json(root / "independent_scoring_time.json", {
                "start_monotonic": scoring_started, "end_monotonic": time.monotonic(),
                "scope": "post_episode_independent_coverage"})
    return code


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())
