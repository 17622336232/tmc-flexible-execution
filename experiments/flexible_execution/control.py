"""Execute locked, bounded development jobs and recover their actual state."""

import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from resource import RUSAGE_CHILDREN, getrusage
from shutil import disk_usage
from statistics import median

from experiments.task_attack import EFFORT, MODEL
from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES
from experiments.task_attack.runtime import write_json

from .candidate_ranking import ONLINE_RANKED_METHODS
from .clean_coverage import audit_run
from .custom_schedule import FINAL_ROLE as CUSTOM_FINAL_ROLE
from .learning_schedule import ROLE as RANKER_ROLE, registered_config
from .live_planning import CONDITIONAL_MODES, STREAM_METHODS
from .pilot_schedule import PAIRED_MODES, STREAM_ROLE, TYPED_ROLE, VISUAL_ROLES, _read, _valid_preflight
from .public_schedule import FINAL_ROLE as PUBLIC_FINAL_ROLE, VALIDATION_ROLE as PUBLIC_VALIDATION_ROLE
from .request_budget import RequestBudget
from .runner import _hash


DEFAULT_AUTH_PROJECT = Path.home() / "uav" / "Diff-VLA"
MAX_ATTEMPTS = 3
REQUEST_LIMIT = 24
HEARTBEAT_SECONDS = 30
STARTUP_CLEANUP_SECONDS = 90
TASK_TERMINAL = {"FINISHED_VALID", "FINISHED_TASK_FAILURE", "TIMEOUT"}
RUN_SOURCES = (
    "scripts/run_flexible_clean.sh", "experiments/flexible_execution/adapters/clean_run.py",
    "experiments/flexible_execution/adapters/gpt_probe.py", "experiments/flexible_execution/adapters/viewpoints.py",
    "experiments/flexible_execution/clean_coverage.py", "experiments/flexible_execution/gpt_vision.py",
    "experiments/flexible_execution/queries.py", "experiments/flexible_execution/evidence.py",
    "experiments/flexible_execution/task_ir.py", "experiments/flexible_execution/executor.py",
    "experiments/flexible_execution/search.py", "experiments/flexible_execution/timeline.py",
    "experiments/flexible_execution/request_budget.py", "experiments/task_attack_integrity/run.py",
    "experiments/task_attack", "bridge/uav_bridge", "sim_ws/src/vla_sim",
)
CONDITIONAL_SOURCES = ("experiments/flexible_execution/adapters/live_inspection.py",
                       "experiments/flexible_execution/live_planning.py")
RANKING_SOURCE = "experiments/flexible_execution/candidate_ranking.py"
PUBLIC_SOURCES = ("experiments/flexible_execution/public_task.py", "experiments/flexible_execution/public_schedule.py",
                  "experiments/flexible_execution/public_visual.py", "experiments/flexible_execution/public_audit.py")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _batch_estimate(control, directory, protocol, count):
    path = control / "jobs.jsonl"
    same_pilot, same_simulator = deque(maxlen=24), deque(maxlen=24)
    unreadable = 0
    if path.is_file():
        with path.open(encoding="utf8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if (row["state"] not in TASK_TERMINAL or not row.get("started_at") or not row.get("finished_at")
                            or row.get("model") != protocol["model"]
                            or row.get("reasoning_effort") != protocol["reasoning_effort"]):
                        continue
                    seconds = (datetime.fromisoformat(row["finished_at"]) -
                               datetime.fromisoformat(row["started_at"])).total_seconds()
                    pilot = Path(row["arm"]).parents[1]
                    stage = row["stage"]
                    if not isinstance(stage, str):
                        raise ValueError("Timing record has an invalid stage")
                except (ValueError, KeyError, TypeError, IndexError):
                    unreadable += 1
                    continue
                if seconds <= 0:
                    continue
                if pilot == directory:
                    same_pilot.append(seconds)
                elif stage.startswith("public-") == protocol["stage"].startswith("public-"):
                    same_simulator.append(seconds)
    observations = same_pilot or same_simulator
    typical = median(observations) if observations else None
    return {"prior_recorded_launches": len(observations), "unreadable_timing_records": unreadable,
            "prior_scope": "same pilot" if same_pilot else "same simulator" if same_simulator else None,
            "typical_seconds_per_launch": typical, "estimated_wall_seconds": typical * count if typical is not None else None}


def _proc(pid):
    try:
        path = Path("/proc") / str(pid)
        stat = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        if stat[0] in {"Z", "X"}:
            return None
        cmd = (path / "cmdline").read_bytes()
        return {"pid": pid, "pgid": int(stat[2]), "start_ticks": stat[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "command_sha256": hashlib.sha256(cmd).hexdigest()}
    except (OSError, IndexError, ValueError):
        return None


def is_live(identity):
    if not identity:
        return False
    current = _proc(identity["pid"])
    # The launcher execs Python in the same process after sourcing ROS.
    return bool(current and all(current[key] == identity[key] for key in ("pid", "pgid", "start_ticks", "boot_id")))


def _find_worker(episode):
    expected = os.fsencode(str(episode))
    for folder in Path("/proc").iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            args = (folder / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if expected not in args or b"--output" not in args:
            continue
        if b"scripts/run_flexible_clean.sh" in args or b"experiments.flexible_execution.adapters.clean_run" in args:
            return _proc(int(folder.name))
        if b"experiments.flexible_execution.public_task" in args and b"--registered-job" in args:
            identity = _proc(int(folder.name))
            if identity and identity["pgid"] == identity["pid"]:
                return identity
    return None


def simulator_running(control):
    state = _read(control / "state.json") or {}
    warmup = state.get("current_job") or {}
    if warmup.get("state") != "RUNNING" or not warmup.get("pid"):
        return None
    live = _proc(warmup["pid"])
    if live and live["start_ticks"] == warmup.get("process_start_ticks"):
        return {"id": warmup["id"], "process": live, "state": "RUNNING"}
    return None


def _runtime_inputs(checkout, catalog, auth_project, ports):
    sources = []
    ranked = catalog.get("role") == RANKER_ROLE
    custom_final = catalog.get("role") == CUSTOM_FINAL_ROLE
    typed = catalog.get("role") == TYPED_ROLE or custom_final
    visual = catalog.get("role") in VISUAL_ROLES.values()
    public = catalog.get("role") in (PUBLIC_VALIDATION_ROLE, PUBLIC_FINAL_ROLE)
    stream = catalog.get("role") == STREAM_ROLE
    conditional = ranked or typed or visual or stream or catalog.get("role") == "CONDITIONAL_CONNECTION_DEVELOPMENT"
    stage_sources = CONDITIONAL_SOURCES if conditional else ()
    if ranked or (conditional and (checkout / RANKING_SOURCE).is_file()):
        stage_sources += (RANKING_SOURCE,)
    if typed:
        stage_sources += ("experiments/flexible_execution/requirement_program.py",)
    value_schema = "experiments/flexible_execution/material_schema.py"
    if conditional and (checkout / value_schema).is_file():
        stage_sources += (value_schema,)
    if any(case.get("study_task_sha256") for case in catalog.get("cases", ())):
        stage_sources += ("experiments/flexible_execution/custom_evaluation.py",)
    if any(case.get("archive_external_rgb") is False for case in catalog.get("cases", ())):
        stage_sources += ("experiments/flexible_execution/storage.py",)
    if any(case.get("shared_frontend") for case in catalog.get("cases", ())):
        stage_sources += ("experiments/flexible_execution/shared_frontend.py",)
    if any(case.get("layout_seed") is not None for case in catalog.get("cases", ())):
        stage_sources += ("experiments/task_attack_operator/scene.py", "experiments/task_attack_recessed/scene.py",
                          "experiments/task_attack_value/scene.py", "experiments/flexible_execution/development_inputs.py")
    if public:
        stage_sources += (*PUBLIC_SOURCES, "experiments/flexible_execution/live_planning.py", RANKING_SOURCE)
        if catalog.get("role") == PUBLIC_FINAL_ROLE:
            stage_sources += ("experiments/flexible_execution/live_calibration.py",)
    if custom_final:
        stage_sources += ("experiments/flexible_execution/custom_schedule.py",)
    if any(method in STREAM_METHODS for case in catalog["cases"] for method in case["order"]):
        stage_sources += ("experiments/flexible_execution/stream_baseline.py",)
    for label in RUN_SOURCES + stage_sources:
        source = checkout / label
        if source.is_dir():
            sources.extend(path for path in source.rglob("*") if path.is_file() and
                           not any(part.startswith(".") or part in ("tests", "__pycache__")
                                   for part in path.relative_to(source).parts))
        else:
            sources.append(source)
    legacy = Path(os.environ.get("VLA_LEGACY_ROOT", str(Path.home() / "uav" / "Embodied-intelligent-behavior")))
    paths = sorted(set(sources))
    system_paths = [Path("/etc/os-release"), Path("/opt/ros/noetic/setup.bash"),
                    legacy / "sim_ws/devel/setup.bash", legacy / "sim_ws/devel/lib/vla_sim/vla_traj_server",
                    legacy / "sim_ws/devel/lib/vla_sim/vla_cloud_bridge", legacy / ".venv/pyvenv.cfg"]
    if public:
        runtime = catalog["runtime"]
        packages = Path(runtime["packages"])
        system_paths = [Path("/etc/os-release"), Path(sys.executable), Path(runtime["archive"]),
                        Path(runtime["unity_executable"]), Path(runtime["xvfb_run"]),
                        Path(runtime["xvfb_run"]).with_name("Xvfb"), packages / "ai2thor/controller.py",
                        packages / "ai2thor/server.py"]
        if any(not path.is_file() for path in system_paths):
            raise ValueError("A registered public simulator input is missing")
        version, stage = ((9, "public-visual-final") if catalog["role"] == PUBLIC_FINAL_ROLE
                          else (7, "public-visual-development"))
    elif visual:
        version, stage = 6, catalog["role"].lower().replace("_", "-")
    elif stream:
        version, stage = 8, "segmented-generation-development"
    elif custom_final:
        version, stage = 10, "custom-visual-final"
    elif typed:
        version, stage = 5, "typed-vision-development"
    elif ranked:
        version, stage = 4, "ranked-development"
    elif conditional:
        version, stage = 3, "conditional-development"
    else:
        version, stage = 2, "matched-baselines"
    return {"version": version, "stage": stage,
            "catalog_sha256": _hash(catalog),
            "model": MODEL, "reasoning_effort": EFFORT, "request_limit": REQUEST_LIMIT,
            "max_infra_attempts": MAX_ATTEMPTS, "startup_cleanup_seconds": STARTUP_CLEANUP_SECONDS,
            "ros_ports": list(ports), "auth_project": str(auth_project), "legacy_project": str(legacy.resolve()),
            "source_sha256": {str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in paths},
            "environment_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in system_paths if path.is_file()}}


def _locked_source(existing, current, checkout, control, catalog, auth_project, ports):
    fields = set(current) - {"source_sha256"}
    if any(existing.get(key) != current[key] for key in fields):
        raise ValueError("Pilot data, settings, or environment changed. Use a new protocol version.")
    revision = existing["code_git_commit"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Invalid locked checkout revision")
    isolated = control / "revisions" / revision
    if not isolated.is_dir():
        isolated.parent.mkdir(exist_ok=True)
        subprocess.run(["git", "worktree", "add", "--detach", str(isolated), revision], cwd=checkout,
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    verified = _runtime_inputs(isolated, catalog, auth_project, ports)
    if verified != {key: value for key, value in existing.items() if key != "code_git_commit"}:
        raise ValueError("The archived checkout does not match the locked pilot. Keep the old result separate.")
    return isolated


def plans(directory, protocol, catalog):
    fingerprint = _hash({key: value for key, value in protocol.items() if key != "code_git_commit"})
    for case in catalog["cases"]:
        for method in case["order"]:
            binding = catalog.get("ranker_bindings", {}).get(method)
            execution_method = binding["method"] if binding else method
            if execution_method not in PAIRED_MODES and execution_method not in CONDITIONAL_MODES:
                raise ValueError("Method is not enabled for active development: " + method)
            identifier = f"v{protocol['version']}-{case['id']}-{method}-{fingerprint[:16]}"
            job = {"id": identifier, "stage": protocol["stage"], "input_sha256": fingerprint,
                   "case": case, "method": method, "model": protocol["model"],
                   "ranker_config_sha256": catalog.get("ranker_config_sha256") if execution_method in ONLINE_RANKED_METHODS else None,
                   "reasoning_effort": protocol["reasoning_effort"], "state": "PENDING", "attempt": 0,
                   "arm": str(directory / case["id"] / method)}
            if binding:
                job.update(execution_method=execution_method, ranker_file=binding["ranker_file"],
                           training_seed=binding["seed"], ranker_config_sha256=binding["sha256"],
                           ranker_check_cap=binding.get("check_cap"))
            if catalog.get("role") in (PUBLIC_VALIDATION_ROLE, PUBLIC_FINAL_ROLE):
                job["public_runtime"] = catalog["runtime"]
            if catalog.get("role") == PUBLIC_FINAL_ROLE:
                job.update(frozen_evaluation=catalog["frozen_evaluation"],
                           frozen_evaluation_sha256=catalog["frozen_evaluation_sha256"])
            yield job


def _attempt_path(job, attempt):
    return Path(job["arm"]) / f"attempt-{attempt:03d}"


def _save(control, job, **patch):
    job = {**job, **patch, "updated_at": _now()}
    write_json(Path(job["arm"]) / "job.json", job)
    write_json(control / "last_transition.json", job)
    with (control / "jobs.jsonl").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write((json.dumps(job, ensure_ascii=False, allow_nan=False) + "\n").encode("utf8"))
        handle.flush()
        os.fsync(handle.fileno())
    return job


def _evidence(root, job):
    pre = _read(root / "clean_preflight.json")
    if pre is None:
        return None, None
    _valid_preflight(pre, job["case"], job.get("execution_method", job["method"]), root)
    if (pre.get("model"), pre.get("reasoning_effort")) != (job["model"], job["reasoning_effort"]):
        raise ValueError("Result uses different model settings: " + str(root))
    expected_limit = job.get("request_limit")
    if expected_limit is not None and pre.get("request_budget", {}).get("limit") != expected_limit:
        raise ValueError("Result uses a different request budget: " + str(root))
    if pre.get("ranker_config_sha256") != job.get("ranker_config_sha256"):
        raise ValueError("Result uses a different ranking profile: " + str(root))
    result = _read(root / "result.json")
    expected_scene = job["case"].get("scene_fingerprint")
    if expected_scene and result and result.get("execution_completed"):
        from .development_inputs import scene_fingerprint

        observed = _read(root / "clean_verified_scene.json")
        if observed is None or scene_fingerprint(observed) != expected_scene:
            raise ValueError("The finished episode does not match its registered visual scene: " + str(root))
    return pre, result


def _logs(root):
    source = root / "private/robot/model_calls.jsonl"
    return [json.loads(line) for line in source.read_text(encoding="utf8").splitlines()] if source.is_file() else []


def _charge(root):
    clock = _read(root / "job_clock.json")
    if clock is None:
        return 0.
    elapsed = clock.get("elapsed_seconds", max(0., time.time() - clock["started_wall"]))
    return min(clock["allowance_seconds"], elapsed)


def _outcome(root, job, returncode, stop=None):
    if job.get("public_runtime"):
        from .public_audit import registered_outcome

        return registered_outcome(root, job, _charge(root), returncode, stop)
    charge = 0.
    try:
        pre, result = _evidence(root, job)
        calls = _logs(root)
        charge = _charge(root)
        request_count = sum("logical_job_attempt" in call for call in calls)
        if stop in {"TIMEOUT", "CANCELLED_BY_USER"}:
            return stop, {"reason": "supervisor", "charged_seconds": charge, "logged_dispatches": request_count}
        errors = [row["error"] for row in calls if row.get("error")]
        if result and result.get("error"):
            errors.append(result["error"])
        external = next((error for error in errors if error.get("code") in EXTERNAL_RESPONSE_CODES or
                         error.get("status") in (401, 403, 429)), None)
        if (pre and not pre["ready_to_attempt_model_call"]) or external:
            return "BLOCKED_EXTERNAL", {"reason": external or "account/model preflight",
                                        "charged_seconds": charge, "logged_dispatches": request_count}
        if result and result.get("execution_completed"):
            clock = _read(root / "job_clock.json") or {}
            report = _read(root / "clean_coverage.json")
            if report is None or report != audit_run(root):
                raise ValueError("Missing or inconsistent independent coverage report")
            metrics = result.get("robot_metrics", {})
            records = {"output": str(root), "score": result["score"]["outcome"],
                       "coverage_complete": report["complete"], "charged_seconds": charge,
                       "logged_dispatches": request_count}
            reason = metrics.get("end_reason")
            if reason in {"episode_deadline", "request_budget_exhausted"} and not report["complete"]:
                return "TIMEOUT", dict(records, reason=reason)
            if clock and clock.get("elapsed_seconds", 0.) >= clock["allowance_seconds"] and not report["complete"]:
                return "TIMEOUT", dict(records, reason="episode_deadline")
            valid = report["complete"] and result["score"]["outcome"] == "correct"
            return ("FINISHED_VALID" if valid else "FINISHED_TASK_FAILURE"), records
        if result and (result.get("error") or {}).get("type") == "KeyboardInterrupt":
            return "CANCELLED_BY_USER", {"reason": "interrupted", "charged_seconds": charge}
        deadline = job["case"]["episode_timeout_seconds"] - job.get("charged_seconds", 0.)
        if charge and charge >= deadline:
            return "TIMEOUT", {"reason": "episode deadline", "charged_seconds": charge}
        if errors and (errors[-1].get("type") == "ModelDecisionInvalid" or errors[-1].get("code") in {
            "invalid_json", "invalid_output", "unsupported_output", "inconsistent_output", "incomplete_output",
        }):
            return "FINISHED_TASK_FAILURE", {"reason": "model response rejected", "error": errors[-1],
                                              "charged_seconds": charge}
        return "INFRA_ERROR", {"reason": "runner exited without a verifiable episode", "returncode": returncode,
                               "error": (result or {}).get("error"), "charged_seconds": charge}
    except (OSError, KeyError, ValueError) as error:
        return "INFRA_ERROR", {"reason": "artifact verification: " + str(error),
                               "returncode": returncode, "charged_seconds": charge}


def _publish(directory, control, all_jobs, current=None, batch=None):
    rows = [(_read(Path(job["arm"]) / "job.json") or job) for job in all_jobs]
    counts = {state: sum(job["state"] == state for job in rows) for state in
              ("PENDING", "RUNNING", *sorted(TASK_TERMINAL), "INFRA_ERROR", "BLOCKED_EXTERNAL", "CANCELLED_BY_USER")}
    live = []
    for row in rows:
        if row["state"] == "RUNNING" and is_live(row.get("process")):
            root = _attempt_path(row, row["attempt"]) / "episode"
            budget = RequestBudget(row["request_ledger"], row["request_limit"]).summary()
            live.append({**row, "request_budget": budget, "attempt_charged_seconds": _charge(root)})
    next_job = next((job for job in rows if job["state"] not in TASK_TERMINAL), None)
    stage = (next_job or all_jobs[0])["stage"]
    state = {"checked_at": _now(), "stage": stage, "pilot_directory": str(directory),
             "job_counts": counts, "current_job": live[0] if live else current, "active_jobs": live,
             "last_batch": batch, "next_job": next_job["id"] if next_job else None,
             "next_command": f"python3 -m experiments.flexible_execution.cli pilot-run --pilot-dir "
                             f"{shlex.quote(str(directory))} --resume" if next_job else None}
    write_json(control / "state.json", state)
    return state


def _wait_running(job, refresh):
    while is_live(job.get("process")):
        remaining = job["hard_deadline_wall"] - time.time()
        if remaining <= 0:
            _stop(job["process"], owned_group=bool(job.get("public_runtime")))
            return "TIMEOUT"
        refresh(job)
        time.sleep(min(HEARTBEAT_SECONDS, remaining))
    return None


def _stop(identity, *, owned_group=False):
    for signum, grace in ((signal.SIGTERM, 12), (signal.SIGKILL, 3)):
        if not is_live(identity):
            break
        try:
            if owned_group and identity["pid"] == identity["pgid"]:
                os.killpg(identity["pgid"], signum)
            else:
                os.kill(identity["pid"], signum)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while is_live(identity) and time.monotonic() < deadline:
            time.sleep(.25)


def _wait_child(child, job, refresh):
    while child.poll() is None:
        remaining = job["hard_deadline_wall"] - time.time()
        if remaining <= 0:
            _stop(job["process"], owned_group=bool(job.get("public_runtime")))
            return child.wait(), "TIMEOUT"
        refresh(job)
        try:
            child.wait(timeout=min(HEARTBEAT_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            continue
    return child.returncode, None


def _record_worker_cpu(attempt, before):
    after = getrusage(RUSAGE_CHILDREN)
    write_json(attempt / "worker_cpu.json", {"scope": "reaped_local_worker_and_descendants",
        "user_seconds": after.ru_utime - before.ru_utime, "system_seconds": after.ru_stime - before.ru_stime})


def _finish(control, job, stop=None, returncode=None):
    root = _attempt_path(job, job["attempt"]) / "episode"
    status, detail = _outcome(root, job, returncode, stop)
    budget = RequestBudget(job["request_ledger"], job["request_limit"]).summary()
    charged = job.get("charged_seconds", 0.) + detail.get("charged_seconds", 0.)
    if status == "INFRA_ERROR" and (not budget["remaining"] or charged >= job["case"]["episode_timeout_seconds"]):
        status = "TIMEOUT"
        detail["reason"] = "cumulative job budget exhausted"
    return _save(control, job, state=status, result=detail, returncode=returncode, process=None,
                 charged_seconds=charged, request_budget=budget, finished_at=_now(), artifact_dir=str(root))


def _verify_terminal(job):
    detail = job.get("result") or {}
    if job["state"] == "TIMEOUT" and detail.get("reason") in {"supervisor", "cumulative job budget exhausted"}:
        return
    root = Path(job["artifact_dir"])
    earlier_charge = job.get("charged_seconds", 0.) - detail.get("charged_seconds", 0.)
    observed, _ = _outcome(root, {**job, "charged_seconds": max(0., earlier_charge)}, job.get("returncode"))
    if observed != job["state"]:
        raise ValueError("Saved result no longer verifies for " + job["id"] + ": " + str(root))


def verified_cohort(name, directory, role, count):
    from .pilot_schedule import status

    if not (directory / "pilot_manifest.json").is_file():
        raise ValueError(f"{name} has 0/{count} recorded cases and no registered catalog")
    checkpoint = status(directory)
    if checkpoint["role"] != role or checkpoint["planned_pairs"] != count or checkpoint["recorded_pairs"] != count:
        raise ValueError(f"{name} has {checkpoint['recorded_pairs']}/{count} recorded cases")
    catalog = _read(directory / "pilot_manifest.json")
    protocol = _read(directory / "run_protocol.json")
    if (not protocol or protocol["catalog_sha256"] != _hash(catalog)
            or (protocol["model"], protocol["reasoning_effort"]) != (MODEL, EFFORT)):
        raise ValueError(f"{name} used an inconsistent model or source assignment")
    outcomes = []
    for case in catalog["cases"]:
        for method in case["order"]:
            job = _read(directory / case["id"] / method / "job.json")
            if not job or job["state"] not in TASK_TERMINAL:
                raise ValueError(f"{name} contains an unresolved job")
            _verify_terminal(job)
            outcomes.append({"id": job["id"], "state": job["state"], "result": job["result"]})
    evidence = {"catalog_sha256": _hash(catalog), "protocol_sha256": _hash(protocol),
                "recorded_jobs": len(outcomes), "outcomes_sha256": _hash(outcomes)}
    return catalog, evidence


def _public_command(job, checkout, control, run, episode, auth_project):
    runtime = job["public_runtime"]
    xvfb = Path(runtime["xvfb_run"])
    display_environment = control / "public-display"
    paths = {f"XDG_{key.upper()}_HOME": display_environment / key for key in ("cache", "config")}
    paths["XDG_RUNTIME_DIR"] = display_environment / "runtime"
    for path in paths.values():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    python_path = os.pathsep.join((runtime["packages"], str(checkout / "bridge"), str(checkout)))
    return ["env", f"PATH={xvfb.parent}{os.pathsep}{os.environ.get('PATH', os.defpath)}", f"PYTHONPATH={python_path}",
            "LIBGL_ALWAYS_SOFTWARE=1", "GALLIUM_DRIVER=llvmpipe", "LP_NUM_THREADS=2",
            *(f"{key}={path}" for key, path in paths.items()), str(xvfb), "-a", "-n", "95", "-s",
            "-screen 0 640x480x24 -nolisten tcp -noreset", sys.executable, "-m", "experiments.flexible_execution.public_task",
            "--registered-job", str(run / "launch.json"), "--output", str(episode), "--auth-project", str(auth_project)]


def _gazebo_command(job, directory, catalog, auth_project, episode, remaining, ledger, ports):
    case = job["case"]
    method = job.get("execution_method", job["method"])
    command = ["bash", "scripts/run_flexible_clean.sh", "--chatgpt-project", str(auth_project),
               "--output", str(episode), "--method", method, "--family", case["family"],
               "--seed", str(case["seed"]), "--equipment-count", str(case["equipment_count"]),
               "--backend", case["backend"], "--source-profile", case["source_profile"],
               "--episode-timeout", str(remaining), "--planned-episode-timeout", str(case["episode_timeout_seconds"]),
               "--max-request-attempts", str(REQUEST_LIMIT), "--request-ledger", str(ledger), "--ros-port", str(ports[0]),
               "--gazebo-port", str(ports[1])]
    if case.get("archive_external_rgb") is False:
        command.append("--onboard-evidence-only")
    if case.get("layout_seed") is not None:
        command.extend(("--layout-seed", str(case["layout_seed"])))
    if case.get("scene_fingerprint"):
        command.extend(("--scene-fingerprint", case["scene_fingerprint"]))
    if case.get("shared_frontend"):
        command.extend(("--frontend-reference", str(directory / case["id"] / "shared_frontend.json")))
    if case.get("study_task_sha256"):
        study = Path(case["study_task_file"])
        if _hash(_read(study)) != case["study_task_sha256"]:
            raise ValueError("The study instruction does not match its registered input")
        command.extend(("--study-task", str(study)))
    if method in CONDITIONAL_MODES:
        command.extend(("--cost-config", str(directory / "calibrated_costs.json")))
        if case.get("early_fields"):
            command.extend(("--early-fields", ",".join(case["early_fields"])))
        if case.get("collection_rule"):
            command.extend(("--collection-rule", case["collection_rule"]))
        if case.get("handoff_at_observation"):
            command.append("--handoff-at-observation")
        if case.get("typed_task_program"):
            command.append("--compile-task-program")
    if method in ONLINE_RANKED_METHODS:
        ranker_path = Path(job.get("ranker_file", directory / "calibrated_rankers.json"))
        if _hash(_read(ranker_path)) != job["ranker_config_sha256"]:
            raise ValueError("The ranking checkpoint assignment changed")
        command.extend(("--ranker-config", str(ranker_path)))
    return command


def pilot_run(directory, control, checkout, *, resume, max_jobs, auth_project=DEFAULT_AUTH_PROJECT,
              unlock_external=False, minimum_free_disk_gb=0):
    directory, control, auth_project = directory.resolve(), control.resolve(), auth_project.resolve()
    if max_jobs < 1:
        raise ValueError("Choose a positive maximum batch size")
    control.mkdir(parents=True, exist_ok=True)
    with (control / ".simulator.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {**control_status(control), "worker_owned_by_another_batch": True}
        running = control_status(control)
        if running["active_jobs"] and running.get("pilot_directory") != str(directory):
            return {**running, "worker_owned_by_another_batch": True}
        return _run_locked(directory, control, checkout, resume, max_jobs, auth_project, unlock_external,
                           minimum_free_disk_gb)


def _run_locked(directory, control, checkout, resume, max_jobs, auth_project, unlock_external, minimum_free_disk_gb):
    warmup = simulator_running(control)
    if warmup:
        return {"execution_status": "RUNNING", "active_jobs": [warmup]}
    catalog = _read(directory / "pilot_manifest.json")
    if not catalog:
        raise ValueError("The existing pilot schedule is required")
    if catalog.get("cost_config_sha256") and _hash(_read(directory / "calibrated_costs.json")) != catalog["cost_config_sha256"]:
        raise ValueError("Conditional cost calibration does not match its frozen catalog")
    if catalog.get("ranker_config_sha256"):
        registered_config(directory, catalog)
    if catalog.get("role") == PUBLIC_FINAL_ROLE:
        freeze = _read(Path(catalog["frozen_evaluation"]))
        if (not freeze or _hash(freeze) != catalog["frozen_evaluation_sha256"]
                or freeze["runtime"] != catalog["runtime"] or freeze["role"] != PUBLIC_FINAL_ROLE):
            raise ValueError("The held-out public evaluation does not match its frozen inputs")
    ports = (11674, 11675)
    protocol = _runtime_inputs(checkout, catalog, auth_project, ports)
    freeze = None
    if catalog.get("role") == CUSTOM_FINAL_ROLE:
        freeze = _read(Path(catalog["frozen_evaluation"]))
        if (not freeze or _hash(freeze) != catalog["frozen_evaluation_sha256"]
                or any(freeze[key] != protocol[key] for key in ("model", "reasoning_effort", "request_limit", "environment_sha256"))
                or freeze["ranker_bindings"] != catalog["ranker_bindings"]
                or freeze["cost_config_sha256"] != catalog["cost_config_sha256"]
                or any(case.get("archive_external_rgb", True) != freeze.get("archive_external_rgb", True)
                       or case.get("shared_frontend", False) != freeze.get("shared_frontend", False)
                       for case in catalog["cases"])):
            raise ValueError("The custom test inputs or execution code differ from the frozen evaluation")
    lock = directory / "run_protocol.json"
    existing = _read(lock)
    if existing:
        if not resume:
            raise ValueError("Use --resume for the locked pilot")
        frozen = existing
    else:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        frozen = {**protocol, "code_git_commit": revision}
        if freeze and freeze["source_sha256"] != frozen["source_sha256"]:
            raise ValueError("The first custom test launch must use the frozen source")
    if frozen.get("source_sha256"):
        # Run the worker from the verified snapshot even when the development
        # tree is currently identical. Later edits cannot race worker startup.
        checkout = _locked_source(frozen, protocol, checkout, control, catalog, auth_project, ports)
    elif {key: value for key, value in frozen.items() if key != "code_git_commit"} != protocol:
        raise ValueError("The existing pilot protocol does not match the current inputs")
    protocol = frozen
    if not existing:
        write_json(lock, protocol)
    if freeze and freeze["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("The custom test launch differs from the frozen source")
    all_jobs = list(plans(directory, protocol, catalog))
    if not all_jobs:
        raise ValueError("Cannot launch an empty pilot schedule")
    launched = 0
    opened = {}
    batch_started = time.monotonic()
    batch = {"opened_at": _now(), "max_launches": max_jobs, "max_new_dispatches": max_jobs * REQUEST_LIMIT,
             "max_seconds_per_launch": max(case["episode_timeout_seconds"] for case in catalog["cases"])
             + STARTUP_CLEANUP_SECONDS, "estimate": _batch_estimate(control, directory, protocol, max_jobs)}

    def refresh(current=None):
        used = sum(RequestBudget(path, REQUEST_LIMIT).summary()["reserved"] - at_start
                   for path, at_start in opened.items())
        return _publish(directory, control, all_jobs, current,
                        {**batch, "launched": launched, "actual_new_dispatches": used,
                         "actual_wall_seconds": time.monotonic() - batch_started})

    refresh()
    for template in all_jobs:
        arm = Path(template["arm"])
        job = _read(arm / "job.json") or template
        if job["id"] != template["id"]:
            raise ValueError("Existing arm belongs to a different protocol: " + str(arm))
        if job["state"] == "PENDING" and (arm / "result.json").is_file():
            legacy = {**job, "result_origin": "pre-supervisor"}
            status, detail = _outcome(arm, legacy, 0)
            if status in TASK_TERMINAL:
                _save(control, job, state=status, result=detail, artifact_dir=str(arm), finished_at=_now())
                continue
            raise ValueError("Existing direct-arm result could not be verified: " + str(arm))
        if job["state"] == "RUNNING":
            episode = _attempt_path(job, job["attempt"]) / "episode"
            if not is_live(job.get("process")):
                job = _save(control, job, process=_find_worker(episode))
            job = _finish(control, job, _wait_running(job, refresh))
        if job["state"] in TASK_TERMINAL:
            _verify_terminal(job)
            continue
        if job["state"] == "BLOCKED_EXTERNAL" and not unlock_external:
            return refresh(job)
        if job["state"] == "CANCELLED_BY_USER" and not resume:
            return refresh(job)
        if job["attempt"] >= MAX_ATTEMPTS:
            return refresh(job)
        if launched >= max_jobs:
            return refresh()
        free, reserve = disk_usage(directory).free, minimum_free_disk_gb * 1024**3
        if free < reserve:
            state = {**refresh(job), "execution_status": "BLOCKED_EXTERNAL", "resource_gate": {
                "condition": "minimum_free_disk", "path": str(directory), "available_bytes": free,
                "required_free_bytes": reserve}}
            write_json(control / "state.json", state)
            return state
        # Later arms in a long batch must still use the version frozen before
        # the first arm, even when the development checkout is being edited.
        current = _runtime_inputs(checkout, catalog, auth_project, ports)
        if current != {key: value for key, value in protocol.items() if key != "code_git_commit"}:
            checkout = _locked_source(protocol, current, checkout, control, catalog, auth_project, ports)
        next_case = job["case"]
        remaining = next_case["episode_timeout_seconds"] - job.get("charged_seconds", 0.)
        ledger = control / "requests" / (job["id"] + ".jsonl")
        budget = RequestBudget(ledger, REQUEST_LIMIT).summary()
        if remaining <= 0 or not budget["remaining"]:
            _save(control, job, state="TIMEOUT", request_budget=budget, finished_at=_now(),
                  result={"reason": "cumulative job budget exhausted"})
            continue
        attempt = job["attempt"] + 1
        while _attempt_path(job, attempt).exists():
            attempt += 1
        run = _attempt_path(job, attempt)
        run.mkdir(parents=True)
        episode = run / "episode"
        if job.get("public_runtime"):
            command = _public_command(job, checkout, control, run, episode, auth_project)
        else:
            command = _gazebo_command(job, directory, catalog, auth_project, episode, remaining, ledger, ports)
        config = {"state": "RUNNING", "process": None, "attempt": attempt, "command": command,
                  "workdir": str(checkout), "started_at": _now(), "finished_at": None, "result": None,
                  "request_ledger": str(ledger), "request_limit": REQUEST_LIMIT, "request_budget": budget,
                  "launcher_log": str(run / "launcher.log"), "code_git_commit": protocol["code_git_commit"],
                  "hard_deadline_wall": time.time() + remaining + STARTUP_CLEANUP_SECONDS}
        write_json(run / "launch.json", {**job, **config})
        job = _save(control, job, **config)
        opened.setdefault(ledger, budget["reserved"])
        before_cpu = getrusage(RUSAGE_CHILDREN)
        with (run / "launcher.log").open("xb") as log:
            try:
                child = subprocess.Popen(command, cwd=checkout, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            except OSError:
                return refresh(_finish(control, job, returncode=127))
            job = _save(control, job, process=_proc(child.pid))
            launched += 1
            try:
                returncode, stop = _wait_child(child, job, refresh)
            except KeyboardInterrupt:
                _stop(job["process"], owned_group=bool(job.get("public_runtime")))
                returncode = child.wait()
                _record_worker_cpu(run, before_cpu)
                refresh(_finish(control, job, "CANCELLED_BY_USER", returncode))
                raise
        _record_worker_cpu(run, before_cpu)
        job = _finish(control, job, stop, returncode)
        if job["state"] not in TASK_TERMINAL:
            return refresh(job)
    return refresh()


def control_status(control):
    state = _read(control / "state.json") or {}
    active = [job for job in state.get("active_jobs", []) if is_live(job.get("process"))]
    last = _read(control / "last_transition.json") or {}
    if not active and last.get("state") == "RUNNING":
        episode = _attempt_path(last, last["attempt"]) / "episode"
        process = last.get("process") if is_live(last.get("process")) else _find_worker(episode)
        if process:
            active = [{**last, "process": process,
                       "request_budget": RequestBudget(last["request_ledger"], last["request_limit"]).summary()}]
        state = {**state, "current_job": last, "pilot_directory": str(Path(last["arm"]).parents[1])}
    warm = simulator_running(control)
    if warm:
        active.append(warm)
    current = active[0] if active else state.get("current_job")
    current_state = (current or {}).get("state")
    reserve = state.get("resource_gate")
    if active:
        execution = "RUNNING"
    elif reserve and disk_usage(reserve["path"]).free < reserve["required_free_bytes"]:
        execution = "BLOCKED_EXTERNAL"
    elif current_state == "RUNNING":
        execution = "EXITED_NEEDS_RECOVERY"
    elif current_state in {"BLOCKED_EXTERNAL", "INFRA_ERROR", "CANCELLED_BY_USER"}:
        execution = current_state
    else:
        execution = "READY" if state.get("next_job") or current_state == "PENDING" else "IDLE"
    return {**state, "current_job": current, "active_jobs": active, "execution_status": execution,
            "scientific_conclusion": "INCONCLUSIVE"}
