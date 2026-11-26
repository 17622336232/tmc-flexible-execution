import gzip
import json
from os.path import relpath
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.flexible_execution import acceleration, cli, control
from experiments.flexible_execution.pilot_report import _attempt_costs
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.storage import archive_sensor_logs
from experiments.flexible_execution.timeline import Timeline
from experiments.task_attack.runtime import write_json


class SupervisorTest(unittest.TestCase):
    def test_diagnostics_archive_is_lossless_and_can_resume_with_another_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagnostic = root / "private/sensor_timing.jsonl"
            diagnostic.parent.mkdir()
            raw = b'{"timing":0.05}\n'
            diagnostic.write_bytes(raw)
            evidence = root / "private/model_calls.jsonl"
            evidence.write_bytes(b"model evidence")
            archive_sensor_logs(root)
            self.assertFalse(diagnostic.exists())
            self.assertEqual(gzip.decompress(diagnostic.with_suffix(".jsonl.gz").read_bytes()), raw)
            self.assertEqual(evidence.read_bytes(), b"model evidence")

            roslog = root / "private/roslog/worker.log"
            roslog.parent.mkdir()
            roslog.write_bytes(b"native simulation log")
            archive_sensor_logs(root)
            manifest = json.loads((root / "storage_archive.json").read_text())
            self.assertEqual(len(manifest["diagnostic_logs"]), 2)
            self.assertEqual(gzip.decompress(roslog.with_suffix(".log.gz").read_bytes()), b"native simulation log")

    def test_truncated_timing_log_does_not_prevent_estimating_the_next_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            control_dir = Path(temp)
            pilot = control_dir / "pilot"
            protocol = {"stage": "matched-baselines", "model": control.MODEL, "reasoning_effort": control.EFFORT}
            record = {**protocol, "state": "FINISHED_VALID", "arm": str(pilot / "case" / "method"),
                      "started_at": "2026-11-24T21:10:00+00:00", "finished_at": "2026-11-24T21:11:40+00:00"}
            (control_dir / "jobs.jsonl").write_text(json.dumps(record) + '\n{"started_at":', encoding="utf8")
            estimate = control._batch_estimate(control_dir, pilot, protocol, 2)
            self.assertEqual(estimate["prior_scope"], "same pilot")
            self.assertEqual(estimate["unreadable_timing_records"], 1)
            self.assertEqual(estimate["estimated_wall_seconds"], 200)
            new = control._save(control_dir, {**record, "state": "PENDING"})
            lines = (control_dir / "jobs.jsonl").read_text(encoding="utf8").splitlines()
            self.assertEqual(lines[-2], '{"started_at":')
            self.assertEqual(json.loads(lines[-1]), new)

    def test_project_spends_remaining_launches_on_the_next_stage(self):
        stages = [({"stage": "first", "last_batch": {"launched": 1}, "active_jobs": []}, 1),
                  ({"stage": "next", "last_batch": {"launched": 2}, "active_jobs": []}, 2)]
        for protocol, module, stage in (("legacy", cli, "_continue_stage"),
                                        ("protocol-v2", acceleration, "advance_stage")):
            with self.subTest(protocol=protocol), patch.object(module, stage, side_effect=stages) as run_stage:
                args = SimpleNamespace(max_jobs=3, protocol=protocol)
                result = cli._continue(Path("out"), Path("run"), {}, args)
                self.assertEqual([call.args[-1] for call in run_stage.call_args_list], [3, 2])
                self.assertEqual([row["stage"] for row in result["project_batches"]], ["first", "next"])
                self.assertEqual(result["remaining_launch_allowance"], 0)

    def test_resume_requests_the_existing_owned_schedule(self):
        state = {"execution_status": "RUNNING", "pilot_directory": "existing", "active_jobs": [{"arm": "owned"}]}
        args = SimpleNamespace(resume=True, max_jobs=2, retry_external=False, chatgpt_project=Path("account"))
        with patch.object(cli, "control_status", return_value=state), patch.object(cli, "pilot_run",
                return_value={**state, "worker_owned_by_another_batch": True}) as run:
            result = cli._continue(Path("out"), Path("run"), {}, args)
        self.assertEqual(run.call_args.args[0], Path("existing"))
        self.assertEqual(result["project_batches"], [])
        self.assertEqual(result["remaining_launch_allowance"], 2)

    def test_project_recovers_the_recorded_interrupted_schedule_first(self):
        for status in ("EXITED_NEEDS_RECOVERY", "INFRA_ERROR"):
            with self.subTest(status=status):
                state = {"execution_status": status, "pilot_directory": "interrupted", "active_jobs": [],
                         "current_job": {"arm": "owned"}}
                args = SimpleNamespace(resume=True, max_jobs=2, retry_external=False, chatgpt_project=Path("account"))
                with patch.object(cli, "control_status", return_value=state), patch.object(
                    cli, "pilot_run", return_value={"stage": "matched", "last_batch": {"launched": 2}}
                ) as run:
                    cli._continue(Path("out"), Path("run"), {}, args)
                self.assertEqual(run.call_args.args[0], Path("interrupted"))

    def test_project_reuses_remaining_batch_after_temporary_overload(self):
        overloaded = {"state": "INFRA_ERROR", "attempt": 1,
                      "result": {"error": {"code": "server_is_overloaded", "status": 200}}}
        stages = [({"stage": "matched", "last_batch": {"launched": 1}, "current_job": overloaded}, 1),
                  ({"stage": "matched", "last_batch": {"launched": 2}}, 2)]
        with patch.object(cli, "_continue_stage", side_effect=stages) as stage, patch.object(cli.time, "sleep") as wait:
            result = cli._continue(Path("out"), Path("run"), {}, SimpleNamespace(max_jobs=3, resume=True))
        self.assertEqual([call.args[-1] for call in stage.call_args_list], [3, 2])
        wait.assert_called_once_with(15)
        self.assertEqual(result["project_batches"][0]["service_retry_backoff_seconds"], 15)
        self.assertEqual(result["remaining_launch_allowance"], 0)

    def test_project_stops_when_temp_retries_or_external_access_are_blocked(self):
        for status, attempt, code in (("INFRA_ERROR", control.MAX_ATTEMPTS, "server_is_overloaded"),
                                      ("INFRA_ERROR", 1, "unclassified"),
                                      ("BLOCKED_EXTERNAL", 1, "rate_limit_exceeded")):
            with self.subTest(status=status, attempt=attempt, code=code):
                job = {"state": status, "attempt": attempt, "result": {"error": {"code": code}}}
                failed = {"stage": "matched", "last_batch": {"launched": 1}, "current_job": job}
                with patch.object(cli, "_continue_stage", return_value=(failed, 1)) as stage, patch.object(
                    cli.time, "sleep"
                ) as wait:
                    result = cli._continue(Path("out"), Path("run"), {}, SimpleNamespace(max_jobs=3, resume=True))
                stage.assert_called_once()
                wait.assert_not_called()
                self.assertEqual(result["remaining_launch_allowance"], 2)

    def test_new_attempt_after_exit_retains_old_result_and_cumulative_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pilot = root / "pilot"
            journal = root / "control"
            script = root / "scripts" / "run_flexible_clean.sh"
            script.parent.mkdir()
            script.write_text("#!/usr/bin/env bash\nexit 1\n")
            case = {"id": "pair-001", "family": "cracks", "seed": 1, "equipment_count": 2,
                    "source_profile": "single", "backend": "cpu", "episode_timeout_seconds": 480,
                    "order": ["paired-serial"]}
            write_json(pilot / "pilot_manifest.json", {"cases": [case]})
            proto = {"version": 2, "stage": "matched-baselines", "model": control.MODEL,
                     "reasoning_effort": control.EFFORT}

            def episode_result(episode, job, *_args):
                budget = RequestBudget(job["request_ledger"], job["request_limit"])
                if not episode.exists():
                    budget.reserve(model=job["model"], effort=job["reasoning_effort"], image_sha256=[])
                    episode.mkdir()
                    (episode / "preserved.txt").write_text(str(job["attempt"]))
                if job["attempt"] == 1:
                    return "INFRA_ERROR", {"charged_seconds": 53., "reason": "fixture runner exited"}
                return "FINISHED_TASK_FAILURE", {"charged_seconds": 29., "reason": "fixture task failed"}

            with patch.object(control, "_runtime_inputs", side_effect=lambda *_args: dict(proto)), patch.object(
                control.subprocess, "check_output", return_value="c" * 40
            ), patch.object(control, "_outcome", side_effect=episode_result):
                start = control.pilot_run(pilot, journal, root, resume=True, max_jobs=1)
                resume = control.pilot_run(pilot, journal, root, resume=True, max_jobs=1)
                repeat = control.pilot_run(pilot, journal, root, resume=True, max_jobs=1)

            arm = pilot / case["id"] / "paired-serial"
            self.assertEqual((arm / "attempt-001/episode/preserved.txt").read_text(), "1")
            self.assertEqual((arm / "attempt-002/episode/preserved.txt").read_text(), "2")
            self.assertFalse((arm / "attempt-003").exists())
            job = json.loads((arm / "job.json").read_text())
            self.assertEqual(job["charged_seconds"], 82.)
            self.assertEqual(job["request_budget"]["reserved"], 2)
            self.assertEqual(job["command"][job["command"].index("--episode-timeout") + 1], "427.0")
            self.assertEqual(start["job_counts"]["INFRA_ERROR"], 1)
            self.assertEqual(resume["job_counts"]["FINISHED_TASK_FAILURE"], 1)
            self.assertEqual(repeat["last_batch"]["launched"], 0)

    def test_dead_or_reused_process_does_not_look_running(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            dead = {"pid": 2000000000, "pgid": 2000000000, "start_ticks": "1", "boot_id": "previous-boot"}
            job = {"state": "RUNNING", "process": dead}
            write_json(directory / "state.json", {"current_job": job, "active_jobs": [job]})
            status = control.control_status(directory)
            self.assertEqual(status["active_jobs"], [])
            self.assertEqual(status["execution_status"], "EXITED_NEEDS_RECOVERY")

    def test_unpublished_live_worker_blocks_a_different_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            control_dir = root / "control"
            original = root / "original-pilot"
            arm = original / "example" / "conditional-a"
            worker = {"pid": 123}
            write_json(control_dir / "last_transition.json", {"state": "RUNNING", "arm": str(arm),
                "attempt": 1, "process": None, "request_limit": 24, "request_ledger": str(control_dir / "requests.jsonl")})
            with patch.object(control, "_find_worker", return_value=worker), patch.object(control, "_run_locked") as launch:
                result = control.pilot_run(root / "different-pilot", control_dir, root, resume=True, max_jobs=1)
            launch.assert_not_called()
            self.assertTrue(result["worker_owned_by_another_batch"])
            self.assertEqual(result["pilot_directory"], str(original))
            self.assertEqual(result["active_jobs"][0]["process"], worker)

    def test_disk_reserve_leaves_the_case_pending_without_starting_a_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pilot = root / "pilot"
            control_dir = root / "control"
            write_json(pilot / "pilot_manifest.json", {"cases": [{"id": "case", "order": ["paired-serial"],
                                                                  "episode_timeout_seconds": 480}]})
            runtime = {"version": 2, "stage": "matched-baselines", "model": control.MODEL,
                       "reasoning_effort": control.EFFORT}
            with patch.object(control, "_runtime_inputs", side_effect=lambda *_args: dict(runtime)), patch.object(
                control.subprocess, "check_output", return_value="b" * 40
            ), patch.object(control, "disk_usage", return_value=SimpleNamespace(free=10)), patch.object(
                control.subprocess, "Popen"
            ) as launch:
                result = control.pilot_run(pilot, control_dir, root, resume=True, max_jobs=1, minimum_free_disk_gb=30)
                self.assertEqual(control.control_status(control_dir)["execution_status"], "BLOCKED_EXTERNAL")
            launch.assert_not_called()
            self.assertEqual(result["job_counts"]["PENDING"], 1)
            self.assertEqual(result["resource_gate"]["available_bytes"], 10)

    def test_cost_rollup_counts_the_final_attempt_once_for_relative_pilot_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = Path(temp).resolve()
            for attempt, wall in enumerate((5., 7.), 1):
                episode = arm / f"attempt-{attempt:03d}" / "episode"
                logs = episode / "private/robot"
                logs.mkdir(parents=True)
                call = {"logical_job_attempt": attempt, "wall_seconds": wall, "started_at": attempt}
                (logs / "model_calls.jsonl").write_text(json.dumps(call) + "\n")
                write_json(episode / "job_clock.json", {"started_wall": 20. * attempt, "elapsed_seconds": wall})
                write_json(episode.parent / "worker_cpu.json", {"scope": "reaped_local_worker_and_descendants",
                                                                "user_seconds": wall, "system_seconds": 0.5})
            costs = _attempt_costs(Path(relpath(arm)), episode, False)
            self.assertEqual(costs["total_model_transport_wall_seconds"], 12.)
            self.assertEqual(costs["transport_calls_with_duration"], 2)
            self.assertEqual(costs["attempts_with_worker_cpu"], 2)
            self.assertEqual(costs["total_local_worker_cpu_seconds"], 13.)
            self.assertEqual(costs["task_elapsed_including_recovery_seconds"], 27.)
            self.assertEqual(costs["task_recovery_gap_seconds"], 15.)

    def test_cost_rollup_preserves_complete_rows_from_interrupted_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = Path(temp)
            first = arm / "attempt-001/episode/private/robot"
            trace = Timeline(first / "timeline.jsonl")
            trace.event("action_started", "capture")
            with trace.path.open("ab") as log:
                log.write(b'{"event":"\xe4')
            calls = first / "model_calls.jsonl"
            call = {"logical_job_attempt": 1, "wall_seconds": 5., "started_at": 1}
            calls.write_bytes(json.dumps(call).encode() + b'\n{"input"')
            complete = arm / "attempt-002/episode"
            complete.mkdir(parents=True)
            costs = _attempt_costs(arm, complete, False)
            self.assertEqual(costs["total_model_transport_wall_seconds"], 5.)
            self.assertIsNone(costs["total_local_worker_cpu_seconds"])
            self.assertNotIn("task_elapsed_including_recovery_seconds", costs)
            self.assertEqual(costs["interrupted_raw_log_tails"], 2)
            self.assertEqual(costs["unclosed_timing_intervals"], 1)
            with self.assertRaises(json.JSONDecodeError):
                _attempt_costs(arm, first.parent.parent, False)
            calls.write_bytes(calls.read_bytes() + b"\n")
            with self.assertRaises(json.JSONDecodeError):
                _attempt_costs(arm, complete, False)
