import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.flexible_execution import cli, control
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.task_attack.runtime import write_json


class SupervisorTest(unittest.TestCase):
    def test_project_spends_remaining_launches_on_the_next_stage(self):
        stages = [({"stage": "first", "last_batch": {"launched": 1}, "active_jobs": []}, 1),
                  ({"stage": "next", "last_batch": {"launched": 2}, "active_jobs": []}, 2)]
        with patch.object(cli, "_continue_stage", side_effect=stages) as run_stage:
            result = cli._continue(Path("out"), Path("run"), {}, SimpleNamespace(max_jobs=3))
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
