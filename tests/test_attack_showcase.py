"""Verify that demonstration commands select the reported implementation."""
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from experiments.task_attack import FAMILIES
from experiments.task_attack_showcase.run import CANDIDATE, parser, task_command, validate_scope
from experiments.task_attack_showcase.display import cleanup_gui, owned_gui_processes


class ShowcaseWiringTests(unittest.TestCase):
    def test_reported_method_uses_exact_frozen_waiting_model(self):
        args = parser().parse_args(['--family', 'cracks'])
        command = task_command(args, Path('/tmp/flight'), Path('/tmp/actual_static.json'))
        self.assertEqual(command[command.index('-m')+1], 'conditional_critic_native_runner')
        self.assertEqual(command[command.index('--timing-model')+1], str(CANDIDATE/'bundle.json'))
        self.assertEqual(command[command.index('--public-perception')+1], '/tmp/actual_static.json')
        self.assertEqual(command[command.index('--backend')+1], 'gazebo')
        with self.assertRaises(ValueError):
            task_command(args, Path('/tmp/flight'))

    def test_generic_actor_supports_every_declared_task_without_attack_model(self):
        for family in FAMILIES:
            args = parser().parse_args(['--family', family, '--policy', 'clean'])
            validate_scope(args)
            command = task_command(args, Path('/tmp/flight'))
            self.assertEqual(command[command.index('-m')+1], 'experiments.task_attack_integrity.run')
            self.assertNotIn('--timing-model', command)
            self.assertNotIn('--public-perception', command)

    def test_unvalidated_attack_scope_is_not_silently_claimed(self):
        with self.assertRaises(ValueError):
            validate_scope(parser().parse_args(['--family', 'inventory']))
        validate_scope(parser().parse_args(['--family', 'inventory', '--view-only']))

    def test_window_cleanup_is_scoped_and_rejects_pid_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'trial'
            neighbor = Path(temporary)/'trial_other'
            owned = subprocess.Popen(['gzclient', '-c', 'import time; time.sleep(30)', str(directory/'gui.log')],
                                     executable=sys.executable)
            other = subprocess.Popen(['gzclient', '-c', 'import time; time.sleep(30)', str(neighbor/'gui.log')],
                                     executable=sys.executable)
            try:
                time.sleep(.05)
                tracked = owned_gui_processes(directory)
                self.assertIn(owned.pid, tracked)
                self.assertNotIn(other.pid, tracked)
                cleanup_gui(directory, {owned.pid: 'different-process-start-time'})
                self.assertIsNone(owned.poll())
                cleanup_gui(directory, tracked)
                owned.wait(timeout=2)
                self.assertIsNone(other.poll())
            finally:
                for process in (owned, other):
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=2)


if __name__ == '__main__':
    unittest.main()
