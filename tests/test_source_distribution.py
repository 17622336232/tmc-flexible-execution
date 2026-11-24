"""Portability checks: procedural scenes and tools need no private old data."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

from experiments.task_attack_integrity.scene import build_scene

ROOT = Path(__file__).resolve().parents[1]


class SourceDistributionTests(unittest.TestCase):
    def test_scenes_are_generated_without_prior_image_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            for family in ('cracks', 'inventory', 'parking'):
                result = build_scene(Path(temporary)/family, 10001, family, count=4, source_profile='single')
                world = ET.parse(result['world']).getroot().find('world')
                self.assertIsNotNone(world)
                self.assertTrue(result['catalog']['editable_actions'])
                for uri in world.findall('.//mesh/uri'):
                    if uri.text.startswith('file://'):
                        self.assertTrue(Path(uri.text[7:]).is_file(), uri.text)
                self.assertTrue(list((Path(temporary)/family/'assets/materials/textures').glob('*.png')))

    def test_launch_help_works_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = ['bash', str(ROOT/'scripts/run_attack_showcase.sh'), '--help']
            process = subprocess.run(command, cwd=temporary, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('--view-only', process.stdout)
            self.assertIn('--sensor-smoke', process.stdout)

    def test_training_help_does_not_open_old_data(self):
        command = ['bash', str(ROOT/'scripts/research.sh'), 'fit_isolated_waiting', '--help']
        process = subprocess.run(command, cwd='/tmp', capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--dataset', process.stdout)

    def test_model_import_maps_old_code_paths_within_checkout(self):
        from scripts.import_attack_models import local_source
        self.assertEqual(local_source('/old/Diff-VLA/experiments/task_attack/scene.py'),
                         ROOT/'experiments/task_attack/scene.py')
        self.assertEqual(local_source('artifacts/task_attack/operator_diagnosis_20261115/conditional_critic_policy.py'),
                         ROOT/'research/attack_learning/conditional_critic_policy.py')
        with self.assertRaises(ValueError):
            local_source('../../outside.py')
        with self.assertRaises(ValueError):
            local_source('artifacts/task_attack/operator_diagnosis_20261115/../../../outside.py')


if __name__ == '__main__':
    unittest.main()
