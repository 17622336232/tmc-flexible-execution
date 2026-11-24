import json
from pathlib import Path
import tempfile
import unittest

from experiments.check_task_attack_scene_truth import AUDITED_IMPLEMENTATIONS, recheck
from experiments.task_attack.evaluate import evaluate


class SceneTruthRecheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, family='blocked_exit', missing=False):
        manifest = dict(implementation_hash=next(iter(AUDITED_IMPLEMENTATIONS)), rows=[])
        for policy in ('clean', 'blind', 'inferred', 'known_task'):
            row = dict(family=family, policy=policy, directory=policy,
                       block=family+'_01', scene_block='campus_01', source_profile='single')
            manifest['rows'].append(row)
            directory = self.root / policy
            (directory / 'private').mkdir(parents=True)
            (directory / 'public').mkdir()
            spec = (dict(required=['blocked'], predicate=dict(blocked=True)) if family=='blocked_exit'
                    else dict(required=['temperature'], greater=dict(temperature=70)))
            truth = dict(scene_name='campus', family=family, score_spec=spec,
                         devices=[dict(identity='CP-01', blocked=False, vehicle_present=True,
                                       temperature=82)])
            report = dict(outcome='accepted', records=[dict(object_id='CP-01',
                          facts=dict(blocked=True) if family=='blocked_exit' else dict(temperature=82),
                          evidence_image_id='onboard_0001')])
            (directory/'private/truth.json').write_text(json.dumps(truth))
            (directory/'public/catalog.json').write_text(json.dumps(dict(scene_conditions={})))
            if not (missing and policy=='known_task'):
                result = dict(score=evaluate(truth, report), report=report)
                (directory/'result.json').write_text(json.dumps(result))
        (self.root/'manifest.json').write_text(json.dumps(manifest))

    def test_all_policies_rechecked_without_rewriting_originals(self):
        self.fixture()
        before = {str(p):p.read_bytes() for p in self.root.rglob('*.json')}
        result = recheck(self.root)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['scene_truth_recheck']['changed_outcomes']), 4)
        for policy, stats in result['policies'].items():
            self.assertEqual(stats['correct'], 1, policy)
            self.assertEqual(stats['wrong_accepted'], 0, policy)
        self.assertEqual(before, {str(p):p.read_bytes() for p in self.root.rglob('*.json')})

    def test_other_tasks_are_not_reinterpreted(self):
        self.fixture(family='temperature')
        result = recheck(self.root)
        self.assertEqual(result['scene_truth_recheck']['affected_rows'], [])
        self.assertTrue(all(s['correct']==1 for s in result['policies'].values()))

    def test_missing_run_keeps_denominator_unavailable(self):
        self.fixture(missing=True)
        result = recheck(self.root)
        self.assertFalse(result['complete'])
        stats = result['policies']['known_task']
        self.assertEqual(stats['missing'], 1)
        self.assertIsNone(stats['wrong_accepted_per_scheduled'])

    def test_unreviewed_scene_implementation_is_rejected(self):
        self.fixture()
        path = self.root/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['implementation_hash'] = 'unreviewed'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'not been audited'):
            recheck(self.root)


if __name__ == '__main__':
    unittest.main()
