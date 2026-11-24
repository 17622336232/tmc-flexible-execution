"""Uniform, offline diagnosis of a known campus-scene annotation error.

The frozen actor implementation puts each parked vehicle across its portal's
approach. Its original `blocked` truth only counted crates. This recheck counts
either physical obstruction, for ALL policies, without changing reports, actor
inputs, truth.json, result.json or the original primary analysis.

This correction is restricted to the audited frozen implementation below. It
must not be applied to arbitrary worlds in which parking is beside a passage.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from experiments.analyze_task_attack import analyze
from experiments.task_attack.evaluate import evaluate


AUDITED_IMPLEMENTATIONS = {
    '265fc6d45a2e5c1d2ef51ead3a03b45527513b64df751a427878b7ff517d0a7b',
}


def recheck(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['implementation_hash'] not in AUDITED_IMPLEMENTATIONS:
        raise ValueError('This correction has not been audited for this scene implementation')
    original = analyze(root)
    checked = copy.deepcopy(original)
    policies = checked['policies']
    for stats in policies.values():
        for outcome in ('correct', 'wrong_accepted', 'refused', 'incomplete'):
            stats[outcome] = 0
    evidence = []
    for row in checked['rows']:
        path = Path(row['result_path'])
        result = json.loads(path.read_text())
        raw_score = copy.deepcopy(result['score'])
        score = raw_score
        changes = []
        if row['family'] == 'blocked_exit':
            truth_path = path.parent / 'private/truth.json'
            truth = json.loads(truth_path.read_text())
            if truth.get('scene_name') != 'campus':
                raise ValueError('Expected the audited campus layout')
            for device in truth['devices']:
                if device.get('vehicle_present') and not device.get('blocked'):
                    changes.append(dict(object_id=device['identity'], field='blocked',
                                        before=False, after=True,
                                        reason='Parked vehicle spans this portal approach'))
                    device['blocked'] = True
            report = result.get('report') or dict(outcome='incomplete', records=[])
            score = evaluate(truth, report)
            evidence.append(dict(result_path=str(path), raw_score=raw_score,
                                 checked_score=score, truth_changes=changes,
                                 original_truth_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest()))
        row['raw_outcome'] = raw_score['outcome']
        row['outcome'] = score['outcome']
        row['fact_errors'] = score.get('fact_errors', [])
        row['scene_truth_changes'] = changes
        policies[row['policy']][score['outcome']] += 1
    for stats in policies.values():
        stats['wrong_accepted_per_scheduled'] = (
            None if stats['missing'] else stats['wrong_accepted'] / stats['scheduled'])
    checked['families'] = {}
    for row in checked['rows']:
        counts = checked['families'].setdefault(row['family'], {}).setdefault(
            row['policy'], dict(correct=0, wrong_accepted=0, refused=0, incomplete=0))
        counts[row['outcome']] += 1
    checked['scene_truth_recheck'] = dict(
        version='campus_approach_obstruction_v1',
        post_hoc=True, primary_scores_modified=False,
        implementation_hash=manifest['implementation_hash'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reason='The original campus blocked flag counted crates but omitted vehicles across the same portal approach.',
        scope='All blocked_exit policies; other task scores unchanged.',
        affected_rows=evidence,
        changed_outcomes=[dict(result_path=x['result_path'],
                               before=x['raw_score']['outcome'],
                               after=x['checked_score']['outcome']) for x in evidence
                          if x['raw_score']['outcome'] != x['checked_score']['outcome']])
    checked['limitations'].append(
        'Post-hoc correction of a physical-scene truth error; original primary scores remain available separately.')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    root = args.batch.resolve()
    checked = recheck(root)
    (root / 'analysis.scene_truth_checked.json').write_text(
        json.dumps(checked, ensure_ascii=False, indent=2))
    if args.plot:
        if not checked['complete']:
            raise SystemExit('Batch incomplete; no full-batch plot emitted')
        from experiments.analyze_task_attack import plot
        plot(checked, root, filename='outcomes.scene_truth_checked.png',
             subtitle='Uniform post-episode scene-truth correction; original scores retained')
    print(json.dumps({k: checked[k] for k in ('complete', 'scheduled', 'available', 'policies',
                                             'scene_truth_recheck')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
