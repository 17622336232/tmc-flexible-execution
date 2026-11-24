"""Keep each held world's rewards out of the ENTIRE waiting-iteration chain.

The old simultaneous OOF recurrence could feed held-world labels back through
other folds' previous-policy targets after iteration1. This file closes that
channel. Upstream features retain their separately documented stacked-crossfit
limitations; this is not a claim of strict end-to-end nested cross-validation.
Calibration is never used for iteration selection or fitting.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from experiments.task_attack.runtime import write_json
from fit_learned_stopping import ARMS, fit, prepare as episode_prepare
from fit_multiscale_waiting import later_indices, replan
from replace_local_quality_features import fold
from world_balanced_learning import prepare as world_prepare

HORIZONS = (5., 15., 30., 60.)


def chain(training, future, prediction_x, iterations, fit_function=fit):
    """No held-world reward, transition, task or label is an argument."""
    previous = training['reward'].copy()
    result = []
    for iteration in range(1, iterations + 1):
        targets = np.zeros((len(previous), 1 + len(HORIZONS)))
        targets[:, 0] = training['reward']
        for j in range(len(HORIZONS)):
            supported = future[:, j] >= 0
            targets[supported, j + 1] = previous[future[supported, j]]
        head = fit_function(training['x'], targets, training['weights'], 161122 + iteration)
        prediction = head.predict(prediction_x)
        result.append((head, prediction))
        previous, _ = replan(training, head.predict(training['x']))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior-unit', choices=('episode', 'world'), default='world')
    p.add_argument('--iterations', type=int, default=6)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('Preserve every isolated-iteration attempt')
    with gzip.open(a.dataset / 'stopping_trajectories.json.gz', 'rt') as f:
        raw = json.load(f)
    a.output.mkdir(parents=True)
    write_json(a.output / 'contract.json', dict(status='Isolated waiting-iteration TRAIN development; not ASR',
        prior_unit=a.prior_unit, planning_horizons=HORIZONS, iterations=a.iterations,
        label_boundary='Each complete-world outer split runs a separate full recurrence. Held-world rewards never enter its targets through any other fold.',
        remaining_limitation='Upstream feature models use the existing stacked OOF scheme, not newly nested full pipelines. Calibration is disjoint from all fits and iteration choices.',
        selection='Pool predictions from five separate TRAIN outer chains; select iteration before any calibration scoring. Same procedure for every information condition.',
        unchanged='Original actions, rewards, sensor/actor/safety/scorer and2+3s dispatch schedule. No task rules or required delay.',
        data_sha256=hashlib.sha256((a.dataset / 'stopping_trajectories.json.gz').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), actual_attacks=0))
    prepare = world_prepare if a.prior_unit == 'world' else episode_prepare
    summaries = {}
    for arm in ARMS:
        training, valid = prepare(raw['training'], arm), prepare(raw['calibration'], arm)
        predictions = np.zeros((a.iterations, len(training['x']), 1 + len(HORIZONS)))
        for k in range(5):
            own = [e for e in raw['training'] if fold(e['seed']) != k]
            own_data = prepare(own, arm)
            held = np.array([fold(int(w)) == k for w in training['worlds']])
            future = later_indices(own, HORIZONS)
            fitted = chain(own_data, future, training['x'][held], a.iterations)
            for i, (_, prediction) in enumerate(fitted):
                predictions[i, held] = prediction
            del fitted
            print('ISOLATED WAIT FOLD', arm, k, flush=True)
        history, best = [], None
        for i in range(a.iterations):
            _, metrics = replan(training, predictions[i])
            score = metrics['main_selected_proxy']
            history.append(dict(iteration=i + 1, main_selected_proxy=score))
            if best is None or score > best[0]:
                best = (score, i + 1)
        score, iteration = best
        # Full TRAIN recurrence is fitted independently after the OOF choice.
        # Only public calibration feature vectors are passed to predict().
        fitted = chain(training, later_indices(raw['training'], HORIZONS), valid['x'], iteration)
        head, vp = fitted[-1]
        head.n_jobs = 1
        _, metrics = replan(valid, vp)
        with (a.output / (arm + '_stopping.pkl')).open('wb') as f:
            pickle.dump(dict(head=head, arm=arm, iteration=iteration, planning_horizons=HORIZONS,
                actual_observation_cycle=5., held_reward_feedback_removed=True), f, protocol=4)
        write_json(a.output / (arm + '_development_result.json'), dict(
            status='Disjoint calibration diagnostic, not ASR', selected_iteration=iteration,
            training_iterations=history, **metrics))
        summaries[arm] = dict(selected_iteration=iteration, own_training_crossfit_proxy=score,
            calibration_main_proxy=metrics['main_selected_proxy'], calibration_main_episodes=metrics['main_episodes'])
        write_json(a.output / 'progress.json', dict(arms_complete=list(summaries)))
        print('ISOLATED WAIT ARM', arm, summaries[arm], flush=True)
    write_json(a.output / 'summary.json', dict(status='Isolated iteration-chain development; not ASR',
        arms=summaries, actual_attacks=0, superiority_proven=False, promoted=False))


if __name__ == '__main__':
    main()
