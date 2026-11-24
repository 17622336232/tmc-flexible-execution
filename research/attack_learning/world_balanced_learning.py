"""Equal family/world prior for crossed TRAIN task parameters.

Repeated target configurations in one world must share that world's mass.
Prefix duration is deliberately NOT normalized: alive tasks retain the same
pre-task prior at a given clock time. All comparison heads use this rule.
No outcome, instruction contents or held-out label enters the weights.
"""
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from experiments.task_attack_value.forest import export, predict
from experiments.task_attack_value.value_features import action_features
from external_observability_audit import descriptor
from learn_read_dependency import context, goal_inputs
from local_operator_kernels import ConditionalDistribution
from local_operator_rollout import quality_input
from research.attack_learning.train_fixed_slot_models import assess
from fit_learned_stopping import prepare as original_prepare


def episode_weights(episodes):
    counts = Counter((e['family'], e['seed']) for e in episodes)
    worlds = defaultdict(set)
    for e in episodes:
        worlds[e['family']].add(e['seed'])
    if len({e['key'] for e in episodes}) != len(episodes):
        raise ValueError('Execution keys must be unique')
    return {e['key']: 1. / len(worlds[e['family']]) / counts[e['family'], e['seed']]
            for e in episodes}


def fit_intent(episodes, quality, hypotheses):
    keys = [descriptor(g, z) for g, z in hypotheses]
    weights = episode_weights(episodes)
    xx, yy, ww = [], [], []
    for e in episodes:
        target = keys.index(descriptor(e['program'], e['target']))
        for t in np.arange(0., min(901., e['events'][-1]['t']), 5.):
            ports, clock = context(e['events'], quality[e['key']], t, e['perception'])
            xx.append(goal_inputs(ports, clock, hypotheses))
            yy.extend(np.eye(len(hypotheses))[target])
            ww.extend([weights[e['key']]] * len(hypotheses))
    head = ExtraTreesRegressor(n_estimators=96, max_depth=12, min_samples_leaf=16,
                              max_features=.8, n_jobs=2, random_state=161120)
    head.fit(np.vstack(xx), np.array(yy), sample_weight=np.array(ww))
    head.n_jobs = 1
    return head


def fit_quality(episodes):
    prior = episode_weights(episodes)
    xx, yy, weights = [], [], []
    for ep in episodes:
        for ex in ep['examples']:
            for source in range(4):
                for j in range(2):
                    q = ex['acquisition_quality'][source * 2 + j]
                    if q is None:
                        continue
                    xx.append(quality_input(ex['event'], ep['perception'], source, j))
                    yy.append([q])
                    weights.append(prior[ep['key']] / len(ep['examples']) / 8)
    head = ConditionalDistribution(trees=64, leaf=3, depth=16, seed=16119).fit(
        np.array(xx, dtype=np.float32), np.array(yy), np.array(weights))
    head.forest.n_jobs = 1
    return head


def matrix(cases, kind):
    x, y, weights, groups = [], [], [], []
    prior = episode_weights(cases)
    known, dynamic = kind.startswith('known_'), kind.endswith('dynamic')
    for case in cases:
        start = len(x)
        for sample in case['samples']:
            values, _ = action_features(np.array(sample['x' if dynamic else 'static']),
                case['perception'], case['actions'], case['known'] if known else None)
            x.extend(values)
            y.extend(sample['y'])
            weights.extend([prior[case['key']] / len(case['actions'])] * len(values))
        groups.append((case, start, len(x)))
    return np.array(x, dtype=np.float32), np.array(y), np.array(weights), groups


def fit_one(training, validation, kind, configuration):
    # Configurations are already fixed on old TRAIN development. Validation
    # metrics are reported only; no configuration or arm is selected here.
    x, y, w, _ = matrix(training, kind)
    vx, vy, _, groups = matrix(validation, kind)
    head = ExtraTreesRegressor(n_estimators=96, n_jobs=2, random_state=1515,
                              max_features=.8, **configuration)
    head.fit(x, y, sample_weight=w)
    prediction = head.predict(vx)
    portable = export(head)
    if float(np.max(np.abs(prediction - predict(portable, vx)))) > 1e-12:
        raise RuntimeError('Portable model prediction mismatch')
    metrics = assess(prediction, vy, groups)
    print('WORLD BALANCED FIT', kind, configuration, metrics, flush=True)
    return portable, configuration, metrics


def prepare(episodes, arm):
    data = original_prepare(episodes, arm)
    weights = episode_weights(episodes)
    for group in data['groups']:
        data['weights'][group['start']:group['end']] = weights[group['key']]
    return data
