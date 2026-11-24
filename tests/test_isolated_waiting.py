import copy
import unittest

import numpy as np

from fit_learned_stopping import prepare
from fit_multiscale_waiting import later_indices
from fit_isolated_waiting import chain, HORIZONS


class MeanHead:
    def __init__(self, y):
        self.values = np.mean(y, axis=0)

    def predict(self, x):
        return np.tile(self.values, (len(x), 1))


class IsolatedWaitingTests(unittest.TestCase):
    def test_held_reward_cannot_affect_any_training_iteration(self):
        def episode(key, seed, reward):
            return dict(key=key, seed=seed, family='f', rows=[dict(t=i * 5.,
                arms=dict(ours_critic=dict(x=[float(i)], reward=reward))) for i in range(6)])
        own = [episode('a', 1, .2), episode('b', 2, .4)]
        held = [episode('h', 3, .3)]
        changed = copy.deepcopy(held)
        for r in changed[0]['rows']:
            r['arms']['ours_critic']['reward'] = -99999.
        recorded = []
        def fit(x, y, weights, seed):
            recorded.append(y.copy())
            return MeanHead(y)
        first = chain(prepare(own, 'ours_critic'), later_indices(own, HORIZONS),
                      prepare(held, 'ours_critic')['x'], 6, fit)
        targets = [r.copy() for r in recorded]
        recorded.clear()
        second = chain(prepare(own, 'ours_critic'), later_indices(own, HORIZONS),
                       prepare(changed, 'ours_critic')['x'], 6, fit)
        for a, b in zip(targets, recorded):
            np.testing.assert_array_equal(a, b)
        for (_, a), (_, b) in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_training_targets_do_affect_fitted_chain(self):
        def data(value):
            return dict(x=np.zeros((2, 1)), reward=np.full(2, value), weights=np.ones(2),
                following=np.array([1, -1]), groups=[dict(key='a', seed=1, family='f', start=0, end=2)])
        future = np.array([[1, -1, -1, -1], [-1, -1, -1, -1]])
        fit = lambda x, y, w, s: MeanHead(y)
        left = chain(data(.1), future, np.zeros((1, 1)), 2, fit)[-1][1]
        right = chain(data(.9), future, np.zeros((1, 1)), 2, fit)[-1][1]
        self.assertFalse(np.array_equal(left, right))


if __name__ == '__main__':
    unittest.main()
