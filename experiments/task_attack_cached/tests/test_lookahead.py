import unittest
from unittest.mock import patch

import numpy as np

from experiments.task_attack_cached.lookahead import select


class CachedTreeTests(unittest.TestCase):
    def compute(self,dynamic,now=.3,later=.2,onset=.1):
        x=np.zeros(11);x[-1]=np.log1p(10.)
        episode={'seed':17,'samples':[[4.,*x.tolist()],[11.,*x.tolist()]]}
        calls=[]
        def predict(model,features,scene,is_dynamic,exclude_seed=None):
            calls.append((features.copy(),is_dynamic,exclude_seed))
            return {'later':exclude_seed is not None,'probability_any_future_read':1.}
        def ranker(response,menu,pred,base,dyn):
            return [dict(action_id='only',expected_incremental_wrong_acceptance=later if pred['later'] else now)],{}
        # The branch is at training time 11 after a current matching time 4;
        # its online clock MUST be current 10 + delta 7, not training time 11.
        with patch('experiments.task_attack_cached.lookahead.features',return_value=x),\
             patch('experiments.task_attack_cached.lookahead.nearest',return_value=[dict(weight=1.,episode=episode,time=4.)]),\
             patch('experiments.task_attack_cached.lookahead.onset_probability',side_effect=([onset,1.] if dynamic else [1.])),\
             patch('experiments.task_attack_cached.lookahead.make_forecast',side_effect=predict),\
             patch('experiments.task_attack_cached.lookahead.rank',side_effect=ranker):
            return select({}, {}, [], [],10.,'equipment',dynamic),calls

    def test_wait_cannot_keep_old_optimistic_value_after_opportunity_loss(self):
        decision,calls=self.compute(True,now=.3,later=.1)
        self.assertEqual(decision['kind'],'act');self.assertAlmostEqual(decision['wait_value'],.1)

    def test_informative_future_can_justify_wait_without_another_api(self):
        decision,calls=self.compute(True,now=.1,later=.3)
        self.assertEqual(decision['kind'],'wait');self.assertEqual(decision['additional_api_requests'],0)
        branch=[c for c in calls if c[2] is not None]
        self.assertTrue(branch)
        for features,dynamic,excluded in branch:
            self.assertEqual(excluded,17)
            self.assertAlmostEqual(np.expm1(features[-1]),17.)

    def test_blind_future_stays_clock_only_even_when_branch_moves(self):
        decision,calls=self.compute(False,now=.1,later=.3)
        self.assertTrue(all(not dynamic for _,dynamic,_ in calls))
        self.assertIsNone(decision['actual_external_onset_probability'])
        self.assertEqual(decision['kind'],'wait')

    def test_positive_tie_executes_immediately(self):
        decision,calls=self.compute(True,now=.2,later=.2)
        self.assertEqual(decision['kind'],'act')


if __name__=='__main__':unittest.main()
