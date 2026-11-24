import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.task_attack_value.belief import run
from experiments.task_attack_value.learned import Policy
from experiments.task_attack_value.forest import predict
from experiments.task_attack_value.vision import candidate_features
from experiments.task_attack_value.value_features import action_features,source_features


def fixture():
    frame=dict(id=1,capture_ros=100.,arrival_wall=100.,robot_pixels=dict(centroid=[.2,.7],bbox=[.18,.68,.22,.72]))
    perception=dict(pre_task_static=True,current_task_used=False,actual_static_rgb_sha256='unit-test',
        anchors=[dict(u=.5,v=.5,w=.05,h=.02)],stations=[dict(index=1,identity='EQ-01',cracked=False,temperature=None)])
    actions=[dict(action_id='place_decal_shell_cracked_01')];x=candidate_features([frame],0,perception['anchors'])
    program=dict(family='cracks',roles=['cracked'],scope='all',filter=dict(op='eq',left=dict(field='cracked'),right=dict(value=True)),reduce=None,literal=False)
    dimension=action_features(x,perception,actions)[0].shape[1]
    def forest(value,known=False):
        return dict(features=dimension+(9 if known else 0),trees=[dict(left=[-1],right=[-1],feature=[-2],threshold=[-2.],value=[value],max_depth=0)])
    heads={name:forest(.7 if name.endswith('dynamic') else .2,name.startswith('known_')) for name in ('static','dynamic','known_static','known_dynamic')}
    model=dict(version=15,value_models=heads,crossfit_value_models={'0':heads},trajectory_library=[],
        informative_prefix=dict(mean=[0.]*10,scale=[1.]*10,beta=np.zeros((11,2)).tolist()),
        metric_scale=np.ones(len(source_features(x,perception,0))).tolist(),programs=[program],calibration={},
        information_blend=1.,use_dynamic_value=True,use_known_dynamic_value=True)
    return frame,perception,actions,model,x


class LocalValuePolicy(unittest.TestCase):
    def test_blind_does_not_request_new_frames(self):
        frame,perception,actions,model,_=fixture();calls=[]
        def get(url,endpoint):
            calls.append(endpoint)
            if endpoint=='catalog':return dict(editable_actions=actions,timing=dict(robot_after_takeoff_budget_seconds=900))
            if endpoint=='initial':return dict(frames=[frame],external_pixel_history=[])
            if endpoint=='state':return dict(started=True,finished=False,edit_used=False,elapsed_seconds=10.)
            raise AssertionError('Blind requested '+endpoint)
        with tempfile.TemporaryDirectory() as directory,patch('experiments.task_attack_value.belief._get',get),patch('urllib.request.urlopen',return_value=io.BytesIO(b'{"accepted":true}')):
            result=run('http://unit-test','belief_blind',directory,model,perception)
        self.assertEqual(result['decision'],'edit');self.assertNotIn('frames',calls)

    def test_common_motion_can_retain_the_same_prior_as_blind(self):
        _,perception,actions,model,x=fixture();policy=Policy(model,perception,actions,True)
        self.assertAlmostEqual(policy.score(x,static_x=x,information_weight=0)[0]['expected_wrong_accepted'],.2)
        self.assertAlmostEqual(policy.score(x,static_x=x,information_weight=1)[0]['expected_wrong_accepted'],.7)

    def test_batched_forecasts_match_individual_predictions(self):
        _,perception,actions,model,x=fixture();policy=Policy(model,perception,actions,True)
        requests=[dict(x=x,static=x,groups=model['value_models'],information=p) for p in (0.,.2,.8,1.)]
        batch=policy.score_requests(requests)
        individual=[policy.score(x,static_x=x,information_weight=p)[0]['expected_wrong_accepted'] for p in (0.,.2,.8,1.)]
        np.testing.assert_allclose(batch,individual,rtol=0,atol=1e-15)

    def test_forecasts_use_actual_future_clock(self):
        _,_,_,_,x=fixture();new=Policy.retime(x,40.)
        self.assertAlmostEqual(new[0,15],np.log1p(40.));self.assertAlmostEqual(new[0,16],x[0,0]*np.log1p(40.))
        np.testing.assert_array_equal(new[:,19:],x[:,19:])

    def test_portable_model_rejects_schema_mismatch(self):
        _,_,_,model,_=fixture()
        with self.assertRaises(ValueError):predict(model['value_models']['static'],np.zeros((1,2)))


if __name__=='__main__':unittest.main()
