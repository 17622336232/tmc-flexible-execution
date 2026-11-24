import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.task_attack_forward.belief import run
from experiments.task_attack_forward.learned import Policy
from experiments.task_attack_forward.vision import candidate_features


def fixture():
    frame=dict(id=1,capture_ros=100.,arrival_wall=100.,robot_pixels=dict(centroid=[.2,.7],bbox=[.18,.68,.22,.72]))
    anchors=[dict(u=.5,v=.5,w=.05,h=.02)];dimension=candidate_features([frame],0,anchors).shape[1]
    head=dict(mean=[0.]*dimension,scale=[1.]*dimension,use_dynamic=True)
    for name in ('dynamic','clock_only'):
        weights=np.zeros((dimension+1,4));weights[0]=np.log([.1,.1,.7,.1]);head[name]=dict(beta=weights.tolist(),calibration=dict(unit_test=True))
    model=dict(version=14,layout='side',dependency=dict(groups={r:copy.deepcopy(head) for r in ('side:cracked','side:temperature')}),
        programs=[dict(family='cracks',roles=['cracked'],scope='all',filter=dict(op='eq',left=dict(field='cracked'),right=dict(value=True)),reduce=None,literal=False)],
        world_particles=[[dict(cracked=False,temperature=34,display_text='EQ-01')]],parameters=dict(temperature_threshold=70),trajectory_library=[],
        endpoint=dict(by_role={r:dict(fresh_adoption=.8,cached_adoption=.2) for r in ('cracked','temperature')}))
    perception=dict(pre_task_static=True,current_task_used=False,actual_static_rgb_sha256='unit-test',anchors=anchors,
        stations=[dict(index=1,identity='EQ-01',cracked=False,temperature=None)])
    actions=[dict(action_id='place_decal_shell_cracked_01')]
    return frame,model,perception,actions


class PolicyBoundaries(unittest.TestCase):
    def test_blind_never_requests_dynamic_gateway_frames(self):
        frame,model,perception,actions=fixture();calls=[]
        def get(url,endpoint):
            calls.append(endpoint)
            if endpoint=='catalog':return dict(editable_actions=actions,timing=dict(robot_after_takeoff_budget_seconds=900))
            if endpoint=='initial':return dict(frames=[frame],external_pixel_history=[])
            if endpoint=='state':return dict(started=True,finished=False,edit_used=False,elapsed_seconds=10.)
            raise AssertionError('Blind condition attempted to read '+endpoint)
        with tempfile.TemporaryDirectory() as directory,patch('experiments.task_attack_forward.belief._get',get),patch('urllib.request.urlopen',return_value=io.BytesIO(b'{"accepted":true}')):
            result=run('http://unit-test','belief_blind',directory,model,perception)
        self.assertEqual(result['decision'],'edit');self.assertNotIn('frames',calls);self.assertEqual(result['calls'],0)

    def test_blind_does_not_change_when_later_motion_changes(self):
        frame,model,perception,actions=fixture();other=copy.deepcopy(frame);other.update(id=2,capture_ros=120.)
        other['robot_pixels']['centroid']=[.8,.2]
        a=Policy(model,perception,actions,False).decide([frame],20.,900)
        b=Policy(model,perception,actions,False).decide([frame,other],20.,900)
        self.assertEqual(a,b)

    def test_terminal_forecast_branch_has_zero_future_value(self):
        frame,model,perception,actions=fixture();x=candidate_features([frame],0,perception['anchors'])
        model['trajectory_library']=[dict(seed=1,samples=[dict(t=0,x=x.tolist(),static=x.tolist())])]
        result=Policy(model,perception,actions,True).decide([frame],0,900)
        self.assertEqual(result['kind'],'edit')
        self.assertTrue(all(branch['expected_wrong_accepted']==0 for branch in result['wait_branches']))

    def test_task_bearing_preparation_is_rejected(self):
        frame,model,perception,actions=fixture();perception['current_task_used']=True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):run('http://unit-test','belief',directory,model,perception)

    def test_common_motion_cannot_invent_task_relevance(self):
        frame,model,perception,actions=fixture();model['informative_prefix']={}
        head=model['dependency']['groups']['side:cracked']
        head['dynamic']['beta'][0]=np.log([.01,.01,.97,.01]).tolist()
        head['clock_only']['beta'][0]=np.log([.7,.1,.1,.1]).tolist()
        policy=Policy(model,perception,actions,True);x=candidate_features([frame],0,perception['anchors'])
        probability=policy.dependency(x,'cracked',static_x=x,information_weight=0.)[0]
        self.assertAlmostEqual(float(probability[1:].sum()),.3)
        self.assertAlmostEqual(float(probability[2]/probability[1:].sum()),.97/.99)

    def test_known_task_conditioning_avoids_probability_cancellation(self):
        frame,model,perception,actions=fixture();head=model['dependency']['groups']['side:cracked']
        head['dynamic']['beta'][0]=np.log([1.,1e-30,1e-30,1e-30]).tolist()
        policy=Policy(model,perception,actions,True,(model['programs'][0],None));x=candidate_features([frame],0,perception['anchors'])
        self.assertAlmostEqual(policy.score(x)[0]['expected_wrong_accepted'],1./3)


if __name__=='__main__':unittest.main()
