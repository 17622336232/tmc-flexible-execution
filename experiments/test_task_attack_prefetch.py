"""Network-free tests for eager scheduling and observation-access isolation."""
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.task_attack_prefetch import belief
from experiments.task_attack.audit import implementation_hash as parent_hash
from experiments.task_attack_prefetch.cleanup import belongs_to_case


class FakeModel:
    instances=[]
    def __init__(self,*args,**kwargs):
        self.calls=0;self.wall_seconds=0.;self.inputs=[];FakeModel.instances.append(self)
    def request(self,system,data,images,**kwargs):
        self.calls+=1;self.wall_seconds+=1;self.inputs.append(copy.deepcopy(data))
        def c(identity,gain):return dict(belief_id=identity,consume_after_ready=1.,fail_without=0.,fail_if_consumed=gain,recover=0.)
        return dict(beliefs=[dict(id='left',p=.5,next_camera_region=[0.,0.,.4,1.]),dict(id='right',p=.5,next_camera_region=[.6,0.,1.,1.])],
            candidates=[dict(action_id='flip_panel_01',conditions=[c('left',.8),c('right',0.)]),dict(action_id='flip_panel_02',conditions=[c('left',0.),c('right',.7)])],
            decision=dict(kind='act',delay_seconds=0),wait_continuations=[],memory='ambiguous initial view')


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.state=dict(elapsed_seconds=.01,finished=False,edit_used=False)
        frame=dict(id=1,capture_ros=10.,arrival_wall=10.,width=960,height=540,jpeg_base64=base64.b64encode(b'initial pixels').decode())
        self.initial=dict(state=self.state,frames=[frame],external_pixel_history=[])
        self.fresh=dict(state=self.state,frames=[dict(frame,id=2)],external_pixel_history=[{'id':2,'robot_pixels':{'centroid':[.8,.5]}}])
        self.paths=[];self.sent=[]
    def get(self,url,path):
        self.paths.append(path)
        if path=='catalog':return dict(editable_actions=[dict(action_id='flip_panel_01'),dict(action_id='flip_panel_02')])
        if path=='state':return self.state
        return copy.deepcopy(self.initial if path=='initial' else self.fresh)
    def predict(self,model,history,elapsed,latency):
        return dict(next_read_robot_pixel_distribution=dict(mean=[.8,.5],std=[.05,.1],conditional_external_motion_model=bool(history)))
    def post(self,request,**kwargs):
        self.sent.append(json.loads(request.data));return io.BytesIO(b'{"acknowledged":true}')
    def run_policy(self,policy):
        with tempfile.TemporaryDirectory() as tmp,patch.object(belief,'ModelClient',FakeModel),patch.object(belief,'_get',self.get),patch('experiments.task_attack.event_timing.predict',self.predict),patch.object(belief.urllib.request,'urlopen',self.post):
            summary=belief.run('http://test.invalid',policy,tmp,call_budget=1,timing_model={'fixture':True})
            data=FakeModel.instances[-1].inputs[0]
            return summary,data
    def test_first_query_is_eager_without_motion_and_fresh_motion_changes_selection(self):
        result,data=self.run_policy('belief')
        self.assertEqual(result['calls'],1)
        self.assertEqual(data['external_pixel_history'],[])
        self.assertEqual([f['id'] for f in data['external_frames']],[1])
        self.assertEqual(self.paths[:3],['catalog','state','initial'])
        self.assertEqual(self.sent,[{'action_id':'flip_panel_02'}])
        self.assertIn('frames',self.paths)
    def test_blind_never_requests_dynamic_frames_or_uses_dynamic_update(self):
        result,data=self.run_policy('belief_blind')
        self.assertNotIn('frames',self.paths)
        self.assertEqual(self.sent,[{'action_id':'flip_panel_01'}])
        self.assertEqual(data['external_pixel_history'],[])
    def test_finished_episode_never_queries_or_edits(self):
        self.state['finished']=True
        with tempfile.TemporaryDirectory() as tmp,patch.object(belief,'ModelClient',FakeModel),patch.object(belief,'_get',self.get),patch.object(belief.urllib.request,'urlopen',self.post):
            result=belief.run('http://test.invalid','belief',tmp)
        self.assertEqual(result['calls'],0);self.assertEqual(self.sent,[])
    def test_model_error_has_no_rule_fallback_or_edit(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(belief,'ModelClient',FakeModel),patch.object(FakeModel,'request',side_effect=RuntimeError('unavailable')),patch.object(belief,'_get',self.get),patch.object(belief.urllib.request,'urlopen',self.post):
            with self.assertRaises(RuntimeError):belief.run('http://test.invalid','belief',tmp)
        self.assertEqual(self.sent,[])
    def test_cleanup_selects_only_exact_case_core(self):
        root='/tmp/owned-case'
        master=['/usr/bin/python3','/opt/ros/noetic/bin/rosmaster','__log:=/tmp/owned-case/private/roslog/run/master.log']
        self.assertTrue(belongs_to_case(master,root))
        self.assertFalse(belongs_to_case(master,'/tmp/other-case'))
        self.assertFalse(belongs_to_case(['/usr/bin/other','__log:=/tmp/owned-case/private/roslog/run/master.log'],root))
        self.assertFalse(belongs_to_case(['/opt/ros/noetic/lib/rosout/rosout','__log:=/tmp/owned-case2/private/roslog/run/log'],root))
    def test_parent_remains_frozen(self):
        self.assertEqual(parent_hash(),'6383e73bdb552ea8b8d9fe5722e81f254091eaf5f2b6f07b4696dbcdfb329cbe')


if __name__=='__main__':unittest.main()
