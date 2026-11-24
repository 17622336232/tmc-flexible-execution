"""Dependency reasoning, negative controls, domain fidelity and frozen protocol."""
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import threading
import time
from collections import deque

import numpy as np
from PIL import Image

from experiments.task_attack import FAMILIES
from experiments.task_attack.belief import rank_candidates,update_from_external_forecast,select_policy_decision,run as run_belief
from experiments.task_attack.baselines import closest_anchor,directional_update,spsa_update,attack_objective
from experiments.task_attack.evaluate import evaluate
from experiments.task_attack.scene import build_scene
from experiments.task_attack.tasks import SCENES
from experiments.task_attack.visual_assets import VISUAL_POLICIES,add_visual_assets,load_bundle
from experiments.task_attack.runtime import GazeboBackend,NativeNavigationError
from experiments.task_attack.batch import account_failure,main as batch_main
from experiments.task_attack.research_report import analyze
from experiments.task_attack.agent import run_agent
from experiments.task_attack.model import ModelClient,ModelDeadlineExceeded


class DependencyTests(unittest.TestCase):
    def response(self):
        return dict(beliefs=[dict(id='a',p=.6),dict(id='b',p=.4)],candidates=[
            dict(action_id='one',conditions=[dict(belief_id='a',consume_after_ready=1,fail_without=0,fail_if_consumed=.5,recover=0),
                dict(belief_id='b',consume_after_ready=0,fail_without=0,fail_if_consumed=1,recover=0)]),
            dict(action_id='two',conditions=[dict(belief_id='a',consume_after_ready=1,fail_without=0,fail_if_consumed=.4,recover=0),
                dict(belief_id='b',consume_after_ready=1,fail_without=0,fail_if_consumed=.4,recover=0)])])

    def test_marginalize_dependencies_instead_of_choosing_top_task(self):
        ranks=rank_candidates(self.response(),[dict(action_id='one'),dict(action_id='two')])
        self.assertEqual(ranks[0]['action_id'],'two')
        self.assertAlmostEqual(ranks[0]['expected_incremental_wrong_acceptance'],.4)

    def test_future_consumption_and_recovery_can_reverse_selection(self):
        response=self.response();response['candidates'][1]['conditions'][0]['recover']=1
        ranks=rank_candidates(response,[dict(action_id='one'),dict(action_id='two')])
        self.assertEqual(ranks[0]['action_id'],'one')
        response['candidates'][0]['conditions'][0]['consume_after_ready']=0
        ranks=rank_candidates(response,[dict(action_id='one'),dict(action_id='two')])
        self.assertEqual(ranks[0]['action_id'],'two')

    def test_reject_unknown_carrier_missing_belief_and_nan(self):
        menu=[dict(action_id='one'),dict(action_id='two')]
        response=self.response();response['candidates'][0]['action_id']='private_truth_action'
        with self.assertRaises(ValueError):rank_candidates(response,menu)
        response=self.response();response['candidates'][0]['conditions'].pop()
        with self.assertRaises(ValueError):rank_candidates(response,menu)
        response=self.response();response['beliefs'][0]['p']=float('nan')
        with self.assertRaises(ValueError):rank_candidates(response,menu)

    def test_fresh_external_transition_updates_spatial_beliefs_without_rewriting_model_response(self):
        response=dict(beliefs=[dict(id='left',p=.5,next_camera_region=[.1,.4,.3,.8]),
                              dict(id='right',p=.5,next_camera_region=[.7,.4,.9,.8])])
        forecast=dict(next_read_robot_pixel_distribution=dict(mean=[.8,.6],std=[.02,.02],conditional_external_motion_model=True))
        updated,audit=update_from_external_forecast(response,forecast)
        self.assertGreater(updated['beliefs'][1]['p'],.99)
        self.assertEqual(response['beliefs'][1]['p'],.5)
        self.assertEqual(len(audit),2)
        self.assertAlmostEqual(sum(b['p'] for b in updated['beliefs']),1.)

    def test_uncalibrated_spatial_forecast_cannot_supply_fictitious_evidence(self):
        response=self.response()
        updated,audit=update_from_external_forecast(response,{})
        self.assertIs(updated,response);self.assertFalse(audit)

    def test_wait_value_keeps_missing_probability_and_respects_finite_query_depth(self):
        ranks=[dict(action_id='one',expected_incremental_wrong_acceptance=.4)]
        response=dict(wait_continuations=[dict(p=.25,best_action_id='one',gain_after_observation_and_next_api=1.)])
        menu=[dict(action_id='one')]
        decision=select_policy_decision(response,ranks,menu,3)
        self.assertEqual(decision['kind'],'act');self.assertEqual(decision['wait_value'],.25)
        response['wait_continuations'][0]['p']=1.
        self.assertEqual(select_policy_decision(response,ranks,menu,3)['kind'],'wait')
        self.assertEqual(select_policy_decision(response,ranks,menu,0)['kind'],'act')
        self.assertEqual(select_policy_decision({},[],menu,0)['kind'],'abstain')

    def test_first_query_waits_for_learned_external_motion_not_private_progress(self):
        import base64
        frames=[dict(id=1,capture_ros=1.,jpeg_base64=base64.b64encode(b'initial').decode()),
                dict(id=2,capture_ros=2.,jpeg_base64=base64.b64encode(b'later').decode())]
        window_calls=[];requests=[]
        class Client:
            def __init__(self,*a,**kw):self.calls=0;self.wall_seconds=0.
            def request(self,instructions,data,images,**kw):
                self.calls+=1;requests.append(data)
                return dict(beliefs=[dict(id='a',p=1.)],candidates=[dict(action_id='one',conditions=[
                    dict(belief_id='a',consume_after_ready=1.,fail_without=0.,fail_if_consumed=0.,recover=0.)])],
                    decision=dict(kind='abstain'),memory='')
        def get(url,path):
            if path=='catalog':return dict(editable_actions=[dict(action_id='one')])
            if path=='state':return dict(finished=False,edit_used=False,elapsed_seconds=1.)
            if path=='frames':
                window_calls.append(path);return dict(frames=frames[:min(len(window_calls),2)],external_pixel_history=[],
                    state=dict(elapsed_seconds=1.))
            raise AssertionError('Unexpected/privileged endpoint: '+path)
        predictions=[dict(behavior_onset_probability=.1,behavior_onset_uses_external_motion_only=True),
                     dict(behavior_onset_probability=.9,behavior_onset_uses_external_motion_only=True),{}]
        with tempfile.TemporaryDirectory() as root,patch('experiments.task_attack.belief.ModelClient',Client),\
             patch('experiments.task_attack.belief._get',get),patch('experiments.task_attack.belief.time.sleep'),\
             patch('experiments.task_attack.event_timing.predict',side_effect=predictions):
            run_belief('public-only','belief',root,timing_model={'fitted':True})
        self.assertEqual(len(requests),1)
        self.assertEqual([f['id'] for f in requests[0]['external_frames']],[1,2])
        self.assertNotIn('authorized_task_reference',requests[0])


class TaskAndControlTests(unittest.TestCase):
    def test_27_requests_share_three_physical_domains_without_task_encoded_layout(self):
        self.assertEqual(len(FAMILIES),27)
        with tempfile.TemporaryDirectory() as directory:
            scenes={}
            for family in FAMILIES:
                scene=build_scene(Path(directory)/family,20261115,family,count=3,source_profile='single')
                name=SCENES[family]
                if name not in scenes:scenes[name]=scene
                else:
                    self.assertEqual(scene['truth']['devices'],scenes[name]['truth']['devices'])
                    self.assertEqual(scene['catalog'],scenes[name]['catalog'])
                self.assertIn('score_spec',scene['truth'])
                self.assertNotIn('score_spec',scene['catalog'])
                self.assertNotIn('authorized_task',scene['catalog'])

    def test_literal_reading_of_edited_sign_is_correct_not_attack_success(self):
        truth=dict(devices=[dict(identity='EQ-01',carrier='panel_01'),dict(identity='EQ-02',carrier='panel_02')],
            score_spec=dict(mode='current_literal_labels',required=['display_text']))
        report=dict(outcome='accepted',records=[dict(object_id='EQ-02',facts=dict(display_text='EQ-02'))])
        self.assertFalse(evaluate(truth,report)['correct'])
        self.assertTrue(evaluate(truth,report,dict(service_success=True,action_id='flip_panel_01'))['correct'])
        self.assertFalse(evaluate(truth,report,dict(service_success=False,action_id='flip_panel_01'))['correct'])

    def test_generic_new_predicates_are_post_episode_only(self):
        with tempfile.TemporaryDirectory() as directory:
            for family in ('safe_equipment','shortage','overstock','crack_and_heat','thermal_minimum','fire_and_access'):
                scene=build_scene(Path(directory)/family,712,family,count=3,source_profile='single')
                from experiments.task_attack.evaluate import _selected
                from experiments.task_attack.tasks import object_facts
                spec=scene['truth']['score_spec']
                rows=[dict(object_id=d['identity'],facts=object_facts(d)) for d in scene['truth']['devices']
                      if _selected(object_facts(d),d['identity'],spec)]
                self.assertTrue(evaluate(scene['truth'],dict(outcome='accepted',records=rows))['correct'])


class BaselineTests(unittest.TestCase):
    def test_spsa_uses_measured_difference_and_legal_pixel_range(self):
        x=np.full((2,2,3),.5);delta=np.ones_like(x)
        self.assertTrue(np.all(spsa_update(x,delta,1.,0.,.2,.1)==1.))
        self.assertTrue(np.allclose(spsa_update(x,delta,.5,.5,.2,.1),x))

    def test_anchor_update_is_directional_and_backtracks_invalid_placements(self):
        anchor=closest_anchor([[0.,0.],[1.,0.]],[.8,.2])
        self.assertTrue(np.array_equal(anchor,[1.,0.]))
        moved=directional_update([0.,0.],[1.,0.],1.,valid=lambda p:p[0]<=.3)
        self.assertTrue(np.allclose(moved,[.25,0.]))

    def test_visual_objectives_depend_on_actual_policy_actions(self):
        clean=dict(action=dict(type='move',goal=dict(x=1,y=0,z=1.2,yaw=0)))
        attack=dict(action=dict(type='move',goal=dict(x=-2,y=0,z=1.2,yaw=0)))
        position=dict(x=0,y=0,z=1.2);texture=np.zeros((4,4,3))
        for policy in VISUAL_POLICIES:
            self.assertGreater(attack_objective(policy,clean,attack,position,texture),
                               attack_objective(policy,clean,clean,position,texture))

    def test_frozen_artifact_rejects_texture_change_and_untrained_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);textures={}
            for policy in VISUAL_POLICIES:
                p=root/(policy+'.png');Image.new('RGB',(128,128),'red').save(p)
                textures[policy]=dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            bundle=dict(status='frozen_after_training',textures=textures)
            path=root/'bundle.json';path.write_text(json.dumps(bundle));load_bundle(path)
            Image.new('RGB',(128,128),'blue').save(root/(VISUAL_POLICIES[0]+'.png'))
            with self.assertRaises(ValueError):load_bundle(path)
            bundle['status']='pending';path.write_text(json.dumps(bundle))
            with self.assertRaises(ValueError):load_bundle(path)

    def test_sphere_radius_and_patch_geometry_are_physical_scene_materials(self):
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);textures={}
            for policy in VISUAL_POLICIES:
                p=root/(policy+'.png');Image.new('RGB',(128,128),'red').save(p);textures[policy]=p
            scene=build_scene(root/'scene',2,'cracks',count=3,source_profile='single')
            before=(root/'scene/private/truth.json').read_bytes()
            add_visual_assets(scene,textures)
            world=ET.parse(scene['world']).getroot()
            sphere=world.find("world/model[@name='visual_uao_spsa_01']/link/visual/geometry/sphere/radius")
            self.assertEqual(float(sphere.text),.03)
            self.assertEqual(before,(root/'scene/private/truth.json').read_bytes())
            self.assertEqual(len([a for a in scene['catalog']['editable_actions'] if a['action_id'].startswith('deploy_')]),9)


class ExecutionFeedbackTests(unittest.TestCase):
    def test_late_accepted_response_cannot_cross_the_common_evaluation_deadline(self):
        now=[0.];events=[]
        class Client:
            def __init__(self,*a,**kw):self.calls=0;self.wall_seconds=0.
            def request(self,*a,**kw):
                self.calls+=1;self.wall_seconds+=11.;now[0]=11.
                return dict(action=dict(type='submit'),report=dict(outcome='accepted',records=[],summary='too late'))
        class Backend:
            scene={}
            def observe(self):
                return dict(data=b'actual-test-frame',position=dict(x=0,y=0,z=1),yaw=0,ros_stamp=0),{}
        with tempfile.TemporaryDirectory() as directory,patch('experiments.task_attack.agent.ModelClient',Client),\
             patch('experiments.task_attack.agent.time.monotonic',side_effect=lambda:now[0]):
            report,metrics=run_agent(Backend(),'test task',Path(directory),lambda kind,**kw:events.append(kind),timeout=10.)
            decision=json.loads((Path(directory)/'decisions.jsonl').read_text())
        self.assertEqual(report['outcome'],'incomplete');self.assertTrue(decision['late_not_executed'])
        self.assertIn('late_model_result_not_executed',events)

    def test_expired_inference_budget_cannot_send_an_extra_api_request(self):
        client=ModelClient();client.deadline_monotonic=time.monotonic()-1.
        with patch('uav_bridge.chatgpt_auth.CredentialStore',side_effect=AssertionError('Must not acquire credentials')):
            with self.assertRaises(ModelDeadlineExceeded):client.request('instructions',{})
        self.assertEqual(client.calls,0)

    def test_availability_failure_cannot_become_primary_integrity_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[]
            for policy,outcome in [('clean','correct'),('belief','incomplete'),('belief_blind','correct')]:
                row=dict(block='b',scene_block='scene',family='cracks',source_profile='single',policy=policy,directory=policy)
                rows.append(row);dest=root/policy;dest.mkdir()
                (dest/'result.json').write_text(json.dumps(dict(execution_completed=True,implementation_hash='fixture',
                    score=dict(outcome=outcome,correct=outcome=='correct',wrong_accepted=False))))
            (root/'manifest.json').write_text(json.dumps(dict(rows=rows,implementation_hash='fixture')))
            report=analyze(root)
            self.assertEqual(report['PRIMARY_paired_wrong_accepted_increase_main_tasks']['belief_blind']['difference'],0.)
            self.assertEqual(report['paired_correct_completion_reduction_main_tasks']['belief_blind']['difference'],1.)

    def test_independent_scene_sampling_retains_exact_method_pairing(self):
        with tempfile.TemporaryDirectory() as directory,patch('sys.argv',['batch','--output',directory,
                '--families','cracks','temperature','inventory','--policies','clean','belief',
                '--instances-per-family','3','--scene-allocation','independent']),\
             patch('experiments.task_attack.batch.snapshot_implementation'),patch('sys.stdout',new_callable=io.StringIO):
            batch_main();manifest=json.loads((Path(directory)/'manifest.json').read_text())
            grouped={}
            for row in manifest['rows']:grouped.setdefault(row['block'],[]).append(row)
            self.assertEqual(len({row['seed'] for row in manifest['rows']}),9)
            self.assertEqual(len({row['scene_block'] for row in manifest['rows']}),9)
            for group in grouped.values():
                self.assertEqual(len(group),2);self.assertEqual(len({row['seed'] for row in group}),1)

    def test_terminal_native_stop_is_returned_without_waiting_full_motion_timeout(self):
        backend=GazeboBackend.__new__(GazeboBackend);backend.lock=threading.RLock()
        backend.planner_failures=deque(maxlen=32);backend.event=lambda *a,**kw:None
        backend.ros=SimpleNamespace(is_shutdown=lambda:False)
        started=(time.monotonic(),2.)
        def msg(name,text,stamp):
            return SimpleNamespace(name=name,msg=text,level=4,WARN=4,
                header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:stamp)))
        backend._planner_log(msg('/unrelated_node','Emergency stop.',4.))
        backend._planner_log(msg('/vla_planner','Intermediate velocity optimization failed',4.))
        backend._planner_log(msg('/vla_planner','Emergency stop.',1.))
        self.assertIsNone(backend._native_failure_since(started))
        backend._planner_log(msg('/vla_planner','Final goal in obstacle, unsafe. Emergency stop.',4.))
        with self.assertRaisesRegex(NativeNavigationError,'Original planner stopped'):
            backend.wait_motion(dict(x=1,y=1,z=1,yaw=0),native_request_started=started)

    def test_account_block_is_not_silently_run_as_hundreds_of_attack_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'private/observer/model_calls.jsonl';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(error=dict(code='http_error',status=429)))+'\n')
            failure=account_failure(root)
            self.assertEqual(failure['status'],429)
            self.assertEqual(failure['role'],'observer')
            path.write_text(json.dumps(dict(output=dict(action='wait')))+'\n')
            self.assertIsNone(account_failure(root))


class CPUSensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import pybullet
        except ImportError:raise unittest.SkipTest('Optional pinned CPU renderer dependency not active')

    def test_tilted_rays_transform_to_actual_first_hit_surfaces(self):
        from experiments.task_attack.cpu_world import SceneRenderer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'assets').mkdir()
            world=root/'scene.world'
            world.write_text('''<sdf version="1.6"><world name="test"><model name="wall"><static>true</static>
<pose>2 0 1 0 0 0</pose><link name="body"><visual name="v"><geometry><box><size>1 4 2</size></box></geometry></visual>
<collision name="c"><geometry><box><size>1 4 2</size></box></geometry></collision></link></model></world></sdf>''')
            renderer=SceneRenderer(world,robot=False)
            try:
                position=np.array([0.,0.,1.]);q=renderer.p.getQuaternionFromEuler([.13,.22,.15])
                points=renderer.lidar(position,q,100.)
                hits=points[np.linalg.norm(points,axis=1)<7.99]
                rotation=np.array(renderer.p.getMatrixFromQuaternion(q)).reshape(3,3)
                world_hits=hits.dot(rotation.T)+position+rotation.dot([0,0,.155])
                self.assertGreater(len(world_hits),100)
                # With a yaw-only acquisition transformed by full attitude,
                # these points leave the actual wall AABB and this test fails.
                self.assertTrue(np.all(world_hits[:,0]>=1.47))
                self.assertTrue(np.all(world_hits[:,0]<=2.53))
                self.assertTrue(np.all(np.abs(world_hits[:,1])<=2.03))
                self.assertTrue(np.all((world_hits[:,2]>=-.03)&(world_hits[:,2]<=2.03)))
            finally:renderer.close()

    def test_range_acquisition_cannot_hit_its_own_lagged_rgb_collision_body(self):
        from experiments.task_attack.cpu_world import SceneRenderer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'assets').mkdir();world=root/'empty.world'
            world.write_text('<sdf version="1.6"><world name="empty"></world></sdf>')
            renderer=SceneRenderer(world)
            try:
                # This old pose gives 998 false near-range hits without
                # synchronization; the real pitched platform has no target.
                renderer.set_pose('vla_uav',[.35,0,1.2],[0,0,0,1])
                q=renderer.p.getQuaternionFromEuler([0,.3,0])
                rays=renderer.lidar([0,0,1.2],q,100.)
                self.assertEqual(int((np.linalg.norm(rays,axis=1)<1.).sum()),0)
                self.assertTrue(np.allclose(renderer.poses['vla_uav'][0],[0,0,1.2]))
                # A real near obstacle remains visible; this must not turn
                # into an implementation that blindly discards short rays.
                collision=renderer.p.createCollisionShape(renderer.p.GEOM_BOX,halfExtents=[.08,.3,.3],
                    physicsClientId=renderer.client)
                renderer.p.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision,
                    basePosition=[.8,0,1.25],physicsClientId=renderer.client)
                real=renderer.lidar([0,0,1.2],q,101.)
                self.assertGreater(int((np.linalg.norm(real,axis=1)<1.).sum()),0)
            finally:renderer.close()

    def test_physical_contact_monitor_uses_original_robot_collision_body(self):
        from experiments.task_attack.cpu_world import SceneRenderer
        with tempfile.TemporaryDirectory() as directory:
            scene=build_scene(directory,3,'cracks',count=3,source_profile='single')
            renderer=SceneRenderer(scene['world'])
            try:
                self.assertIsNone(renderer.contacts())
                renderer.set_pose('vla_uav',[0.,2.5,1.2],[0,0,0,1])
                self.assertEqual(renderer.contacts(),'equipment_station_02')
            finally:renderer.close()


if __name__=='__main__':unittest.main()
