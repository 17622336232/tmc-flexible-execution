"""Offline black-box texture optimization on explicitly separate training scenes.

These are ONE-STEP high-level API adaptations, not original white-box OpenVLA
reproductions. Every optimization score comes from actual Max-model decisions
on physically rendered images. Never mix query counts with full task episodes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from .agent import SYSTEM as ROBOT_SYSTEM
from .audit import implementation_hash,snapshot_implementation
from .baselines import SOURCES,attack_objective,spsa_update
from .cpu_world import SceneRenderer
from .model import ModelClient
from .runtime import append_json,write_json
from .scene import build_scene
from .visual_assets import VISUAL_POLICIES,add_visual_assets


def texture_image(latent):
    return Image.fromarray(np.uint8(np.clip(latent,0,1)*255)).resize((128,128),Image.NEAREST)


def measured_observation(renderer,position,yaw):
    p=np.array([position[k] for k in ('x','y','z')]);c,s=math.cos(yaw),math.sin(yaw)
    renderer.set_pose('vla_uav',p.tolist(),renderer.p.getQuaternionFromEuler([0,0,yaw]))
    origin=p+[.17*c,.17*s,.02];fx=640/(2*math.tan(1.3962634016/2))
    rgb=renderer.camera(origin,yaw)
    raw=renderer.lidar(p,yaw,time.time())
    ranges=np.linalg.norm(raw,axis=1);raw=raw[(ranges>.1)&(ranges<7.99)]
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    points=raw.dot(rotation.T)+p+[0,0,.155]
    basis=np.array([[s,-c,0],[0,0,-1],[c,s,0]])
    optical=(points-origin).dot(basis.T)
    z=optical[:,2];valid=z>.05;points=points[valid];optical=optical[valid];z=z[valid]
    uv=optical[:,:2]*fx/z[:,None]+[320,180]
    cells=[]
    for row in range(4):
        for col in range(8):
            indices=np.flatnonzero((uv[:,0]>=80*col)&(uv[:,0]<80*(col+1))&
                                  (uv[:,1]>=90*row)&(uv[:,1]<90*(row+1)))
            if len(indices):
                j=indices[np.argsort(z[indices])[len(indices)//2]]
                cells.append(dict(pixel_cell=[col,row],world_xyz=points[j].round(3).tolist(),measured_points=len(indices)))
    transform=np.eye(4);transform[:3,:3]=basis.T;transform[:3,3]=origin
    camera=dict(width=640,height=360,world_frame='world',optical_frame='camera_optical',
        optical_axes='x_right_y_down_z_forward',image_rectified=False,projection_model='plumb_bob',
        distortion_calibrated=True,intrinsics=dict(fx=fx,fy=fx,cx=320.,cy=180.),world_T_camera=transform.tolist())
    geometry=dict(position=position,yaw=yaw,camera=camera,
        measured_surface_grid=dict(columns=8,rows=4,cells=cells,
            description='Median actual finite-range collision ray hit per image cell; obstacle surfaces, not free viewpoints.'))
    buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=88)
    return buffer.getvalue(),geometry


def request_input(task,geometry,conditions):
    return dict(authorized_task=task,current_image_id='training_observation',geometry=geometry,
        scene_conditions={k:v for k,v in conditions.items() if k!='source_graph'},
        execution_clock=dict(elapsed_seconds=0.,remaining_seconds=300.),memory='',records=[],
        recent_actions=[],last_tool_feedback=None,recent_images=[],view_history=[])


def train(root,iterations=12,examples_per_step=2,workers=4):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    final=root/'bundle.json'
    if final.exists():raise ValueError('Training already frozen; use a fresh directory')
    if iterations<1 or examples_per_step<1 or not 1<=workers<=8:raise ValueError('Invalid training budget')
    # These seeds and viewpoints are training-only; no evaluation result is
    # observed when selecting a texture. Three common scene types are retained.
    seeds=[910001,910002,910003]
    families=[('cracks','temperature','thermal_minimum'),('inventory','color_search','order_check'),
              ('parking','blocked_exit','fire_safety')]
    samples=[]
    rng=np.random.RandomState(20261115)
    initial={}
    for policy in VISUAL_POLICIES:
        path=root/(policy+'_initial.png');texture_image(rng.uniform(.1,.9,(16,16,3))).save(path);initial[policy]=path
    for scene_index,seed in enumerate(seeds):
        for view_index,family in enumerate(families[scene_index]):
            dest=root/'samples'/('%d_%d'%(scene_index,view_index))
            scene=build_scene(dest,seed,family,count=3,source_profile='single')
            add_visual_assets(scene,initial)
            x=(-1.6,0.,1.6)[view_index]
            position=dict(x=x,y=-.9,z=1.25);yaw=math.pi/2
            renderer=SceneRenderer(scene['world'])
            try:jpeg,geometry=measured_observation(renderer,position,yaw)
            finally:renderer.close()
            image=dest/'clean.jpg';image.write_bytes(jpeg)
            data=request_input(scene['truth']['task'],geometry,scene['catalog']['scene_conditions'])
            write_json(dest/'input.json',data)
            samples.append(dict(root=str(dest),seed=seed,family=family,position=position,yaw=yaw,
                                input=data,image=str(image),world=scene['world'],scene=scene))
    config=dict(version=1,iterations=iterations,examples_per_step=examples_per_step,workers=workers,
        latent_resolution=[16,16,3],physical_texture_resolution=[128,128],training_scene_seeds=seeds,
        model='gpt-6.1-terra',effort='max',implementation_hash_at_start=implementation_hash(),
        nature='Actual image-to-decision queries; not complete task rollouts and not paper-result reproduction')
    write_json(root/'training_config.json',config)
    snapshot_implementation(root/'source_snapshot')
    def clean_query(index):
        sample=samples[index];path=Path(sample['root'])/'clean_decision.json'
        if path.exists():return json.loads(path.read_text())
        model=ModelClient(Path(sample['root'])/'clean_calls.jsonl')
        response=model.request(ROBOT_SYSTEM,sample['input'],[dict(label='CURRENT onboard RGB training_observation',
                                data=Path(sample['image']).read_bytes())])
        if not isinstance(response.get('action'),dict):raise ValueError('Training policy returned no action')
        write_json(path,response);return response
    with ThreadPoolExecutor(max_workers=workers) as pool:
        clean=list(pool.map(clean_query,range(len(samples))))
        for policy_index,policy in enumerate(VISUAL_POLICIES):
            state=root/(policy+'_state.npz');log=root/(policy+'_optimization.jsonl')
            if state.exists():
                saved=np.load(str(state));latent=saved['latent'];start=int(saved['iteration'])
            else:latent=np.asarray(Image.open(initial[policy]).resize((16,16)),dtype=float)/255.;start=0
            prng=np.random.RandomState(20261115+policy_index)
            # Advance deterministic noise through an interrupted prefix.
            for _ in range(start):prng.choice([-1.,1.],latent.shape)
            for iteration in range(start,iterations):
                delta=prng.choice([-1.,1.],latent.shape)
                perturbation=.20/(iteration+1)**.101;step=.10/(iteration+1)**.602
                examples=[(iteration*examples_per_step+j)%len(samples) for j in range(examples_per_step)]
                def evaluate_trial(job):
                    sign,index=job;sample=samples[index]
                    trial=root/'optimization'/policy/('%03d_%+d_%02d'%(iteration,sign,index))
                    trial.mkdir(parents=True,exist_ok=True)
                    candidate=np.clip(latent+sign*perturbation*delta,0,1)
                    image_path=trial/'texture.png';texture_image(candidate).save(image_path)
                    scene=build_scene(trial,sample['seed'],sample['family'],count=3,source_profile='single')
                    textures=dict(initial);textures[policy]=image_path;add_visual_assets(scene,textures)
                    renderer=SceneRenderer(scene['world'])
                    # Rotate over three physical placements, independently of
                    # task truth. Positioning is training access, not online data.
                    carrier='deploy_%s_%02d'%(policy,1+(index%3))
                    pose=scene['action_states'][carrier]
                    try:
                        renderer.set_pose(pose['model_name'],pose['position'],renderer.p.getQuaternionFromEuler([0,0,pose['yaw']]))
                        jpeg,geometry=measured_observation(renderer,sample['position'],sample['yaw'])
                    finally:renderer.close()
                    (trial/'onboard.jpg').write_bytes(jpeg)
                    data=request_input(sample['input']['authorized_task'],geometry,scene['catalog']['scene_conditions'])
                    model=ModelClient(trial/'model_calls.jsonl')
                    response=model.request(ROBOT_SYSTEM,data,[dict(label='CURRENT onboard RGB training_observation',data=jpeg)])
                    if not isinstance(response.get('action'),dict):raise ValueError('No model action in optimization sample')
                    score=attack_objective(policy,clean[index],response,sample['position'],candidate)
                    record=dict(policy=policy,iteration=iteration,sign=sign,sample=index,seed=sample['seed'],
                        objective=score,model_calls=model.calls,wall_seconds=model.wall_seconds,
                        physical_carrier=carrier,response=response,image_sha256=hashlib.sha256(jpeg).hexdigest())
                    write_json(trial/'result.json',record);return record
                jobs=[(sign,index) for sign in (1,-1) for index in examples]
                results=list(pool.map(evaluate_trial,jobs))
                plus=float(np.mean([r['objective'] for r in results if r['sign']==1]))
                minus=float(np.mean([r['objective'] for r in results if r['sign']==-1]))
                latent=spsa_update(latent,delta,plus,minus,step,perturbation)
                append_json(log,dict(iteration=iteration,plus=plus,minus=minus,step=step,perturbation=perturbation,
                    examples=examples,queries=sum(r['model_calls'] for r in results)))
                np.savez(str(state),latent=latent,iteration=iteration+1)
                texture_image(latent).save(root/(policy+'.png'))
                print('%s offline optimization %d/%d; measured objectives %.4f / %.4f'%(policy,iteration+1,iterations,plus,minus),flush=True)
    textures={policy:dict(path=policy+'.png',sha256=hashlib.sha256((root/(policy+'.png')).read_bytes()).hexdigest())
              for policy in VISUAL_POLICIES}
    bundle=dict(config,status='frozen_after_training',textures=textures,sources=SOURCES,
        optimized_models_are_textures=True,trained_task_inference_model=False,
        original_uao_complete_rollout_losses=False,original_iccv_whitebox_logits=False,
        implementation_hash_at_end=implementation_hash())
    if bundle['implementation_hash_at_end']!=bundle['implementation_hash_at_start']:
        bundle['status']='development_code_changed_not_for_frozen_test'
        write_json(root/'unfrozen_bundle.json',bundle)
        raise RuntimeError('Code changed during optimization; preserved development artifacts, no evaluation bundle frozen')
    write_json(final,bundle);print('Frozen training bundle: '+str(final),flush=True)
    return bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--iterations',type=int,default=12);p.add_argument('--examples-per-step',type=int,default=2)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    train(a.output,a.iterations,a.examples_per_step,a.workers)


if __name__=='__main__':main()
