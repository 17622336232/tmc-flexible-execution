"""Fit field/source acquisition opportunity from actual proxy camera geometry.

Offline collision rays establish occlusion, not merely an in-frustum rectangle.
No target camera, map or 3-D state is read at deployment. Apparent area is a
continuous quality proxy, NOT a claim that a field was recognized or accepted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from experiments.task_attack.runtime import write_json
from experiments.task_attack.scene import build_scene as old_scene
from experiments.task_attack_recessed.scene import build_scene as side_scene
from experiments.task_attack.cpu_world import SceneRenderer
from experiments.task_attack.visual_assets import add_visual_assets,load_bundle
from .relational_model_development import anchors_from_rgb,candidate_features


def opportunity(renderer,camera,center,width,height,yaw):
    transform=np.array(camera['world_T_camera']);k=camera['intrinsics'];origin=transform[:3,3]
    axis=np.array([math.cos(yaw),math.sin(yaw),0.])
    points=np.array([np.array(center)+axis*x+np.array([0.,0.,z]) for x in np.linspace(-width/2,width/2,9) for z in np.linspace(-height/2,height/2,5)])
    p=(points-origin).dot(transform[:3,:3]);in_front=p[:,2]>.05
    uv=np.column_stack([k['fx']*p[:,0]/np.maximum(.001,p[:,2])+k['cx'],k['fy']*p[:,1]/np.maximum(.001,p[:,2])+k['cy']])
    inside=in_front&(uv[:,0]>=0)&(uv[:,0]<camera['width'])&(uv[:,1]>=0)&(uv[:,1]<camera['height'])
    rays=renderer.p.rayTestBatch([origin.tolist()]*len(points),points.tolist(),numThreads=1,physicsClientId=renderer.client)
    clear=np.array([r[0]<0 or r[2]>.9999 for r in rays])
    fraction=float(np.mean(inside&clear))
    area=float(np.ptp(uv[:,0])*np.ptp(uv[:,1])) if in_front.all() else 0.
    return fraction,max(0.,area)


def collect(batches,cache):
    if cache.exists():return json.loads(cache.read_text())
    episodes=[];failed=[];references={'cracked':[],'temperature':[]}
    for batch in batches:
        batch=Path(batch);manifest=json.loads((batch/'manifest.json').read_text())
        bundle=manifest.get('baseline_bundle') or 'models/task_attack/baseline_bundle/bundle.json'
        bundle_data,textures,bundle_hash=load_bundle(Path(bundle))
        for row in manifest['rows']:
            if row['policy']!='clean':continue
            directory=batch/row['directory'];framefile=directory/'public/external_frames.jsonl'
            if not framefile.exists():continue
            truth=json.loads((directory/'private/truth.json').read_text())
            if truth['scene_name']!='equipment':continue
            track=[json.loads(s) for s in framefile.read_text().splitlines() if s.strip()]
            if not track:continue
            count=len(truth['devices']);anchors=anchors_from_rgb((directory/'public/external'/('%06d.jpg'%track[0]['id'])).read_bytes(),count)
            if anchors is None:failed.append(str(directory));continue
            catalog=json.loads((directory/'public/catalog.json').read_text());protocol=catalog.get('experiment_conditions',{}).get('carrier_protocol')
            side=protocol=='side_state_decals_v11';calls=directory/'private/robot/model_calls.jsonl'
            if not calls.exists():continue
            callrows=[json.loads(s) for s in calls.read_text().splitlines() if s.strip()];cameras={}
            cited=set();identities={d['identity'].casefold():i for i,d in enumerate(truth['devices'])}
            for call in callrows:
                camera=call.get('input',{}).get('geometry',{}).get('camera',{});image=call.get('input',{}).get('current_image_id')
                if 'world_T_camera' in camera and image:cameras[image]=camera
                output=call.get('output') or {};records=output.get('records',[])+output.get('report',{}).get('records',[])
                for record in records:
                    station=identities.get(str(record.get('object_id','')).casefold());evidence=record.get('evidence_image_id')
                    if station is None:continue
                    for role in ('cracked','temperature'):
                        if record.get('facts',{}).get(role) is not None:cited.add((evidence,station,role))
            start=track[0]['capture_ros'];read_labels=[]
            # Rebuild only a SELF-OWNED temporary proxy. SceneRenderer creates
            # a mesh cache; it must never write into recorded experiment data.
            with tempfile.TemporaryDirectory(prefix='vla-read-source-') as tmp:
                constructor=side_scene if side else old_scene
                scene=constructor(tmp,row['seed'],row['family'],count,row['source_profile']);add_visual_assets(scene,textures,bundle_hash)
                renderer=SceneRenderer(scene['world'],robot=False);shapes={}
                for i in range(count):
                    for role in ('cracked','temperature'):
                        key=('place_decal_shell_smooth_%02d'%(i+1) if role=='cracked' else 'place_decal_display_05_%02d'%(i+1)) if side else ('cover_shell_%02d'%(i+1) if role=='cracked' else 'cover_display_%02d'%(i+1))
                        state=scene['action_states'][key]
                        size=(.64,.70) if role=='cracked' and side else (.64,.78) if role=='cracked' else (.70,.35) if side else (1.1,.25)
                        shapes[(i,role)]=(state['position'],*size,state['yaw'])
                for image,camera in cameras.items():
                    values={}
                    for (station,role),shape in shapes.items():
                        visible,area=opportunity(renderer,camera,*shape);values['%d:%s'%(station,role)]=[visible,area]
                        if (image,station,role) in cited and visible>.1:references[role].append(area)
                    read_labels.append(dict(t=camera['ros_stamp']-start,values=values))
                renderer.p.disconnect(renderer.client)
            samples=[];last=-1e9
            for i,frame in enumerate(track):
                elapsed=frame['capture_ros']-start
                if elapsed-last<6:continue
                x=candidate_features(track[:i+1],elapsed,anchors);static=candidate_features(track[:1],elapsed,anchors)
                if x is not None and static is not None:samples.append(dict(t=elapsed,x=x.tolist(),static=static.tolist()));last=elapsed
            episodes.append(dict(seed=row['seed'],layout='side' if side else 'front',samples=samples,reads=read_labels,
                count=count,source_sha256=hashlib.sha256(framefile.read_bytes()).hexdigest(),directory=str(directory)))
            print('labelled',len(episodes),directory,flush=True)
    result=dict(episodes=episodes,anchor_failures=failed,area_reference={role:float(np.median(values)) if values else None for role,values in references.items()},
        area_reference_count={role:len(values) for role,values in references.items()},
        label='Actual image geometry plus scene collision occlusion rays; continuous source visibility/size opportunity; no recognition/adoption assertion')
    write_json(cache,result);return result


def matrix(episodes,role,reference,mean=None,scale=None,dynamic=True):
    xs=[];ys=[];weights=[]
    for e in episodes:
        for sample in e['samples']:
            values=[]
            for station in range(e['count']):
                past=[];future=[]
                for read in e['reads']:
                    visible,area=read['values']['%d:%s'%(station,role)];value=visible*min(1.,area/max(1.,reference))
                    (future if read['t']>sample['t'] else past).append(value)
                values.append([max(future or [0.]),max(past or [0.])])
            xs.extend(sample['x' if dynamic else 'static']);ys.extend(values);weights.extend([1./(len(e['samples'])*e['count'])]*e['count'])
    x=np.array(xs);y=np.array(ys);weights=np.array(weights);weights/=weights.sum()
    if mean is None:mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-4)
    x=np.column_stack([np.ones(len(x)),(x-mean)/scale]);return x,y,weights,mean,scale


def fit(data,output):
    episodes=data['episodes'];calibration=set()
    for layout in sorted({e['layout'] for e in episodes}):
        seeds=sorted({e['seed'] for e in episodes if e['layout']==layout},key=lambda s:hashlib.sha256(('source-op-v12:'+str(s)).encode()).hexdigest())
        calibration.update(seeds[:max(2,len(seeds)//4)])
    model=dict(version=12,groups={},area_reference=data['area_reference'],area_reference_count=data['area_reference_count'],
        all_development_scene_seeds=sorted(e['seed'] for e in episodes),training_scene_seeds=sorted(e['seed'] for e in episodes if e['seed'] not in calibration),calibration_scene_seeds=sorted(calibration),
        source_sha256=[e['source_sha256'] for e in episodes],
        source_geometry_used_offline_only=True,external_rgb_only_at_deployment=True,
        labels='Continuous future and cached field/source acquisition opportunity; downstream adoption separately learned from edited proxy execution')
    for layout in sorted({e['layout'] for e in episodes}):
        train=[e for e in episodes if e['layout']==layout and e['seed'] not in calibration];valid=[e for e in episodes if e['layout']==layout and e['seed'] in calibration]
        for role in ('cracked','temperature'):
            reference=data['area_reference'][role];group=dict(training_scenes=len(train),calibration_scenes=len(valid))
            x,y,w,mean,scale=matrix(train,role,reference);group.update(mean=mean.tolist(),scale=scale.tolist());scores={}
            for dynamic in (False,True):
                x,y,w,_,_=matrix(train,role,reference,mean,scale,dynamic);vx,vy,vw,_,_=matrix(valid,role,reference,mean,scale,dynamic);candidates=[]
                for penalty in (.001,.01,.1,1.):
                    regularizer=penalty*np.eye(x.shape[1]);regularizer[0,0]=0.
                    beta=np.linalg.solve(x.T.dot(w[:,None]*x)+regularizer,x.T.dot(w[:,None]*y));prediction=np.clip(vx.dot(beta),0,1)
                    mse=np.sum(vw[:,None]*(prediction-vy)**2,axis=0)
                    candidates.append(dict(beta=beta.tolist(),penalty=penalty,calibration_mse=mse.tolist(),criterion=float(mse.mean())))
                best=min(candidates,key=lambda c:(c['criterion'],-c['penalty']));group['dynamic' if dynamic else 'clock_only']=best
            group['use_dynamic']=[group['dynamic']['calibration_mse'][i]<group['clock_only']['calibration_mse'][i] for i in (0,1)]
            model['groups'][layout+':'+role]=group
    write_json(output,model);return model


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=fit(collect(a.batches,a.cache),a.output)
    print(json.dumps({key:{'dynamic':g['dynamic']['calibration_mse'],'clock':g['clock_only']['calibration_mse'],'use_dynamic':g['use_dynamic']} for key,g in result['groups'].items()},indent=2))
