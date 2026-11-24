"""Offline carrier-frustum labels; camera poses/geometry are NEVER online inputs.

An in-frustum surface is only an opportunity, not OCR or evidence adoption.
"""
import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from experiments.task_attack.scene import build_scene
from experiments.task_attack.runtime import write_json
from .continuation import extract,nearest


def projection(camera,center,width,height):
    transform=np.array(camera['world_T_camera']);k=camera['intrinsics']
    corners=np.array([[center[0]+dx,center[1],center[2]+dz] for dx in (-width/2,width/2) for dz in (-height/2,height/2)])
    points=(corners-transform[:3,3]).dot(transform[:3,:3])
    if points[:,2].min()<=.1:return 0.
    u=k['fx']*points[:,0]/points[:,2]+k['cx'];v=k['fy']*points[:,1]/points[:,2]+k['cy']
    w=max(0.,min(camera['width'],u.max())-max(0.,u.min()));h=max(0.,min(camera['height'],v.max())-max(0.,v.min()))
    return float(np.clip(w*h/max(1e-9,(u.max()-u.min())*(v.max()-v.min())),0.,1.))


def label_episode(directory,episode):
    manifest=json.loads((directory/'manifest.json').read_text());catalog=json.loads((directory/'public/catalog.json').read_text())
    shapes={};sdf=ET.parse(directory/'inspection.world')
    # Reconstruct the exact ORIGINAL public menu placements on a temporary
    # self-owned proxy; never mutate a recorded or running target scene.
    with tempfile.TemporaryDirectory(prefix='vla-source-labels-') as tmp:
        proxy=build_scene(tmp,manifest['seed'],manifest['family'],manifest['equipment_count'],manifest['source_profile'])
        for action in catalog['editable_actions']:
            key=action['action_id'];state=proxy['action_states'].get(key)
            if state is None:continue
            node=sdf.find('.//model[@name="'+state['model_name']+'"]')
            if node is None:continue
            if key.startswith('flip_'):
                visual=node.find('.//visual[@name="front"]');size=list(map(float,visual.findtext('geometry/mesh/scale').split()))
                center=list(state['position']);center[1]-=.014;width,height=size[0],size[2]
            else:
                box=node.find('.//visual/geometry/box/size')
                if box is None:continue
                size=list(map(float,box.text.split()));width,height=size[0],size[2];center=state['position']
            shapes[key]=(center,width,height)
    frames=[json.loads(s) for s in (directory/'public/external_frames.jsonl').read_text().splitlines()]
    start=frames[0]['capture_ros'];observations=[];calls=directory/'private/robot/model_calls.jsonl'
    for call in (json.loads(s) for s in calls.read_text().splitlines()) if calls.exists() else []:
        camera=call.get('input',{}).get('geometry',{}).get('camera',{})
        if 'world_T_camera' in camera:
            observations.append((camera['ros_stamp']-start,{key:projection(camera,*shape) for key,shape in shapes.items()}))
    for read in episode['reads']:
        near=min(observations,key=lambda o:abs(o[0]-read[0])) if observations else None
        values=near[1] if near and abs(near[0]-read[0])<.1 else {}
        if len(read)==3:read.append(values)
        else:read[3]=values
    return episode


def source_probability(prediction,action_id):
    total=0.;supported=False
    for rollout in prediction['rollouts']:
        values=[r[3][action_id] for r in rollout['future'] if len(r)>3 and action_id in r[3]]
        if values:supported=True
        total+=rollout['weight']*max(values or [0.])
    return total if supported else None


def validate(model,heldout,dynamic):
    errors_by_scene=[]
    for episode in heldout:
        channels=sorted({key for r in episode['reads'] for key in r[3]});errors=[]
        for sample in episode['samples']:
            t=sample[0];near=nearest(model,sample[1:],episode['scene'],dynamic)
            for action in channels:
                true=max([r[3].get(action,0.) for r in episode['reads'] if r[0]>t] or [0.])
                pred=sum(n['weight']*max([r[3].get(action,0.) for r in n['episode']['reads'] if r[0]>n['time']] or [0.]) for n in near)
                errors.append((pred-true)**2)
        if errors:errors_by_scene.append(float(np.mean(errors)))
    return dict(heldout_scenes=len(errors_by_scene),mean_scene_source_frustum_mse=float(np.mean(errors_by_scene)),
        label='Any future in-frustum fraction of named carrier surface; not OCR/adoption')


def augment(model_path,batch,output):
    model=json.loads(Path(model_path).read_text());root=Path(batch);manifest=json.loads((root/'manifest.json').read_text())
    paths={r['seed']:root/r['directory'] for r in manifest['rows'] if r['policy']=='clean'}
    calibration=[];training={e['seed']:e for e in model['episodes']}
    for episode in extract(root):
        label_episode(paths[episode['seed']],episode)
        if episode['seed'] in training:training[episode['seed']]['reads']=episode['reads']
        elif episode['seed'] in model['calibration_scene_seeds']:calibration.append(episode)
    model['source_emission']=dict(geometry_used_offline_only=True,current_target_camera_not_accessible=True,
        dynamic=validate(model,calibration,True),clock_only=validate(model,calibration,False),
        caution='Learned opportunity proxy from external tracks, never target camera/geometry/OCR/adoption access')
    write_json(output,model);return model


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--development-batch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=augment(a.model,a.development_batch,a.output)
    print(json.dumps(result['source_emission'],ensure_ascii=False,indent=2))
