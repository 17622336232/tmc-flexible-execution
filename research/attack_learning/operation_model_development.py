"""Fit external-prefix -> subsequent evidence BINDING, not task-name guessing.

Only offline proxy training reads robot evidence logs. Deployment receives an
RGB marker track, public clock and scene type. Recorded binding is a measured
operation label; absence of a recorded binding is not proof of no perception.
"""
import argparse
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.task_attack.event_timing import features as motion_features
from experiments.task_attack_cached.continuation import extract,lines
from experiments.task_attack_cached.source_emission import label_episode
from experiments.task_attack.runtime import write_json

ROLES=('object_id','cracked','temperature','box_count','color','vehicle_present','restricted','blocked','extinguisher_present')


def features(history,elapsed):
    base=motion_features(history,elapsed)
    entries=[e for e in history if e.get('robot_pixels')]
    if base is None:return None
    # Uniformly sample the ACTUAL whole video prefix. Unlike the last 12-frame
    # velocity alone, this preserves visit order and earlier departures.
    points=np.array([e['robot_pixels']['centroid'] for e in entries]);ts=np.array([e['capture_ros'] for e in entries])
    query=np.linspace(ts[0],ts[-1],8)
    trail=np.column_stack([np.interp(query,ts,points[:,axis]) for axis in (0,1)]).flatten()
    return np.concatenate([base[:-1],trail,base[-1:]])


def collect(batches):
    episodes=[];source_hashes=[]
    for batch in batches:
        batch=Path(batch);manifest=json.loads((batch/'manifest.json').read_text())
        rows={r['seed']:r for r in manifest['rows'] if r['policy']=='clean'}
        for episode in extract(batch):
            directory=batch/rows[episode['seed']]['directory'];truth=json.loads((directory/'private/truth.json').read_text())
            source_ids={d['identity'].casefold():'station_%02d'%(i+1) for i,d in enumerate(truth['devices'])}
            track=lines(directory/'public/external_frames.jsonl');start=track[0]['capture_ros']
            captures={}
            for path in sorted((directory/'private/robot').glob('onboard_*.json')):
                item=json.loads(path.read_text());captures[item['image_id']]=item['ros_stamp']-start
            operations={};inputs=[]
            for path in [directory/'private/robot/decisions.jsonl',directory/'private/robot/report.json']:
                if not path.exists():continue
                source_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
                decisions=lines(path) if path.suffix=='.jsonl' else [dict(report=json.loads(path.read_text()))]
                for decision in decisions:
                    output=decision.get('decision',decision)
                    records=output.get('records',[])+output.get('report',{}).get('records',[])
                    for row in records:
                        source=source_ids.get(str(row.get('object_id','')).strip().casefold())
                        image=row.get('evidence_image_id');stamp=captures.get(image)
                        if source is None or stamp is None:continue
                        fields=['object_id']+[k for k,v in row.get('facts',{}).items() if k in ROLES and v is not None]
                        for role in fields:operations[(stamp,source+':'+role)]=True
            label_episode(directory,episode)
            for read in episode['reads']:
                read.append({key:1. for (stamp,key) in operations if abs(stamp-read[0])<.15})
            samples=[];last=-math.inf
            for i,frame in enumerate(track):
                elapsed=frame['capture_ros']-start
                if elapsed-last<6:continue
                x=features(track[:i+1],elapsed)
                if x is not None:samples.append([elapsed,*x.tolist()]);last=elapsed
            if not samples:continue
            episode.update(samples=samples,recorded_binding_events=len(operations),
                count=len(truth['devices']))
            episodes.append(episode)
    # A repeated physical scene must remain one unit across all split sets.
    counts={}
    for episode in episodes:counts[episode['seed']]=counts.get(episode['seed'],0)+1
    if any(v!=1 for v in counts.values()):raise ValueError('Duplicate training scene seeds require explicit clustering')
    return episodes,source_hashes


def nearest(model,x,scene,dynamic=True,exclude_seed=None):
    query=(np.array(x)-model['mean'])/model['scale'];found=[]
    for episode in model['episodes']:
        if episode['scene']!=scene or episode['seed']==exclude_seed:continue
        samples=np.asarray(episode['samples']);residual=(samples[:,1:]-model['mean'])/model['scale']-query
        if dynamic:
            # Equalize original short-window motion and the whole-prefix trail;
            # tune their mixture on held-out OPERATION prediction, not ASR.
            alpha=model.get('history_weight',.5)
            distance=(1-alpha)*np.mean(residual[:,:10]**2,axis=1)+alpha*np.mean(residual[:,10:-1]**2,axis=1)+.1*residual[:,-1]**2
        else:distance=residual[:,-1]**2
        index=int(np.argmin(distance));found.append((float(distance[index]),episode,float(samples[index,0])))
    found.sort(key=lambda a:(a[0],a[1]['seed']));found=found[:model['neighbors']]
    if not found:raise ValueError('No supported proxy operations for public scene')
    distances=np.array([f[0] for f in found]);weights=np.exp(-.5*(distances-distances.min())/model.get('temperature',1.))
    weights/=weights.sum()
    return [dict(weight=float(w),episode=e,time=t,distance=d) for w,(d,e,t) in zip(weights,found)]


def future_vector(episode,t,channels):
    keys={key for read in episode['reads'] if read[0]>t for key,value in (read[4] if len(read)>4 else {}).items() if value>0}
    return np.array([float(key in keys) for key in channels])


def evaluate(model,heldout,dynamic):
    scores=[];positive=[];negative=[];camera=[]
    for episode in heldout:
        channels=['station_%02d:%s'%(station,role) for station in range(1,episode['count']+1) for role in ROLES]
        loss=[]
        for sample in episode['samples'][::3]:
            ns=nearest(model,sample[1:],episode['scene'],dynamic)
            predicted=sum(n['weight']*future_vector(n['episode'],n['time'],channels) for n in ns)
            actual=future_vector(episode,sample[0],channels);error=(predicted-actual)**2
            loss.extend(error);positive.extend(error[actual>0]);negative.extend(error[actual==0])
            p=sum(n['weight'] for n in ns if any(read[0]>n['time'] for read in n['episode']['reads']))
            camera.append((p-any(read[0]>sample[0] for read in episode['reads']))**2)
        if loss:scores.append(float(np.mean(loss)))
    return dict(scenes=len(scores),mean_scene_binding_brier=float(np.mean(scores)),
        positive_binding_brier=float(np.mean(positive)),negative_binding_brier=float(np.mean(negative)),
        future_camera_brier=float(np.mean(camera)),
        metric='Actual future evidence-image-to-source/fact recorded bindings; not hidden perception state or attack success')


def fit(batches,base_model,output):
    episodes,source_hashes=collect(batches)
    ranked=sorted(episodes,key=lambda e:hashlib.sha256(('bind-model-v10:'+str(e['seed'])).encode()).hexdigest())
    calibration=ranked[:len(ranked)//4];training=ranked[len(calibration):]
    values=np.concatenate([np.asarray(e['samples'])[:,1:] for e in training])
    model=dict(version=10,mean=values.mean(axis=0).tolist(),scale=np.maximum(values.std(axis=0),1e-4).tolist(),
        episodes=training,neighbors=5,temperature=1.,history_weight=.5,pixel_std=base_model['pixel_std'],
        behavior_onset=base_model['behavior_onset'],
        all_development_scene_seeds=sorted(e['seed'] for e in episodes),
        training_scene_seeds=sorted(e['seed'] for e in training),calibration_scene_seeds=sorted(e['seed'] for e in calibration),
        offline_operation_label_hashes=source_hashes,external_only_at_deployment=True,
        no_task_name_or_private_task_feature=True,terminal_prefixes_retained=True,
        learned_operation='Image acquisition -> later explicit source/fact binding; whole observed external prefix -> joint continuation',
        limitations='Not full paper K/D/E: no inferred hidden binding is asserted to be observed; endpoint intervention effect is a separately calibrated proxy.')
    choices=[]
    for alpha in (0.,.5,1.):
        for k in (3,5,9):
            model.update(history_weight=alpha,neighbors=k)
            dynamic=evaluate(model,calibration,True);static=evaluate(model,calibration,False)
            choices.append(dict(history_weight=alpha,neighbors=k,dynamic=dynamic,clock_only=static,
                criterion=dynamic['mean_scene_binding_brier']))
    best=min(choices,key=lambda a:(a['criterion'],a['neighbors'],a['history_weight']))
    model.update(history_weight=best['history_weight'],neighbors=best['neighbors'])
    model['calibration']=dict(candidates=choices,selected=best,split='Whole scene held out; calibration chooses operation prediction only, never final ASR')
    write_json(output,model);return model


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',type=Path,required=True)
    p.add_argument('--base-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=fit(a.batches,json.loads(a.base_model.read_text()),a.output)
    print(json.dumps(dict(training=len(result['episodes']),calibration=result['calibration']['selected']),ensure_ascii=False,indent=2))
