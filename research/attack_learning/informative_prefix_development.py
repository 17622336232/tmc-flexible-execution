"""Calibrate task-dependent external evidence vs common pre-task motion.

This is an observation-likelihood component, never a robot flight gate. Private
proxy tool timestamps supervise it OFFLINE; deployment receives image pixels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.task_attack.event_timing import features
from experiments.task_attack.runtime import write_json
from .program_posterior_development import train,softmax


def collect(bundle):
    root=Path('artifacts/task_attack/forward_training_01');manifest=json.loads((root/'manifest.json').read_text());result=[]
    eligible=set(bundle['dependency']['training_scene_seeds']+bundle['dependency']['calibration_scene_seeds'])
    for row in manifest['rows']:
        if row['stratum']!='clean' or row['seed'] not in eligible:continue
        directory=root/row['directory'];events=[json.loads(line) for line in (directory/'private/events.jsonl').read_text().splitlines()]
        dispatches=[r['wall'] for r in events if r['event']=='task_decision' and r.get('decision',{}).get('action',{}).get('type') in ('move','approach')]
        first=min(dispatches) if dispatches else float('inf')
        frames=[json.loads(line) for line in (directory/'public/external_frames.jsonl').read_text().splitlines()];samples=[]
        for i,frame in enumerate(frames):
            x=features(frames[:i+1],frame['capture_ros']-frames[0]['capture_ros'])
            if x is not None:samples.append(dict(x=x[:-1].tolist(),y=int(frame['arrival_wall']>=first)))
        result.append(dict(seed=row['seed'],samples=samples))
    return result


def matrix(episodes,mean,scale):
    x=[];y=[];w=[]
    for episode in episodes:
        samples=episode['samples'];labels=np.array([s['y'] for s in samples]);positive=float(labels.mean())
        for sample in samples:
            x.append([1.,*((np.array(sample['x'])-mean)/scale)]);y.append(sample['y'])
            # Shared equal-scene, balanced pre/post evidence likelihood loss.
            w.append(.5/len(samples)/max(1e-9,positive if sample['y'] else 1-positive))
    return np.array(x),np.array(y),np.array(w)


def metric(beta,x,y,w):
    p=softmax(x.dot(beta))[:,1];w=w/w.sum()
    return dict(balanced_brier=float(np.sum(w*(p-y)**2)),
        pre_task_mean_probability=float(np.average(p[y==0],weights=w[y==0])),
        pre_task_false_positive_rate=float(np.average(p[y==0]>=.5,weights=w[y==0])),
        post_task_mean_probability=float(np.average(p[y==1],weights=w[y==1])))


def fit(bundle):
    episodes=collect(bundle);validation=set(bundle['dependency']['calibration_scene_seeds']);tr=[e for e in episodes if e['seed'] not in validation];va=[e for e in episodes if e['seed'] in validation]
    values=np.array([s['x'] for e in tr for s in e['samples']]);mean=values.mean(axis=0);scale=np.maximum(1e-4,values.std(axis=0))
    x,y,w=matrix(tr,mean,scale);vx,vy,vw=matrix(va,mean,scale);choices=[]
    for penalty in (.001,.01,.1,1.,10.):
        beta=train(x,y,w,2,penalty);choices.append(dict(beta=beta.tolist(),penalty=penalty,calibration=metric(beta,vx,vy,vw)))
    best=min(choices,key=lambda c:(c['calibration']['balanced_brier'],-c['penalty']))
    best.update(mean=mean.tolist(),scale=scale.tolist(),features='Image-only motion features excluding clock',
        training_scene_seeds=[e['seed'] for e in tr],calibration_scene_seeds=[e['seed'] for e in va],
        target='Probability observed external prefix supports autonomous task-conditioned behavior, not common ascent. Soft evidence likelihood, no hard victim/attack phase rule.')
    oof={}
    for episode in tr:
        tx,ty,tw=matrix([e for e in tr if e['seed']!=episode['seed']],mean,scale)
        oof[episode['seed']]=train(tx,ty,tw,2,best['penalty']).tolist()
    return best,oof


def predict(model,history,elapsed,beta=None):
    x=features(history,elapsed)
    if x is None:return 0.
    z=np.r_[1.,(x[:-1]-np.array(model['mean']))/np.array(model['scale'])]
    return float(softmax(z.dot(beta if beta is not None else model['beta'])[None,:])[0,1])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    model,oof=fit(json.loads(a.bundle.read_text()));write_json(a.output,dict(model=model,oof=oof));print(json.dumps(model['calibration'],indent=2))
