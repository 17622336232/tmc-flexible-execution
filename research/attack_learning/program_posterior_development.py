"""Supervised external evidence-dependency posterior, without whole-task labels.

Offline proxy authorization defines required fact roles and whether a single
referent or the row is requested. Online features contain only external video
pixels and clock. Different thresholds/wordings share the same operation state.
"""
import argparse
import collections
import copy
import json
from pathlib import Path

import numpy as np

from experiments.task_attack.runtime import write_json


def softmax(z):
    z=z-z.max(axis=1,keepdims=True);p=np.exp(np.clip(z,-50,0));return p/p.sum(axis=1,keepdims=True)


def annotate(model,batches):
    paths={}
    for batch in batches:
        batch=Path(batch);manifest=json.loads((batch/'manifest.json').read_text())
        for row in manifest['rows']:
            if row['policy']=='clean':paths[row['seed']]=batch/row['directory']
    for episode in model['episodes']:
        truth=json.loads((paths[episode['seed']]/'private/truth.json').read_text());spec=truth['score_spec']
        target=spec.get('target');station=next((i+1 for i,d in enumerate(truth['devices']) if d['identity']==target),None)
        roles=sorted(spec['required']);episode['program']=dict(roles=roles,target_station=station,
            scope='single' if target is not None else 'all',literal=spec.get('mode')=='current_literal_labels')
    return model


def signature(program):
    return ','.join(program['roles'])+';'+str(program['target_station'])+';'+str(program['literal'])


def matrices(episodes,classes,mean,scale,dynamic):
    xs=[];ys=[];weights=[]
    for episode in episodes:
        if signature(episode['program']) not in classes:continue
        y=classes.index(signature(episode['program']));samples=np.array(episode['samples'])[:,1:]
        x=(samples-mean)/scale
        if not dynamic:x=x[:,-1:]
        xs.extend(np.column_stack([np.ones(len(x)),x]));ys.extend([y]*len(x));weights.extend([1./len(x)]*len(x))
    return np.array(xs),np.array(ys),np.array(weights)


def train(x,y,weights,k,penalty):
    onehot=np.eye(k)[y];weights=weights/weights.sum();beta=np.zeros((x.shape[1],k))
    frequency=np.bincount(y,weights=weights,minlength=k);beta[0]=np.log(frequency+1e-4)
    # Convex multinomial regression. Equal total weight per complete scene;
    # post-onset hovers do not create hundreds of independent task examples.
    lipschitz=np.linalg.norm((np.sqrt(weights)[:,None]*x),ord=2)**2+penalty
    step=1/max(lipschitz,1e-6)
    for _ in range(600):
        prob=softmax(x.dot(beta));gradient=x.T.dot(weights[:,None]*(prob-onehot))
        gradient[1:]+=penalty*beta[1:];delta=step*gradient;beta-=delta
        if np.max(np.abs(delta))<1e-7:break
    return beta


def evaluate(beta,x,y,weights):
    prob=softmax(x.dot(beta));weights=weights/weights.sum();truth=np.eye(prob.shape[1])[y]
    return dict(brier=float(np.sum(weights*np.sum((prob-truth)**2,axis=1))),
        log_loss=float(np.sum(weights*-np.log(np.maximum(1e-9,prob[np.arange(len(y)),y])))),
        accuracy=float(np.sum(weights*(prob.argmax(axis=1)==y))),scene_weighting=True)


def fit(model,batches,output):
    # Restore all 120 development episodes, using the SAME fixed scene split as
    # the continuation model. No new target scene is consulted.
    from .operation_model_development import collect
    episodes,_=collect(batches);model=copy.deepcopy(model);model['episodes']=episodes;annotate(model,batches)
    training=[e for e in episodes if e['seed'] in model['training_scene_seeds']]
    heldout=[e for e in episodes if e['seed'] in model['calibration_scene_seeds']]
    heads={}
    for scene in sorted({e['scene'] for e in episodes}):
        train_episodes=[e for e in training if e['scene']==scene];valid_episodes=[e for e in heldout if e['scene']==scene]
        classes=sorted({signature(e['program']) for e in train_episodes});programs=[next(e['program'] for e in train_episodes if signature(e['program'])==key) for key in classes]
        item=dict(classes=classes,programs=programs,unsupported_calibration_programs=sum(signature(e['program']) not in classes for e in valid_episodes))
        for dynamic in (False,True):
            x,y,w=matrices(train_episodes,classes,model['mean'],model['scale'],dynamic)
            vx,vy,vw=matrices(valid_episodes,classes,model['mean'],model['scale'],dynamic);candidates=[]
            for penalty in (.01,.1,1.,10.):
                beta=train(x,y,w,len(classes),penalty);metrics=evaluate(beta,vx,vy,vw)
                candidates.append(dict(penalty=penalty,beta=beta.tolist(),calibration=metrics))
            best=min(candidates,key=lambda c:(c['calibration']['log_loss'],-c['penalty']))
            item['dynamic' if dynamic else 'clock_only']=best;item[('dynamic' if dynamic else 'clock_only')+'_grid']=[{k:v for k,v in c.items() if k!='beta'} for c in candidates]
        # Explicitly permit the dynamic policy class to ignore an unhelpful
        # feature. The fallback is selected on development likelihood, never
        # from which method won a target episode.
        item['use_dynamic']=item['dynamic']['calibration']['log_loss']<item['clock_only']['calibration']['log_loss']
        heads[scene]=item
    model['program_heads']=heads;model['episodes']=training
    model['program_posterior_meaning']='Required fact-role and referent scope posterior; exact instruction, threshold and private target of a current episode are not inputs.'
    write_json(output,model);return model


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--batches',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    model=fit(json.loads(a.model.read_text()),a.batches,a.output)
    print(json.dumps({scene:{'dynamic':head['dynamic']['calibration'],'clock_only':head['clock_only']['calibration'],
        'use_dynamic':head['use_dynamic'],'unsupported':head['unsupported_calibration_programs']} for scene,head in model['program_heads'].items()},ensure_ascii=False,indent=2))
