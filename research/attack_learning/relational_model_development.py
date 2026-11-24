"""Entity-relative external behavior model and factored operation posterior.

Goal operands share ONE scorer: moving toward station 4 is represented by the
same relative geometry as moving toward station 1. Training authorization labels
are offline only. No current task, XYZ, robot log or simulator state is an input.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from experiments.task_attack.event_timing import features as motion_features
from experiments.task_attack.runtime import write_json
from .program_posterior_development import softmax,train,evaluate


def anchors_from_rgb(data,count):
    image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:return None
    height,width=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    for threshold in (185,175,195,150):
        mask=cv2.inRange(hsv,np.array([0,0,threshold]),np.array([180,55,255]))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,_,stats,centers=cv2.connectedComponentsWithStats(mask);candidates=[]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if .018*width<=w<=.12*width and .008*height<=h<=.065*height and 2<=w/h<=8 and area/(w*h)>.40:
                candidates.append(dict(u=float((x+w/2)/width),v=float((y+h/2)/height),
                    w=float(w/width),h=float(h/height),y=int(y)))
        candidates.sort(key=lambda a:a['y'])
        for first in candidates:
            row=[c for c in candidates if abs(c['y']-first['y'])<=height*.015]
            if len(row)==count:
                row.sort(key=lambda a:a['u'])
                if min(b['u']-a['u'] for a,b in zip(row,row[1:]))>width*.00001:
                    return [{k:v for k,v in c.items() if k!='y'} for c in row]
    return None


def candidate_features(history,elapsed,anchors):
    entries=[x for x in history if x.get('robot_pixels')]
    base=motion_features(history,elapsed)
    if base is None:return None
    points=np.array([e['robot_pixels']['centroid'] for e in entries]);ts=np.array([e['capture_ros'] for e in entries])
    queries=np.linspace(ts[0],ts[-1],5);trail=np.column_stack([np.interp(queries,ts,points[:,axis]) for axis in (0,1)])
    old=np.array([np.interp(max(ts[0],ts[-1]-15),ts,points[:,axis]) for axis in (0,1)])
    values=[]
    for anchor in anchors:
        # Normalize by the observed label width, not a true world scale.
        scale=max(.005,anchor['w']);center=np.array([anchor['u'],anchor['v']]);delta=(points[-1]-center)/scale
        velocity=np.array(base[4:6])/scale;distance=np.linalg.norm(delta);previous=np.linalg.norm((old-center)/scale)
        local=((trail-center)/scale).flatten()
        values.append([*delta,*np.abs(delta),*(delta**2),*velocity,float(delta.dot(velocity)),distance,
            previous-distance,base[2]/scale,base[3]/scale,base[7]/scale,base[9],math.log1p(elapsed),
            delta[0]*math.log1p(elapsed),delta[1]*math.log1p(elapsed),distance*math.log1p(elapsed),*local])
    return np.array(values)


def context_features(candidate):
    # Keep two likely local source relations without identifying the task.
    order=np.argsort(candidate[:,9]);return np.concatenate([candidate[order[0]],candidate[order[1]],candidate.mean(axis=0)])


def collect(batches):
    episodes=[];failed=[]
    for batch in batches:
        batch=Path(batch);manifest=json.loads((batch/'manifest.json').read_text())
        for row in manifest['rows']:
            if row['policy']!='clean':continue
            directory=batch/row['directory'];frames=directory/'public/external_frames.jsonl'
            if not frames.exists():continue
            track=[json.loads(s) for s in frames.read_text().splitlines() if s.strip()]
            if not track:continue
            truth=json.loads((directory/'private/truth.json').read_text());count=len(truth['devices'])
            image=directory/'public/external'/('%06d.jpg'%track[0]['id'])
            if not image.exists():failed.append(dict(directory=str(directory),reason='no initial external image'));continue
            anchors=anchors_from_rgb(image.read_bytes(),count)
            if anchors is None:failed.append(dict(directory=str(directory),reason='ambiguous RGB label anchors'));continue
            spec=truth['score_spec'];target=spec.get('target');station=next((i for i,d in enumerate(truth['devices']) if d['identity']==target),None)
            roles=sorted(spec['required']);start=track[0]['capture_ros'];samples=[];last=-1e9
            for i,frame in enumerate(track):
                elapsed=frame['capture_ros']-start
                if elapsed-last<6:continue
                x=candidate_features(track[:i+1],elapsed,anchors)
                static=candidate_features(track[:1],elapsed,anchors)
                if x is not None and static is not None:
                    samples.append(dict(t=elapsed,x=x.tolist(),static=static.tolist(),context=context_features(x).tolist(),static_context=context_features(static).tolist()));last=elapsed
            if samples:episodes.append(dict(seed=row['seed'],scene=truth['scene_name'],roles=roles,target=station,
                directory=str(directory),count=count,anchors=anchors,samples=samples,
                source_sha256=hashlib.sha256(frames.read_bytes()).hexdigest()))
    return episodes,failed


def target_train(episodes,mean,scale,dynamic,penalty):
    xs=[];ys=[];weights=[]
    for e in episodes:
        if e['target'] is None:continue
        for sample in e['samples']:
            xs.append((np.array(sample['x' if dynamic else 'static'])-mean)/scale);ys.append(e['target']);weights.append(1./len(e['samples']))
    x=np.array(xs);y=np.array(ys);weights=np.array(weights);weights/=weights.sum();beta=np.zeros(x.shape[-1]);one=np.eye(x.shape[1])[y]
    lipschitz=np.linalg.norm((np.sqrt(weights)[:,None,None]*x).reshape(-1,x.shape[-1]),ord=2)**2+penalty
    step=1/max(lipschitz,1e-6)
    for _ in range(800):
        prob=softmax(x.dot(beta));grad=np.einsum('n,nk,nkf->f',weights,prob-one,x)+penalty*beta
        delta=step*grad;beta-=delta
        if np.max(np.abs(delta))<1e-7:break
    return beta


def target_evaluate(episodes,beta,mean,scale,dynamic):
    losses=[];briers=[];accuracies=[]
    for e in episodes:
        if e['target'] is None:continue
        x=np.array([s['x' if dynamic else 'static'] for s in e['samples']]);p=softmax(((x-mean)/scale).dot(beta));y=e['target']
        losses.append(float(np.mean(-np.log(np.maximum(1e-9,p[:,y])))))
        briers.append(float(np.mean(np.sum((p-np.eye(e['count'])[y])**2,axis=1))))
        accuracies.append(float(np.mean(p.argmax(axis=1)==y)))
    return dict(scenes=len(losses),log_loss=float(np.mean(losses)),brier=float(np.mean(briers)),accuracy=float(np.mean(accuracies)))


def context_matrix(episodes,classes,mean,scale,dynamic,part):
    x=[];y=[];w=[]
    for e in episodes:
        label=','.join(e['roles']) if part=='roles' else str(e['target'] is not None)
        for sample in e['samples']:
            value=(np.array(sample['context' if dynamic else 'static_context'])-mean)/scale
            x.append(np.concatenate([[1.],value]));y.append(classes.index(label));w.append(1./len(e['samples']))
    return np.array(x),np.array(y),np.array(w)


def fit(batches,output):
    episodes,failed=collect(batches)
    ranked=sorted({e['seed'] for e in episodes},key=lambda s:hashlib.sha256(('relative-program-v11:'+str(s)).encode()).hexdigest())
    calibration=set()
    for scene in sorted({e['scene'] for e in episodes}):
        seeds=[seed for seed in ranked if any(e['seed']==seed and e['scene']==scene for e in episodes)]
        calibration.update(seeds[:max(2,len(seeds)//4)])
    training=[e for e in episodes if e['seed'] not in calibration];validation=[e for e in episodes if e['seed'] in calibration]
    values=np.concatenate([np.array(s['x']) for e in training for s in e['samples']]);mean=values.mean(axis=0);scale=np.maximum(values.std(axis=0),1e-4)
    model=dict(version=11,all_development_scene_seeds=ranked,training_scene_seeds=sorted(e['seed'] for e in training),calibration_scene_seeds=sorted(calibration),
        mean=mean.tolist(),scale=scale.tolist(),anchor_detection_failures=failed,external_only_at_deployment=True,
        local_goal_scorer_shared_across_station_operands=True,no_task_classification_required=True,
        source_frame_sha256=[e['source_sha256'] for e in episodes],
        target={},contexts={},deployment_inputs='Actual external RGB initial anchors, image-only robot pixel history, clock, public scene and menu. NO current task except the labelled task-known diagnostic.')
    for dynamic in (False,True):
        candidates=[]
        for penalty in (.01,.1,1.,10.):
            beta=target_train(training,mean,scale,dynamic,penalty);metrics=target_evaluate(validation,beta,mean,scale,dynamic)
            candidates.append(dict(beta=beta.tolist(),penalty=penalty,calibration=metrics))
        best=min(candidates,key=lambda c:(c['calibration']['log_loss'],-c['penalty']));model['target']['dynamic' if dynamic else 'clock_only']=best
    model['target']['use_dynamic']=model['target']['dynamic']['calibration']['log_loss']<model['target']['clock_only']['calibration']['log_loss']
    for scene in sorted({e['scene'] for e in episodes}):
        tr=[e for e in training if e['scene']==scene];va=[e for e in validation if e['scene']==scene]
        context_values=np.array([s['context'] for e in tr for s in e['samples']]);cm=context_values.mean(axis=0);cs=np.maximum(context_values.std(axis=0),1e-4)
        context=dict(mean=cm.tolist(),scale=cs.tolist())
        for part in ('roles','single_scope'):
            classes=sorted({','.join(e['roles']) if part=='roles' else str(e['target'] is not None) for e in tr+va})
            head=dict(classes=classes)
            for dynamic in (False,True):
                x,y,w=context_matrix(tr,classes,cm,cs,dynamic,part);vx,vy,vw=context_matrix(va,classes,cm,cs,dynamic,part);candidates=[]
                for penalty in (.01,.1,1.,10.):
                    beta=train(x,y,w,len(classes),penalty);metrics=evaluate(beta,vx,vy,vw)
                    candidates.append(dict(beta=beta.tolist(),penalty=penalty,calibration=metrics))
                head['dynamic' if dynamic else 'clock_only']=min(candidates,key=lambda c:(c['calibration']['log_loss'],-c['penalty']))
            head['use_dynamic']=head['dynamic']['calibration']['log_loss']<head['clock_only']['calibration']['log_loss'];context[part]=head
        model['contexts'][scene]=context
    write_json(output,model);return model


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batches',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();model=fit(a.batches,a.output)
    print(json.dumps(dict(training=len(model['training_scene_seeds']),calibration=len(model['calibration_scene_seeds']),
        anchor_failures=len(model['anchor_detection_failures']),target=model['target'],contexts={scene:{part:context[part] for part in ('roles','single_scope')} for scene,context in model['contexts'].items()}),ensure_ascii=False,indent=2))
