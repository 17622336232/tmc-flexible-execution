"""Learn joint camera-read continuations from development episodes only.

Whole scenes are split, not frames. Terminal prefixes remain in training.
Online features contain only external RGB marker tracks, clock and public scene.
An empirical read-position proxy cannot establish carrier visibility or adoption.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.task_attack.event_timing import FEATURES, features
from experiments.task_attack.runtime import write_json


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def extract(batch):
    root=Path(batch);manifest=json.loads((root/'manifest.json').read_text());episodes=[]
    for row in manifest['rows']:
        if row['policy']!='clean':continue
        directory=root/row['directory'];path=directory/'public/external_frames.jsonl'
        if not path.exists():continue
        track=lines(path);visible=[f for f in track if f.get('robot_pixels')]
        if not visible:continue
        start=track[0]['capture_ros'];reads=[];sources=[path]
        for file in sorted((directory/'private/robot').glob('onboard_*.json')):
            stamp=json.loads(file.read_text())['ros_stamp']
            nearest=min(visible,key=lambda f:abs(f['capture_ros']-stamp))
            if abs(nearest['capture_ros']-stamp)<=2:
                reads.append([stamp-start,*nearest['robot_pixels']['centroid']]);sources.append(file)
        if not reads:continue
        reads.sort();samples=[];last=-math.inf
        for i,frame in enumerate(track):
            t=frame['capture_ros']-start
            if t-last<3:continue
            x=features(track[max(0,i-35):i+1],t)
            if x is not None:
                samples.append([t,*x.tolist()]);last=t
        if not samples:continue
        episodes.append(dict(seed=row['seed'],scene=json.loads((directory/'manifest.json').read_text())['scene_name'],
            samples=samples,reads=reads,end=track[-1]['capture_ros']-start,
            source_sha256=[hashlib.sha256(f.read_bytes()).hexdigest() for f in sources]))
    return episodes


def nearest(model, x, scene, dynamic=True, exclude_seed=None):
    mean=np.array(model['mean']);scale=np.array(model['scale']);query=(np.array(x)-mean)/scale
    found=[]
    for episode in model['episodes']:
        if episode['scene']!=scene or episode['seed']==exclude_seed:continue
        samples=np.array(episode['samples']);z=(samples[:,1:]-mean)/scale
        # Static access has no hidden current pixels or motion, even if the
        # caller happens to have them. Match only the public elapsed clock.
        residual=z-query
        dist=np.mean(residual**2,axis=1) if dynamic else residual[:,-1]**2
        ix=int(np.argmin(dist));found.append((float(dist[ix]),episode,float(samples[ix,0])))
    found.sort(key=lambda v:(v[0],v[1]['seed']));found=found[:model['neighbors']]
    if not found:raise ValueError('No development continuations for this public scene')
    ds=np.array([r[0] for r in found]);weights=np.exp(-.5*(ds-ds.min()))
    weights/=weights.sum()
    return [dict(weight=float(w),episode=e,time=t,distance=d) for w,(d,e,t) in zip(weights,found)]


def forecast(model,history,elapsed,scene,dynamic=True):
    x=features(history,elapsed)
    if x is None:
        # The initial RGB marker can be temporarily occluded; a clock-only
        # prior is legitimate. Never replace it with a simulator pose.
        x=np.array(model['mean']);x[-1]=math.log1p(max(0,elapsed));dynamic=False
    result=[]
    for entry in nearest(model,x,scene,dynamic):
        e=entry['episode'];t=entry['time'];past=[r for r in e['reads'] if r[0]<=t]
        future=[r for r in e['reads'] if r[0]>t]
        result.append(dict(weight=entry['weight'],training_seed=e['seed'],distance=entry['distance'],
            future=[[r[0]-t,*r[1:]] for r in future],past=[[r[0]-t,*r[1:]] for r in past]))
    return dict(rollouts=result,dynamic=dynamic,pixel_std=model['pixel_std'],
        probability_any_future_read=sum(r['weight'] for r in result if r['future']),
        deployment_inputs='external pixel tracks, public scene and clock only',
        caution='Empirical camera-position continuations; not observed target visibility, cache identity or adoption')


def region_mass(point,region,std):
    if region is None:return 1.
    value=1.
    for axis in (0,1):
        cdf=lambda a:.5*(1+math.erf((a-point[axis])/(std[axis]*math.sqrt(2))))
        value*=max(0.,cdf(region[axis+2])-cdf(region[axis]))
    return value


def summarize(prediction,region):
    mass=next_mass=cache=0.;alive=prediction['probability_any_future_read'];std=prediction['pixel_std']
    for r in prediction['rollouts']:
        if r['future']:
            next_mass+=r['weight']*region_mass(r['future'][0][1:],region,std)
        # Repeated closely spaced reads are correlated. Taking their maximum
        # avoids pretending that ten pictures of one hover are ten independent
        # opportunities to have cached this region.
        cache+=r['weight']*max([region_mass(p[1:],region,std) for p in r['past']] or [0.])
        mass+=r['weight']*max([region_mass(p[1:],region,std) for p in r['future']] or [0.])
    return dict(next_region_probability_given_read=next_mass/alive if alive else 0.,
        any_future_region_probability=mass,cached_region_probability=cache)


def evaluate(model,heldout,dynamic):
    briers=[];distances=[];counts=[]
    for episode in heldout:
        b=[];d=[]
        for sample in episode['samples']:
            t=sample[0];future=[r for r in episode['reads'] if r[0]>t]
            ns=nearest(model,sample[1:],episode['scene'],dynamic)
            available=[(a['weight'],next((r for r in a['episode']['reads'] if r[0]>a['time']),None)) for a in ns]
            p=sum(w for w,r in available if r is not None);b.append((p-bool(future))**2)
            if future and p>0:
                pred=sum(w*np.array(r[1:]) for w,r in available if r is not None)/p
                d.append(np.array(future[0][1:])-pred)
        if b:briers.append(float(np.mean(b)))
        distances.extend(d);counts.append(len(b))
    error=np.array(distances)
    return dict(episodes=len(briers),prefixes=sum(counts),mean_episode_future_read_brier=float(np.mean(briers)),
        next_pixel_distance=float(np.mean(np.linalg.norm(error,axis=1))),
        residual_std=np.maximum(.005,np.std(error,axis=0)).tolist())


def fit(batch,output):
    episodes=extract(batch)
    if len(episodes)<15:raise ValueError('Need at least 15 independent scene episodes')
    # Stable random split by scene seed, independent of outcome and task name.
    ranked=sorted(episodes,key=lambda e:hashlib.sha256(('continuation-v8:'+str(e['seed'])).encode()).hexdigest())
    heldout=ranked[:max(9,len(ranked)//4)];train=ranked[len(heldout):]
    values=np.concatenate([np.array(e['samples'])[:,1:] for e in train])
    model=dict(version=1,features=FEATURES,mean=values.mean(axis=0).tolist(),scale=np.maximum(values.std(axis=0),1e-4).tolist(),
        episodes=train,neighbors=5,pixel_std=[.05,.05],external_only_at_deployment=True,
        training_scene_seeds=sorted(e['seed'] for e in train),calibration_scene_seeds=sorted(e['seed'] for e in heldout),
        all_development_scene_seeds=sorted(e['seed'] for e in episodes),
        task_or_outcome_features=False,terminal_prefixes_retained=True,
        meaning='Nonparametric joint camera-event/timing surrogate, not full compositional paper K/D/E')
    choices=[]
    for k in (3,5,9):
        model['neighbors']=k;dynamic=evaluate(model,heldout,True);static=evaluate(model,heldout,False)
        # Both evaluation terms measure camera prediction, never attack reward.
        score=dynamic['mean_episode_future_read_brier']+dynamic['next_pixel_distance']
        choices.append(dict(neighbors=k,dynamic=dynamic,clock_only=static,prediction_score=score))
    best=min(choices,key=lambda x:(x['prediction_score'],x['neighbors']));model['neighbors']=best['neighbors']
    model['pixel_std']=best['dynamic']['residual_std'];model['calibration']=dict(candidates=choices,selected=best,
        split='whole scenes, 25% held out; no final-report correctness used to choose parameters')
    write_json(output,model);return model


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--development-batch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=fit(args.development_batch,args.output)
    print(json.dumps(dict(training=len(result['episodes']),calibration=result['calibration']),ensure_ascii=False,indent=2))
