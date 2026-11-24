"""Small learned external-view next-observation timing forecaster.

Training labels use recorded training robot image timestamps. Deployment uses
ONLY external RGB pixel tracks. No task/family/scene coordinates are features.
This is a fitted timing component, not the paper's complete K/D/E framework.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .runtime import write_json


FEATURES=('u','v','box_width','box_height','du_dt','dv_dt','speed','path_length',
          'observed_span','stationary_fraction','log_elapsed')


def fit_behavior_onset(episodes):
    """Learn when external motion contains instruction-dependent execution.

    Training-only labels are the first autonomous move/approach dispatch, not
    a task name, answer, privileged deployment phase or a success label. Clock
    elapsed is excluded: the deployed scheduler must use observed motion.
    """
    if len(episodes)<4:return None
    training,validation=episodes[:-2],episodes[-2:]
    x=np.concatenate([a for a,b in training]);y=np.concatenate([b for a,b in training])
    if not 0<float(y.mean())<1:return None
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-4)
    z=np.column_stack((np.ones(len(x)),(x-mean)/scale));beta=np.zeros(z.shape[1])
    penalty=3*np.eye(len(beta));penalty[0,0]=0.
    # Long post-onset hovers must not swamp the short pre-onset segment.
    # Balanced logistic loss is shared across tasks and learned from labels.
    weights=np.where(y>0,.5/y.mean(),.5/(1-y.mean()))
    for _ in range(35):
        p=1/(1+np.exp(-np.clip(z.dot(beta),-30,30)));w=p*(1-p)*weights
        step=np.linalg.solve(z.T.dot(w[:,None]*z)+penalty+1e-8*np.eye(len(beta)),
            z.T.dot((p-y)*weights)+penalty.dot(beta))
        beta-=step
        if np.max(np.abs(step))<1e-6:break
    squared=[];constant=[]
    for vx,vy in validation:
        pro=1/(1+np.exp(-np.clip(np.column_stack((np.ones(len(vx)),(vx-mean)/scale)).dot(beta),-30,30)))
        vw=np.where(vy>0,.5/max(1e-8,vy.mean()),.5/max(1e-8,1-vy.mean()))
        vw/=max(1e-8,float(vw.mean()))
        squared.extend(vw*(pro-vy)**2);constant.extend(vw*(.5-vy)**2)
    brier=float(np.mean(squared));prior_brier=float(np.mean(constant))
    return dict(features=FEATURES[:-1],mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),
        prior_probability=.5,training_class_fraction=float(y.mean()),class_balanced_loss=True,
        training_frames=len(x),training_episodes=len(training),
        calibration_episodes=len(validation),calibration_brier=brier,prior_brier=prior_brier,
        use_conditional=brier<prior_brier,
        label='First autonomous move/approach dispatched; labels only used offline',
        deployment='External pixel motion only, no task semantics, elapsed-clock feature or private tool event')


def features(history,elapsed):
    entries=[x for x in history if x.get('robot_pixels')]
    if not entries:return None
    last=entries[-1]['robot_pixels'];u,v=last['centroid'];box=last['bbox']
    recent=entries[-12:];velocities=[];length=0.
    for first,second in zip(recent,recent[1:]):
        dt=second['capture_ros']-first['capture_ros']
        if not 0<dt<30:continue
        delta=np.array(second['robot_pixels']['centroid'])-first['robot_pixels']['centroid']
        velocities.append(delta/dt);length+=float(np.linalg.norm(delta))
    velocity=np.mean(velocities[-3:],axis=0) if velocities else np.zeros(2)
    speeds=[float(np.linalg.norm(x)) for x in velocities]
    return np.array([u,v,box[2]-box[0],box[3]-box[1],*velocity,float(np.linalg.norm(velocity)),
        length,max(0.,recent[-1]['capture_ros']-recent[0]['capture_ros']),
        float(np.mean(np.array(speeds)<.002)) if speeds else 1.,math.log1p(max(0.,elapsed))])


def fit(batch,output):
    batch=Path(batch);manifest=json.loads((batch/'manifest.json').read_text())
    episodes=[];source_hashes=[];seeds=set();latencies=[];onset_episodes=[];onset_source_hashes=[]
    for row in manifest['rows']:
        observer=batch/row['directory']/'private/observer/model_calls.jsonl'
        if observer.exists():
            for line in observer.read_text().splitlines():
                call=json.loads(line)
                if call.get('output') is not None and isinstance(call.get('wall_seconds'),(int,float)):
                    latencies.append(call['wall_seconds'])
        if row['policy']!='clean':continue
        directory=batch/row['directory'];frames=directory/'public/external_frames.jsonl'
        images=sorted((directory/'private/robot').glob('onboard_*.json'))
        if not frames.exists() or not images:continue
        track=[json.loads(line) for line in frames.read_text().splitlines() if line.strip()]
        reads=sorted(json.loads(p.read_text())['ros_stamp'] for p in images)
        start=track[0]['capture_ros'];xs=[];ys=[];future_pixels=[]
        visible=[f for f in track if f.get('robot_pixels')]
        read_pixels=[]
        for stamp in reads:
            nearest=min(visible,key=lambda f:abs(f['capture_ros']-stamp)) if visible else None
            read_pixels.append(nearest['robot_pixels']['centroid'] if nearest and abs(nearest['capture_ros']-stamp)<=2. else [float('nan')]*2)
        for i,frame in enumerate(track):
            j=bisect.bisect_right(reads,frame['capture_ros'])
            if j>=len(reads):continue
            x=features(track[max(0,i-35):i+1],frame['capture_ros']-start)
            dt=reads[j]-frame['capture_ros']
            if x is not None and 0<dt<=120:
                xs.append(x);ys.append(math.log1p(dt));future_pixels.append(read_pixels[j])
        if xs:
            episodes.append((np.array(xs),np.array(ys),np.array(future_pixels)))
            seeds.add(row['seed'])
            source_hashes.append(hashlib.sha256(frames.read_bytes()).hexdigest())
        decisions=directory/'private/robot/decisions.jsonl'
        if decisions.exists():
            dispatched=[d['wall'] for d in (json.loads(line) for line in decisions.read_text().splitlines() if line.strip())
                if d.get('decision',{}).get('action',{}).get('type') in ('move','approach')]
            # No-motion completions remain negative observations. Dropping
            # them would teach a biased scheduler that every task must move.
            onset=min(dispatched) if dispatched else float('inf');ox=[];oy=[]
            for i,frame in enumerate(track):
                if frame['capture_ros']-start>120:break
                value=features(track[max(0,i-35):i+1],frame['capture_ros']-start)
                if value is not None:
                    ox.append(value[:-1])
                    # ROS simulation seconds and wall timestamps are NOT
                    # interchangeable. Frame arrivals and dispatch share
                    # wall time, even for historical Gazebo data.
                    oy.append(float(frame['arrival_wall']>=onset))
            if ox:
                onset_episodes.append((np.array(ox),np.array(oy)))
                onset_source_hashes.append(hashlib.sha256(decisions.read_bytes()).hexdigest())
    if len(episodes)<4:raise ValueError('Need at least four independent training episodes for timing calibration')
    # Deterministic episode-level holdout; neighboring frames never leak across
    # the calibration split. Task outcomes are not used to select episodes.
    train=episodes[:-2];validation=episodes[-2:]
    x=np.concatenate([a for a,b,c in train]);y=np.concatenate([b for a,b,c in train])
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-4)
    z=np.column_stack((np.ones(len(x)),(x-mean)/scale))
    penalty=np.eye(z.shape[1]);penalty[0,0]=0.
    beta=np.linalg.solve(z.T.dot(z)+10*penalty,z.T.dot(y))
    validation_errors=[];prior_errors=[]
    prior=float(np.median(y))
    for vx,vy,vc in validation:
        predicted=np.column_stack((np.ones(len(vx)),(vx-mean)/scale)).dot(beta)
        validation_errors.extend((np.expm1(np.clip(predicted,0,math.log1p(120)))-np.expm1(vy)).tolist())
        prior_errors.extend((np.expm1(prior)-np.expm1(vy)).tolist())
    residual=y-z.dot(beta);sigma=max(.1,float(np.std(residual)))
    # A fast transition component forecasts the EXTERNALLY OBSERVED robot
    # pixel position at its next camera read. Actual task/semantic labels are
    # unnecessary, and no simulator coordinate is used as an input or output.
    targets=np.concatenate([c for a,b,c in train]);mask=np.isfinite(targets).all(axis=1)
    future_model=None
    if mask.sum()>=30:
        design=z[mask];target=targets[mask]
        coefficients=np.linalg.solve(design.T.dot(design)+10*penalty,design.T.dot(target))
        prior_pixel=target.mean(axis=0);fit_residual=target-design.dot(coefficients)
        errors=[];constant=[]
        for vx,vy,vc in validation:
            valid=np.isfinite(vc).all(axis=1)
            if not valid.any():continue
            vz=np.column_stack((np.ones(valid.sum()),(vx[valid]-mean)/scale))
            errors.extend((vc[valid]-np.clip(vz.dot(coefficients),0,1)).tolist())
            constant.extend((vc[valid]-prior_pixel).tolist())
        if errors:
            error=np.array(errors);base=np.array(constant)
            future_model=dict(coefficients=coefficients.tolist(),prior=prior_pixel.tolist(),
                residual_std=np.maximum(.005,np.maximum(fit_residual.std(axis=0),error.std(axis=0))).tolist(),
                validation_mean_pixel_distance=float(np.linalg.norm(error,axis=1).mean()),
                prior_mean_pixel_distance=float(np.linalg.norm(base,axis=1).mean()),
                training_frames=int(mask.sum()),
                use_conditional=bool(np.mean(error**2)<np.mean(base**2)))
    model=dict(version=1,features=FEATURES,mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),
        log_residual_std=sigma,prior_log_seconds=prior,
        training_episodes=len(train),calibration_episodes=len(validation),training_frames=len(x),
        training_scene_seeds=sorted(seeds),source_frame_log_sha256=source_hashes,
        training_manifest_sha256=hashlib.sha256((batch/'manifest.json').read_bytes()).hexdigest(),
        external_only_at_deployment=True,task_semantics_used_as_features=False,
        latency_prior_seconds=float(np.median(latencies)) if latencies else None,
        latency_prior_requests=len(latencies),
        future_external_read_position=future_model,
        external_behavior_onset=fit_behavior_onset(onset_episodes),
        behavior_training_decision_log_sha256=onset_source_hashes,
        validation_mae_seconds=float(np.mean(np.abs(validation_errors))),
        unconditional_prior_mae_seconds=float(np.mean(np.abs(prior_errors))))
    # Preserve a genuine negative calibration: if the fitted predictor fails to
    # beat a constant prior, use that prior and expose the failed validation.
    # This is model selection on training holdout, not a hand-written task rule.
    model['use_conditional_predictor']=model['validation_mae_seconds']<model['unconditional_prior_mae_seconds']
    write_json(output,model);return model


def predict(model,history,elapsed,latency):
    x=features(history,elapsed)
    conditional=bool(model.get('use_conditional_predictor') and x is not None)
    value=float(np.r_[1.,(x-np.array(model['mean']))/np.array(model['scale'])].dot(model['beta'])) if conditional else model['prior_log_seconds']
    value=float(np.clip(value,0,math.log1p(120)))
    std=model['log_residual_std'];delay=max(0.,(model.get('latency_prior_seconds') or 0.) if latency is None else latency)
    before=.5*(1+math.erf((math.log1p(delay)-value)/(std*math.sqrt(2))))
    result=dict(estimated_seconds_to_next_camera_read=math.expm1(value),
        assumed_model_latency_seconds=delay,
        probability_read_before_expected_model_return=before,
        conditional_external_motion_model=conditional,
        calibration_mae_seconds=model['validation_mae_seconds'],
        caution='Learned timing uncertainty, not observed private progress, cache contents or a dispatch gate')
    onset=model.get('external_behavior_onset')
    if onset and onset.get('use_conditional') and x is not None:
        logit=float(np.r_[1.,(x[:-1]-np.array(onset['mean']))/np.array(onset['scale'])].dot(onset['beta']))
        result['behavior_onset_probability']=1/(1+math.exp(-max(-30,min(30,logit))))
        result['behavior_onset_uses_external_motion_only']=True
    future=model.get('future_external_read_position')
    if future:
        conditional_position=bool(future['use_conditional'] and x is not None)
        center=np.r_[1.,(x-np.array(model['mean']))/np.array(model['scale'])].dot(future['coefficients']) if conditional_position else np.array(future['prior'])
        result['next_read_robot_pixel_distribution']=dict(mean=np.clip(center,0,1).tolist(),
            std=future['residual_std'],conditional_external_motion_model=conditional_position,
            calibration_mean_distance=future['validation_mean_pixel_distance'],
            meaning='Robot marker position in fixed external camera when its NEXT image is acquired; not target/object world coordinates')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-batch',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();model=fit(a.training_batch,a.output)
    print(json.dumps({k:model[k] for k in ('training_episodes','calibration_episodes','training_frames',
        'validation_mae_seconds','unconditional_prior_mae_seconds','use_conditional_predictor')},indent=2))


if __name__=='__main__':main()
