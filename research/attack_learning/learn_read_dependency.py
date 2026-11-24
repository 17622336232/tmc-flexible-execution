"""Amortized task evidence from predicted, actually observed read opportunities.

This is an APPROXIMATE inference alternative to the misspecified full-path
likelihood. It does not prescribe any victim action. Structured candidate
arguments are public hypotheses; only an offline training label is the true
task. Acquisition estimates use world-cross-fitted quality models trained on
OWN data. No actual ray label, onboard image/log or future frame is a feature.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from experiments.task_attack.runtime import write_json
from experiments.task_attack_value.value_features import program_features
from external_observability_audit import descriptor
from local_operator_rollout import quality_input
from replace_local_quality_features import fit_quality,fold


def quality_history(events,perception,head):
    queries=[quality_input(e['value'],perception,i,j) for e in events for i in range(4) for j in range(2)]
    return np.clip(head.mean_outcome(np.array(queries))[:,0],0,1).reshape(len(events),4,2)


def context(events,qualities,t,perception):
    n=sum(e['t']<=t for e in events)
    if not n:raise ValueError('Only a nonempty actual prefix can be encoded')
    used=events[:n];times=np.array([e['t'] for e in used]);qs=qualities[:n]
    dt=np.diff(np.r_[times[0],times]);body=np.array(used[-1]['value']);ports=[]
    for i,a in enumerate(perception['anchors']):
        x=[]
        for j in range(2):
            q=qs[:,i,j];peak=int(q.argmax())
            x.extend([q[-1],q.max(),float(dt.dot(q)/max(1.,times[-1]-times[0])),
                float(dt.dot(q*q)/max(1.,times[-1]-times[0])),np.log1p(t-times[peak])])
            for seconds in (5.,15.,40.):
                recent=times>=max(times[0],t-seconds);x.append(float(q[recent].mean()) if recent.any() else float(q[-1]))
        station=perception['stations'][i]
        for field,unit in (('cracked',1.),('temperature',100.)):
            value=station.get(field);known=isinstance(value,(bool,int,float));x.extend([float(known),float(value)/unit if known else 0.])
        w=max(.005,a['w']);x.extend([(body[0]-a['u'])/w,(body[1]-a['v'])/w,body[2]/w,body[3]/w,*body[4:6]])
        ports.append(np.array(x))
    return np.array(ports),np.r_[np.log1p(t),t-times[-1],used[-1]['mask']]


def goal_inputs(ports,clock,hypotheses):
    rows=[]
    for program,target in hypotheses:
        p=program_features((program,target),0);p=p[:3]+p[4:]
        # No family ID. The target is represented by its observed source port.
        ref=ports[target] if target is not None else np.zeros(ports.shape[1])
        other=np.delete(ports,target,axis=0).mean(axis=0) if target is not None else ports.mean(axis=0)
        rows.append(np.r_[clock,ports.ravel(),p,float(target is not None),ref,other])
    return np.array(rows,dtype=np.float32)


def probabilities(head,x,prior,temperature=1.,blend=0.):
    scores=np.clip(head.predict(x),1e-9,1.)
    scores=np.exp(np.log(scores)/temperature-np.max(np.log(scores)/temperature));scores/=scores.sum()
    return (1-blend)*scores+blend*prior


def equal_world_loss(rows,probabilities_):
    worlds={r['seed'] for r in rows};values=[]
    for world in worlds:
        indices=[i for i,r in enumerate(rows) if r['seed']==world];episodes={rows[i]['episode'] for i in indices};episode_values=[]
        for ep in episodes:
            ids=[i for i in indices if rows[i]['episode']==ep]
            episode_values.append(float(np.mean([-np.log(max(1e-300,probabilities_[i][rows[i]['y']])) for i in ids])))
        values.append(np.mean(episode_values))
    return float(np.mean(values))


def calibration_half(seed):return int(hashlib.sha256(('read_dependency_goalcal_v1:'+str(seed)).encode()).hexdigest()[:8],16)%2


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve versions')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'dataset.json.gz','rt') as f:data=json.load(f)
    with (a.model/'local_kernels.pkl').open('rb') as f:models=pickle.load(f)
    models['quality']['distribution'].forest.n_jobs=1
    train_all=[e for e in data['episodes'] if e['train']];clean=[e for e in data['episodes'] if Path(e['directory']).name=='clean'];qh={}
    for k in range(5):
        head=fit_quality([e for e in train_all if fold(e['seed'])!=k])
        held=[e for e in clean if e['train'] and fold(e['seed'])==k]
        for e in held:qh[e['key']]=quality_history(e['events'],e['perception'],head)
        print('READ DEPENDENCY OOF',k,'episodes',len(held),flush=True)
    for e in clean:
        if e['calibration']:qh[e['key']]=quality_history(e['events'],e['perception'],models['quality']['distribution'])
    hypotheses=[(p,z) for p in models['programs'] for z in (range(4) if p['scope']=='target' else [None])]
    keys=[descriptor(p,z) for p,z in hypotheses];prior=np.array([1/len(models['programs'])/(4 if p['scope']=='target' else 1) for p,z in hypotheses]);rows=[]
    counts={f:sum(e['train'] and e['family']==f for e in clean) for f in {e['family'] for e in clean}}
    for e in clean:
        for t in np.arange(10.,min(301.,e['events'][-1]['t']),5.):
            ports,clock=context(e['events'],qh[e['key']],t,e['perception']);x=goal_inputs(ports,clock,hypotheses)
            rows.append(dict(seed=e['seed'],episode=e['key'],family=e['family'],train=e['train'],calibration=e['calibration'],t=float(t),x=x,y=keys.index(descriptor(e['program'],e['target'])),weight=1/max(1,counts[e['family']])))
    train=[r for r in rows if r['train']];valid=[r for r in rows if r['calibration']];cal=[r for r in valid if calibration_half(r['seed'])==0];test=[r for r in valid if calibration_half(r['seed'])==1]
    if not cal or not test:raise ValueError('Need independent development calibration halves')
    x=np.vstack([r['x'] for r in train]);y=np.concatenate([np.eye(len(hypotheses))[r['y']] for r in train]);weights=np.repeat([r['weight'] for r in train],len(hypotheses));choices=[]
    write_json(a.output/'contract.json',dict(status='Approximate read-dependency inference DEVELOPMENT; no actual attacks or ASR',dataset_sha256=hashlib.sha256((a.dataset/'dataset.json.gz').read_bytes()).hexdigest(),
        parent_model_sha256=hashlib.sha256((a.model/'local_kernels.pkl').read_bytes()).hexdigest(),training_worlds=sorted({r['seed'] for r in train}),tuning_worlds=sorted({r['seed'] for r in cal}),diagnostic_worlds=sorted({r['seed'] for r in test}),
        split='Existing world split, then SHA256 read_dependency_goalcal_v1:seed modulo2 within calibration; fixed before any head results',
        feature_boundary='Only actual external RGB-derived events, c0, clock and predicted field opportunities. Quality inputs train-world OOF. No teacher instruction, declared log, true world, future frame or actual ray labels online.',
        caveat='Hypothesis vocabulary remains finite. Amortized discriminative goal weights are an inference approximation, not exact Bayesian filtering under the generative K/D/E. Diagnostic half reused in earlier development; not a formal untouched test.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for depth,leaf in ((8,64),(12,64),(12,16)):
        head=ExtraTreesRegressor(n_estimators=96,max_depth=depth,min_samples_leaf=leaf,max_features=.8,n_jobs=2,random_state=161120)
        head.fit(x,y,sample_weight=weights);raw=head.predict(np.vstack([r['x'] for r in cal])).reshape(len(cal),len(hypotheses));variants=[]
        for temperature in (.25,.5,1.,2.,4.):
            logp=np.log(np.clip(raw,1e-9,1))/temperature;p0=np.exp(logp-logp.max(axis=1,keepdims=True));p0/=p0.sum(axis=1,keepdims=True)
            for blend in (0.,.1,.25,.5,1.):
                ps=(1-blend)*p0+blend*prior;variants.append((equal_world_loss(cal,ps),temperature,blend))
        loss,temperature,blend=min(variants);choices.append((loss,head,temperature,blend,dict(depth=depth,leaf=leaf)))
        print('READ DEPENDENCY FIT',depth,leaf,'cal_logloss',loss,'temperature',temperature,'prior_blend',blend,flush=True)
    loss,head,temperature,blend,configuration=min(choices,key=lambda x:x[0]);head.n_jobs=1
    output_rows=[]
    for r in valid:
        ps=probabilities(head,r['x'],prior,temperature,blend);chosen=int(ps.argmax());actual,target=hypotheses[r['y']];predicted,predtarget=hypotheses[chosen]
        output_rows.append(dict(seed=r['seed'],episode=r['episode'],family=r['family'],t=r['t'],tuning=calibration_half(r['seed'])==0,y=r['y'],probabilities=ps.tolist(),true_mass=float(ps[r['y']]),prior_true_mass=float(prior[r['y']]),top1=chosen==r['y'],roles_correct=set(actual['roles'])==set(predicted['roles']),target_correct=target==predtarget))
    summaries=[]
    for tuning in (True,False):
        for lo,hi in ((10,30),(30,60),(60,120),(120,301)):
            selected=[r for r in output_rows if r['tuning']==tuning and lo<=r['t']<hi]
            if not selected:continue
            episodes={r['episode'] for r in selected};w=np.array([1/sum(s['episode']==r['episode'] for s in selected) for r in selected]);w/=w.sum()
            summaries.append(dict(tuning=tuning,start=lo,end=hi,episodes=len(episodes),top1=float(w.dot([r['top1'] for r in selected])),roles_accuracy=float(w.dot([r['roles_correct'] for r in selected])),mean_true_mass=float(w.dot([r['true_mass'] for r in selected])),log_loss=float(w.dot([-np.log(r['true_mass']) for r in selected])),prior_log_loss=float(w.dot([-np.log(r['prior_true_mass']) for r in selected]))))
    with (a.output/'read_dependency.pkl').open('wb') as f:pickle.dump(dict(head=head,hypotheses=hypotheses,prior=prior,temperature=temperature,blend=blend,configuration=configuration,quality=models['quality']['distribution']),f,protocol=4)
    write_json(a.output/'rows.json',dict(rows=output_rows,hypotheses=hypotheses))
    write_json(a.output/'summary.json',dict(status='Read-dependency inference diagnostic only; no actual ASR',configuration=configuration,temperature=temperature,prior_blend=blend,epochs=summaries,training_prefixes=len(train),tuning_prefixes=len(cal),diagnostic_prefixes=len(test),actual_attacks_executed=0,superiority_proven=False))
    print('READ DEPENDENCY COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
