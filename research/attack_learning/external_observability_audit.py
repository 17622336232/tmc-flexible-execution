"""Diagnostic: how much can an external RGB prefix identify in OWN data?

This is a finite-program supervised observer, not a trained VLA, an ASR
measurement, or a proof of an information-theoretic ceiling. No teacher text,
onboard geometry, world facts or future external frame enters its features.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from build_local_operator_dataset import canonical
from experiments.task_attack.runtime import write_json


def descriptor(program,target):
    return json.dumps({k:program.get(k) for k in ('roles','scope','filter','reduce','literal')},sort_keys=True)+':'+str(target)


def prefix_features(events,t,perception):
    used=[e for e in events if e['t']<=t]
    if not used:raise ValueError('No actual prefix')
    times=np.array([e['t'] for e in used]);body=np.array([canonical(e['value'],perception) for e in used]);mask=np.array([e['mask'] for e in used])
    # Missing RGB heading is explicitly masked. No future interpolation.
    for c in range(6):
        valid=mask[:,c]
        body[:,c]=np.interp(times,times[valid],body[valid,c]) if valid.any() else 0.
    values=[np.log1p(t),float(t-times[-1]),*body[-1],*mask[-1].astype(float)]
    for ago in (2.,5.,10.,20.,40.,80.):
        recent=times>=max(times[0],t-ago);observations=int(recent.sum())
        if not observations:recent[-1]=True
        b=body[recent]
        values.extend(b.mean(axis=0));values.extend(b.std(axis=0));values.extend(b[-1]-b[0]);values.append(float(observations))
    for ago in np.linspace(0,60,13):
        target=max(times[0],t-ago)
        values.extend([np.interp(target,times,body[:,c]) for c in range(6)])
    # Source ports are static observed geometry, not hidden panel contents.
    for source,a in enumerate(perception['anchors']):
        anchor=np.array(canonical([a['u'],a['v'],0,0,0,0],perception))
        delta=body[:,:2]-anchor[:2];distance=np.linalg.norm(delta,axis=1);j=int(distance.argmin())
        values.extend([*delta[-1],float(distance[-1]),float(distance.min()),*body[j,2:],float(t-times[j]),a['w'],a['h']])
        station=perception['stations'][source]
        for field in ('cracked','temperature'):
            v=station.get(field);known=isinstance(v,(bool,int,float));values.extend([float(known),float(v)/(100 if field=='temperature' else 1) if known else 0.])
    return np.array(values,dtype=np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve development results')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'dataset.json.gz','rt') as f:data=json.load(f)
    episodes=[e for e in data['episodes'] if Path(e['directory']).name=='clean'];programs=json.loads(Path('models/task_attack/integrity_v19/bundle.json').read_text())['programs']
    hypotheses=[(p,t) for p in programs for t in (range(4) if p['scope']=='target' else [None])];keys=[descriptor(p,t) for p,t in hypotheses]
    priors=np.array([1/len(programs)/(4 if p['scope']=='target' else 1) for p,t in hypotheses]);samples=[]
    for e in episodes:
        for t in np.arange(10.,min(301.,e['events'][-1]['t']),5.):
            samples.append(dict(ep=e,t=float(t),x=prefix_features(e['events'],t,e['perception']),y=keys.index(descriptor(e['program'],e['target']))))
    train=[s for s in samples if s['ep']['train']];valid=[s for s in samples if s['ep']['calibration']]
    worlds_train={s['ep']['seed'] for s in train};worlds_valid={s['ep']['seed'] for s in valid}
    if worlds_train&worlds_valid:raise RuntimeError('World leakage')
    counts=np.bincount([s['y'] for s in train],minlength=len(keys));weights=np.array([priors[s['y']]/max(1,counts[s['y']]) for s in train])
    model=ExtraTreesClassifier(n_estimators=160,max_depth=18,min_samples_leaf=16,max_features=.8,n_jobs=2,random_state=161116)
    model.fit(np.array([s['x'] for s in train]),np.array([s['y'] for s in train]),sample_weight=weights)
    raw=model.predict_proba(np.array([s['x'] for s in valid]));prediction=np.tile(priors*.05,(len(valid),1));prediction[:,model.classes_]+=.95*raw
    rows=[]
    for s,pred in zip(valid,prediction):
        y=s['y'];chosen=int(pred.argmax());program,target=hypotheses[chosen]
        true_program,true_target=hypotheses[y]
        rows.append(dict(seed=s['ep']['seed'],episode=s['ep']['key'],family=s['ep']['family'],t=s['t'],true_mass=float(pred[y]),prior_true_mass=float(priors[y]),
            top1=chosen==y,roles_correct=set(program['roles'])==set(true_program['roles']),scope_correct=program['scope']==true_program['scope'],target_correct=target==true_target,
            probabilities=pred.tolist(),truth_label=y))
    epochs=[]
    for start,end in [(10,30),(30,60),(60,120),(120,301)]:
        r=[r for r in rows if start<=r['t']<end]
        counts={e:sum(x['episode']==e for x in r) for e in {x['episode'] for x in r}};w=np.array([1/counts[x['episode']] for x in r]);w/=w.sum()
        epochs.append(dict(start=start,end=end,prefixes=len(r),episodes=len(counts),top1_accuracy=float(w.dot([x['top1'] for x in r])),roles_accuracy=float(w.dot([x['roles_correct'] for x in r])),
            mean_true_mass=float(w.dot([x['true_mass'] for x in r])),mean_prior_true_mass=float(w.dot([x['prior_true_mass'] for x in r])),
            log_loss=float(w.dot([-np.log(x['true_mass']) for x in r])),prior_log_loss=float(w.dot([-np.log(x['prior_true_mass']) for x in r]))))
    summary=dict(status='EXTERNAL OBSERVABILITY DEVELOPMENT DIAGNOSTIC ONLY; no attacks and no ASR or impossibility proof',train_worlds=len(worlds_train),validation_worlds=len(worlds_valid),train_prefixes=len(train),validation_prefixes=len(valid),
        encoding='Retained actual external RGB-derived6D history and actual static RGB context; no true task online',
        epochs=epochs,limitations=['Finite9 public program families, development calibration reused; this is not an independent superiority study.',
            'Classifier errors can reflect insufficient training or model bias. Low accuracy alone does not prove that visible behavior cannot identify a task.',
            'Uses retained RGB for every actual event; a future runtime must expose the same richer stream to every compared method. Frozen405 used a more limited JPEG stream and is not altered.'])
    write_json(a.output/'summary.json',summary);write_json(a.output/'rows.json',dict(rows=rows,hypotheses=hypotheses));print(json.dumps(summary,ensure_ascii=False),flush=True)
    with (a.output/'observer.pkl').open('wb') as f:pickle.dump(dict(model=model,hypotheses=hypotheses,prior=priors,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,protocol=4)


if __name__=='__main__':main()
