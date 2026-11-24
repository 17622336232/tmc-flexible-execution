"""Cross-fit acquisition-quality INPUTS from external RGB descriptors.

Privileged ray/geometry measurements remain targets for the quality head and
offline motion labels. They no longer masquerade as known predictor inputs
when learning READ and DECIDE. Folds keep entire physical worlds together;
calibration worlds are predicted only by a head trained on training worlds.
No declared fact, actual outcome, raw observation or victim rule is changed.
"""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

from local_operator_kernels import ConditionalDistribution,ACTIVE
from local_operator_rollout import quality_input
from primitive_state import CandidateState,Capture,Binding,NamedIdentityClaim
from indexed_operator_features import global_features
from augment_named_local_dataset import assertions_from_features
from experiments.task_attack.runtime import write_json


def fold(seed):return int(hashlib.sha256(('quality_input_oof_v1:'+str(seed)).encode()).hexdigest()[:8],16)%5


def fit_quality(episodes):
    families={f:sum(e['family']==f for e in episodes) for f in {e['family'] for e in episodes}};xx=[];yy=[];weights=[]
    for ep in episodes:
        for ex in ep['examples']:
            for source in range(4):
                for j in range(2):
                    q=ex['acquisition_quality'][source*2+j]
                    if q is None:continue
                    xx.append(quality_input(ex['event'],ep['perception'],source,j));yy.append([q]);weights.append(1/families[ep['family']]/len(ep['examples'])/8)
    head=ConditionalDistribution(trees=64,leaf=3,depth=16,seed=16119).fit(np.array(xx,dtype=np.float32),np.array(yy),np.array(weights));head.forest.n_jobs=1
    return head


def predict(episodes,head,predictions,diagnostics):
    for ep in episodes:
        queries=[quality_input(ex['event'],ep['perception'],source,j) for ex in ep['examples'] for source in range(4) for j in range(2)]
        values=np.clip(head.mean_outcome(np.array(queries))[:,0],0,1).reshape(len(ep['examples']),4,2)
        for ex,pred in zip(ep['examples'],values):
            key=(ep['directory'],ex['t'])
            if key in predictions:raise RuntimeError('Repeated capture time needs explicit grouping')
            predictions[key]=pred
            for source in range(4):
                for j in range(2):
                    actual=ex['acquisition_quality'][source*2+j]
                    if actual is not None:diagnostics.append(dict(train=ep['train'],actual=float(actual),predicted=float(pred[source,j])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve derived versions')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'dataset.json.gz','rt') as f:data=json.load(f)
    train=[e for e in data['episodes'] if e['train']];valid=[e for e in data['episodes'] if e['calibration']];predictions={};diagnostics=[];folds=[]
    for k in range(5):
        fit=[e for e in train if fold(e['seed'])!=k];held=[e for e in train if fold(e['seed'])==k]
        if {e['seed'] for e in fit}&{e['seed'] for e in held}:raise RuntimeError('World leakage')
        predict(held,fit_quality(fit),predictions,diagnostics);folds.append(dict(fold=k,fit_worlds=sorted({e['seed'] for e in fit}),predicted_worlds=sorted({e['seed'] for e in held})));print('OOF QUALITY fold',k,len(held),flush=True)
    predict(valid,fit_quality(train),predictions,diagnostics);parameters=json.loads(Path('models/task_attack/integrity_v19/bundle.json').read_text())['parameters'];changed=0
    for ep in data['episodes']:
        for ex in ep['examples']:
            caps=[]
            for row in ex['captures']:
                cap=Capture(**row)
                if cap.field in ACTIVE:
                    if cap.time>ex['t']+1e-6:raise RuntimeError('Do not use a future capture')
                    pred=predictions.get((ep['directory'],cap.time))
                    if pred is None:raise RuntimeError('Capture has no actual external teacher descriptor')
                    cap=replace(cap,view_quality=float(pred[cap.source,ACTIVE.index(cap.field)]),quality_measured=True)
                caps.append(cap)
            ex['captures']=[vars(c) for c in caps]
            for feature_key,binding_key,name_key in (('before_features','before_bindings','before_named_identities'),('after_features','after_bindings','after_named_identities')):
                named=name_key in ex
                state=CandidateState((),(),tuple(caps),tuple(Binding(**b) for b in ex[binding_key]),assertions_from_features(ex[feature_key]),named_identities=tuple(NamedIdentityClaim(**c) for c in ex.get(name_key,[])))
                ex[feature_key]=global_features(ep['program'],ep['target'],state,ep['perception'],ex['event'],parameters,ex['t'],named=named)
            ex['quality_input_version']='world_OOF_external_RGB_v1';changed+=1
    summary=[]
    for train_flag in (True,False):
        rows=[r for r in diagnostics if r['train']==train_flag];err=np.array([r['predicted']-r['actual'] for r in rows]);bins=[]
        for lo,hi in ((0.,.05),(.05,.25),(.25,.75),(.75,1.000001)):
            group=[r for r in rows if lo<=r['actual']<hi]
            bins.append(dict(actual_range=[lo,hi],count=len(group),mean_predicted=float(np.mean([r['predicted'] for r in group])) if group else None,mean_actual=float(np.mean([r['actual'] for r in group])) if group else None))
        summary.append(dict(training=train_flag,values=len(rows),mae=float(np.mean(abs(err))),rmse=float(np.sqrt(np.mean(err**2))),bins=bins))
    data['quality_input_crossfit']=dict(status='Derived deployment-consistent INPUTS; not an ASR result or proven improvement',parent_sha256=hashlib.sha256((a.dataset/'dataset.json.gz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fold_rule='SHA256 quality_input_oof_v1:world_seed modulo5',folds=folds,
        actual_ray_acquisition_quality_labels_preserved=True,raw_labels_reports_and_outcomes_unchanged=True,calibration_worlds_never_in_quality_training=True,changed_steps=changed,diagnostics=summary)
    with gzip.open(a.output/'dataset.json.gz','wt') as f:json.dump(data,f,separators=(',',':'))
    write_json(a.output/'summary.json',data['quality_input_crossfit']);print('DEPLOYMENT QUALITY INPUTS',json.dumps(summary),flush=True)


if __name__=='__main__':main()
