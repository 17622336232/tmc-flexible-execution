"""Versioned declared-name state augmentation; no label/truth substitution."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from primitive_state import CandidateState,Fact,Binding,Capture,Assertion,NamedIdentityClaim
from epistemic_operator_features import global_features
from experiments.task_attack.runtime import write_json


def assertions_from_features(vector):
    width=len(vector)//4;count=round(math.expm1(vector[-2]));sources=[i for i in range(4) if vector[width*(i+1)-1]>.5]
    if count<len(sources):raise ValueError('Inconsistent ledger counters')
    records=[Assertion(i,'_count_only',None,0) for i in sources]
    if sources:records += [Assertion(sources[0],'_count_only',None,0)]*(count-len(sources))
    return tuple(records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve derived dataset versions')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'dataset.json.gz','rt') as f:data=json.load(f)
    manifest=json.loads((a.annotations/'manifest.json').read_text());progress=json.loads((a.annotations/'progress.json').read_text())
    if progress['recorded']!=progress['planned']:raise RuntimeError('Finish declared annotation plan first')
    if not (a.annotations/'smoke_manual_review.json').exists():raise RuntimeError('Explicit recognition semantics need source review')
    review=json.loads((a.annotations/'smoke_manual_review.json').read_text());masks={(m['key'],m['step'],m['identity_index']) for m in review.get('exclusions',[])};masked=[]
    annotations={r['directory']:r for r in manifest['rows']};episodes=[];excluded=[];unbound=[];parameters=json.loads(Path('models/task_attack/integrity_v19/bundle.json').read_text())['parameters'];added=0
    for ep in data['episodes']:
        row=annotations.get(ep['directory'])
        if row is None:raise RuntimeError('Missing attempted identifier annotation')
        directory=a.annotations/row['key'];status=json.loads((directory/'result.json').read_text())
        if not status['valid']:
            excluded.append(dict(directory=ep['directory'],reason='Identifier annotation rejected; actual episode outcome preserved and never excluded from future actual evaluation'));continue
        source=Path(ep['directory'])/'private/robot/model_calls.jsonl'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['source_sha256']:raise RuntimeError('Actual returned source changed')
        calls=[json.loads(x) for x in source.read_text().splitlines()];steps={s['step']:s['identity_observations'] for s in json.loads((directory/'annotation.json').read_text())['steps']}
        names={s['identity'].strip().casefold():i for i,s in enumerate(ep['perception']['stations'])};history=[];by_step={}
        origin=calls[ep['examples'][0]['step']-1]['input']['geometry']['camera']['ros_stamp']-ep['examples'][0]['t']
        for index,call in enumerate(calls,1):
            before=history[:];new=[];when=call['started_at']+call['wall_seconds']-origin
            for identity_index,declaration in enumerate(steps.get(index,[])):
                if (row['key'],index,identity_index) in masks:
                    masked.append(dict(key=row['key'],step=index,identity_index=identity_index,declaration=declaration));continue
                port=names.get(declaration['identifier'].strip().casefold())
                if port is None:
                    unbound.append(dict(directory=ep['directory'],step=index,declaration=declaration,reason='Name not in actual public c0 vocabulary; no true-world correction or invented physical binding'));continue
                claim=NamedIdentityClaim(port,declaration['status'],when);history.append(claim);new.append(dict(name_port=port,status=claim.status,time=when,excerpt=declaration['excerpt'],image_ids=declaration['image_ids']))
            by_step[index]=(before,history[:],new)
        truth=tuple(Fact(i,k,v) for i,w in enumerate(ep['world']) for k,v in w.items())
        for ex in ep['examples']:
            before,after,declarations=by_step[ex['step']];captures=tuple(Capture(**c) for c in ex['captures'])
            for feature_key,bindings_key,claims in [('before_features','before_bindings',before),('after_features','after_bindings',after)]:
                state=CandidateState(truth,truth,captures,tuple(Binding(**b) for b in ex[bindings_key]),assertions_from_features(ex[feature_key]),named_identities=tuple(claims))
                ex[feature_key]=global_features(ep['program'],ep['target'],state,ep['perception'],ex['event'],parameters,ex['t'])
            ex['before_named_identities']=[asdict(c) for c in before];ex['after_named_identities']=[asdict(c) for c in after];ex['named_identity_declarations']=declarations;added+=len(declarations)
        episodes.append(ep)
    result=dict(status='OWN declared NAME-awareness development dataset only; no attack ASR',episodes=episodes,excluded=excluded,unbound_named_declarations=unbound,manual_semantic_masks=masked,
        input_boundary='Annotation uses actual returned text only; name ports bind to actual public c0 vocabulary, never to hidden world coordinates.',
        parent_sha256=hashlib.sha256((a.dataset/'dataset.json.gz').read_bytes()).hexdigest(),annotation_manifest_sha256=hashlib.sha256((a.annotations/'manifest.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with gzip.open(a.output/'dataset.json.gz','wt') as f:json.dump(result,f,separators=(',',':'))
    write_json(a.output/'summary.json',dict(status=result['status'],episodes=len(episodes),named_declarations=added,unbound_names=len(unbound),exclusions=excluded,parent_sha256=result['parent_sha256']))
    print('NAMED DATASET',len(episodes),'declarations',added,'unbound',len(unbound),flush=True)


if __name__=='__main__':main()
