"""Give comparison rankers the same available clean development resources.

Actual task/world/future reads are OFFLINE targets only. Runtime descriptors
contain observed external RGB/history, static perception and a public clock.
Repeated tasks in one physical world retain the predeclared WORLD split.
These clean-route counterfactual targets are diagnostics, not executed ASR.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.task_attack.runtime import write_json
from experiments.task_attack_value.programs import answer,payload
from experiments.task_attack_value.vision import candidate_features,rgb_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve versioned development evidence')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'dataset.json.gz','rt') as f:data=json.load(f)
    base=json.loads(Path('models/task_attack/integrity_v19/bundle.json').read_text())
    delay=2.+base['operation_model']['implementation_delay_seconds'];cases=[];started=time.time()
    source_names=[Path(__file__),Path('experiments/task_attack_value/vision.py'),Path('experiments/task_attack_value/value_features.py')]
    source_hashes={str(n):hashlib.sha256(n.read_bytes()).hexdigest() for n in source_names}
    selected=[e for e in data['episodes'] if Path(e['directory']).name=='clean']
    contract=dict(status='COMMON-RESOURCE baseline development targets; no actual attacks or ASR',
        dataset_sha256=hashlib.sha256((a.dataset/'dataset.json.gz').read_bytes()).hexdigest(),
        clean_episodes=len(selected),training_worlds=sorted({e['seed'] for e in selected if e['train']}),calibration_worlds=sorted({e['seed'] for e in selected if e['calibration']}),
        attacked_episodes='Available to all methods for response/adoption modeling; NOT mislabeled as clean counterfactual routes',
        noncompletion='Retain every available clean prefix, including incomplete episodes. No actual evaluation outcome is removed.',
        input_boundary='Actual external RGB/history and static scene only. Known-task head alone receives authorized task arguments.',
        target='Whole-query effect times actual clean future acquisition and prior empirical adoption. Auxiliary proxy, not observed counterfactual ASR.',
        dispatch_seconds=2.,implementation_delay_seconds=base['operation_model']['implementation_delay_seconds'],source_sha256=source_hashes)
    if set(contract['training_worlds'])&set(contract['calibration_worlds']):raise ValueError('World leakage')
    write_json(a.output/'contract.json',contract)
    for e in selected:
        root=Path(e['directory']);catalog=json.loads((root/'public/catalog.json').read_text());actions=[x for x in catalog['editable_actions'] if payload(x)]
        baseline=answer(e['program'],e['world'],base['parameters'],e['target']);effects=[]
        for action in actions:
            change=payload(action);alternative=copy.deepcopy(e['world']);alternative[change['source']][change['role']]=change['value']
            effects.append(float(answer(e['program'],alternative,base['parameters'],e['target'])!=baseline))
        frames=[json.loads(s) for s in (root/'public/external_frames.jsonl').read_text().splitlines()];samples=[];last=-1e9
        for i,frame in enumerate(frames):
            t=frame['capture_ros']-frames[0]['capture_ros']
            if i and t-last<3.:continue
            image=root/'public/external'/('%06d.jpg'%frame['id'])
            frame['rgb_orientation_features']=rgb_features(image.read_bytes(),frame['robot_pixels']) if image.exists() else None
            x=candidate_features(frames[:i+1],t,e['perception']['anchors']);initial=candidate_features(frames[:1],t,e['perception']['anchors'])
            if x is None or initial is None:continue
            y=[];future=[];cached=[]
            for action,effect in zip(actions,effects):
                change=payload(action);column=2*change['source']+int(change['role']=='temperature');before=[];after=[]
                for read in e['examples']:
                    q=read['acquisition_quality'][column]
                    if q is not None:(after if read['t']>t+delay else before).append(q)
                f=max(after or [0.]);c=max(before or [0.]);head=base['endpoint']['by_role'][change['role']]
                y.append(effect*f*((1-c)*head['fresh_adoption']+c*head['cached_adoption']));future.append(f);cached.append(c)
            samples.append(dict(t=t,x=x.tolist(),static=initial.tolist(),y=y,future_opportunity=future,cached_opportunity=cached))
            last=t
        cases.append(dict(key=e['key'],seed=e['seed'],family=e['family'],directory=e['directory'],train=e['train'],calibration=e['calibration'],
            perception=e['perception'],known=[e['program'],e['target']],actions=actions,whole_query_effects=effects,samples=samples))
        if len(cases)%20==0:print('COMMON BASELINE',len(cases),'/',len(selected),'prefixes',sum(len(c['samples']) for c in cases),flush=True)
    if any(hashlib.sha256(Path(n).read_bytes()).hexdigest()!=sha for n,sha in source_hashes.items()):raise RuntimeError('Feature sources changed during preparation')
    with gzip.open(a.output/'training_values.json.gz','wt') as f:json.dump(dict(cases=cases,contract=contract),f,separators=(',',':'))
    write_json(a.output/'summary.json',dict(**contract,prefixes=sum(len(c['samples']) for c in cases),training_episodes=sum(c['train'] for c in cases),calibration_episodes=sum(c['calibration'] for c in cases),seconds=time.time()-started))
    print('COMMON BASELINE COMPLETE',len(cases),'prefixes',sum(len(c['samples']) for c in cases),flush=True)


if __name__=='__main__':main()
