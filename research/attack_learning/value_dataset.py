"""Build offline OWN counterfactual value labels, never target-time inputs."""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.task_attack.runtime import write_json
from experiments.task_attack_value.programs import answer,payload,known_program
from experiments.task_attack_value.vision import rgb_features,candidate_features
from experiments.task_attack_value.value_features import action_features,source_features
from .source_read_model_development import collect
from .informative_prefix_development import predict as informative


def acquire(batch,cache):
    manifest=json.loads((batch/'manifest.json').read_text())
    if not all((batch/row['directory']/'result.json').exists() for row in manifest['rows']):raise ValueError('Finish the entire declared proxy collection first')
    return manifest,collect([batch],cache)


def build(batches,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    parent=json.loads(Path('models/task_attack/forward_v14/bundle.json').read_text());cases=[];excluded=[];input_hashes=[]
    image_feature_sha=hashlib.sha256(Path('experiments/task_attack_value/vision.py').read_bytes()).hexdigest()
    area_reference=parent['dependency']['area_reference'];programs=parent['programs'];parameters=parent['parameters'];endpoint=parent['endpoint']['by_role']
    for batch in batches:
        batch=Path(batch)
        if batch.name=='forward_training_01':
            manifest=json.loads((batch/'manifest.json').read_text());acquisitions=json.loads(Path('models/task_attack/forward_v13/side_acquisition_labels.json').read_text());static_root=Path('artifacts/task_attack/operator_diagnosis_20261115/training_static')
        else:
            manifest,acquisitions=acquire(batch,destination/(batch.name+'_acquisition.json'));static_root=Path('artifacts/task_attack/operator_diagnosis_20261115/balanced_static')
        input_hashes.append(hashlib.sha256((batch/'manifest.json').read_bytes()).hexdigest());acquisitions={e['seed']:e for e in acquisitions['episodes']}
        for row in manifest['rows']:
            if row['policy']!='clean':continue
            directory=batch/row['directory'];raw=json.loads((directory/'result.json').read_text());fatal=False
            for log in (directory/'private/roslog').glob('*/rosout.log'):
                if 'Depth Lost! EMERGENCY_STOP' in log.read_text(errors='replace'):fatal=True;break
            if fatal or not raw.get('execution_completed') and raw.get('error',{}).get('type')=='TransportError':
                excluded.append(dict(seed=row['seed'],reason='Known infrastructure censor; exclude only from healthy future-value proxy fitting. Actual evaluation/endpoint record remains.',result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest()));continue
            static_path=static_root/('%d.json'%row['seed'])
            if not static_path.exists():raise ValueError('Missing actual static scene interpretation '+str(row['seed']))
            static=json.loads(static_path.read_text());truth=json.loads((directory/'private/truth.json').read_text());catalog=json.loads((directory/'public/catalog.json').read_text());actions=[a for a in catalog['editable_actions'] if payload(a)]
            program=next(p for p in programs if p['family']==row['family']);world=[dict(cracked=d['cracked'],temperature=d['temperature'],display_text=d['identity']) for d in truth['devices']]
            target=next((i for i,d in enumerate(truth['devices']) if d['identity']==truth['score_spec'].get('target')),None);baseline=answer(program,world,parameters,target);effects=[]
            for action in actions:
                change=payload(action);alternative=copy.deepcopy(world);alternative[change['source']][change['role']]=change['value'];effects.append(float(baseline!=answer(program,alternative,parameters,target)))
            frames=[json.loads(s) for s in (directory/'public/external_frames.jsonl').read_text().splitlines()];labels=acquisitions[row['seed']];samples=[];last=-1e9
            for index,frame in enumerate(frames):
                image=directory/'public/external'/('%06d.jpg'%frame['id'])
                frame['rgb_orientation_features']=rgb_features(image.read_bytes(),frame['robot_pixels']) if image.exists() else None
                elapsed=frame['capture_ros']-frames[0]['capture_ros']
                if elapsed-last<3.:continue
                prefix=frames[:index+1];x=candidate_features(prefix,elapsed,static['anchors']);initial=candidate_features(frames[:1],elapsed,static['anchors'])
                if x is None or initial is None:continue
                y=[];future_opportunity=[];cached_opportunity=[]
                for action,effect in zip(actions,effects):
                    change=payload(action);key='%d:%s'%(change['source'],change['role']);past=[];future=[]
                    for read in labels['reads']:
                        visible,area=read['values'][key];quality=visible*min(1.,area/max(1.,area_reference[change['role']]))
                        (future if read['t']>elapsed else past).append(quality)
                    f=max(future or [0.]);c=max(past or [0.]);transition=endpoint[change['role']]
                    future_opportunity.append(f);cached_opportunity.append(c)
                    y.append(effect*f*((1-c)*transition['fresh_adoption']+c*transition['cached_adoption']))
                # Only actual image-derived geometry/color and actual static
                # scene interpretation will become predictor inputs.
                samples.append(dict(t=elapsed,x=x.tolist(),static=initial.tolist(),y=y,future_opportunity=future_opportunity,cached_opportunity=cached_opportunity,
                    informative_probability=informative(parent['informative_prefix'],prefix,elapsed)))
                last=elapsed
            cases.append(dict(seed=row['seed'],family=row['family'],directory=str(directory),batch=batch.name,perception=static,
                known=(program,target),actions=actions,whole_query_effects=effects,samples=samples,training_target_station=row.get('training_target_station'),
                source_frame_sha256=hashlib.sha256((directory/'public/external_frames.jsonl').read_bytes()).hexdigest(),
                source_result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest()))
            print('VALUE LABELLED',len(cases),row['seed'],len(samples),flush=True)
    # The large private training dataset is compressed. Runtime model bundles
    # strip all labels, private paths, known task references and world truth.
    if hashlib.sha256(Path('experiments/task_attack_value/vision.py').read_bytes()).hexdigest()!=image_feature_sha:raise RuntimeError('Image feature source changed while building dataset')
    output=dict(cases=cases,excluded=excluded,image_feature_source_sha256=image_feature_sha,parent_bundle_sha256=hashlib.sha256(Path('models/task_attack/forward_v14/bundle.json').read_bytes()).hexdigest(),input_manifest_sha256=input_hashes,
        target='Whole-query change in OWN true world times actual clean future source acquisition and a separately real-outcome-fitted adoption transition. Counterfactual proxy, not an executed attack outcome.')
    with gzip.open(destination/'training_values.json.gz','wt',encoding='utf-8') as stream:json.dump(output,stream,separators=(',',':'))
    write_json(destination/'dataset_summary.json',dict(cases=len(cases),excluded=excluded,input_manifest_sha256=input_hashes,world_truth_used_offline_only=True,current_target_truth_used=False))
    return output
