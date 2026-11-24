"""Versioned OFFLINE local examples from OWN actor executions.

Robot text, camera geometry and scene truth are teacher labels only. This file
is deliberately not imported by an online observer. Event inputs come from the
actual external JPEGs. Every label has its original episode/step provenance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.task_attack.runtime import write_json
from experiments.task_attack_value.forest import predict
from experiments.task_attack_value.vision import rgb_features
from experiments.task_attack_value.programs import known_program
from primitive_state import CandidateState, Fact
from operator_features import FIELDS, local_features

BASE=Path('models/task_attack/integrity_v19/bundle.json')
ANNOTATIONS=Path('artifacts/task_attack/declared_memory_annotation_01')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def event(frame,directory,heads):
    pixels=frame.get('robot_pixels')
    if not pixels:return None
    image=directory/'public/external'/('%06d.jpg'%frame['id'])
    raw=rgb_features(image.read_bytes(),pixels) if image.exists() else None
    u,v=pixels['centroid'];box=pixels['bbox'];w=box[2]-box[0];h=box[3]-box[1]
    result=[u,v,w,h,0.,0.];mask=[True]*4+[False]*2
    if raw is not None:
        x=np.r_[raw,u,v,w,h][None]
        result[4:]=[2*float(predict(head,x)[0])-1 for head in heads];mask[4:]=[True,True]
    return dict(t=frame['capture_ros'],value=result,mask=mask,id=frame['id'],sha256=frame['sha256'])


def canonical(value,perception):
    """Only image coordinates: origin/scale follow actual static label boxes."""
    anchors=perception['anchors'];scale=max(.01,anchors[-1]['u']-anchors[0]['u'])
    v=np.array(value,dtype=float,copy=True)
    v[0]=(v[0]-anchors[0]['u'])/scale
    v[1]=(v[1]-np.mean([a['v'] for a in anchors]))/scale
    v[2:4]/=scale
    return v.tolist()


def denormalize(value,perception):
    anchors=perception['anchors'];scale=max(.01,anchors[-1]['u']-anchors[0]['u'])
    v=np.array(value,dtype=float,copy=True)
    v[0]=v[0]*scale+anchors[0]['u'];v[1]=v[1]*scale+np.mean([a['v'] for a in anchors]);v[2:4]*=scale
    return v.tolist()


def global_features(program,target,state,perception,body,parameters,clock):
    return np.concatenate([local_features(program,target,i,state,perception,body,parameters,clock)
                           for i in range(len(perception['stations']))]).tolist()


def structured_facts(output):
    records=list(output.get('records') or [])
    records+=list((output.get('report') or {}).get('records') or [])
    result=[]
    for record in records:
        for field,value in (record.get('facts') or {}).items():
            if field in FIELDS and value is not None:
                result.append(dict(object_id=record.get('object_id'),field=field,status='certain',values=[value],
                    image_ids=[record['evidence_image_id']] if record.get('evidence_image_id') else [],structured=True))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise RuntimeError('Never overwrite a completed local dataset')
    args.output.mkdir(parents=True)
    base=json.loads(BASE.read_text());manifest=json.loads((ANNOTATIONS/'manifest.json').read_text())
    progress=json.loads((ANNOTATIONS/'progress.json').read_text())
    if progress['recorded']!=progress['planned']:raise RuntimeError('Finish declared annotation plan first')
    review=json.loads((ANNOTATIONS/'smoke_manual_review.json').read_text())
    masks={(m['seed'],m['step'],m['fact_index']) for m in review['exclusions']}
    acquisitions={}
    for path in ('models/task_attack/forward_v13/side_acquisition_labels.json',
                 'artifacts/task_attack/value_training_01/balanced_proxy_01_acquisition.json'):
        acquisitions.update({e['seed']:e for e in json.loads(Path(path).read_text())['episodes']})
    areas=json.loads(Path('models/task_attack/forward_v13/bundle.json').read_text())['dependency']['area_reference']
    with gzip.open('artifacts/task_attack/direct_slot_training_01/training_values.json.gz','rt') as f:
        cases={r['directory']:r for r in json.load(f)['cases']}
    episodes=[];exclusions=[];audits=[]
    for row in manifest['rows']:
        directory=Path(row['directory']);ap=ANNOTATIONS/str(row['seed']);status=json.loads((ap/'result.json').read_text())
        if not status['valid']:
            exclusions.append(dict(directory=str(directory),seed=row['seed'],reason='Annotation rejected by frozen validator; actual episode retained elsewhere.'))
            continue
        if sha(directory/'private/robot/model_calls.jsonl')!=row['source_sha256']:raise RuntimeError('Teacher log changed')
        case=cases[str(directory)];perception=case['perception']
        truth=json.loads((directory/'private/truth.json').read_text())
        world=[{**{k:d[k] for k in FIELDS if k in d},'display_text':d['identity']} for d in truth['devices']]
        identities={d['identity']:i for i,d in enumerate(truth['devices'])}
        # Authorized text is offline supervision, not a run-time input.
        program,target=known_program(truth['task'],base['programs'],perception['stations'])
        frames=[json.loads(x) for x in (directory/'public/external_frames.jsonl').read_text().splitlines()]
        events=[e for e in (event(f,directory,base['operation_model']['pose_heads']) for f in frames) if e]
        if not events:raise ValueError('No actual external event')
        origin=events[0]['t']
        for e in events:e['t']-=origin
        reads=acquisitions[row['seed']]['reads']
        ann={s['step']:s['facts'] for s in json.loads((ap/'annotation.json').read_text())['steps']}
        calls=[json.loads(x) for x in (directory/'private/robot/model_calls.jsonl').read_text().splitlines()]
        facts=tuple(Fact(i,k,v) for i,w in enumerate(world) for k,v in w.items())
        state=CandidateState(facts,facts);examples=[];capture_images={};mask_log=[]
        for step,call in enumerate(calls,1):
            request=call.get('input') or {};out=call.get('output')
            camera=request.get('geometry',{}).get('camera',{});image=request.get('current_image_id')
            if 'ros_stamp' not in camera or not out:continue
            t=camera['ros_stamp']-origin;near=min(events,key=lambda e:abs(e['t']-t))
            if abs(near['t']-t)>1.2:continue
            read=min(reads,key=lambda r:abs(r['t']-t))
            if abs(read['t']-t)>1.:raise ValueError('Acquisition timestamp mismatch')
            quality=[];before_binding={(b.object_source,b.field):b for b in state.bindings if b.status=='certain'}
            # Raw capture values are latched but never supplied to DECIDE.
            for i,w in enumerate(world):
                for field in w:
                    vi,ar=read['values'].get('%d:%s'%(i,field),(None,None))
                    q=vi*min(1.,ar/max(1.,areas[field])) if vi is not None and field in areas else None
                    state=state.capture(i,field,t,q);capture_images[(image,i,field)]=len(state.captures)-1
                    if field in ('cracked','temperature'):quality.append(q)
            before=state;declared=[]
            af=[]
            for j,fact in enumerate(ann.get(step,[])):
                if (row['seed'],step,j) in masks:
                    mask_log.append(dict(step=step,fact=j,reason='Explicit smoke semantic mask'));continue
                if fact['field']=='box_count':
                    # These data are cabinet inspections. Do not turn a count
                    # of cabinets into a loose-box fact, even with a quote.
                    mask_log.append(dict(step=step,fact=j,reason='No explicit loose-box referent annotation in this corpus'));continue
                if fact['field']=='display_text':
                    mask_log.append(dict(step=step,fact=j,reason='Text-as-object does not establish physical source identity'));continue
                af.append(fact)
            af+=structured_facts(out)
            for fact in af:
                source=identities.get(fact.get('object_id'))
                if source is None or fact['field'] not in world[source]:continue
                values=fact['values'] or [None]
                for value in values:
                    # Without an explicit image, this is a declaration time
                    # upper bound, never an exact new-acquisition annotation.
                    sequences=[capture_images[(im,source,fact['field'])] for im in fact['image_ids']
                               if (im,source,fact['field']) in capture_images]
                    sequence=max(sequences) if sequences else capture_images[(image,source,fact['field'])]
                    state=state.bind(source,fact['field'],value,sequence,fact['status'])
                    declared.append(dict(source=source,field=fact['field'],value=value,status=fact['status'],
                        image_attribution='mentioned-available' if sequences else 'declaration-upper-bound-only',
                        structured=fact.get('structured',False)))
            after=state
            next_camera=None
            for later in calls[step:]:
                next_camera=later.get('input',{}).get('geometry',{}).get('camera',{})
                if 'ros_stamp' in next_camera:break
            next_t=next_camera['ros_stamp']-origin if next_camera and 'ros_stamp' in next_camera else None
            next_event=min(events,key=lambda e:abs(e['t']-next_t)) if next_t is not None else None
            # Time intervals are actual observations; absent successors are
            # right censored, not successful terminal decisions.
            action=out.get('action') or {};report=out.get('report') or {}
            terminal=action.get('type')=='submit'
            current_records=list(request.get('records') or [])+list(out.get('records') or [])
            for record in out.get('records') or []:
                source=identities.get(record.get('object_id'));im=record.get('evidence_image_id')
                if source is None:continue
                for field,value in (record.get('facts') or {}).items():
                    seq=capture_images.get((im,source,field))
                    if seq is not None and value is not None:state=state.register(source,field,value,seq)
            examples.append(dict(step=step,image=image,t=t,call_seconds=call.get('wall_seconds'),
                event=near['value'],event_mask=near['mask'],canonical_event=canonical(near['value'],perception),
                before_features=global_features(program,target,before,perception,near['value'],base['parameters'],t),
                after_features=global_features(program,target,after,perception,near['value'],base['parameters'],t),
                before_bindings=[vars(b) for b in before.bindings],after_bindings=[vars(b) for b in after.bindings],
                captures=[vars(c) for c in before.captures],declared=declared,acquisition_quality=quality,
                records_before=request.get('records') or [],records_returned=out.get('records') or [],
                action_type=action.get('type'),terminal=terminal,report=report,
                next_t=next_t,next_event=next_event['value'] if next_event else None,
                next_canonical_event=canonical(next_event['value'],perception) if next_event else None,
                censored=not terminal and next_event is None))
        if not examples:
            exclusions.append(dict(directory=str(directory),seed=row['seed'],reason='No aligned complete local example'));continue
        episodes.append(dict(key=sha(directory/'private/robot/model_calls.jsonl')[:20],directory=str(directory),
            seed=row['seed'],family=row['family'],program=program,target=target,perception=perception,world=world,
            train=row['seed'] in base['training_scene_seeds'],calibration=row['seed'] in base['calibration_scene_seeds'],
            events=events,examples=examples,annotation_sha256=sha(ap/'annotation.json'),
            source_sha256=row['source_sha256'],semantic_masks=mask_log))
        print('LOCAL DATA',len(episodes),row['seed'],len(examples),flush=True)
    result=dict(status='OFFLINE OWN TEACHER DATA ONLY; no deployed policy or new ASR',episodes=episodes,
        excluded=exclusions,base_sha256=sha(BASE),annotation_manifest_sha256=sha(ANNOTATIONS/'manifest.json'),
        source_sha256=sha(Path(__file__)),warnings=[
            'Declared memory is not hidden perception. Missing mention does not mean a negative sensor fact.',
            'Image mentions give uncertain provenance; no exact-read attribution is invented.',
            'All episode records remain preserved; annotation exclusions only limit supervised fitting.',
            'Source keys use complete episode paths, never a globally unique-seed assumption.'])
    with gzip.open(args.output/'dataset.json.gz','wt') as f:json.dump(result,f,separators=(',',':'))
    write_json(args.output/'summary.json',dict(status=result['status'],episodes=len(episodes),
        calls=sum(len(e['examples']) for e in episodes),train_worlds=len({e['seed'] for e in episodes if e['train']}),
        calibration_worlds=len({e['seed'] for e in episodes if e['calibration']}),excluded=exclusions,
        semantic_masks=sum(len(e['semantic_masks']) for e in episodes),source_sha256=result['source_sha256']))


if __name__=='__main__':main()
