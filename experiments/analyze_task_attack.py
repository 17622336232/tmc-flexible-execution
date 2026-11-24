"""Read-only, post-episode analysis. Never imported by an online actor.

Reads actual reports/logs, preserves primary scores and reports missing runs.
Plots are emitted only for complete batches. No model calls or live ROS access.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import copy
import hashlib


def read_lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def median(values):
    return statistics.median(values) if values else None


def analyze(root):
    manifest=json.loads((root/'manifest.json').read_text())
    rows=[];policies={};family_counts={}
    for case in manifest['rows']:
        directory=root/case['directory'];path=directory/'result.json'
        policy=case['policy']
        stats=policies.setdefault(policy,dict(scheduled=0,correct=0,wrong_accepted=0,refused=0,
                                            incomplete=0,missing=0,actual_edits=0,later_capture_edits=0,
                                            observer_seconds=[],observer_completed_seconds=[],
                                            observer_failed_or_cancelled_requests=0,robot_seconds=[],terminal_errors={},
                                            alias_only_contract_flags=0,incomplete_reasons={}))
        stats['scheduled']+=1
        if not path.exists():stats['missing']+=1;continue
        result=json.loads(path.read_text());outcome=result['score']['outcome'];stats[outcome]+=1
        edit=result.get('actual_edit') or {}
        robot_calls=read_lines(directory/'private/robot/model_calls.jsonl')
        observer_calls=read_lines(directory/'private/observer/model_calls.jsonl')
        observer=result.get('observer',{});last=observer.get('last_response',{})
        seconds=result.get('robot_metrics',{}).get('episode_wall_seconds')
        duration_source='robot_metrics'
        if seconds is None:
            events=read_lines(directory/'private/events.jsonl')
            takeoff=[x for x in events if x['event']=='takeoff_complete']
            if events:
                beginning=takeoff[-1] if takeoff else events[0]
                seconds=max(0.,events[-1]['wall']-beginning['wall'])
                duration_source='event_interval_after_takeoff' if takeoff else 'startup_events'
        if seconds is not None:stats['robot_seconds'].append(seconds)
        stats['observer_seconds'].extend(x['wall_seconds'] for x in observer_calls)
        stats['observer_completed_seconds'].extend(x['wall_seconds'] for x in observer_calls if 'output' in x)
        stats['observer_failed_or_cancelled_requests']+=sum('error' in x for x in observer_calls)
        stats['actual_edits']+=int(bool(edit.get('service_success')))
        later=result.get('attack_timing',{}).get('onboard_images_captured_after_edit',[])
        stats['later_capture_edits']+=int(bool(later))
        error=result.get('error') or {}
        category=error.get('code') or error.get('type')
        if category:stats['terminal_errors'][category]=stats['terminal_errors'].get(category,0)+1
        if outcome=='incomplete':
            summary=(result.get('report') or {}).get('summary','')
            why=category or ('resource_budget' if summary=='Experimental episode budget exhausted' else 'model_incomplete')
            stats['incomplete_reasons'][why]=stats['incomplete_reasons'].get(why,0)+1
        refs=[];target=None;alias_diagnostic=None;aliases=[]
        truth_path=directory/'private/truth.json'
        catalog_path=directory/'public/catalog.json'
        if truth_path.exists() and catalog_path.exists() and result.get('report'):
            # A diagnostic explicitly separate from the primary task contract:
            # does a PUBLIC object handle uniquely name the correct physical
            # object, even though it is not the requested equipment-number namespace?
            # Never rewrite actors, original reports, or primary endpoint scores.
            from experiments.task_attack.evaluate import evaluate
            truth=json.loads(truth_path.read_text());catalog=json.loads(catalog_path.read_text())
            handles={}
            graph=catalog.get('scene_conditions',{}).get('source_graph',[])
            for i,group in enumerate(graph):
                if i>=len(truth['devices']):continue
                identity=truth['devices'][i]['identity']
                handles[group['object_handle']]=identity
                for source in group['sources']:handles[source['source_id']]=identity
            normalized=copy.deepcopy(result['report'])
            for record in normalized.get('records',[]):
                key='object_id' if 'object_id' in record else 'equipment_id'
                value=record.get(key)
                if value in handles:
                    record[key]=handles[value];aliases.append(dict(reported=value,physical_identity=handles[value]))
            if aliases:
                alias_diagnostic=evaluate(truth,normalized)
                if outcome=='wrong_accepted' and alias_diagnostic['correct']:stats['alias_only_contract_flags']+=1
        if edit and truth_path.exists():
            truth=json.loads(truth_path.read_text());index=re.search(r'_(\d+)$',edit.get('action_id',''))
            if index and int(index[1])<=len(truth['devices']):
                target=truth['devices'][int(index[1])-1]['identity']
            for record in result.get('report',{}).get('records',[]):
                if record.get('object_id',record.get('equipment_id'))!=target:continue
                image_id=record.get('evidence_image_id','')
                if not re.fullmatch(r'onboard_\d+',image_id):continue
                metadata=directory/'private/robot'/(image_id+'.json')
                if metadata.exists():
                    frame=json.loads(metadata.read_text())
                    if frame.get('ros_stamp',float('inf'))<=edit.get('ack_ros',-1):refs.append(image_id)
        item=dict(family=case['family'],policy=policy,source_profile=case.get('source_profile'),
                  scene_block=case.get('scene_block'),
                  outcome=outcome,fact_errors=result['score'].get('fact_errors',[]),
                  report=result.get('report'),terminal_error=error or None,
                  robot_episode_seconds=seconds,duration_source=duration_source,robot_requests=len(robot_calls),
                  observer_requests=len(observer_calls),
                  observer_request_seconds=[x['wall_seconds'] for x in observer_calls],
                  observer_efforts=sorted({x['reasoning_effort'] for x in observer_calls}),
                  observer_decision=observer.get('decision'),actual_edit=edit or None,
                  edited_object_identity=target,later_capture_ids=later,
                  report_contract_aliases=aliases,alias_normalized_diagnostic_score=alias_diagnostic,
                  final_records_named_as_target_referencing_pre_edit_images=refs,
                  observer_task_hypotheses=last.get('task_hypotheses',[]),
                  attack_reason=last.get('reason'),counterfactuals=last.get('counterfactuals',[]),
                  result_path=str(path))
        rows.append(item)
        count=family_counts.setdefault(case['family'],{}).setdefault(policy,dict(correct=0,wrong_accepted=0,refused=0,incomplete=0))
        count[outcome]+=1
    complete=all(not s['missing'] for s in policies.values())
    for stats in policies.values():
        stats['observer_request_median_seconds']=median(stats.pop('observer_seconds'))
        stats['observer_completed_request_median_seconds']=median(stats.pop('observer_completed_seconds'))
        stats['robot_episode_median_seconds']=median(stats.pop('robot_seconds'))
        stats['wrong_accepted_per_scheduled']=None if stats['missing'] else stats['wrong_accepted']/stats['scheduled']
    inference_rows=[]
    for row in rows:
        if row['policy']!='inferred':continue
        hypotheses=[x for x in row['observer_task_hypotheses']
                    if isinstance(x,dict) and isinstance(x.get('probability'),(int,float))]
        top=max(hypotheses,key=lambda x:x['probability']) if hypotheses else None
        inference_rows.append(dict(family=row['family'],predicted_family=top.get('family') if top else None,
                                   correct=bool(top and top.get('family')==row['family']),
                                   reported_confidence=top.get('probability') if top else None))
    return dict(complete=complete,scheduled=len(manifest['rows']),available=len(rows),
                initial_scene_instances=len({x.get('scene_block',x['block']) for x in manifest['rows']}),
                task_blocks=len({x['block'] for x in manifest['rows']}),
                analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                robot_model=manifest.get('robot_model'),attacker_model=manifest.get('attacker_model'),
                implementation_hash=manifest['implementation_hash'],policies=policies,
                inferred_task_diagnostic=dict(available=len(inference_rows),
                    top1_correct=sum(x['correct'] for x in inference_rows),rows=inference_rows,
                    note='Post-episode family diagnostic only; reported probabilities are uncalibrated and this is not the attack endpoint.'),
                families=family_counts,rows=rows,
                limitations=['Zero-shot VLM pilot; no learned operation/duration/event model.',
                             'A later image does not prove the edited carrier was visible or adopted.',
                             'Evidence references are model claims; not automated proof of image content.',
                             'Public-handle alias normalization is a post-hoc diagnostic, not a replacement primary score.',
                             'No result or primary score is modified by this analysis.'])


def plot(data, root, filename='outcomes.png', subtitle=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['clean','blind','inferred','known_task']
    colors={'correct':'#3e8f7b','wrong_accepted':'#cf493f','refused':'#e5ae43','incomplete':'#8c99a8'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),gridspec_kw={'width_ratios':[1.65,1]})
    left=[0]*len(names)
    for outcome,color in colors.items():
        values=[data['policies'][p][outcome] for p in names]
        axes[0].barh(names,values,left=left,color=color,label=outcome.replace('_',' '))
        for i,n in enumerate(values):
            if n:axes[0].text(left[i]+n/2,i,str(n),ha='center',va='center',fontsize=11)
        left=[a+b for a,b in zip(left,values)]
    axes[0].invert_yaxis();axes[0].set_xlabel('Scheduled episodes (all outcomes retained)')
    axes[0].set_title('Robot task outcomes');axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False)
    active=[p for p in names if p!='clean']
    values=[data['policies'][p]['observer_request_median_seconds'] or 0 for p in active]
    axes[1].bar(active,values,color='#4a78a8')
    for i,v in enumerate(values):axes[1].text(i,v,'%.1fs'%v,ha='center',va='bottom')
    axes[1].set_ylabel('Seconds');axes[1].set_title('Actual attacker request median')
    axes[1].set_ylim(0,max(values+[1])*1.25)
    for ax in axes:
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    robot=(data.get('robot_model') or {}).get('reasoning_effort','legacy')
    attacker=(data.get('attacker_model') or {}).get('reasoning_effort','legacy')
    title='Task attack pilot: %d task families | robot %s | attacker %s' % (len(data['families']),robot,attacker)
    if subtitle:title += '\n'+subtitle
    fig.suptitle(title,fontsize=12)
    fig.tight_layout(rect=[0,.06,1,.91 if subtitle else .95])
    fig.savefig(str(root/filename),dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();root=args.batch.resolve();data=analyze(root)
    (root/'analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    if args.plot:
        if not data['complete']:raise SystemExit('Batch is incomplete; no full-batch plot emitted')
        plot(data,root)
    print(json.dumps({k:v for k,v in data.items() if k not in ('rows','families')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
