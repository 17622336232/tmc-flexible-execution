"""Fixed paired pilot manifest and resumable sequential backend execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import random
import statistics
import shutil
import subprocess
import threading
import time

from experiments.task_attack import FAMILIES, POLICIES, RESEARCH_POLICIES, MODEL, EFFORT, ATTACK_EFFORT, ATTACK_EFFORTS, SOURCE_PROFILES
from .audit import implementation_hash, snapshot_implementation
from experiments.task_attack.evaluate import evaluate, SCORE_VERSION
from .run import ROOT
from experiments.task_attack.runtime import write_json
from experiments.task_attack.tasks import SCENES


def account_failure(directory):
    """Stop new dispatch on account/authorization blocks, preserving all runs."""
    directory=Path(directory);errors=[]
    result=directory/'result.json'
    if result.exists():
        data=json.loads(result.read_text())
        errors.extend([data.get('error') or {},data.get('observer') or {}])
    for role in ('robot','observer'):
        path=directory/'private'/role/'model_calls.jsonl'
        if path.exists():
            errors.extend(dict(x.get('error') or {},role=role) for x in
                (json.loads(line) for line in path.read_text().splitlines() if line.strip()))
    codes={'subscription_sharing_usage_limit_exceeded','insufficient_quota','quota_exceeded',
           'invalid_api_key','invalid_access_token','invalid_grant','authentication_required',
           'not_logged_in','refresh_expired','refresh_interrupted','refresh_not_ready',
           'invalid_refresh_token','token_expired','refresh_token_expired',
           'refresh_token_invalidated','refresh_token_reused','plan_usage_disabled','invalid_storage','unsafe_storage'}
    for error in errors:
        code=error.get('code',error.get('error_code'));status=error.get('status')
        if status in (401,402,403,429) or code in codes:
            return dict(status=status,code=code,role=error.get('role'),
                reason='Account/authorization/usage block: no new episodes dispatched; completed results preserved.')
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--instances-per-family',type=int,default=1)
    p.add_argument('--seed',type=int,default=2026111502)
    p.add_argument('--execute',action='store_true',help='Without this flag, write only the fixed manifest')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--rescore',action='store_true',help='Offline: uniformly recompute scores from unchanged actual reports; preserve original result backups')
    p.add_argument('--episode-timeout',type=float,default=1800)
    p.add_argument('--equipment-count',type=int,default=6)
    p.add_argument('--workers',type=int,default=1,help='Independent ROS/Gazebo processes; paired policies within a block remain sequential')
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=[f for f in FAMILIES if not f.endswith('_labels')])
    p.add_argument('--attack-effort',choices=ATTACK_EFFORTS,default=ATTACK_EFFORT)
    p.add_argument('--source-profiles',nargs='+',choices=SOURCE_PROFILES,default=list(SOURCE_PROFILES))
    p.add_argument('--policies',nargs='+',choices=('clean','belief_blind','belief'),default=['clean','belief_blind','belief'])
    p.add_argument('--backend',choices=('gazebo','cpu'),default='gazebo')
    p.add_argument('--scene-allocation',choices=('crossed','independent'),default='independent',
        help='crossed reuses a scene across task families; independent samples a fresh scene per task instance, paired across methods')
    p.add_argument('--baseline-bundle',type=Path)
    p.add_argument('--timing-model',type=Path)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.rescore and a.execute:p.error('Rescoring is an offline operation; do not combine it with execution')
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
    else:
        if a.summarize or a.rescore:p.error('Manifest does not exist')
        if a.instances_per_family<1:p.error('At least one instance per family is required')
        if len(set(a.families))!=len(a.families):p.error('Task families must not be duplicated')
        if len(set(a.policies))!=len(a.policies):p.error('Policies must not be duplicated')
        if any(x in a.policies for x in ('uao_spsa','iccv_uada_spsa','iccv_upa_spsa')) and not a.baseline_bundle:
            p.error('Visual comparisons require a trained frozen --baseline-bundle')
        baseline_hash=None
        timing_hash=None
        if a.timing_model:
            timing=json.loads(a.timing_model.read_text())
            shutil.copy2(str(a.timing_model),str(root/'event_timing.json'))
            timing_hash=hashlib.sha256((root/'event_timing.json').read_bytes()).hexdigest()
        if a.baseline_bundle:
            from experiments.task_attack.visual_assets import load_bundle
            bundle,files,baseline_hash=load_bundle(a.baseline_bundle)
            frozen=root/'baseline_bundle';frozen.mkdir(parents=True,exist_ok=True)
            for policy,source in files.items():
                relative=Path(bundle['textures'][policy]['path'])
                if relative.is_absolute() or '..' in relative.parts:p.error('Baseline texture paths must stay within the bundle')
                target=frozen/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(source),str(target))
            shutil.copy2(str(a.baseline_bundle),str(frozen/'bundle.json'))
        rng=random.Random(a.seed);rows=[]
        # Every task samples the SAME physical scene distribution; no task
        # instruction/answer is encoded in a special layout or public catalog.
        # Crossed scenes support strict observation-ambiguity studies. Fresh
        # independent task instances give the larger benchmark more distinct
        # environments while retaining exact initial pairing across methods.
        scene_keys=sorted({(f if a.scene_allocation=='independent' else SCENES[f],i)
            for f in a.families for i in range(a.instances_per_family)})
        scene_seeds={key:rng.randrange(1,2**31) for key in scene_keys}
        balanced_profiles=[a.source_profiles[i%len(a.source_profiles)] for i in range(len(a.families)*a.instances_per_family)]
        rng.shuffle(balanced_profiles)
        for fi,family in enumerate(a.families):
            for i in range(a.instances_per_family):
                scene_key=family if a.scene_allocation=='independent' else SCENES[family]
                scene_seed=scene_seeds[(scene_key,i)]
                policies=list(a.policies);rng.shuffle(policies)
                block='%s_%02d'%(family,i+1)
                for policy in policies:
                    rows.append(dict(block=block,family=family,seed=scene_seed,policy=policy,
                                     scene_block='%s_%02d'%(scene_key,i+1),
                                     source_profile=balanced_profiles[fi*a.instances_per_family+i],
                                     directory=block+'/'+policy))
        block_order=sorted({x['block'] for x in rows});rng.shuffle(block_order)
        rows=[row for block in block_order for row in rows if row['block']==block]
        if a.workers<1 or a.workers>(6 if a.backend=='cpu' else 3):p.error('Workers must be 1..6 for CPU, 1..3 for Gazebo')
        if not 2<=a.equipment_count<=6:p.error('Pilot equipment count must be 2..6')
        manifest=dict(version=1,model=MODEL,reasoning_effort=EFFORT,equipment_count=a.equipment_count,workers=a.workers,
                      backend=a.backend,policies=a.policies,
                      scene_allocation=a.scene_allocation,
                      baseline_bundle=str(root/'baseline_bundle/bundle.json') if a.baseline_bundle else None,
                      baseline_bundle_sha256=baseline_hash,
                      timing_model=str(root/'event_timing.json') if a.timing_model else None,
                      timing_model_sha256=timing_hash,
                      robot_model=dict(model=MODEL,reasoning_effort=EFFORT),
                      attacker_model=dict(model=MODEL,reasoning_effort=a.attack_effort,trained=False),
                      implementation_hash=implementation_hash(),
                      implementation='eager_dependency_prefetch_v7_development',phase_policy_included=False,
                      episode_timeout_seconds=a.episode_timeout,rows=rows)
        if a.baseline_bundle and set(scene_seeds.values())&set(bundle.get('training_scene_seeds',[])):
            p.error('Frozen test scenes overlap baseline training')
        if a.timing_model and set(scene_seeds.values())&set(timing.get('training_scene_seeds',[])):
            p.error('Frozen test scenes overlap external timing training')
        manifest['protocol']=dict(primary='Representation-normalized wrong ACCEPTED main-task reports via the frozen semantic_fact_audit.py; paired across methods and clean. Raw v5 strict-format scores are retained as auxiliary.',
            secondary='Correct completion, refusal, incomplete, interface/flight errors reported separately',
            inferential_unit='Shared scene instance; multiple task families in one scene are clustered',
            paper_methods='No paper baselines are rerun in this scheduler pilot. The seven-method frozen v6 study remains separate.',
            tuning='Training/development scenes separate; no changes while held-out batch executes',
            literal_label_controls=[f for f in a.families if f.endswith('_labels')])
        write_json(manifest_path,manifest)
        snapshot_implementation(root/'source_snapshot')
    if a.execute:
        if manifest.get('implementation_hash')!=implementation_hash():
            p.error('Implementation changed after the manifest was frozen; use a fresh batch directory')
        if manifest.get('baseline_bundle'):
            from experiments.task_attack.visual_assets import load_bundle
            if load_bundle(manifest['baseline_bundle'])[2]!=manifest['baseline_bundle_sha256']:
                p.error('Frozen baseline bundle changed')
        if manifest.get('timing_model') and hashlib.sha256(Path(manifest['timing_model']).read_bytes()).hexdigest()!=manifest['timing_model_sha256']:
            p.error('Frozen event timing model changed')
        blocks={}
        for row in manifest['rows']:blocks.setdefault(row['block'],[]).append(row)
        workers=manifest.get('workers',1)
        shards=[list(blocks.values())[i::workers] for i in range(workers)]
        stop_dispatch=threading.Event();stop_lock=threading.Lock()
        def run_shard(worker,groups):
            for group in groups:
                for row in group:
                    if stop_dispatch.is_set():return
                    if manifest.get('implementation_hash')!=implementation_hash():
                        raise RuntimeError('Implementation changed during the frozen test batch; remaining runs stopped')
                    if manifest.get('baseline_bundle') and load_bundle(manifest['baseline_bundle'])[2]!=manifest['baseline_bundle_sha256']:
                        raise RuntimeError('Baseline artifacts changed during frozen test')
                    if manifest.get('timing_model') and hashlib.sha256(Path(manifest['timing_model']).read_bytes()).hexdigest()!=manifest['timing_model_sha256']:
                        raise RuntimeError('Event timing model changed during frozen test')
                    dest=root/row['directory']
                    if (dest/'result.json').exists():continue
                    dest.mkdir(parents=True,exist_ok=True)
                    cmd=['bash',str(ROOT/'scripts/run_attack_prefetch.sh'),'--headless','--family',row['family'],
                         '--policy',row['policy'],'--seed',str(row['seed']),'--output',str(dest),
                         '--equipment-count',str(manifest.get('equipment_count',6)),
                         '--attack-effort',manifest.get('attacker_model',{}).get('reasoning_effort',ATTACK_EFFORT),
                         '--source-profile',row.get('source_profile','independent'),
                         '--ros-port',str(11381+worker),'--gazebo-port',str(11481+worker),
                         '--episode-timeout',str(manifest['episode_timeout_seconds'])]
                    cmd+=['--backend',manifest.get('backend','gazebo')]
                    if manifest.get('baseline_bundle'):cmd+=['--baseline-bundle',manifest['baseline_bundle']]
                    if manifest.get('timing_model'):cmd+=['--timing-model',manifest['timing_model']]
                    print('worker %d：%s %s'%(worker+1,row['block'],row['policy']),flush=True)
                    with (dest/'runner.log').open('w') as log:
                        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                    if not (dest/'result.json').exists():
                        write_json(dest/'result.json',dict(score=dict(outcome='incomplete',wrong_accepted=False,correct=False),
                            error=dict(type='runner_failed',exit_code=result.returncode),manifest=row))
                    failure=account_failure(dest)
                    if failure:
                        with stop_lock:
                            if not stop_dispatch.is_set():
                                write_json(root/'execution_stop.json',dict(wall=time.time(),episode=row['directory'],
                                    account_failure=failure,remaining='Not run; no fabricated episode results'))
                                print('账户／接口受限，停止派发新回合，保留已运行结果。',flush=True)
                                stop_dispatch.set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(run_shard,i,shard) for i,shard in enumerate(shards)]
            for future in futures:future.result()
    if a.rescore:
        scorer_hash=hashlib.sha256(Path(__file__).with_name('evaluate.py').read_bytes()).hexdigest()
        previous_summary=root/'summary.json'
        if (previous_summary.exists() and not (root/'summary.score_v1.json').exists()
                and 1 in json.loads(previous_summary.read_text()).get('score_versions_present',[1])):
            shutil.copy2(str(previous_summary),str(root/'summary.score_v1.json'))
        for row in manifest['rows']:
            path=root/row['directory']/'result.json'
            if not path.exists():continue
            result=json.loads(path.read_text())
            truth=json.loads((path.parent/'private/truth.json').read_text())
            report=result.get('report')
            saved=path.parent/'private/robot/report.json'
            if saved.exists() and report is not None and json.loads(saved.read_text())!=report:
                raise ValueError('Stored report differs from the actual robot report: '+str(path.parent))
            backup=path.with_name('result.score_v1.json')
            if result.get('score',{}).get('score_version',1)==1:
                if not backup.exists():shutil.copy2(str(path),str(backup))
                result.setdefault('score_original_v1',result.get('score'))
            result['score']=evaluate(truth,report,result.get('actual_edit'))
            result['evaluation']=dict(version=SCORE_VERSION,code_sha256=scorer_hash,
                report_sha256=hashlib.sha256(saved.read_bytes()).hexdigest() if saved.exists() else None,
                correction='Inspection ledger may contain explicit normal findings; those are not positive alarms. Raw reports and actor execution are unchanged.')
            write_json(path,result)
    summary=dict(planned_episodes=len(manifest['rows']),completed_episodes=0,policies={},score_versions_present=[],
                 paired_comparisons={},task_families={},source_profiles={},
                 note='All scheduled instances stay in denominators; missing results are shown separately.')
    durations={};robot_times={};edit_counts={};later_counts={}
    outcomes={}
    for row in manifest['rows']:
        path=root/row['directory']/'result.json'
        stats=summary['policies'].setdefault(row['policy'],dict(planned=0,available=0,correct=0,wrong_accepted=0,refused=0,incomplete=0,missing=0))
        stats['planned']+=1
        if not path.exists():stats['missing']+=1;continue
        result=json.loads(path.read_text());score=result.get('score',{})
        if score.get('score_version',1) not in summary['score_versions_present']:
            summary['score_versions_present'].append(score.get('score_version',1))
        stats['available']+=1;summary['completed_episodes']+=1
        outcome=score.get('outcome','incomplete');stats[outcome]+=1
        for section,label in (('task_families',row.get('family','legacy')),('source_profiles',row.get('source_profile','legacy'))):
            group=summary[section].setdefault(label,{})
            count=group.setdefault(row['policy'],dict(available=0,correct=0,wrong_accepted=0,refused=0,incomplete=0))
            count['available']+=1;count[outcome]+=1
        policy=row['policy'];edit=result.get('actual_edit')
        edit_counts[policy]=edit_counts.get(policy,0)+int(bool(edit and edit.get('service_success')))
        later_counts[policy]=later_counts.get(policy,0)+int(bool(result.get('attack_timing',{}).get('onboard_images_captured_after_edit')))
        seconds=result.get('robot_metrics',{}).get('episode_wall_seconds')
        if seconds is not None:robot_times.setdefault(policy,[]).append(seconds)
        calls=path.parent/'private/observer/model_calls.jsonl'
        if calls.exists():
            durations.setdefault(policy,[]).extend(json.loads(line)['wall_seconds'] for line in calls.read_text().splitlines() if line.strip())
        outcomes[(row['block'],row['policy'])]=int(bool(score.get('wrong_accepted')))
    for stats in summary['policies'].values():
        stats['wrong_accepted_per_scheduled']=(stats['wrong_accepted']/stats['planned']
            if stats['missing']==0 else None)
    for policy,stats in summary['policies'].items():
        samples=durations.get(policy,[])
        stats.update(actual_edits=edit_counts.get(policy,0),edits_with_later_captures=later_counts.get(policy,0),
                     observer_requests=len(samples),observer_request_median_seconds=statistics.median(samples) if samples else None,
                     robot_episode_median_seconds=statistics.median(robot_times[policy]) if robot_times.get(policy) else None)
    focus='belief' if 'belief' in manifest.get('policies',[]) else 'inferred'
    for comparator in sorted({x['policy'] for x in manifest['rows']}-{focus}):
        pairs=[(v,outcomes[(b,comparator)]) for (b,p),v in outcomes.items()
               if p==focus and (b,comparator) in outcomes]
        summary['paired_comparisons'][focus+'_minus_'+comparator]=dict(
            complete_pairs=len(pairs),risk_difference=None if not pairs else sum(x-y for x,y in pairs)/len(pairs))
    write_json(root/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
