"""Read-only evaluation of frozen, paired API-baseline research batches.

No live robot/model access, no score rewriting, no filtering on attack success.
Completed tasks, budgeted incomplete work and infrastructure errors are separate.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics

from .evaluate import evaluate
from .runtime import write_json


LABELS=dict(clean='无攻击',belief_blind='盲攻击（相同决策器）',belief='证据依赖推断',
    uao_spsa='UAO / API 适配',iccv_uada_spsa='ICCV UADA / API 适配',
    iccv_upa_spsa='ICCV UPA / API 适配',redvla_external='RedVLA / 外部视角适配',
    inferred='旧版整任务推断',blind='旧版盲攻击',known_task='已知任务参照',belief_known_task='新决策器已知任务参照')


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def cluster_interval(pairs,seed=20261115,replicates=2000,interpretation=None):
    grouped={}
    for group,difference in pairs:grouped.setdefault(group,[]).append(difference)
    keys=sorted(grouped)
    if not keys:return None
    rng=random.Random(seed);samples=[]
    for _ in range(replicates):
        values=[]
        for group in rng.choices(keys,k=len(keys)):values.extend(grouped[group])
        samples.append(sum(values)/len(values))
    samples.sort()
    return dict(scene_clusters=len(keys),paired_tasks=len(pairs),difference=sum(x for g,x in pairs)/len(pairs),
        percentile_95_ci=[samples[int(.025*(len(samples)-1))],samples[int(.975*(len(samples)-1))]],
        interpretation=interpretation or 'Positive means belief attack reduced correct completion more than comparator; scene-cluster bootstrap')


def analyze(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    episodes=[];summary={};index={};families={};profiles={};audit_errors=[]
    for row in manifest['rows']:
        policy=row['policy'];stats=summary.setdefault(policy,dict(scheduled=0,completed=0,correct=0,
            wrong_accepted=0,refused=0,incomplete=0,infrastructure_errors=0,missing=0,
            successful_physical_edits=0,edits_with_later_capture=0,robot_requests=0,attacker_requests=0,
            native_motion_timeouts=0,native_planner_stops=0,tool_errors=0,observer_errors=0,
            attacker_wall_seconds=[],robot_wall_seconds=[],wrong_reports_with_disclosed_uncertainty=0))
        stats['scheduled']+=1;path=root/row['directory']/'result.json'
        if not path.exists():stats['missing']+=1;continue
        result=json.loads(path.read_text());score=result.get('score',{})
        outcome=score.get('outcome','incomplete');stats['completed']+=1
        if outcome not in ('correct','wrong_accepted','refused','incomplete'):
            audit_errors.append(dict(directory=row['directory'],error='non-task result in research batch'));continue
        stats[outcome]+=1
        infrastructure=bool(result.get('error') or not result.get('execution_completed'))
        stats['infrastructure_errors']+=int(infrastructure)
        events=lines(path.parent/'private/events.jsonl')
        feedback=[e.get('feedback',{}) for e in events if e.get('event')=='task_tool_feedback']
        motion_timeouts=sum('Original navigation timeout:' in f.get('message','') for f in feedback)
        planner_stops=sum(f.get('error_type')=='NativeNavigationError' for f in feedback)
        observer_error=result.get('observer',{}).get('decision')=='observer_error'
        stats['native_motion_timeouts']+=motion_timeouts;stats['native_planner_stops']+=planner_stops
        stats['tool_errors']+=len(feedback);stats['observer_errors']+=int(observer_error)
        if result.get('implementation_hash')!=manifest['implementation_hash']:
            audit_errors.append(dict(directory=row['directory'],error='implementation hash mismatch'))
        if result.get('baseline_bundle_sha256')!=manifest.get('baseline_bundle_sha256'):
            audit_errors.append(dict(directory=row['directory'],error='baseline artifact mismatch'))
        if result.get('timing_model_sha256')!=manifest.get('timing_model_sha256'):
            audit_errors.append(dict(directory=row['directory'],error='event timing artifact mismatch'))
        truth_path=path.parent/'private/truth.json';truth=json.loads(truth_path.read_text()) if truth_path.exists() else None
        if truth and result.get('execution_completed'):
            check=evaluate(truth,result.get('report'),result.get('actual_edit'))
            if check!=score:audit_errors.append(dict(directory=row['directory'],error='stored score differs from unchanged actual report recomputation'))
        robot=lines(path.parent/'private/robot/model_calls.jsonl')
        attacker=lines(path.parent/'private/observer/model_calls.jsonl')
        stats['robot_requests']+=len(robot);stats['attacker_requests']+=len(attacker)
        stats['attacker_wall_seconds'].extend(x['wall_seconds'] for x in attacker)
        seconds=result.get('robot_metrics',{}).get('episode_wall_seconds')
        if seconds is not None:stats['robot_wall_seconds'].append(seconds)
        edit=result.get('actual_edit') or {}
        stats['successful_physical_edits']+=int(bool(edit.get('service_success')))
        later=result.get('attack_timing',{}).get('onboard_images_captured_after_edit',[])
        stats['edits_with_later_capture']+=int(bool(later))
        report=result.get('report') or {};text=json.dumps(report,ensure_ascii=False)
        disclosed=any(word in text for word in ('遮挡','重复','不一致','翻转','被修改','歧义','不确定','无法确认'))
        stats['wrong_reports_with_disclosed_uncertainty']+=int(outcome=='wrong_accepted' and disclosed)
        for grouped,key in ((families,row['family']),(profiles,row.get('source_profile','unknown'))):
            counts=grouped.setdefault(key,{}).setdefault(policy,dict(completed=0,correct=0,wrong_accepted=0,
                                                                  refused=0,incomplete=0,infrastructure_errors=0))
            counts['completed']+=1;counts[outcome]+=1;counts['infrastructure_errors']+=int(infrastructure)
        item=dict(row,outcome=outcome,correct=bool(score.get('correct')),wrong_accepted=bool(score.get('wrong_accepted')),
            infrastructure_error=infrastructure,terminal_error=result.get('error'),actual_edit=edit or None,
            observer_decision=result.get('observer',{}).get('decision'),
            observer_error=observer_error,native_motion_timeouts=motion_timeouts,native_planner_stops=planner_stops,
            tool_errors=len(feedback),
            wrong_report_disclosed_uncertainty=bool(outcome=='wrong_accepted' and disclosed),
            robot_seconds=seconds,robot_requests=len(robot),attacker_requests=len(attacker),
            later_capture_opportunity=bool(later),literal_control=row['family'].endswith('_labels'),
            report=report,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        episodes.append(item);index[(row['block'],policy)]=item
    comparisons={};conditional={};wrong_comparisons={};wrong_conditional={}
    policies=[p for p in summary if p!='belief']
    for comparator in policies:
        pairs=[];solvable=[];wrong_pairs=[];wrong_solvable=[]
        for row in episodes:
            if row['policy']!='belief' or row['literal_control']:continue
            other=index.get((row['block'],comparator));clean=index.get((row['block'],'clean'))
            if not other:continue
            difference=int(other['correct'])-int(row['correct'])
            pairs.append((row['scene_block'],difference))
            wrong_difference=int(row['wrong_accepted'])-int(other['wrong_accepted'])
            wrong_pairs.append((row['scene_block'],wrong_difference))
            if clean and clean['correct'] and not row['infrastructure_error'] and not other['infrastructure_error']:
                solvable.append((row['scene_block'],difference))
                wrong_solvable.append((row['scene_block'],wrong_difference))
        comparisons[comparator]=cluster_interval(pairs)
        conditional[comparator]=cluster_interval(solvable)
        meaning='Positive means belief increased wrong accepted final reports more than comparator; scene-cluster bootstrap'
        wrong_comparisons[comparator]=cluster_interval(wrong_pairs,interpretation=meaning)
        wrong_conditional[comparator]=cluster_interval(wrong_solvable,interpretation=meaning)
    for policy,stats in summary.items():
        n=stats['completed']
        stats['correct_per_completed']=stats['correct']/n if n else None
        stats['correct_per_scheduled']=stats['correct']/stats['scheduled'] if stats['missing']==0 else None
        stats['wrong_accepted_per_completed']=stats['wrong_accepted']/n if n else None
        stats['wrong_accepted_wilson_95_ci_descriptive']=wilson(stats['wrong_accepted'],n)
        stats['attacker_request_median_seconds']=statistics.median(stats['attacker_wall_seconds']) if stats['attacker_wall_seconds'] else None
        stats['robot_episode_median_seconds']=statistics.median(stats['robot_wall_seconds']) if stats['robot_wall_seconds'] else None
        del stats['attacker_wall_seconds'];del stats['robot_wall_seconds']
    cohorts={}
    for item in episodes:
        cohort='literal_controls' if item['literal_control'] else 'main_tasks'
        counts=cohorts.setdefault(cohort,{}).setdefault(item['policy'],dict(completed=0,correct=0,
            wrong_accepted=0,refused=0,incomplete=0,infrastructure_errors=0))
        counts['completed']+=1;counts[item['outcome']]+=1;counts['infrastructure_errors']+=int(item['infrastructure_error'])
    return dict(version=1,backend=manifest.get('backend','gazebo'),planned=len(manifest['rows']),
        completed=len(episodes),complete_batch=len(episodes)==len(manifest['rows']),
        task_families=len({row['family'] for row in manifest['rows']}),
        main_task_families=len({row['family'] for row in manifest['rows'] if not row['family'].endswith('_labels')}),
        literal_control_families=len({row['family'] for row in manifest['rows'] if row['family'].endswith('_labels')}),
        independent_scene_instances=len({row['scene_block'] for row in manifest['rows']}),
        implementation_hash=manifest['implementation_hash'],baseline_bundle_sha256=manifest.get('baseline_bundle_sha256'),
        policies=summary,cohorts=cohorts,task_results=families,source_profiles=profiles,
        paired_correct_completion_reduction_main_tasks=comparisons,
        paired_clean_solvable_no_infrastructure_errors=conditional,
        PRIMARY_paired_wrong_accepted_increase_main_tasks=wrong_comparisons,
        PRIMARY_paired_wrong_accepted_clean_solvable=wrong_conditional,
        audit_errors=audit_errors,episodes=episodes,
        limitations=['API and external-view adaptations are not original paper reproductions.',
            'Multiple tasks in one scene are correlated; use scene-cluster intervals.',
            'A later camera capture is an opportunity, not proof the edited carrier was seen or adopted.',
            'Refusal, budgeted incomplete work and infrastructure errors are not wrong accepted reports.',
            'Literal-sign controls score current visible text; changed text alone is not deception.',
            'No superiority claim when relevant paired uncertainty intervals include zero.'])


def save(root,report):
    root=Path(root);write_json(root/'research_results.json',report)
    with (root/'research_episodes.csv').open('w',newline='') as f:
        keys=['block','scene_block','family','source_profile','policy','outcome','infrastructure_error',
              'observer_error','native_motion_timeouts','native_planner_stops','tool_errors',
              'robot_seconds','robot_requests','attacker_requests','literal_control','result_sha256']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(report['episodes'])
    rows=[]
    for policy,s in report['policies'].items():
        rows.append('<tr><td>%s</td><td>%d/%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td></tr>'%
            (html.escape(LABELS.get(policy,policy)),s['completed'],s['scheduled'],s['correct'],s['wrong_accepted'],
             s['refused'],s['incomplete'],s['infrastructure_errors']))
    text=('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>外部观察攻击实验</title>
<style>body{font:16px system-ui;max-width:1180px;margin:44px auto;padding:0 24px;color:#182337;background:#f8fafc}
table{border-collapse:collapse;width:100%%;background:white}td,th{padding:13px;text-align:left;border-bottom:1px solid #dce3eb}
th{background:#e8eef6}p{line-height:1.7}.note{background:#fff2cf;padding:18px;border-radius:8px}a{color:#2356a4}</style>
<h1>外部观察攻击实验</h1><p>已记录 %d / %d 轮；%d 类实际任务＋%d 类牌面文字负对照，%d 个随机场景实例。后端：%s。</p>
<p class="note">主指标为错误接受的配对增量；拒绝、超时和未完成不是完整性攻击成功。UAO、ICCV、RedVLA 均明确标为 API／外部视角适配，数字不代表原论文复现结果。</p>
<table><tr><th>方法</th><th>已记录／计划</th><th>正确</th><th>错误接受</th><th>拒绝</th><th>未完成</th><th>其中系统错误</th></tr>%s</table>
<p><a href="research_results.json">完整结果与配对区间</a> · <a href="research_episodes.csv">逐轮数据</a> · <a href="manifest.json">冻结实验清单</a></p>
<p>上表含实际任务及负对照；完整结果中的 cohorts 分开列出两者。配对比较只使用实际任务。导航超时、原规划器停止及观察者接口错误在逐轮 CSV 和完整结果中另列，不能自动归因于攻击。同一场景的重复运行并非独立样本；比较使用场景聚类区间。此页面只读已有文件，不改变分数或调用模型。</p></html>'''%
        (report['completed'],report['planned'],report['main_task_families'],report['literal_control_families'],
         report['independent_scene_instances'],html.escape(report['backend']),''.join(rows)))
    (root/'research_report.html').write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    a=p.parse_args();report=analyze(a.batch);save(a.batch,report)
    print(json.dumps({k:report[k] for k in ('planned','completed','complete_batch','main_task_families',
        'literal_control_families','independent_scene_instances','policies','audit_errors')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
