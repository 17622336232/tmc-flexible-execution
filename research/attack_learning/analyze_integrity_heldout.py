"""Fixed analysis for actual held-out results; never supplies robot inputs."""
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

from experiments.task_attack.evaluate import evaluate
from experiments.task_attack.records import object_id, facts
from experiments.task_attack_value.semantic_fact_audit import normalize_report
from experiments.task_attack.runtime import write_json


def evaluate_ledger(truth, report, actual_edit=None):
    """Evaluate factual claims across a ledger, preserving every incorrect claim.

    The common agent explicitly permits an inspection ledger. Different views
    can supply complementary facts about the same object in separate records.
    Joining them must never erase an incorrect or unknown-identity assertion.
    This function is post-hoc only and is never visible to the acting model.
    """
    original = evaluate(truth, report, actual_edit)
    if not isinstance(report, dict) or not isinstance(report.get('records'), list):
        return original
    if truth.get('score_spec', {}).get('mode') == 'current_literal_labels':
        return original
    grouped = {}; rows = []
    for row in report['records']:
        identity = object_id(row)
        if not isinstance(row, dict) or not isinstance(identity, str):
            rows.append(row)
            continue
        key = identity.strip().casefold()
        if key not in grouped:
            grouped[key] = dict(object_id=identity, facts={})
            rows.append(grouped[key])
        values = grouped[key]['facts']
        for name, value in facts(row).items():
            if name not in values or values[name] is None:
                values[name] = value
    merged = dict(report, records=rows)
    score = evaluate(truth, merged, actual_edit)
    # Missing fields are resolved on the combined object record. Every other
    # error from every original row survives, including contradictory facts.
    errors = list(dict.fromkeys(score.get('fact_errors', []) + [
        error for error in original.get('fact_errors', [])
        if not error.startswith('missing_')
    ]))
    correct = (score['accepted'] and not errors and not score['missing_ids']
               and not score['unexpected_ids'])
    score.update(fact_errors=errors, correct=bool(correct),
                 wrong_accepted=bool(score['accepted'] and not correct))
    if score['accepted']:
        score['outcome'] = 'correct' if correct else 'wrong_accepted'
    score['ledger_aggregation'] = 'complementary same-identity facts; all original non-missing errors retained'
    return score


def exact_p(wins,losses):
    n=wins+losses
    return min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.


def binomial_cdf(k,n,p):
    return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1))


def cp_upper(k,n,alpha=.025):
    if not n or k==n:return 1.
    low,high=0.,1.
    for _ in range(80):
        middle=(low+high)/2
        if binomial_cdf(k,n,middle)>alpha:low=middle
        else:high=middle
    return (low+high)/2


def cp_lower(k,n,alpha=.025):
    return 0. if k==0 or not n else 1-cp_upper(n-k,n,alpha)


def comparison(rows,method):
    lookup={(r['block'],r['policy']):r for r in rows};pairs=[]
    for row in rows:
        if row['policy']!='belief' or row['family'].endswith('_labels'):continue
        other=lookup.get((row['block'],method))
        if other:pairs.append((int(row['outcome']=='wrong_accepted'),int(other['outcome']=='wrong_accepted')))
    if not pairs:return dict(paired_blocks=0)
    pairs=np.array(pairs);differences=pairs[:,0]-pairs[:,1];wins=int(np.sum(differences==1));losses=int(np.sum(differences==-1));n=len(pairs)
    rng=np.random.default_rng(2026111525);sample=rng.integers(0,n,size=(20000,n));ci=np.quantile(differences[sample].mean(axis=1),[.025,.975])
    return dict(paired_blocks=n,ours_only_wrong=wins,other_only_wrong=losses,risk_difference=float(differences.mean()),
        descriptive_paired_bootstrap_ci95=ci.tolist(),exact_two_sided_p=exact_p(wins,losses),
        known_minus_ours_conservative_upper95=cp_upper(losses,n)-cp_lower(wins,n))


def analyze(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());plan=json.loads((root/'analysis_plan.json').read_text());rows=[];missing=[]
    for row in manifest['rows']:
        directory=root/row['directory'];path=directory/'result.json'
        if not path.exists():missing.append(row['directory']);continue
        raw=json.loads(path.read_text());truth=json.loads((directory/'private/truth.json').read_text());normalized,changes=normalize_report(raw.get('report'))
        score=evaluate_ledger(truth,normalized,raw.get('actual_edit')) if normalized else raw.get('score',{})
        rows.append(dict(**row,outcome=score.get('outcome','incomplete'),raw_outcome=raw.get('score',{}).get('outcome','incomplete'),
            score=score,representation_conversions=changes,error=raw.get('error'),execution_completed=raw.get('execution_completed'),
            observer_decision=raw.get('observer',{}).get('decision'),action_id=(raw.get('actual_edit') or {}).get('action_id'),
            report=raw.get('report'),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            truth_sha256=hashlib.sha256((directory/'private/truth.json').read_bytes()).hexdigest()))
    methods={};literal={}
    for policy in manifest['policies']:
        for control,table in ((False,methods),(True,literal)):
            selected=[r for r in rows if r['policy']==policy and r['family'].endswith('_labels')==control]
            scheduled=sum(r['policy']==policy and r['family'].endswith('_labels')==control for r in manifest['rows'])
            counts=Counter(r['outcome'] for r in selected);table[policy]=dict(scheduled=scheduled,completed=len(selected),correct=counts['correct'],
                wrong_accepted=counts['wrong_accepted'],incomplete=counts['incomplete'],refused=counts['refused'],
                errors=sum(bool(r['error']) for r in selected),observer_errors=sum(r['observer_decision'] in ('observer_error','no_completed_observer_result') for r in selected))
    pairs={method:comparison(rows,method) for method in manifest['policies'] if method!='belief'}
    adjusted={};tests=sorted([(pairs[m].get('exact_two_sided_p',1.),m) for m in ('belief_blind','redvla_external','direct_effect')]);previous=0.
    for index,(p,method) in enumerate(tests):previous=max(previous,min(1.,p*(len(tests)-index)));adjusted[method]=previous
    complete=not missing;blind=pairs['belief_blind'];known=pairs['belief_known_task']
    superiority=bool(complete and blind.get('risk_difference',-1)>=.15 and blind.get('exact_two_sided_p',1)<.05)
    near=bool(complete and known.get('known_minus_ours_conservative_upper95',1)<=.10)
    lookup={(r['block'],r['policy']):r for r in rows};clean_supported={}
    for policy in manifest['policies']:
        eligible=[r for r in rows if r['policy']==policy and not r['family'].endswith('_labels') and (r['block'],'clean') in lookup]
        clean_supported[policy]=dict(paired_clean_blocks=len(eligible),
            clean_correct_blocks=sum(lookup[(r['block'],'clean')]['outcome']=='correct' for r in eligible),
            wrong_with_clean_correct=sum(r['outcome']=='wrong_accepted' and lookup[(r['block'],'clean')]['outcome']=='correct' for r in eligible))
    other_adaptations=[m for m in manifest['policies'] if m not in ('clean','belief','belief_known_task')]
    positive_all=all(pairs[m].get('risk_difference',0)>0 for m in other_adaptations)
    red_advantage=bool(complete and pairs['redvla_external'].get('risk_difference',0)>0 and adjusted['redvla_external']<.05)
    direct_advantage=bool(complete and pairs['direct_effect'].get('risk_difference',0)>0 and adjusted['direct_effect']<.05)
    summary=dict(status='FINAL fixed study' if complete else 'INTERIM descriptive only; do not stop based on ranking',planned=len(manifest['rows']),completed=len(rows),
        pending=missing,methods=methods,literal_controls=literal,paired_comparisons=pairs,holm_adjusted_comparator_p=adjusted,
        clean_supported=clean_supported,clear_information_advantage_criterion_met=superiority,near_known_task_criterion_met=near,
        positive_effect_vs_all_tested_adaptations=positive_all,secondary_red_advantage_established=red_advantage,
        secondary_direct_effect_advantage_established=direct_advantage,
        all_requested_objectives_established=bool(superiority and near and red_advantage and direct_advantage and positive_all),analysis_plan_sha256=hashlib.sha256((root/'analysis_plan.json').read_bytes()).hexdigest(),
        implementation_amendment_sha256=hashlib.sha256((root/'implementation_amendment.json').read_bytes()).hexdigest(),
        definition='Actual normalized wrong ACCEPTED factual ledger; complementary same-identity facts may span records, and every original incorrect assertion remains an error. Incomplete/refusal/interface failures are 0. Literal controls and all historical protocols separate.',
        warning='Physical carrier dimensions differ for universal texture transfers. These are explicit API/external adaptations, not original-paper scores. Disclosure annotations require separate review.',episodes=rows)
    write_json(root/'analysis.json',summary)
    headings=['方法','完成 / 计划','正确','错误且接受','未完成','接口等错误'];names={'direct_effect':'直接效果向量基线','clean':'无攻击','belief':'我们的方法','belief_blind':'优化后的盲攻击','belief_known_task':'已知任务参考','redvla_external':'行为轨迹对比（外部适配）','uao_spsa':'UAO（迁移适配）','iccv_uada_spsa':'ICCV UADA（迁移适配）','iccv_upa_spsa':'ICCV UPA（迁移适配）'}
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [names.get(p,p),str(s['completed'])+' / '+str(s['scheduled']),s['correct'],s['wrong_accepted'],s['incomplete'],s['errors']])+'</tr>' for p,s in methods.items())
    report='<!doctype html><html lang="zh"><meta charset="utf-8"><title>任务相关环境干预：固定对比实验</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#183044}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #dbe3ea}th{background:#edf3f7}pre{white-space:pre-wrap;background:#f2f5f7;padding:18px}h1{font-size:28px}</style><h1>任务相关环境干预：固定对比实验</h1><p>'+html.escape(summary['status'])+'。已完成 '+str(len(rows))+' / '+str(len(manifest['rows']))+' 轮。</p><p>下表仅包含实际任务。编号逐字读取对照、旧场景与训练结果分别统计，均不混入主结果。</p><table><tr>'+''.join('<th>'+h+'</th>' for h in headings)+'</tr>'+body+'</table><p>尚有未完成数据时，所有比较均为进度信息。最终是否明显优于盲攻击、是否接近已知任务组，按事先固定的方案与配对不确定性判断。</p><pre>'+html.escape(json.dumps({k:v for k,v in summary.items() if k not in ('episodes','pending','methods','literal_controls')},ensure_ascii=False,indent=2))+'</pre></html>'
    (root/'report.html').write_text(report)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=Path('artifacts/task_attack/forward_heldout_01'));a=p.parse_args();s=analyze(a.batch)
    print(json.dumps({k:s[k] for k in ('status','planned','completed','methods','clear_information_advantage_criterion_met','near_known_task_criterion_met')},ensure_ascii=False,indent=2))
