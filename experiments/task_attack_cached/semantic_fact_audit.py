"""Predeclared v9 factual representation audit; raw strict-format scores stay untouched.

Only complete, unambiguous numeric/Celsius/count expressions are normalized.
No task names, policies, outcomes or expected answers affect conversion.
"""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from experiments.task_attack.evaluate import evaluate
from experiments.task_attack.research_report import cluster_interval

NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
TEMP=re.compile(r'^\s*('+NUMBER+r')\s*(?:°\s*C|℃|C|摄氏度)?\s*$',re.IGNORECASE)
COUNT=re.compile(r'^\s*('+NUMBER+r')\s*(?:箱|boxes|box)?\s*$',re.IGNORECASE)


def numeric_representation(key,value):
    if not isinstance(value,str):return value
    pattern=TEMP if key=='temperature' else COUNT if key in ('box_count','order_count') else None
    if pattern is None:return value
    match=pattern.fullmatch(value)
    if not match:return value
    number=float(match.group(1))
    if not math.isfinite(number):return value
    if key!='temperature':
        if number<0 or not number.is_integer():return value
        return int(number)
    return number


def normalize_report(report):
    normalized=copy.deepcopy(report);changes=[]
    if not isinstance(normalized,dict):return normalized,changes
    rows=normalized.get('records',[])
    if not isinstance(rows,list):return normalized,changes
    for index,row in enumerate(rows):
        if not isinstance(row,dict):continue
        values=row.get('facts')
        if not isinstance(values,dict):continue
        for key,value in list(values.items()):
            replacement=numeric_representation(key,value)
            if type(replacement)!=type(value):
                values[key]=replacement
                changes.append(dict(record_index=index,key=key,original=value,normalized=replacement))
    return normalized,changes


def audit_batch(root,manifest=None):
    root=Path(root);manifest=manifest or json.loads((root/'manifest.json').read_text());methods={};entries=[];lookup={};changes=[]
    for row in manifest['rows']:
        if row['family'].endswith('_labels'):continue
        stats=methods.setdefault(row['policy'],dict(scheduled=0,completed=0,correct=0,wrong_accepted=0,refused=0,incomplete=0,infrastructure_errors=0,changed_outcomes=0,normalized_reports=0))
        stats['scheduled']+=1;directory=root/row['directory'];path=directory/'result.json'
        if not path.exists():continue
        raw=json.loads(path.read_text());truth=json.loads((directory/'private/truth.json').read_text())
        report=raw.get('report');normalized,conversions=normalize_report(report)
        if report is None:score=raw.get('score',{})
        else:score=evaluate(truth,normalized,raw.get('actual_edit'))
        rawscore=raw.get('score',{});outcome=score.get('outcome','incomplete');changed=outcome!=rawscore.get('outcome','incomplete')
        stats['completed']+=1;stats[outcome]+=1
        stats['infrastructure_errors']+=int(bool(raw.get('error') or not raw.get('execution_completed')))
        stats['changed_outcomes']+=int(changed);stats['normalized_reports']+=int(bool(conversions))
        entry=dict(directory=row['directory'],block=row['block'],family=row['family'],policy=row['policy'],seed=row['seed'],source_profile=row['source_profile'],outcome=outcome,raw_outcome=rawscore.get('outcome','incomplete'),conversions=conversions,changed_outcome=changed,raw_score=rawscore,audited_score=score,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),truth_sha256=hashlib.sha256((directory/'private/truth.json').read_bytes()).hexdigest())
        entries.append(entry);lookup[(row['block'],row['policy'])]=entry
        if conversions:changes.append(entry)
    paired={}
    for policy in manifest['policies']:
        if policy=='belief':continue
        pairs=[]
        for entry in entries:
            if entry['policy']!='belief':continue
            other=lookup.get((entry['block'],policy))
            if other:pairs.append((entry['block'],int(entry['outcome']=='wrong_accepted')-int(other['outcome']=='wrong_accepted')))
        paired[policy]=cluster_interval(pairs,interpretation='Predeclared v9 representation-normalized factual outcome; positive means more wrong accepted reports under belief. Raw strict-format scores remain separate.')
    return dict(version=1,status='predeclared_v7_factual_primary_with_raw_auxiliary',rule='Full-match unambiguous Celsius or numeric count representations only; no task, policy, expected value or outcome used in normalization.',raw_scores_unchanged=True,methods=methods,paired_wrong_accepted_main_tasks=paired,changed_or_normalized_reports=changes,episodes=entries)


def save(root,audit):
    root=Path(root);(root/'semantic_fact_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    fields=['directory','block','family','policy','seed','source_profile','raw_outcome','outcome','changed_outcome','result_sha256']
    with (root/'semantic_fact_episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(audit['episodes'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True);args=parser.parse_args()
    root=args.batch;result=audit_batch(root);save(root,result)
    print(json.dumps(dict(methods=result['methods'],changes=[dict(directory=x['directory'],conversions=x['conversions'],changed_outcome=x['changed_outcome']) for x in result['changed_or_normalized_reports']]),ensure_ascii=False,indent=2))
