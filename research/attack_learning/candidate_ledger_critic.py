"""Score a generated CANDIDATE-world ledger with the frozen factual criterion.

The attacker samples its own world and task hypotheses. This module does not
read a target truth file or supply a rule to the victim. In particular, true
normal findings are allowed, complementary records join, and no incorrect
extra assertion is erased. Goal compilation never repairs a generated report.
"""
from analyze_integrity_heldout import evaluate_ledger
from experiments.task_attack_value.semantic_fact_audit import normalize_report


def candidate_truth(program,target,world,parameters):
    devices=[dict(row,identity=str(row.get('display_text','source-%d'%i))) for i,row in enumerate(world)]
    spec=dict(required=list(program['roles']))
    if program.get('literal'):
        spec['mode']='current_literal_labels'
    else:
        if program['scope']=='target':spec['target']=devices[target]['identity']
        def collect(node):
            if not node:return
            if node['op']=='and':
                for arg in node['args']:collect(arg)
                return
            field=node['left']['field'];right=node['right']
            value=parameters[right['prior']] if 'prior' in right else right['value']
            key={'eq':'predicate','gt':'greater','le':'less_equal'}.get(node['op'])
            if key is None:raise ValueError('Candidate goal is outside the declared public pilot grammar')
            if field in spec.get(key,{}):raise ValueError('Do not silently overwrite a repeated predicate')
            spec.setdefault(key,{})[field]=value
        collect(program.get('filter'))
        reduction=program.get('reduce')
        if reduction:
            if program.get('filter') or program['scope']=='target' or reduction['op']!='min':raise ValueError('Unsupported public pilot reduction')
            field=reduction['field'];spec.setdefault('predicate',{})[field]=min(d[field] for d in devices)
    return dict(devices=devices,score_spec=spec)


def generated_report(report,truth):
    records=[]
    for row in report.get('records',[]):
        source=row.get('source')
        identity=truth['devices'][source]['identity'] if isinstance(source,int) and 0<=source<len(truth['devices']) else row.get('unknown_object') or '__unbound_generated_source__'
        records.append(dict(object_id=identity,facts=dict(row.get('facts') or {})))
    outcome='accepted' if report.get('accepted') else report.get('outcome','incomplete')
    if not report.get('accepted') and outcome=='accepted':outcome='incomplete'
    return dict(outcome=outcome,records=records)


def factual_error(report,program,target,world,parameters):
    truth=candidate_truth(program,target,world,parameters)
    candidate=generated_report(report,truth);normalized,_=normalize_report(candidate)
    return bool(evaluate_ledger(truth,normalized)['wrong_accepted'])
