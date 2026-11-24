"""Compositional hypotheses over public tasks and OWN proxy world particles.

These operators belong to the attacker, not the robot. They do not prescribe
robot routes, observations or answers. No current target truth is read here.
"""
import copy
import math
import re


def operand(expression, fact, parameters):
    if 'field' in expression:return fact[expression['field']]
    if 'value' in expression:return expression['value']
    if 'prior' in expression:return parameters[expression['prior']]
    raise ValueError('Unknown public program operand')


def predicate(expression, fact, parameters):
    if expression is None:return True
    op=expression['op']
    if op=='and':return all(predicate(e,fact,parameters) for e in expression['args'])
    if op=='or':return any(predicate(e,fact,parameters) for e in expression['args'])
    left=operand(expression['left'],fact,parameters);right=operand(expression['right'],fact,parameters)
    if op=='eq':return left==right
    if op=='ne':return left!=right
    if op=='gt':return left>right
    if op=='ge':return left>=right
    if op=='lt':return left<right
    if op=='le':return left<=right
    raise ValueError('Unsupported public relational operator')


def answer(program, world, parameters, target=None):
    rows=[(i,f) for i,f in enumerate(world) if
          (program['scope']!='target' or i==target) and predicate(program.get('filter'),f,parameters)]
    reduce=program.get('reduce')
    if reduce and rows:
        field=reduce['field'];op=reduce['op']
        extreme=(min if op=='min' else max if op=='max' else None)
        if extreme is None:raise ValueError('Unknown reduction')
        value=extreme(f[field] for _,f in rows);rows=[(i,f) for i,f in rows if f[field]==value]
    return tuple((i,tuple((role,f[role]) for role in program['roles'])) for i,f in rows)


def payload(action):
    identifier=action['action_id'];match=re.fullmatch(r'place_decal_(shell|display)_(smooth|cracked|05|35|95)_(\d+)',identifier)
    if not match:return None
    kind,value,source=match.groups()
    return dict(role='cracked' if kind=='shell' else 'temperature',source=int(source)-1,
                value=(value=='cracked') if kind=='shell' else int(value))


def known_program(task, programs, stations):
    """Diagnostic reference from authorized natural language, never a score spec.

    The public program family is compiled from the actual text, not selected
    from a hidden trial name. Unknown/ambiguous text fails instead of guessing.
    """
    target=None
    for station in stations:
        identity=station.get('identity')
        if identity and re.search(r'(?<![A-Za-z0-9-])'+re.escape(identity)+r'(?![A-Za-z0-9-])',task,re.I):
            if target is not None:raise ValueError('Ambiguous authorized referent')
            target=station['index']-1
    fields=[]
    if '裂纹' in task:fields.append('cracked')
    if '温度' in task:fields.append('temperature')
    if '文字' in task or '面板' in task and not fields:fields=['display_text']
    candidates=[p for p in programs if set(p['roles'])==set(fields) and
        (p['scope']=='target')==(target is not None)]
    if '最低' in task:candidates=[p for p in candidates if p.get('reduce',{} or None) and p['reduce']['op']=='min']
    elif target is None:
        candidates=[p for p in candidates if not p.get('reduce')]
        if len(candidates)>1:
            false='无裂纹' in task or '没有裂纹' in task or '未见裂纹' in task
            candidates=[p for p in candidates if ('false' in str(p.get('filter')).lower())==false]
    if len(candidates)!=1:raise ValueError('Authorized language has no unique public program')
    return candidates[0],target


def semantic_effect(programs, worlds, parameters, action, stations, known=None):
    """P(answer changes | the selected role/source is relevant).

    Relevance is a separate JOINT learned dependency. Conditioning here avoids
    counting the task prior a second time. Whole-world particles preserve min,
    conjunction and identifier relationships instead of independent cell flips.
    """
    change=payload(action)
    if change is None:return 0.
    role,source,value=change['role'],change['source'],change['value']
    if source>=len(stations):return 0.
    hypotheses=[known[0]] if known else programs;total=effect=0.
    for program in hypotheses:
        if program.get('literal') or role not in program['roles']:continue
        if known and program['scope']=='target' and known[1]!=source:continue
        # A uniformly unknown requested identity makes this source relevant
        # with 1/N probability; all-source programs have relevance 1.
        relevance=1./len(stations) if not known and program['scope']=='target' else 1.
        for original in worlds:
            world=copy.deepcopy(original)
            # Condition only on actual STATIC external RGB measurements.
            for i,station in enumerate(stations):
                for key in ('cracked','temperature'):
                    if station.get(key) is not None:world[i][key]=station[key]
            target=(known[1] if known else source) if program['scope']=='target' else None
            before=answer(program,world,parameters,target);world[source][role]=value
            after=answer(program,world,parameters,target)
            total+=relevance;effect+=relevance*float(before!=after)
    return effect/total if total else 0.
