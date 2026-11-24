"""Declared local operator inputs, independent of whole-task names.

These are inputs to learned heads, not decisions or victim rules. DECIDE and
CHECK features use declared bindings and acquisition quality; they cannot read
candidate physical truth or unrecognized raw carrier/capture contents.
"""
import math
from numbers import Real
import numpy as np

FIELDS=('cracked','temperature','display_text','box_count','color','vehicle_present','restricted','blocked','extinguisher_present')
OPS=('eq','ne','gt','ge','lt','le')


def numeric(value,field):
    if isinstance(value,bool):return float(value),1.
    if isinstance(value,Real) and math.isfinite(value):
        return float(value)/(100. if field=='temperature' else 10. if field=='box_count' else 1.),1.
    return 0.,0.


def leaves(expression):
    if not expression:return []
    if expression.get('op') in ('and','or'):
        return [v for node in expression['args'] for v in leaves(node)]
    return [expression]


def constant(expression,parameters):
    if 'value' in expression:return expression['value']
    if 'prior' in expression:return parameters[expression['prior']]
    return None


def declared_values(state,source):
    out={}
    for b in state.bindings:
        if b.object_source==source and b.status=='certain' and b.declared_value is not None:out[b.field]=b.declared_value
    return out


def compare(expression,values,parameters):
    if not expression:return 1.,1.
    op=expression['op']
    if op in ('and','or'):
        parts=[compare(v,values,parameters) for v in expression['args']]
        if op=='and' and any(k and not v for v,k in parts):return 0.,1.
        if op=='or' and any(k and v for v,k in parts):return 1.,1.
        known=all(k for _,k in parts)
        if not known:return 0.,0.
        return float((all if op=='and' else any)(v for v,_ in parts)),1.
    args=[]
    for operand in (expression['left'],expression['right']):
        if 'field' in operand:
            if operand['field'] not in values:return 0.,0.
            args.append(values[operand['field']])
        else:args.append(constant(operand,parameters))
    left,right=args
    try:
        result={'eq':lambda:left==right,'ne':lambda:left!=right,'gt':lambda:left>right,
                'ge':lambda:left>=right,'lt':lambda:left<right,'le':lambda:left<=right}[op]()
    except (TypeError,KeyError):return 0.,0.
    return float(result),1.


def local_features(program,target,source,state,perception,body_event,parameters,clock):
    values=declared_values(state,source);roles=set(program['roles']);reduction=program.get('reduce') or {}
    result=[float(field in roles) for field in FIELDS]
    result.extend([float(program['scope']=='target'),float(target==source),float(program.get('literal',False)),
        float(reduction.get('op')=='min'),float(reduction.get('op')=='max'),
        float(program.get('filter',{}).get('op')=='and') if program.get('filter') else 0.,
        float(program.get('filter',{}).get('op')=='or') if program.get('filter') else 0.])
    result.extend([float(reduction.get('field')==field) for field in FIELDS])
    constraints=leaves(program.get('filter'))
    for field in FIELDS:
        constraints_here=[e for e in constraints if e.get('left',{}).get('field')==field]
        node=constraints_here[0] if constraints_here else None
        operand=constant(node['right'],parameters) if node else None
        v,isnum=numeric(values.get(field),field);bound=[b for b in state.bindings if b.object_source==source and b.field==field]
        certain=[b.declared_value for b in bound if b.status=='certain' and b.declared_value is not None]
        captures=[c for c in state.captures if c.source==source and c.field==field]
        # Critically, no Capture.carrier_value is accessed here.
        qualities=[c.view_quality for c in captures if c.quality_measured]
        visible=perception['stations'][source].get(field)
        static_value,static_numeric=numeric(visible,field)
        threshold,threshold_numeric=numeric(operand,field)
        result.extend([float(field in values),v,isnum,float(len(set(certain))>1),
            float(any(b.status=='conflict' for b in bound)),float(any(b.status in ('uncertain','unreadable') for b in bound)),
            math.log1p(len(captures)),float(bool(qualities)),max(qualities or [0.]),qualities[-1] if qualities else 0.,
            static_value,static_numeric,float(visible is not None),
            threshold,threshold_numeric,float(field in values and operand is not None and values[field]==operand),
            (v-threshold) if isnum and threshold_numeric else 0.,
            float(node is not None)])
        result.extend([float(node is not None and node.get('op')==op) for op in OPS])
        for op in OPS:
            operands=[numeric(constant(e['right'],parameters),field)[0] for e in constraints_here if e.get('op')==op]
            result.extend([float(len(operands)),min(operands or [0.]),max(operands or [0.])])
    result.extend(compare(program.get('filter'),values,parameters))
    all_known=[declared_values(state,i) for i in range(len(perception['stations']))]
    for field in FIELDS:
        known=[numeric(v[field],field)[0] for v in all_known if field in v and numeric(v[field],field)[1]]
        result.extend([sum(field in v for v in all_known)/len(all_known),len(known)/len(all_known),min(known or [0.]),max(known or [0.])])
    anchor=perception['anchors'][source];event=np.asarray(body_event)
    width=max(.005,anchor['w'])
    result.extend([(event[0]-anchor['u'])/width,(event[1]-anchor['v'])/width,event[2]/width,event[3]/width,*event[4:6]])
    result.extend([math.log1p(max(0.,clock)),math.log1p(len(state.assertions)),
                   float(any(a.object_source==source for a in state.assertions))])
    return np.array(result,dtype=np.float32)
