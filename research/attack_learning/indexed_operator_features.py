"""Index the SAME declared-feature schema once, without scanning raw values.

Only an execution speed optimization. The original encoder remains the parity
reference. No captured carrier value or hidden-world truth is accessed here.
"""
import math
import numpy as np
from operator_features import FIELDS,OPS,numeric,leaves,constant,compare


def global_features(program,target,state,perception,body,parameters,clock,named=False,cache=None):
    n=len(perception['stations']);values=[{} for _ in range(n)];bound={};captures={}
    for b in state.bindings:
        bound.setdefault((b.object_source,b.field),[]).append(b)
        if b.status=='certain' and b.declared_value is not None:values[b.object_source][b.field]=b.declared_value
    for c in state.captures:
        key=(c.source,c.field)
        if key not in captures:captures[key]=[0,False,0.,0.]
        a=captures[key];a[0]+=1
        if c.quality_measured:
            a[2]=max(a[2],c.view_quality) if a[1] else c.view_quality;a[1]=True;a[3]=c.view_quality
    if cache is not None:
        key=(id(program),target,id(perception),id(parameters),float(clock),tuple(body),named,state.bindings,
             tuple((k,tuple(v)) for k,v in captures.items()),len(state.assertions),tuple(sorted({a.object_source for a in state.assertions})),state.named_identities)
        if key in cache:return cache[key]
    roles=set(program['roles']);reduction=program.get('reduce') or {};constraints=leaves(program.get('filter'));event=np.asarray(body)
    prefix=[float(f in roles) for f in FIELDS]+[float(program['scope']=='target'),0.,float(program.get('literal',False)),
        float(reduction.get('op')=='min'),float(reduction.get('op')=='max'),float(program['filter'].get('op')=='and') if program.get('filter') else 0.,
        float(program['filter'].get('op')=='or') if program.get('filter') else 0.]+[float(reduction.get('field')==f) for f in FIELDS]
    field_static=[]
    for field in FIELDS:
        here=[e for e in constraints if e.get('left',{}).get('field')==field];node=here[0] if here else None
        operand=constant(node['right'],parameters) if node else None;threshold,threshold_numeric=numeric(operand,field)
        tail=[float(node is not None and node.get('op')==op) for op in OPS]
        for op in OPS:
            operands=[numeric(constant(e['right'],parameters),field)[0] for e in here if e.get('op')==op]
            tail.extend([float(len(operands)),min(operands or [0.]),max(operands or [0.])])
        field_static.append((operand,threshold,threshold_numeric,float(node is not None),tail))
    summary=[]
    for field in FIELDS:
        known=[numeric(v[field],field)[0] for v in values if field in v and numeric(v[field],field)[1]]
        summary.extend([sum(field in v for v in values)/n,len(known)/n,min(known or [0.]),max(known or [0.])])
    asserted={a.object_source for a in state.assertions};result=[]
    for source in range(n):
        row=prefix[:];row[10]=float(target==source);local=values[source]
        for field,(operand,threshold,threshold_numeric,present,tail) in zip(FIELDS,field_static):
            v,isnum=numeric(local.get(field),field);bs=bound.get((source,field),());certain=[b.declared_value for b in bs if b.status=='certain' and b.declared_value is not None]
            count,hasq,maxq,lastq=captures.get((source,field),(0,False,0.,0.));visible=perception['stations'][source].get(field);static_value,static_numeric=numeric(visible,field)
            row.extend([float(field in local),v,isnum,float(len(set(certain))>1),float(any(b.status=='conflict' for b in bs)),float(any(b.status in ('uncertain','unreadable') for b in bs)),
                math.log1p(count),float(hasq),maxq,lastq,static_value,static_numeric,float(visible is not None),threshold,threshold_numeric,
                float(field in local and operand is not None and local[field]==operand),(v-threshold) if isnum and threshold_numeric else 0.,present]);row.extend(tail)
        row.extend(compare(program.get('filter'),local,parameters));row.extend(summary)
        if named:
            from epistemic_operator_features import named_features
            row.extend(named_features(state,source));row.extend(named_features(state,target if program['scope']=='target' else None))
        anchor=perception['anchors'][source];width=max(.005,anchor['w'])
        row.extend([(event[0]-anchor['u'])/width,(event[1]-anchor['v'])/width,event[2]/width,event[3]/width,*event[4:6],math.log1p(max(0.,clock)),math.log1p(len(state.assertions)),float(source in asserted)])
        result.extend(np.array(row,dtype=np.float32).tolist())
    if cache is not None:cache[key]=result
    return result
