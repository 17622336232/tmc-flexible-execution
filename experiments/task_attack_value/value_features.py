"""Shared source/payload relations using actual visible scene facts only."""
import numpy as np

from .programs import payload


def program_features(known,source):
    if known is None:return []
    program,target=known;crack_test=temperature_test=0.;threshold=0.
    def visit(node):
        nonlocal crack_test,temperature_test,threshold
        if not node:return
        if node.get('op') in ('and','or'):
            for child in node['args']:visit(child)
        elif node.get('left',{}).get('field')=='cracked' and node.get('op')=='eq':
            crack_test=1. if node['right'].get('value') else -1.
        elif node.get('left',{}).get('field')=='temperature':
            temperature_test=1. if node.get('op') in ('gt','ge') else -1.
            threshold=float(node.get('right',{}).get('value',70))/100.
    visit(program.get('filter'))
    return [float('cracked' in program['roles']),float('temperature' in program['roles']),
        float(program['scope']=='target'),float(target==source),crack_test,temperature_test,threshold,
        float(bool(program.get('reduce'))),float(program.get('literal',False))]


def source_features(x,perception,source):
    stations=perception['stations'];anchors=perception['anchors'];row=x[source];anchor=anchors[source]
    cracks=np.array([float(s.get('cracked') or 0.) for s in stations]);known=np.array([float(s.get('cracked') is not None) for s in stations])
    temperature=stations[source].get('temperature');own=cracks[source];own_known=known[source]
    points=row[19:29].reshape(5,2)*anchor['w']+np.array([anchor['u'],anchor['v']])
    centers=np.array([[a['u'],a['v']] for a in anchors]);width=max(.005,float(np.mean([a['w'] for a in anchors])))
    context=[]
    for point in points:
        logits=-.5*np.sum(((centers-point)/width)**2,axis=1);weights=np.exp(logits-logits.max());weights/=weights.sum()
        context.extend([float(weights.dot(cracks*known)),float(weights.dot(known))])
    # In-image forward asymmetry comes from cyan front rotors/top vs gray rear
    # rotors. It is a noisy measured descriptor, not a simulator heading.
    axis=np.array([row[29]*row[11],row[30]*row[12]])
    delta=-row[:2];alignment=float(delta.dot(axis)/max(1e-8,np.linalg.norm(delta)*np.linalg.norm(axis)))
    return np.concatenate([row,np.array([own,own_known,0. if temperature is None else temperature/100.,float(temperature is not None),
        float((cracks*known).mean()),float(known.mean()),*context,alignment,*axis,float(np.linalg.norm(axis)),
        row[0]*anchor['w']+anchor['u'],row[1]*anchor['w']+anchor['v'],row[11]*anchor['w'],row[12]*anchor['w']])])


def action_features(x,perception,actions,known=None):
    rows=[];selected=[];cache={}
    for action in actions:
        change=payload(action)
        if change is None:continue
        source=change['source']
        if source not in cache:cache[source]=source_features(x,perception,source)
        role=float(change['role']=='temperature');value=float(change['value'])/(100. if role else 1.)
        rows.append(np.concatenate([cache[source],np.array([role,value,*program_features(known,source)])]));selected.append(action)
    return np.array(rows),selected
