"""Keep the referent argument when sharing a local operator across sources.

Previously a non-target source only saw target_match=False, so different
remote targets became indistinguishable to its motion head. Relative target
geometry and the declared neighbour set are inputs, never route rules.
"""
import numpy as np
from factorized_local_operators import relative_features as base_features


def relative_features(global_vector,source,bindings):
    x=list(base_features(global_vector,source,bindings));width=len(global_vector)//4
    ports=[global_vector[width*i:width*(i+1)] for i in range(4)]
    referents=[p for p in ports if p[10]>.5]
    if referents:
        x.extend([1.,*referents[0][-9:-3]])
    else:x.extend([0.]*7)
    own=ports[source];own_width=max(1e-8,own[-7])
    # A set of other declared local states, ordered by their RELATIVE image
    # displacement. It has no object name or whole-task/family ID feature.
    neighbours=[]
    for i,p in enumerate(ports):
        ratio=own_width/max(1e-8,p[-7])
        dx=own[-9]-p[-9]*ratio;dy=own[-8]-p[-8]*ratio
        known=[]
        for field in ('cracked','temperature'):
            current=[b for b in bindings if b['object_source']==i and b['field']==field and b['status']=='certain' and b['declared_value'] is not None]
            value=current[-1]['declared_value'] if current else None
            number=float(value) if isinstance(value,(bool,int,float)) else 0.
            known.extend([float(bool(current)),number/(100. if field=='temperature' else 1.)])
        neighbours.append((dx,[dx,dy,float(p[10]),*known]))
    for _,values in sorted(neighbours,key=lambda r:r[0]):x.extend(values)
    return x


def configure(models=None):
    import factorized_local_operators
    import local_gate_modes
    if models is None or models.get('port_encoder')=='referent_and_set_v1':
        factorized_local_operators.relative_features=relative_features
        local_gate_modes.relative_features=relative_features
