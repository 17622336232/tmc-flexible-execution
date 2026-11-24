"""Local name-awareness INPUTS; no manual search/approach/submit rules.

The request 'find EQ-04' and a declared recognition of EQ-04 are separate.
Physical target geometry may be known to an external observer without being
recognized by the acting robot. These features let shared learned kernels
represent that distinction; the model still chooses every action.
"""
import math
import numpy as np

from operator_features import local_features as original_local

STATES=('unknown','recognized','not_recognized','uncertain','conflict')


def named_features(state,name_port):
    claims=[r for r in state.named_identities if r.name_port==name_port] if name_port is not None else []
    status=claims[-1].status if claims else 'unknown'
    return [float(status==s) for s in STATES]+[float(any(r.status=='recognized' for r in claims)),float(any(r.status=='conflict' for r in claims)),math.log1p(len(claims))]


def local_features(program,target,source,state,perception,body,parameters,clock):
    base=original_local(program,target,source,state,perception,body,parameters,clock)
    requested=target if program['scope']=='target' else None
    extras=named_features(state,source)+named_features(state,requested)
    # The shared relational ports deliberately retain their original header
    # and final9 geometry/count channels. Append epistemic inputs before them.
    return np.r_[base[:-9],extras,base[-9:]].astype(np.float32)


def global_features(program,target,state,perception,body,parameters,clock):
    return np.concatenate([local_features(program,target,i,state,perception,body,parameters,clock) for i in range(len(perception['stations']))]).tolist()


def configure():
    from relational_local_ports import configure as configure_relations
    configure_relations()
    import build_local_operator_dataset
    import local_operator_rollout
    import local_execution_particles
    import factorized_local_operators
    import rao_blackwell_local
    build_local_operator_dataset.global_features=global_features
    local_operator_rollout.global_features=global_features
    local_execution_particles.global_features=global_features
    factorized_local_operators.global_features=global_features
    rao_blackwell_local.global_features=global_features
