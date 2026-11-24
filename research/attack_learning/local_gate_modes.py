"""MAP prediction of the fitted local branch classifier, not a victim rule.

Leaf frequencies contain epistemic uncertainty. Independently drawing an
early-stop coin at every model step can manufacture failures never observed in
the actual near-deterministic Max actor. This alternative is evaluated openly;
it is not a calibrated stochastic-kernel claim.
"""
import numpy as np
from factorized_local_operators import relative_features
from build_local_operator_dataset import canonical


def decide(state,features,perception,models,rng):
    bindings=[vars(b) for b in state.bindings]
    xs=np.array([relative_features(features,i,bindings) for i in range(4)])
    p=np.clip(models['gate']['distribution'].mean_outcome(xs),0,1)
    if float(np.mean(p[:,0]))>=.5:return dict(terminal=True)
    if 'global_motion' in models:
        from direct_local_motion import sample
        return sample(features,models,rng)
    source=int(p[:,1].argmax())
    index=int(models['motion']['distribution'].sample(xs[source:source+1],rng)[0])
    outcome=models['motion']['outcomes'][index];endpoint=np.array(outcome['relative_endpoint'],copy=True)
    a=perception['anchors'][source];anchor=canonical([a['u'],a['v'],0,0,0,0],perception)
    endpoint[:2]+=anchor[:2]
    return dict(terminal=False,endpoint=endpoint.tolist(),travel_seconds=outcome['travel_seconds'])
