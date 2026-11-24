"""Learn actual next motion without inventing an intended-object label.

The brightest/clearest device in a future image is not necessarily the object
the actor intended to inspect. This alternative learns the observed LOCAL
endpoint/time directly from declared state and structured goal arguments.
It contains no whole-route template, task-family feature or victim rule.
"""
import numpy as np
from local_operator_kernels import ConditionalDistribution


def inputs(features):
    width=len(features)//4;ports=[features[width*i:width*(i+1)] for i in range(4)];refs=[p for p in ports if p[10]>.5]
    if len(refs)>1:raise ValueError('Ambiguous structured candidate argument')
    return np.r_[features,[float(bool(refs))],refs[0] if refs else np.zeros(width)]


def fit(episodes):
    counts={f:sum(e['family']==f for e in episodes) for f in {e['family'] for e in episodes}};rows=[]
    for ep in episodes:
        for ex in ep['examples']:
            if ex['terminal'] or ex['next_canonical_event'] is None:continue
            travel=max(0.,ex['next_t']-ex['t']-ex['call_seconds']);endpoint=ex['next_canonical_event']
            rows.append(dict(x=inputs(ex['after_features']),y=np.r_[endpoint,travel],weight=1/counts[ep['family']]/len(ep['examples']),outcome=dict(endpoint=endpoint,travel_seconds=travel)))
    head=ConditionalDistribution(trees=64,leaf=3,depth=16,seed=161119).fit(np.array([r['x'] for r in rows],dtype=np.float32),np.array([r['y'] for r in rows]),np.array([r['weight'] for r in rows]))
    return dict(distribution=head,outcomes=[r['outcome'] for r in rows],examples=len(rows),meaning='Actual external-RGB next body endpoint/time; no guessed source-purpose supervision')


def sample(features,models,rng):
    index=int(models['global_motion']['distribution'].sample(inputs(features)[None],rng)[0]);out=models['global_motion']['outcomes'][index]
    return dict(terminal=False,endpoint=list(out['endpoint']),travel_seconds=out['travel_seconds'])
