"""Cheap external observation branches over cached conditional forecasts.

Both NOW and WAIT use the same empirical continuation, propagated to their
actual times. Future branches are model predictions, not future target logs.
Branch rollout scenes are excluded when evaluating their later observations.
"""
import math
import numpy as np

from experiments.task_attack.event_timing import features
from .forecast import rank as rank_dependencies
from experiments.task_attack_cached.continuation import nearest
from .source import source_probability


def rank(response,menu,current,static,dynamic):
    ranks,audit=rank_dependencies(response,menu,current,static,dynamic)
    alive=current['probability_any_future_read']
    for candidate in ranks:
        exposure=source_probability(current,candidate['action_id'])
        if exposure is not None and alive>0:
            factor=min(1.,exposure/alive)
            candidate['expected_incremental_wrong_acceptance']*=factor
            candidate['components']=[v*factor for v in candidate['components']]
        candidate['predicted_source_frustum_opportunity']=exposure
    ranks.sort(key=lambda r:(-r['expected_incremental_wrong_acceptance'],r['action_id']))
    audit['source_emission']='Learned future carrier frustum opportunity; not recognition or adoption'
    return ranks,audit


def onset_probability(model,x):
    onset=model.get('behavior_onset')
    if not onset or not onset.get('use_conditional'):raise ValueError('A calibrated motion-onset model is required')
    z=(np.array(x[:-1])-onset['mean'])/onset['scale']
    logit=onset['beta'][0]+np.dot(z,onset['beta'][1:])
    return float(1/(1+math.exp(-float(np.clip(logit,-30,30)))))


def make_forecast(model,x,scene,dynamic,exclude_seed=None):
    rollouts=[]
    for entry in nearest(model,x,scene,dynamic,exclude_seed):
        e=entry['episode'];t=entry['time']
        rollouts.append(dict(weight=entry['weight'],training_seed=e['seed'],distance=entry['distance'],
            future=[[r[0]-t,*r[1:]] for r in e['reads'] if r[0]>t],
            past=[[r[0]-t,*r[1:]] for r in e['reads'] if r[0]<=t]))
    return dict(rollouts=rollouts,dynamic=dynamic,pixel_std=model['pixel_std'],
        probability_any_future_read=sum(r['weight'] for r in rollouts if r['future']))


def select(model,response,menu,history,elapsed,scene,dynamic):
    x=features(history,elapsed)
    if x is None:
        x=np.array(model['mean']);x[-1]=math.log1p(max(0,elapsed));dynamic=False
    static=make_forecast(model,x,scene,False)
    current=make_forecast(model,x,scene,True) if dynamic else static
    ranks,audit=rank(response,menu,current,static,dynamic)
    now=max(0.,ranks[0]['expected_incremental_wrong_acceptance']) if ranks else 0.
    branches=[];wait_value=0.
    # An already informative actual motion is used immediately. Before that,
    # integrate possible future informative external prefixes. In the blind
    # condition those hypothetical images remain hidden and only time updates.
    observed_onset=onset_probability(model,x) if dynamic else None
    if observed_onset is None or observed_onset<.5:
        for neighbor in nearest(model,x,scene,dynamic):
            e=neighbor['episode'];t=neighbor['time'];branch=None
            for sample in e['samples']:
                if sample[0]<=t:continue
                if onset_probability(model,sample[1:])>=.5:
                    branch=sample;break
            if branch is None:
                branches.append(dict(p=neighbor['weight'],event='no supported informative prefix',value=0.));continue
            delta=branch[0]-t
            future_x=np.array(branch[1:] if dynamic else x,copy=True)
            future_x[-1]=math.log1p(elapsed+delta)
            # Cross-fit: do not use this training branch's own future camera
            # labels to evaluate the policy after its predicted observation.
            future_static=make_forecast(model,future_x,scene,False,e['seed'])
            future=make_forecast(model,future_x,scene,dynamic,e['seed']) if dynamic else future_static
            later,post=rank(response,menu,future,future_static,dynamic)
            value=max(0.,later[0]['expected_incremental_wrong_acceptance']) if later else 0.
            wait_value+=neighbor['weight']*value
            branches.append(dict(p=neighbor['weight'],predicted_external_delay=delta,value=value,
                action_id=later[0]['action_id'] if later else None,
                probability_future_read=future['probability_any_future_read'],crossfit_excluded_seed=e['seed']))
    kind='wait' if wait_value>now else 'act' if now>0 else 'abstain'
    return dict(kind=kind,current_value=now,wait_value=wait_value,branches=branches,
        actual_external_onset_probability=observed_onset,ranked_candidates=ranks,
        posterior_update=audit,additional_api_requests=0,
        meaning='Approximate empirical observation-tree value; no target future or task phase was read')
