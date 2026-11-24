"""Learn stop/wait choices from OWN trajectories, with identical controls.

Policy iteration uses returns under the PREVIOUS learned policy, not an
oracle maximum over future true rewards. No task-specific timer, route,
confidence threshold, or forced waiting is added to the victim or attacker.
Rewards remain a clean-route counterfactual PROXY until actual paired trials.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from experiments.task_attack.runtime import write_json
from replace_local_quality_features import fold

ARMS=('ours_critic','blind','direct','known_critic')


def prepare(episodes,arm):
    x=[];reward=[];following=[];weights=[];groups=[];worlds=[];counts=Counter(e['family'] for e in episodes)
    for e in episodes:
        start=len(x);times=np.array([r['t'] for r in e['rows']]);n=len(times)
        for i,r in enumerate(e['rows']):
            row=r['arms'][arm];x.append(row['x']);reward.append(row['reward']);weights.append(1/counts[e['family']]);worlds.append(e['seed'])
            nxt=int(np.searchsorted(times,r['t']+5.-1e-6));following.append(start+nxt if nxt<n else -1)
        groups.append(dict(key=e['key'],seed=e['seed'],family=e['family'],start=start,end=len(x)))
    return dict(x=np.array(x,dtype=np.float32),reward=np.array(reward),following=np.array(following),weights=np.array(weights),worlds=np.array(worlds),groups=groups)


def evaluate(data,prediction):
    returns=np.zeros(len(data['reward']));selected=np.full(len(returns),-1,dtype=int)
    for group in data['groups']:
        for i in range(group['end']-1,group['start']-1,-1):
            if prediction[i,1]>prediction[i,0]:
                nxt=data['following'][i]
                if nxt>=0:returns[i]=returns[nxt];selected[i]=selected[nxt]
            else:returns[i]=data['reward'][i];selected[i]=i
    rows=[]
    for group in data['groups']:
        i=group['start'];rows.append(dict(key=group['key'],seed=group['seed'],family=group['family'],selected_proxy=float(returns[i]),selected_index=int(selected[i]-i) if selected[i]>=0 else None))
    main=[r for r in rows if r['family']!='equipment_labels']
    return returns,dict(all_episodes=len(rows),main_episodes=len(main),main_selected_proxy=float(np.mean([r['selected_proxy'] for r in main])),rows=rows)


def fit(x,y,weights,seed):
    head=ExtraTreesRegressor(n_estimators=64,max_depth=10,min_samples_leaf=64,max_features=.8,n_jobs=2,random_state=seed)
    head.fit(x,y,sample_weight=weights);return head


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=6);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve learned-policy versions')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'stopping_trajectories.json.gz','rt') as f:raw=json.load(f)
    write_json(a.output/'contract.json',dict(status='OWN learned stopping DEVELOPMENT; not actual attack success',arms=ARMS,iterations=a.iterations,
        data_sha256=hashlib.sha256((a.dataset/'stopping_trajectories.json.gz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        teacher='Actual clean-route counterfactual proxy reward at the policy-selected action. Wait targets are previous learned-policy returns, never max future truth.',
        prior_control='Blind feature vectors contain only first observation and public clock; dynamic controls get their public behavior features; known alone gets authorized goal.',
        selection='Each arm independently chooses its best OWN training crossfit iteration. Calibration worlds are not used by this policy fitting or iteration selection.',
        caveat='Stacked crossfit is development only. Calibration worlds were used in earlier component development. All rewards are acquisition/adoption proxies; actual victim recovery requires fresh paired trials.',actual_attacks_executed=0))
    summaries={}
    for arm in ARMS:
        train=prepare(raw['training'],arm);valid=prepare(raw['calibration'],arm);previous=train['reward'].copy();best=None;iterations=[]
        for iteration in range(1,a.iterations+1):
            following=train['following'];wait=np.zeros(len(previous));supported=following>=0;wait[supported]=previous[following[supported]];targets=np.column_stack([train['reward'],wait]);predictions=np.zeros_like(targets)
            for k in range(5):
                held=np.array([fold(w)==k for w in train['worlds']]);own=~held
                head=fit(train['x'][own],targets[own],train['weights'][own],161121+iteration)
                predictions[held]=head.predict(train['x'][held])
            previous,metrics=evaluate(train,predictions);iterations.append(dict(iteration=iteration,main_selected_proxy=metrics['main_selected_proxy']))
            print('LEARNED STOP',arm,'iteration',iteration,'OWN crossfit proxy',metrics['main_selected_proxy'],flush=True)
            if best is None or metrics['main_selected_proxy']>best[0]:
                best=(metrics['main_selected_proxy'],iteration,targets.copy())
            write_json(a.output/(arm+'_iterations.json'),dict(status='OWN fitting diagnostics, not ASR',iterations=iterations))
        _,iteration,targets=best;head=fit(train['x'],targets,train['weights'],161121+iteration);head.n_jobs=1
        _,metrics=evaluate(valid,head.predict(valid['x']))
        with (a.output/(arm+'_stopping.pkl')).open('wb') as f:pickle.dump(dict(head=head,arm=arm,iteration=iteration),f,protocol=4)
        np.savez_compressed(a.output/(arm+'_selected_targets.npz'),targets=targets)
        write_json(a.output/(arm+'_development_result.json'),dict(status='HELD-WORLD development counterfactual PROXY, not ASR',selected_iteration=iteration,training_iterations=iterations,**metrics))
        summaries[arm]=dict(selected_iteration=iteration,own_training_crossfit_proxy=best[0],calibration_main_proxy=metrics['main_selected_proxy'],calibration_main_episodes=metrics['main_episodes'])
    write_json(a.output/'summary.json',dict(status='Learned stop/wait proxy diagnostic, actual paired attacks NOT yet executed',arms=summaries,actual_attacks_executed=0,superiority_proven=False))
    print('LEARNED STOPPING COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
