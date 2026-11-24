"""Learn several planning horizons, execute the same next5s observation cycle.

Targets forecast PREVIOUS learned-policy returns at later public prefixes.
Evaluation always rechecks after the common2+3s cycle, never oracle-chooses
future truth or commits to a task-specific timer. It is receding-horizon
proxy planning, not an exact generative Bellman/KDE reproduction.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np

from fit_learned_stopping import ARMS,prepare,evaluate,fit
from replace_local_quality_features import fold
from experiments.task_attack.runtime import write_json


def later_indices(episodes,seconds):
    indices=[];start=0
    for e in episodes:
        times=np.array([r['t'] for r in e['rows']]);n=len(times)
        for t in times:
            row=[]
            for delay in seconds:
                j=int(np.searchsorted(times,t+delay-1e-6));row.append(start+j if j<n else -1)
            indices.append(row)
        start+=n
    return np.array(indices)


def replan(data,prediction):
    # All methods actually receive/reconsider the same next5s window.
    return evaluate(data,np.column_stack([prediction[:,0],prediction[:,1:].max(axis=1)]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=6);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve policy attempts')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'stopping_trajectories.json.gz','rt') as f:raw=json.load(f)
    horizons=(5.,15.,30.,60.);future=later_indices(raw['training'],horizons);summaries={}
    write_json(a.output/'contract.json',dict(status='MULTISCALE waiting DEVELOPMENT; proxy only, not ASR',planning_horizons=horizons,actual_observation_cycle=5.,iterations=a.iterations,
        target='Previous learned-policy returns at each future public prefix, NOT oracle best future rewards. Every actual policy reconsiders after5s.',
        comparison='Identical planning options and training/selection for every arm; no minimum action time, task-specific workflow or victim rule.',
        data_sha256=hashlib.sha256((a.dataset/'stopping_trajectories.json.gz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),actual_attacks_executed=0))
    for arm in ARMS:
        train=prepare(raw['training'],arm);valid=prepare(raw['calibration'],arm);previous=train['reward'].copy();best=None;iterations=[]
        for iteration in range(1,a.iterations+1):
            targets=np.zeros((len(previous),1+len(horizons)));targets[:,0]=train['reward']
            for j in range(len(horizons)):
                available=future[:,j]>=0;targets[available,j+1]=previous[future[available,j]]
            prediction=np.zeros_like(targets)
            for k in range(5):
                held=np.array([fold(w)==k for w in train['worlds']]);own=~held;head=fit(train['x'][own],targets[own],train['weights'][own],161122+iteration);prediction[held]=head.predict(train['x'][held])
            previous,metrics=replan(train,prediction);iterations.append(dict(iteration=iteration,main_selected_proxy=metrics['main_selected_proxy']))
            print('MULTISCALE STOP',arm,'iteration',iteration,'OWN proxy',metrics['main_selected_proxy'],flush=True)
            if best is None or metrics['main_selected_proxy']>best[0]:best=(metrics['main_selected_proxy'],iteration,targets.copy())
            write_json(a.output/(arm+'_iterations.json'),dict(iterations=iterations))
        _,iteration,targets=best;head=fit(train['x'],targets,train['weights'],161122+iteration);head.n_jobs=1;_,metrics=replan(valid,head.predict(valid['x']))
        with (a.output/(arm+'_stopping.pkl')).open('wb') as f:pickle.dump(dict(head=head,arm=arm,iteration=iteration,planning_horizons=horizons,actual_observation_cycle=5.),f,protocol=4)
        write_json(a.output/(arm+'_development_result.json'),dict(status='HELD-WORLD development counterfactual PROXY, not ASR',selected_iteration=iteration,training_iterations=iterations,**metrics))
        summaries[arm]=dict(selected_iteration=iteration,own_training_crossfit_proxy=best[0],calibration_main_proxy=metrics['main_selected_proxy'],calibration_main_episodes=metrics['main_episodes'])
    write_json(a.output/'summary.json',dict(status='Receding-horizon waiting PROXY development only; not executed ASR',arms=summaries,actual_attacks_executed=0,superiority_proven=False));print('MULTISCALE STOP COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
