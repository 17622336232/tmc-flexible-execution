"""Generate world-out-of-fold stopping trajectories for every control.

All reward fields are OFFLINE clean-route counterfactual proxies, not ASR.
Nested quality fits exclude the complete outer held world from both fitted
read-context features and the structured-goal regressor. The conditional
critic and direct/prior comparison heads are fitted with the same outer split.
No real robot, task actor, safety check or online observation is changed.
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
from experiments.task_attack_value.value_features import action_features
from experiments.task_attack_value.forest import predict
from experiments.task_attack_value.programs import payload
from learn_read_dependency import quality_history,context,goal_inputs
from external_observability_audit import descriptor
from replace_local_quality_features import fit_quality,fold
from research.attack_learning.train_fixed_slot_models import fit_one


def choose(values,actions):
    order=np.array(sorted(range(len(actions)),key=lambda i:actions[i]['action_id']))
    return order[np.asarray(values)[:,order].argmax(axis=1)]


def fit_intent(episodes,quality,hypotheses):
    keys=[descriptor(g,z) for g,z in hypotheses];counts=Counter(e['family'] for e in episodes);xx=[];yy=[];ww=[]
    for e in episodes:
        target=keys.index(descriptor(e['program'],e['target']))
        for t in np.arange(0.,min(901.,e['events'][-1]['t']),5.):
            ports,clock=context(e['events'],quality[e['key']],t,e['perception'])
            xx.append(goal_inputs(ports,clock,hypotheses));yy.extend(np.eye(len(hypotheses))[target]);ww.extend([1/counts[e['family']]]*len(hypotheses))
    head=ExtraTreesRegressor(n_estimators=96,max_depth=12,min_samples_leaf=16,max_features=.8,n_jobs=2,random_state=161120)
    head.fit(np.vstack(xx),np.array(yy),sample_weight=np.array(ww));head.n_jobs=1
    return head


def stopping_features(x,port,clock,values,index,action,posterior,hypotheses,known=None):
    feature=action_features(np.array(x),dict(),[],None) if False else None
    # x below is already the selected action's PUBLIC feature vector.
    ordered=np.sort(values)[::-1];p=np.clip(posterior,1e-12,1.);roles=[]
    for field in ('cracked','temperature'):
        roles.append(sum(q for q,(g,z) in zip(posterior,hypotheses) if field in g['roles']))
    scope=sum(q for q,(g,z) in zip(posterior,hypotheses) if g['scope']=='target')
    information=np.r_[np.sort(posterior)[-4:],-np.sum(p*np.log(p)),roles,scope]
    if known is not None:
        from experiments.task_attack_value.value_features import program_features
        information=np.r_[information,program_features(known,payload(action)['source'])]
    return np.r_[x,port.ravel(),clock,ordered[:4],ordered.mean(),ordered.std(),ordered[0]-ordered[1],information].astype(np.float32).tolist()


def trajectories(cases,episodes,quality_head,intent,heads,hypotheses,prior):
    by_key={e['key']:e for e in episodes};keys=[descriptor(g,z) for g,z in hypotheses];outputs=[]
    for case in cases:
        e=by_key[case['key']];samples=case['samples'];qs=quality_history(e['events'],e['perception'],quality_head);contexts=[];inputs=[]
        for s in samples:
            ports,clock=context(e['events'],qs,s['t'],e['perception']);contexts.append((ports,clock));inputs.append(goal_inputs(ports,clock,hypotheses))
        if not inputs:continue
        posterior=np.clip(intent.predict(np.vstack(inputs)).reshape(len(samples),len(hypotheses)),1e-9,1);posterior/=posterior.sum(axis=1,keepdims=True)
        features=np.vstack([action_features(np.array(s['x']),e['perception'],case['actions'],g)[0] for s in samples for g in hypotheses]);critic=predict(heads['known_dynamic'],features).reshape(len(samples),len(hypotheses),len(case['actions']))
        dynamic_x=[action_features(np.array(s['x']),e['perception'],case['actions'])[0] for s in samples];static_x=[action_features(np.array(s['static']),e['perception'],case['actions'])[0] for s in samples]
        blind=predict(heads['static'],np.vstack(static_x)).reshape(len(samples),-1);direct=predict(heads['dynamic'],np.vstack(dynamic_x)).reshape(len(samples),-1)
        # Only known_critic uses the actual authorized task in inference.
        true_goal=keys.index(descriptor(*case['known']));known=critic[:,true_goal,:]
        scores=dict(ours_critic=np.einsum('sg,sga->sa',posterior,critic),prior_same_critic=np.einsum('g,sga->sa',prior,critic),blind=blind,direct=direct,known_critic=known)
        selected={k:choose(v,case['actions']) for k,v in scores.items()};rows=[]
        for i,s in enumerate(samples):
            ports,clock=contexts[i];arms={}
            for arm,indices in selected.items():
                index=int(indices[i]);action=case['actions'][index]
                visible_port=contexts[0][0] if arm=='blind' else ports
                # Blind gets only the first image-derived port and public clock.
                visible_clock=np.r_[np.log1p(s['t']),0.,contexts[0][1][2:]] if arm=='blind' else clock
                probabilities=posterior[i] if arm=='ours_critic' else prior
                x=static_x[i][index] if arm=='blind' else dynamic_x[i][index]
                vector=stopping_features(x,visible_port,visible_clock,scores[arm][i],index,action,probabilities,hypotheses,case['known'] if arm=='known_critic' else None)
                arms[arm]=dict(x=vector,action_id=action['action_id'],reward=s['y'][index],estimated_value=float(scores[arm][i,index]))
            rows.append(dict(t=s['t'],arms=arms))
        outputs.append(dict(key=e['key'],seed=e['seed'],family=e['family'],rows=rows))
    return outputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--values',type=Path,required=True);p.add_argument('--rankers',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve crossfit versions')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'dataset.json.gz','rt') as f:episodes=json.load(f)['episodes']
    with gzip.open(a.values/'training_values.json.gz','rt') as f:cases=json.load(f)['cases']
    ranker=json.loads((a.rankers/'ranker_candidates.json').read_text());base=json.loads(Path('models/task_attack/integrity_v19/bundle.json').read_text());hypotheses=[(g,z) for g in base['programs'] for z in (range(4) if g['scope']=='target' else [None])];prior=np.array([1/len(base['programs'])/(4 if g['scope']=='target' else 1) for g,z in hypotheses])
    train_all=[e for e in episodes if e['train']];clean=[e for e in train_all if Path(e['directory']).name=='clean'];train_cases=[c for c in cases if c['train']];valid_cases=[c for c in cases if c['calibration']];all_rows=[]
    write_json(a.output/'contract.json',dict(status='NEW stopping-policy training; clean-route counterfactual PROXY only, no actual attacks',
        input_boundary='Public external RGB/history, c0 and clock. Only known-task control receives actual authorized task. Rewards and labels are never observer inputs.',
        crossfit='5 complete-world outer folds. Nested four-fold quality inputs for outer-fold intent training, all exclude outer held worlds. Conditional critic and comparison rankers fit identical outer splits.',
        wait_transition='First actual public prefix at least5 seconds later; common2s compute plus3s observation. No minimum start time or task-specific route/threshold.',
        finite_vocabulary=True,paper_exact_KDE_reproduction=False,actual_attacks_executed=0,
        hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (Path(__file__),a.dataset/'dataset.json.gz',a.values/'training_values.json.gz',a.rankers/'ranker_candidates.json')}))
    for k in range(5):
        outer=[e for e in train_all if fold(e['seed'])!=k];own=[e for e in clean if fold(e['seed'])!=k];held_cases=[c for c in train_cases if fold(c['seed'])==k];quality={}
        for j in range(5):
            if j==k:continue
            head=fit_quality([e for e in outer if fold(e['seed'])!=j])
            for e in own:
                if fold(e['seed'])==j:quality[e['key']]=quality_history(e['events'],e['perception'],head)
        intent=fit_intent(own,quality,hypotheses);qhead=fit_quality(outer);heads={};tr=[c for c in train_cases if fold(c['seed'])!=k]
        for kind in ('static','dynamic','known_dynamic'):
            heads[kind]=fit_one(tr,held_cases,kind,ranker['selected_configurations'][kind])[0]
        rows=trajectories(held_cases,episodes,qhead,intent,heads,hypotheses,prior);all_rows.extend(rows)
        with gzip.open(a.output/('fold_%d_rows.json.gz'%k),'wt') as f:json.dump(dict(episodes=rows),f,separators=(',',':'))
        write_json(a.output/('fold_%d_contract.json'%k),dict(fit_worlds=sorted({e['seed'] for e in outer}),held_worlds=sorted({c['seed'] for c in held_cases}),episodes=len(rows)))
        print('STOPPING OUTER FOLD',k,'episodes',len(rows),'total',len(all_rows),flush=True)
    # Full-training deployment candidate gets OOF quality inputs too.
    quality={}
    for k in range(5):
        qhead=fit_quality([e for e in train_all if fold(e['seed'])!=k])
        for e in clean:
            if fold(e['seed'])==k:quality[e['key']]=quality_history(e['events'],e['perception'],qhead)
    intent=fit_intent(clean,quality,hypotheses);qhead=fit_quality(train_all)
    final=dict(intent=intent,quality=qhead,heads=ranker['value_models'],hypotheses=hypotheses,prior=prior)
    with (a.output/'conditional_critic_candidate.pkl').open('wb') as f:pickle.dump(final,f,protocol=4)
    validation=trajectories(valid_cases,episodes,qhead,intent,ranker['value_models'],hypotheses,prior)
    with gzip.open(a.output/'stopping_trajectories.json.gz','wt') as f:json.dump(dict(training=all_rows,calibration=validation),f,separators=(',',':'))
    write_json(a.output/'summary.json',dict(status='Crossfit stopping data COMPLETE; not executed ASR',training_episodes=len(all_rows),training_worlds=len({e['seed'] for e in all_rows}),calibration_episodes=len(validation),calibration_worlds=len({e['seed'] for e in validation}),actual_attacks_executed=0))
    print('STOPPING DATA COMPLETE',len(all_rows),len(validation),flush=True)


if __name__=='__main__':main()
