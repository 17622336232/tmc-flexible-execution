"""A shared hidden execution process for filtering and prospective observation.

States are OWN proxy operation/clock nodes with actual acquisition schedules.
The same time-warp transition and normalized event density drive real filtering
and hypothetical windows. Counterfactual read adoption is trained from OWN
paired executions. This bounded pilot does not claim a full learned recovery
or arbitrary-operation implementation of the research proposal.
"""
from collections import Counter
import math

import numpy as np

from experiments.task_attack_value.forest import predict
from experiments.task_attack_value.programs import answer,payload


def observed_event(x,perception,pose_heads):
    row=np.asarray(x)[0];anchor=perception['anchors'][0]
    raw=np.r_[row[29:41],row[0]*anchor['w']+anchor['u'],row[1]*anchor['w']+anchor['v'],row[11]*anchor['w'],row[12]*anchor['w']]
    event=np.r_[raw[-4:],0.,0.];mask=np.array([True]*4+[False]*2)
    if row[53]>0:
        heading=np.array([predict(h,raw[None])[0] for h in pose_heads])*2-1
        event[4:]=heading;mask[4:]=True
    return event,mask


def counterfactual_effect(program,worlds,parameters,action,target):
    change=payload(action)
    if not change or program.get('literal') or change['role'] not in program['roles'] or program['scope']=='target' and change['source']!=target:return 0.
    changed=0
    for world in worlds:
        before=answer(program,world,parameters,target)
        alternative=[dict(row) for row in world];alternative[change['source']][change['role']]=change['value']
        changed+=before!=answer(program,alternative,parameters,target)
    return changed/max(1,len(worlds))


class ExecutionBelief:
    def __init__(self,model,perception,actions,known=None):
        self.model=model;self.perception=perception;self.actions=[a for a in actions if payload(a)];self.known=known
        self.sigma=np.array(model['event_sigma']);self.step=model['state_step_seconds'];self.pose_heads=model['pose_heads']
        self.background_mean=np.array(model['background_mean']);self.background_sigma=np.array(model['background_sigma'])
        self.templates=[t for t in model['templates'] if known is None or t['program']==known[0] and t['target']==known[1]]
        if not self.templates:raise ValueError('No supported authorized program; never substitute another task')
        # Persistent joint OWN-world particles. The static observation conditions
        # them once; future scoring never samples independent per-read truths.
        self.worlds=[]
        for original in model['world_prior']:
            world=[dict(row) for row in original]
            for i,station in enumerate(perception['stations']):
                for role in ('cracked','temperature'):
                    if station.get(role) is not None:world[i][role]=station[role]
            self.worlds.append(world)
        starts=[];ends=[];nodes=[];successor=[];initial=[];prior=[];self.template_slices=[];self.template_keys=[]
        counts=Counter(t['program_key'] for t in self.templates)
        conditional={}
        for t in self.templates:
            mismatch=sum(float(a.get('cracked')!=b.get('cracked')) for a,b in zip(t['static_stations'],perception['stations']) if a.get('cracked') is not None and b.get('cracked') is not None)
            conditional[t['seed']]=math.exp(-model['static_context_precision']*mismatch)
        normalizer={key:sum(conditional[t['seed']] for t in self.templates if t['program_key']==key) for key in counts}
        family_keys={}
        for t in self.templates:family_keys.setdefault(t['program']['family'],set()).add(t['program_key'])
        weights={t['seed']:conditional[t['seed']]/normalizer[t['program_key']]/len(family_keys)/len(family_keys[t['program']['family']]) for t in self.templates}
        self.template_effects=[];effect_cache={}
        for t in self.templates:
            start=len(nodes);n=len(t['states']);end=start+n-1;self.template_slices.append(slice(start,end+1));self.template_keys.append(t['program_key'])
            if t['program_key'] not in effect_cache:
                effect_cache[t['program_key']]=np.array([counterfactual_effect(t['program'],self.worlds,model['parameters'],a,t['target']) for a in self.actions])
            self.template_effects.append(effect_cache[t['program_key']])
            for j,state in enumerate(t['states']):
                nodes.append(state);successor.append(min(end,start+j+1));initial.append(weights[t['seed']] if j==0 else 0.)
        self.nodes=nodes;self.event=np.array([s['event'] for s in nodes]);self.successor=np.array(successor,dtype=np.int32)
        self.unknown=len(nodes);self.p=np.r_[np.array(initial)*(1-model['initial_unknown_mass']),model['initial_unknown_mass']]
        self.clock=0.;self.initialized=False
        self.values=self._local_read_values()

    def _local_read_values(self):
        """READ snapshots precede interpretation; an edit changes only FUTURE captures.

        This is the analytic expectation of the fitted two-context read/bind
        kernel. Earlier source snapshots remain intact. Nonlinear task effect
        is scored on the whole persistent candidate world, not a changed cell.
        """
        rows=[];delay=self.model['implementation_delay_seconds'];kernel=self.model['read_adoption_kernel']
        changes=[payload(a) for a in self.actions]
        indices=np.array([c['source']*2+int(c['role']=='temperature') for c in changes])
        fresh=np.array([kernel[c['role']]['fresh_adoption'] for c in changes])
        cached=np.array([kernel[c['role']]['cached_adoption'] for c in changes])
        for t,effect in zip(self.templates,self.template_effects):
            times=np.array([s['t'] for s in t['states']]);rt=np.array([r['t'] for r in t['acquisitions']])
            if len(rt):
                quality=np.array([r['quality'] for r in t['acquisitions']]);after=rt[None,:]>times[:,None]+delay
                f=np.max(quality[None,:,:]*after[:,:,None],axis=1)[:,indices]
                c=np.max(quality[None,:,:]*(~after)[:,:,None],axis=1)[:,indices]
            else:f=c=np.zeros((len(times),len(self.actions)))
            adoption=(1-c)*fresh+c*cached
            rows.extend(effect[None,:]*f*adoption*self.model['implementation_success_probability'])
        # Unsupported trajectories have a conservative increment of -1.
        return np.vstack([np.array(rows),-np.ones((1,len(self.actions)))])

    def advance(self,p,dt):
        if dt<=0:return np.array(p,copy=True)
        if dt>30:
            value=np.array(p,copy=True)
            for _ in range(int(math.ceil(dt/30))):value=self.advance(value,dt/math.ceil(dt/30))
            return value
        rate=dt/self.step;term=np.array(p[:-1],copy=True);prob=math.exp(-rate);out=prob*term;total=prob
        for k in range(1,80):
            term=np.bincount(self.successor,weights=term,minlength=self.unknown)
            prob*=rate/k;out+=prob*term;total+=prob
            if 1-total<1e-12:break
        if total<1:out+=(1-total)*term
        return np.r_[out,p[-1]]

    def density(self,event,mask):
        if not np.any(mask):return np.ones(len(self.p))
        a=np.asarray(event)[mask];sigma=self.sigma[mask]
        logp=-.5*np.sum(((self.event[:,mask]-a)/sigma)**2,axis=1)-np.sum(np.log(sigma))-.5*mask.sum()*math.log(2*math.pi)
        bs=self.background_sigma[mask]
        background=-.5*np.sum(((self.background_mean[mask]-a)/bs)**2)-np.sum(np.log(bs))-.5*mask.sum()*math.log(2*math.pi)
        values=np.r_[logp,background];return np.exp(values-values.max())

    def correct(self,p,event,mask):
        weighted=p*self.density(event,mask);normalizer=float(weighted.sum())
        if normalizer<=0 or not math.isfinite(normalizer):
            result=np.zeros_like(p);result[-1]=1.;return result
        return weighted/normalizer

    def observe(self,event,mask,t):
        if self.initialized and t<self.clock-1e-6:raise ValueError('Past observation cannot rewind execution')
        if not self.initialized:self.clock=t;self.initialized=True
        else:self.p=self.advance(self.p,t-self.clock);self.clock=t
        self.p=self.correct(self.p,event,mask)

    def rank(self,p):
        values=p.dot(self.values)
        order=sorted(range(len(self.actions)),key=lambda i:(-values[i],self.actions[i]['action_id']))
        return [dict(action_id=self.actions[i]['action_id'],expected_increment=float(values[i]),expected_wrong_accepted=float(values[i]),
                     source=payload(self.actions[i])['source'],field=payload(self.actions[i])['role']) for i in order]

    def posterior_summary(self):
        groups={}
        for key,section in zip(self.template_keys,self.template_slices):groups[key]=groups.get(key,0.)+float(self.p[section].sum())
        return dict(program_hypotheses=sorted([dict(hypothesis=k,mass=v) for k,v in groups.items()],key=lambda r:-r['mass']),
                    unknown_mass=float(self.p[-1]),estimated_operation_clock=self.clock,
                    status='MODEL hypotheses only; no current task, pose or evidence log has been read')

    def decide(self,budget,elapsed,dynamic):
        slot=self.model['fixed_compute_dispatch_seconds'];current=self.advance(self.p,slot);ranked=self.rank(current)
        immediate=ranked[0]['expected_increment'] if ranked else 0.;branches=[]
        rng=np.random.RandomState(17017)
        if dynamic:
            for delay in self.model['observation_delays_seconds']:
                if elapsed+slot+delay+slot>=budget:continue
                predicted=self.advance(current,delay);cumulative=np.cumsum(predicted);future_values=[]
                for quantile in (np.arange(self.model['observation_branches'])+.5)/self.model['observation_branches']:
                    index=min(len(predicted)-1,int(np.searchsorted(cumulative,quantile)))
                    event=(self.event[index]+rng.normal(size=6)*self.sigma) if index<self.unknown else self.background_mean+rng.normal(size=6)*self.background_sigma
                    posterior=self.correct(predicted,event,np.ones(6,dtype=bool));dispatch=self.advance(posterior,slot)
                    future_values.append(max(0.,float(np.max(dispatch.dot(self.values)))))
                branches.append(dict(delay_seconds=delay,expected_wrong_accepted=float(np.mean(future_values))))
        else:
            for delay in self.model['observation_delays_seconds']:
                if elapsed+slot+delay+slot>=budget:continue
                predicted=self.advance(current,delay+slot)
                branches.append(dict(delay_seconds=delay,expected_wrong_accepted=max(0.,float(np.max(predicted.dot(self.values))))))
        best=max(branches,key=lambda b:b['expected_wrong_accepted']) if branches else None
        if best and best['expected_wrong_accepted']>max(0.,immediate)+1e-9:kind='wait';reason='Same execution/event model predicts positive value from another complete observation window'
        elif immediate>0:kind='edit';reason='Current counterfactual operation gain is at least as high as waiting'
        else:kind='abstain';reason='No supported positive counterfactual gain'
        return dict(kind=kind,reason=reason,ranked_candidates=ranked,immediate_value=immediate,wait_branches=branches,
                    selected_delay_seconds=best['delay_seconds'] if best and kind=='wait' else None,
                    execution_belief=self.posterior_summary(),fixed_compute_dispatch_seconds=slot,
                    model_limitations=self.model['limitations'])
