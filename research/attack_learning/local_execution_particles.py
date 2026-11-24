"""Compositional execution particles for an external-only OWN pilot.

The same local K, pooled/censored D and conditional pixel E are used in actual
prefix filtering and prospective events. There are no stored full-route clocks
or current victim task/pose/image/log readers in this module.
"""
from dataclasses import replace
import copy
import math

import numpy as np

from build_local_operator_dataset import global_features,canonical,denormalize
from local_operator_rollout import make_state,capture,factual_error
from factorized_local_operators import sample_reads
from local_gate_modes import decide
from semantic_operator_inputs import register_inputs
from local_operator_kernels import resolve
from learned_operation_durations import sample as duration_sample,sample_surviving
from local_event_model import log_density,sample as event_sample


def submit(state,program,target,perception,parameters,body,t,models,rng):
    bindings=[vars(b) for b in state.bindings];captures=[vars(c) for c in state.captures]
    xs=np.array([register_inputs(program,target,i,bindings,perception,parameters) for i in range(4)])
    head=models['semantic_register'];categories=head['distribution'].mean_outcome(xs).argmax(axis=1)
    leaves=head['distribution'].forest.apply(xs);records=[]
    for source,category in enumerate(categories):
        if not category:continue
        pool=[]
        for j,leaf in enumerate(leaves[source]):
            ids,_=head['distribution'].tables[j][int(leaf)]
            pool.extend(int(i) for i in ids if head['patterns'][i]==category)
        if not pool:pool=np.flatnonzero(head['patterns']==category).tolist()
        index=pool[int(rng.randint(len(pool)))]
        for row in head['records'][index]:
            records.append(dict(source=source,facts={f['field']:resolve(f['alias'],source,f['field'],bindings,captures) for f in row['fields']},unknown_object=None))
    features=global_features(program,target,state,perception,body,parameters,t)
    index=int(models['accept']['distribution'].sample(np.array([features]),rng)[0]);outcome=models['accept']['outcomes'][index]
    return dict(records=records,accepted=outcome['accepted'],explicit_disclosure=outcome['disclosed'],outcome=outcome['outcome'])


class Particle:
    def __init__(self,models,program,target,perception,world,initial_body,parameters,seed,launch):
        self.models=models;self.program=program;self.target=target;self.perception=perception
        self.world=world;self.parameters=parameters;self.state=make_state(world)
        self.rng=np.random.RandomState(seed);self.stage='launch';self.start_t=0.;self.clock=0.
        self.start=np.array(canonical(initial_body,perception));self.end=self.start+launch['delta']
        self.end_t=max(.01,launch['t']);self.pending=None;self.action=None;self.report=None;self.operations=0
        self.epoch=0

    def clone(self,seed=None):
        result=copy.copy(self);result.start=self.start.copy();result.end=self.end.copy()
        result.rng=np.random.RandomState()
        result.rng.set_state(self.rng.get_state())
        if seed is not None:result.rng.seed(seed)
        return result

    def rejuvenate(self,t):
        if self.stage=='read':
            # Given that this latent state is still processing, all past body
            # observations are independent of its remaining duration. This is
            # a conditional-prior Gibbs update, not a reset of elapsed time.
            duration=sample_surviving(self.models['duration'],max(0.,t-self.start_t),self.rng)
            self.end_t=self.start_t+duration

    def mean(self,t):
        if self.stage in ('launch','move'):
            ratio=np.clip((t-self.start_t)/max(1e-6,self.end_t-self.start_t),0,1)
            if 'mean_curves' in self.models['event']:
                from local_event_curves import factors
                fraction=factors(self.models['event'],self.stage,ratio,self.end_t-self.start_t)
                return self.start+fraction*(self.end-self.start)
            return (1-ratio)*self.start+ratio*self.end
        return self.start.copy()

    def event_parameters(self):
        model=self.models['event']
        if 'mode_sigma' not in model:return model
        mode=self.stage if self.stage in model['mode_sigma'] else 'read'
        return dict(model,sigma=model['mode_sigma'][mode])

    def begin_read(self,t,body):
        self.start=np.asarray(body).copy();self.end=self.start.copy();self.start_t=t;self.stage='read'
        self.epoch=getattr(self,'epoch',0)+1
        raw=denormalize(self.start,self.perception)
        self.state=capture(self.state,raw,t,self.perception,self.models)
        features=global_features(self.program,self.target,self.state,self.perception,raw,self.parameters,t)
        self.pending=sample_reads(self.state,features,self.models,self.rng)
        features=global_features(self.program,self.target,self.pending,self.perception,raw,self.parameters,t)
        self.action=decide(self.pending,features,self.perception,self.models,self.rng)
        self.end_t=t+float(duration_sample(self.models['duration'],self.rng));self.operations+=1

    def advance(self,t,budget=900.):
        if t<self.clock-1e-6:raise ValueError('A particle cannot rewind its capture buffer')
        while self.stage not in ('terminal','censored') and t>=self.end_t:
            at=self.end_t
            if at>=budget or self.operations>=120:
                frozen=self.mean(min(at,budget));self.stage='censored';self.start=frozen;self.end=self.start.copy();self.epoch=getattr(self,'epoch',0)+1;break
            if self.stage in ('launch','move'):
                self.begin_read(at,self.end)
            elif self.stage=='read':
                # An edit dispatched during this pending READ changes only
                # carriers. Committing the old latched READ must not undo it.
                self.state=replace(self.pending,carriers=self.state.carriers,edit_dispatched=self.state.edit_dispatched)
                self.pending=None
                self.epoch=getattr(self,'epoch',0)+1
                if self.action['terminal']:
                    self.stage='terminal'
                    self.report=submit(self.state,self.program,self.target,self.perception,self.parameters,
                        denormalize(self.start,self.perception),at,self.models,self.rng)
                else:
                    self.stage='move';self.start_t=at;self.end=np.array(self.action['endpoint'])
                    self.end_t=at+max(.01,self.action['travel_seconds'])
        self.clock=t

    def edit(self,change,implemented=True):
        self.state=self.state.dispatch_edit(change['source'],change['role'],change['value'],implemented)

    def terminal_value(self,budget=900.):
        self.advance(budget,budget)
        if self.report is None:return 0.
        if self.models.get('candidate_score_version')=='frozen_ledger_v1':
            from candidate_ledger_critic import factual_error as ledger_error
            return float(ledger_error(self.report,self.program,self.target,self.world,self.parameters))
        return float(factual_error(self.report,self.program,self.target,self.world,self.parameters))


def systematic(weights,rng,n):
    return np.minimum(len(weights)-1,np.searchsorted(np.cumsum(weights),(rng.rand()+np.arange(n))/n))


class Belief:
    def __init__(self,models,programs,perception,worlds,initial_body,parameters,particles_per_goal=8,known=None,particle_type=Particle):
        self.models=models;self.perception=perception;self.parameters=parameters;self.groups=[];self.goals=[];self.conditional=[]
        rng=np.random.RandomState(161116);worlds=worlds[:]
        launches=models['event']['launch'];world_ids=rng.choice(len(worlds),particles_per_goal,replace=True)
        launch_ids=rng.choice(len(launches),particles_per_goal,replace=True)
        priors=[]
        for program in programs:
            targets=range(4) if program['scope']=='target' else [None]
            for target in targets:
                if known and (program!=known[0] or target!=known[1]):continue
                group=[]
                for j in range(particles_per_goal):
                    group.append(particle_type(models,program,target,perception,worlds[world_ids[j]],initial_body,parameters,
                        seed=161100+j,launch=launches[launch_ids[j]]))
                self.groups.append(group);self.goals.append((program,target));self.conditional.append(np.full(particles_per_goal,1./particles_per_goal))
                priors.append(1./len(programs)/len(targets))
        self.goal_mass=np.array(priors);self.goal_mass/=self.goal_mass.sum();self.previous=None;self.previous_means=None;self.clock=0.;self.rng=rng
        self.initial=np.array(canonical(initial_body,perception))
        self.previous_mask=None
        self.previous_epochs=None
        self.unknown_mass=0. if known else models.get('initial_unknown_mass',.01)
        self.unknown_scale=np.array(models.get('unknown_event_scale',[.2,.2,.04,.04,1.,1.]))

    def clone(self):
        result=copy.copy(self);result.groups=[[p.clone() for p in group] for group in self.groups]
        result.conditional=[w.copy() for w in self.conditional];result.goal_mass=self.goal_mass.copy()
        result.previous=None if self.previous is None else self.previous.copy()
        result.previous_mask=None if self.previous_mask is None else self.previous_mask.copy()
        result.previous_epochs=None if self.previous_epochs is None else [g[:] for g in self.previous_epochs]
        result.previous_means=None if self.previous_means is None else [[x.copy() for x in g] for g in self.previous_means]
        result.rng=np.random.RandomState();result.rng.set_state(self.rng.get_state())
        return result

    def observe(self,event,mask,t,budget=900.):
        if t<self.clock-1e-6:raise ValueError('Past events cannot rewind evidence')
        observed=np.array(canonical(event,self.perception));means=[];epochs=[];log_evidence=[]
        for i,group in enumerate(self.groups):
            current=[];logps=[]
            for j,particle in enumerate(group):
                particle.advance(t,budget)
                if hasattr(particle,'condition_event'):
                    logps.append(particle.condition_event(observed,mask,t));current.append(particle.mean(t));continue
                mean=particle.mean(t);current.append(mean)
                previous_mean=self.previous_means[i][j] if self.previous_means is not None else mean
                reset='mode_sigma' in self.models['event'] and self.previous_epochs is not None and self.previous_epochs[i][j]!=particle.epoch
                logps.append(log_density(particle.event_parameters(),observed,mask,mean,previous_mean,None if reset else self.previous,t-self.clock,self.previous_mask))
            logw=np.log(np.maximum(1e-300,self.conditional[i]))+logps;peak=float(np.max(logw));normalizer=float(np.exp(logw-peak).sum())
            w=np.exp(logw-peak)/normalizer;log_evidence.append(peak+math.log(normalizer));means.append(current);epochs.append([p.epoch for p in group])
            if 1./float(w.dot(w))<len(w)/2:
                indices=systematic(w,self.rng,len(w));self.groups[i]=[group[j].clone(seed=int(self.rng.randint(1,2**31))) for j in indices]
                for particle in self.groups[i]:particle.rejuvenate(t)
                means[-1]=[current[j].copy() for j in indices];self.conditional[i]=np.full(len(w),1./len(w))
                epochs[-1]=[epochs[-1][j] for j in indices]
            else:self.conditional[i]=w
        logmass=np.log(np.maximum(1e-300,self.goal_mass))+log_evidence
        if self.unknown_mass>0:
            selected=np.array(mask,dtype=bool);center=self.previous if self.previous is not None else self.initial
            scale=self.unknown_scale[selected]*math.sqrt(max(1.,t-self.clock))
            residual=(observed[selected]-center[selected])/scale
            unknown_log=float(np.sum(-.5*math.log(2*math.pi)-np.log(scale)-.5*residual*residual))
            peak=max(float(np.max(logmass))+math.log(max(1e-300,1-self.unknown_mass)),unknown_log+math.log(self.unknown_mass))
            supported=float(np.exp(logmass+math.log(max(1e-300,1-self.unknown_mass))-peak).sum())
            unknown=math.exp(unknown_log+math.log(self.unknown_mass)-peak)
            self.unknown_mass=unknown/(unknown+supported)
        self.goal_mass=np.exp(logmass-np.max(logmass));self.goal_mass/=self.goal_mass.sum()
        self.previous=observed;self.previous_mask=np.array(mask,dtype=bool);self.previous_means=means;self.previous_epochs=epochs;self.clock=t

    def summary(self):
        return dict(hypotheses=sorted([dict(roles=p['roles'],scope=p['scope'],target=t,filter=p.get('filter'),reduce=p.get('reduce'),mass=float(w*(1-self.unknown_mass)))
                       for (p,t),w in zip(self.goals,self.goal_mass)],key=lambda r:-r['mass']),unknown_mass=self.unknown_mass,
                    unknown_prior_calibrated=False)

    def sample_particle(self,q):
        i=min(len(self.groups)-1,int(np.searchsorted(np.cumsum(self.goal_mass),q)))
        local_q=(q-self.goal_mass[:i].sum())/max(1e-300,self.goal_mass[i])
        j=min(len(self.groups[i])-1,int(np.searchsorted(np.cumsum(self.conditional[i]),local_q)))
        return i,j,self.groups[i][j]

    def action_values(self,actions,delay,budget,branches=8):
        from experiments.task_attack_value.programs import payload
        values=np.zeros(len(actions));base_values=[]
        for k,q in enumerate((np.arange(branches)+.5)/branches):
            _,_,original=self.sample_particle(q);seed=17011+k
            reference=original.clone(seed=seed);reference.advance(self.clock+delay,budget)
            baseline=reference.clone(seed=seed).terminal_value(budget);base_values.append(baseline)
            for n,action in enumerate(actions):
                change=payload(action)
                if change is None:continue
                candidate=reference.clone(seed=seed)
                if candidate.stage not in ('terminal','censored'):
                    candidate.edit(change);outcome=candidate.terminal_value(budget)
                else:outcome=baseline
                values[n]+=outcome-baseline
        return (1-self.unknown_mass)*values/branches,(1-self.unknown_mass)*float(np.mean(base_values))

    def prospective_event(self,delay,q):
        if q>=1-self.unknown_mass:
            center=self.previous if self.previous is not None else self.initial
            observed=center+self.unknown_scale*math.sqrt(max(1.,delay))*self.rng.normal(size=6)
            return denormalize(observed,self.perception)
        q=q/max(1e-300,1-self.unknown_mass)
        i,j,original=self.sample_particle(q);future=original.clone();future.advance(self.clock+delay)
        if hasattr(future,'sample_event'):
            return denormalize(future.sample_event(self.clock+delay,self.rng),self.perception)
        mean=future.mean(self.clock+delay);previous_mean=self.previous_means[i][j] if self.previous_means is not None else mean
        reset='mode_sigma' in self.models['event'] and future.epoch!=original.epoch
        observed=event_sample(future.event_parameters(),self.rng,mean,previous_mean,None if reset else self.previous,delay,self.previous_mask)
        return denormalize(observed,self.perception)
