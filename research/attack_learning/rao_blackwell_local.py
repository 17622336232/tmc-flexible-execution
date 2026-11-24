"""Integrate local endpoint uncertainty instead of treating it as pixel noise.

Each MOVE has a diagonal Gaussian endpoint innovation fitted from OOF local
predictions. Its posterior is updated by the actual external observations.
The same conditional event density generates prospective observations. READ
snapshots latch content at capture time, while their uncertain acquisition
geometry is integrated until the READ completes (delayed latent sampling).
No current task, victim camera, report, pose or truth is read here.
"""
from dataclasses import replace
import math

import numpy as np

from local_execution_particles import Particle,Belief,submit
from build_local_operator_dataset import denormalize,global_features
from local_operator_rollout import quality_input
from factorized_local_operators import sample_reads
from local_gate_modes import decide
from learned_operation_durations import sample as duration_sample


class RBParticle(Particle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.offset_mean=np.zeros(6);self.offset_var=np.zeros(6)
        self.observed=None;self.observed_mask=None;self.observed_epoch=None;self.observed_t=0.
        self.observed_base=None;self.observed_factor=None;self.pending_first_sequence=None

    def clone(self,seed=None):
        result=super().clone(seed)
        for name in ('offset_mean','offset_var','observed','observed_mask','observed_base','observed_factor'):
            value=getattr(self,name);setattr(result,name,None if value is None else value.copy())
        return result

    def base_and_factor(self,t):
        if self.stage in ('launch','move'):
            s=np.clip((t-self.start_t)/max(1e-6,self.end_t-self.start_t),0,1)
            from local_event_curves import factors
            f=factors(self.models['event'],self.stage,s,self.end_t-self.start_t)
            return self.start+f*(self.end-self.start),f
        return self.start.copy(),np.ones(6)

    def mean(self,t):
        base,factor=self.base_and_factor(t);return base+factor*self.offset_mean

    def event_prediction(self,t,mask):
        base,factor=self.base_and_factor(t);model=self.event_parameters();variance=np.array(model['sigma'])**2
        use_previous=self.observed is not None and self.observed_epoch==self.epoch
        if use_previous:
            rho=math.exp(-max(1e-8,t-self.observed_t)/model['tau']) if model['tau'] else 0.
            r=np.where(self.observed_mask,rho,0.)
            center=base+r*(self.observed-self.observed_base)
            coefficient=factor-r*self.observed_factor
            variance=variance*np.maximum(1e-6,1-r*r)
        else:center=base;coefficient=factor
        center=center+coefficient*self.offset_mean
        predictive=variance+coefficient**2*self.offset_var
        return center,predictive,coefficient,base,factor

    def condition_event(self,observed,mask,t):
        selected=np.array(mask,dtype=bool);observed=np.asarray(observed)
        center,variance,h,base,factor=self.event_prediction(t,selected)
        error=observed-center
        value=float(np.sum(-.5*(math.log(2*math.pi)+np.log(variance[selected])+error[selected]**2/variance[selected])))
        gain=np.where(selected,self.offset_var*h/variance,0.)
        self.offset_mean=self.offset_mean+gain*error
        self.offset_var=np.maximum(0.,self.offset_var-gain*h*self.offset_var)
        self.observed=observed.copy();self.observed_mask=selected.copy();self.observed_epoch=self.epoch;self.observed_t=t
        self.observed_base=base;self.observed_factor=factor
        return value

    def sample_event(self,t,rng):
        center,variance,_,_,_=self.event_prediction(t,[True]*6)
        return center+np.sqrt(variance)*rng.normal(size=6)

    def begin_read(self,t,body):
        self.start=np.asarray(body).copy();self.end=self.start.copy();self.start_t=t;self.stage='read';self.epoch+=1
        self.pending_first_sequence=len(self.state.captures)
        # Latch carrier content NOW. Later evidence about a stationary body's
        # uncertain position may inform geometry, never replace these values.
        for fact in self.state.carriers:self.state=self.state.capture(fact.source,fact.field,t,None)
        self.pending=None;self.action=None
        self.end_t=t+float(duration_sample(self.models['duration'],self.rng));self.operations+=1

    def complete_read(self,t):
        body=self.start+self.offset_mean+np.sqrt(self.offset_var)*self.rng.normal(size=6)
        raw=denormalize(body,self.perception)
        xs=[quality_input(raw,self.perception,i,j) for i in range(4) for j in range(2)]
        qs=np.clip(self.models['quality']['distribution'].mean_outcome(np.array(xs))[:,0],0,1)
        from local_operator_kernels import ACTIVE
        captures=[]
        for c in self.state.captures:
            if c.sequence>=self.pending_first_sequence and c.field in ACTIVE:
                c=replace(c,quality_measured=True,view_quality=float(qs[c.source*2+ACTIVE.index(c.field)]))
            captures.append(c)
        self.state=replace(self.state,captures=tuple(captures))
        features=global_features(self.program,self.target,self.state,self.perception,raw,self.parameters,self.start_t)
        if self.models.get('named_read'):
            self.state=sample_reads(self.state,features,self.models,self.rng,declaration_time=t)
        else:self.state=sample_reads(self.state,features,self.models,self.rng)
        features=global_features(self.program,self.target,self.state,self.perception,raw,self.parameters,self.start_t)
        self.action=self.choose_decision(features)
        self.start=body;self.end=body.copy();self.offset_mean=np.zeros(6);self.offset_var=np.zeros(6)
        return raw

    def choose_decision(self,features):
        return decide(self.state,features,self.perception,self.models,self.rng)

    def advance(self,t,budget=900.):
        if t<self.clock-1e-6:raise ValueError('A particle cannot rewind its capture buffer')
        while self.stage not in ('terminal','censored') and t>=self.end_t:
            at=self.end_t
            if at>=budget or self.operations>=120:
                self.start=self.mean(min(at,budget));self.end=self.start.copy();self.offset_mean=np.zeros(6);self.offset_var=np.zeros(6)
                self.stage='censored';self.epoch+=1;break
            if self.stage in ('launch','move'):
                # The endpoint Gaussian becomes the physical hover-position
                # uncertainty. Do not discard it or count it as sensor noise.
                self.begin_read(at,self.end)
            elif self.stage=='read':
                raw=self.complete_read(at);self.epoch+=1
                if self.action['terminal']:
                    # Training acceptance inputs use the actual capture clock,
                    # while completion/episode budgeting still occurs at at.
                    self.stage='terminal';self.report=submit(self.state,self.program,self.target,self.perception,self.parameters,raw,self.start_t,self.models,self.rng)
                else:
                    self.stage='move';self.start_t=at;self.end=np.array(self.action['endpoint'])
                    self.end_t=at+max(.01,self.action['travel_seconds'])
                    self.offset_var=np.array(self.models['predictive_endpoint_sigma'])**2
        self.clock=t


class RBBelief(Belief):
    def __init__(self,*args,**kwargs):super().__init__(*args,particle_type=RBParticle,**kwargs)
