"""Plan successful delivery under a fixed deadline and explicitly declared per-request failure exposure."""
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

from .event_policy import bits,order
from .event_lookahead import LookaheadPlanner


@dataclass(frozen=True)
class Score:
    restricted:float
    delivered:float
    sends:float
    delivered_time_mass:float


class DeadlineWorkflow:
    def __init__(self,model,hazards,deadline,quotient=True):
        self.model=model
        self.ordinary=LookaheadPlanner(model,quotient=quotient)
        self.deadline=deadline
        self.hazard=tuple(hazards)
        if (model.m.model_slots!=1 or len(hazards)!=len(model.m.service) or hazards[0]!=0 or
             any(not isfinite(h) or not 0<=h<=1 for h in hazards) or deadline<=0):
            raise ValueError("Expected a finite deadline, one service slot, and one hazard for each supported arity")
        self.cache={};self.rollouts={};self.nodes=0
        self.trans=lru_cache(None)(self.model.successors)

    def terminal(self,state,clock):
        if self.model.complete(state) and clock<=self.deadline:
            return Score(clock,1.,0.,clock)
        if clock>=self.deadline:
            return Score(self.deadline,0.,0.,0.)
        return None

    def survival(self,action):
        return 1-self.hazard[bits(action.reads[0][1])] if action.reads else 1.

    def after(self,state,clock,action,fn):
        dt,branches=self.trans(state,action)
        survive=self.survival(action)
        next_scores=[(p,fn(child,clock+dt)) for p,child in branches]
        def mean(key):return sum(p*getattr(v,key) for p,v in next_scores)
        return Score((1-survive)*self.deadline+survive*mean("restricted"),survive*mean("delivered"),
                     len(action.reads)+survive*mean("sends"),survive*mean("delivered_time_mass"))

    @staticmethod
    def ranking(v,cmd):
        # Rounded comparisons suppress tiny summation-order changes only.
        return (round(v.restricted,8),-round(v.delivered,10),round(v.sends,8),order(cmd))

    def rollout(self,state,clock,style):
        term=self.terminal(state,clock)
        if term:return term
        key=(style,self.ordinary.key(state),clock)
        if key not in self.rollouts:
            cmd=self.ordinary.greedy(state,style)
            self.rollouts[key]=self.after(state,clock,cmd,lambda y,t:self.rollout(y,t,style))
        return self.rollouts[key]

    def solve(self,state,clock=0,horizon=None):
        term=self.terminal(state,clock)
        if term:return term,None
        key=(self.ordinary.key(state),clock,horizon)
        if key in self.cache:return self.cache[key]
        self.nodes+=1
        fallback=[(self.rollout(state,clock,style),self.ordinary.greedy(state,style)) for style in ("collect","prepare","cost")]
        best,chosen=min(fallback,key=lambda v:self.ranking(*v))
        if horizon!=0:
            candidates=[]
            for cmd in self.ordinary.options(state):
                dt,beside=self.trans(state,cmd);survive=self.survival(cmd)
                low=(1-survive)*self.deadline+survive*sum(p*min(self.deadline,clock+dt+self.ordinary.cost_lb(y)) for p,y in beside)
                candidates.append((low,order(cmd),cmd))
            for low,_,cmd in sorted(candidates,key=lambda t:(t[0],t[1])):
                if low>best.restricted+1e-7:break
                value=self.after(state,clock,cmd,lambda y,t:self.solve(y,t,None if horizon is None else horizon-1)[0])
                if self.ranking(value,cmd)<self.ranking(best,chosen):
                    best,chosen=value,cmd
        self.cache[key]=(best,chosen)
        return best,chosen

    def fixed(self,state,chooser,clock=0):
        memo={}
        def evaluate(x,t):
            term=self.terminal(x,t)
            if term:return term
            key=(x,t)
            if key not in memo:
                cmd=chooser(x,t)
                if cmd not in self.ordinary.options(x):raise ValueError("Fixed policy violates a grant")
                memo[key]=self.after(x,t,cmd,evaluate)
            return memo[key]
        return evaluate(state,clock)
