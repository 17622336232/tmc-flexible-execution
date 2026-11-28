"""Stronger common incumbents and two-dispatch conditional prefetch on the complete action set."""
from functools import lru_cache

from .event_policy import Planner,bits,order
from .event_workflow import ALWAYS,UNKNOWN,indices


class LookaheadPlanner(Planner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.one_step=lru_cache(None)(self._one_step)

    def _greedy(self,s,style):
        if style in ("collect","prepare"):
            return super()._greedy(s,style)
        guards=self.model.guards(s.values)
        def priority(a):
            dt,branches=self.trans(s,a)
            bound=dt+sum(p*self.cost_lb(y) for p,y in branches)
            mask=sum(mask for _,mask in a.reads)
            due=sum(guards[i]==ALWAYS for i in indices(mask))
            optional=bits(mask)-due
            delays=sum(max(0,self.m.service[bits(b)]-bits(b)*self.m.service[1]) for _,b in a.reads)
            captured=bool(a.motion and a.motion[1])
            full=bool(captured and any(guards[i]==ALWAYS and v==UNKNOWN and a.motion[0] in self.m.queries[i].sites
                                       for i,v in enumerate(s.values)))
            # A cost-aware rollout avoids a large group premium and values work on a required site.
            return (delays,bound,-due,-full,optional,-bits(mask),order(a))
        return min(self.options(s),key=priority)

    def incumbent(self,s):
        value,style=min((self.roll(s,style),style) for style in ("collect","prepare","cost"))
        return value,self.greedy(s,style)

    def _one_step(self,s):
        if self.model.complete(s):return 0.,None
        return super().strong(s)

    def strong(self,s):
        best,chosen=self.one_step(s)
        for a in self.options(s):
            t,sub=self.trans(s,a)
            value=t+sum(p*self.one_step(y)[0] for p,y in sub)
            if value<best-1e-8 or (abs(value-best)<1e-8 and order(a)<order(chosen)):
                best,chosen=value,a
        return best,chosen
