"""Anytime nonanticipating planning on the same physical and semantic actions for all comparisons."""
from functools import lru_cache
from math import inf
from time import perf_counter

from .event_workflow import ALWAYS, UNKNOWN, NO_MOTION, indices


def bits(mask):
    return bin(mask).count("1")


def order(a):
    return (a.motion,a.reads)


class Planner:
    def __init__(self,model,*,quotient=True,ablation=None):
        self.model=model
        self.m=model.m
        self.quotient=quotient
        self.ablation=ablation
        self.solutions={}
        self.default_values={}
        self.expanded=0
        self.rollout_nodes=0
        self.hit=0
        self.action_count=0
        self.cost_lb=lru_cache(None)(self._lower)
        self.trans=lru_cache(None)(self.model.successors)
        self.greedy=lru_cache(None)(self._greedy)
        self.options=lru_cache(None)(self._options)
        self.travel_lb=lru_cache(None)(self._route)
        # An optimistic shortest connection remains valid for nonmetric public edge tables.
        self.shortest=[list(row) for row in self.m.travel]
        for k in range(len(self.m.names)):
            for i in range(len(self.m.names)):
                for j in range(len(self.m.names)):
                    self.shortest[i][j]=min(self.shortest[i][j],self.shortest[i][k]+self.shortest[k][j])

    def key(self,s):
        return self.model.signature(s) if self.quotient else s

    def _options(self,s):
        actions=self.model.dispatches(s)
        if self.ablation=="single-read":
            actions=tuple(a for a in actions if all(bits(mask)==1 for _,mask in a.reads))
        if self.ablation=="serial":
            actions=tuple(a for a in actions if not ((a.motion or s.motion!=NO_MOTION) and (a.reads or s.jobs))
                          and len(a.reads)+len(s.jobs)<=1)
        if self.ablation=="no-speculative":
            g=self.model.guards(s.values)
            actions=tuple(a for a in actions if
                (not a.motion or not a.motion[1] or self.m.mandatory_raw&(1<<a.motion[0])
                 or any(g[i]==ALWAYS and s.values[i]==UNKNOWN and a.motion[0] in q.sites for i,q in enumerate(self.m.queries)))
                and all(g[i]==ALWAYS for _,mask in a.reads for i in indices(mask)))
        if not actions and not self.model.complete(s):
            raise ValueError("Ablation removed every legal continuation")
        return actions

    def _route(self,at,required):
        if not required:
            return self.shortest[at][self.m.finish]
        return min(self.shortest[at][i]+self.m.capture[i]+self.travel_lb(i,required^(1<<i)) for i in indices(required))

    def _lower(self,s):
        if self.model.complete(s):return 0.
        required=self.m.mandatory_raw & ~s.raw
        jobs=sum(mask for _,mask,_ in s.jobs)
        unread=0
        for i,(q,g,v) in enumerate(zip(self.m.queries,self.model.guards(s.values),s.values)):
            if g==ALWAYS and v==UNKNOWN and not jobs&(1<<i):
                unread+=1
                if len(q.sites)==1 and not (1<<q.sites[0])&s.raw:
                    required|=1<<q.sites[0]
        at=s.site;motion=0
        if s.motion!=NO_MOTION:
            at,cap,motion=s.motion
            if cap:required&=~(1<<at)
        min_per_question=min(t/n for n,t in enumerate(self.m.service) if n)
        semantic=(sum(rem for _,_,rem in s.jobs)+unread*min_per_question)/self.m.model_slots
        return max(semantic,max([0]+[r for _,_,r in s.jobs]),motion+self.travel_lb(at,required))

    def _greedy(self,s,style):
        guards=self.model.guards(s.values)
        actions=self.options(s)
        def score(a):
            missing=[i for i,v in enumerate(s.values) if v==UNKNOWN and guards[i]]
            due=sum(1 for _,mask in a.reads for i in indices(mask) if guards[i]==ALWAYS)
            total=sum(bits(mask) for _,mask in a.reads)
            children=sum(1 for _,mask in a.reads for i in indices(mask) for j in missing
                         if any(any(parent==i for parent,_ in clause) for clause in guards[j]))
            read_time=sum(self.m.service[bits(b)] for _,b in a.reads)
            travel=importance=0
            if a.motion:
                site,cap=a.motion
                travel=self.m.travel[s.site][site]+(self.m.capture[site] if cap else 0)
                if cap:
                    interested=[i for i in missing if site in self.m.queries[i].sites and not any(s.raw&(1<<k) for k in self.m.queries[i].sites)]
                    importance=(5*sum(guards[i]==ALWAYS for i in interested)+
                                7*bool(self.m.mandatory_raw&(1<<site))+
                                (1.5 if style=="prepare" else .3)*sum(guards[i]!=ALWAYS for i in interested))
                elif site==self.m.finish:
                    importance=1. if not any(not s.raw&(1<<k) for i in missing for k in self.m.queries[i].sites) else .08
                else:importance=.15
            # Two feasible rollouts trade early service against readiness of shared captures.
            quality=(8*due+2.5*children+1.5*(total-due))/(max(1,read_time)**.33)
            value=quality+importance/(1+travel/5)
            if style=="prepare" and a.reads and not children:
                value-=sum(2. for site,b in a.reads if any(guards[i]!=ALWAYS and site in self.m.queries[i].sites
                                                          for i in missing if not b&(1<<i)))
            return (-value,-total,-int(bool(a.motion)),order(a))
        return min(actions,key=score)

    def roll(self,s,style):
        if self.model.complete(s):return 0.
        # Identical public fallback for the generic and specialized searches. The greedy policy
        # depends only on guards, observed/missing status, owned photos, pose, and pending work.
        key=(style,self.model.signature(s))
        if key not in self.default_values:
            self.rollout_nodes+=1
            a=self.greedy(s,style)
            t,sub=self.trans(s,a)
            self.default_values[key]=t+sum(p*self.roll(x,style) for p,x in sub)
        return self.default_values[key]

    def incumbent(self,s):
        vals=[(self.roll(s,style),style) for style in ("collect","prepare")]
        best,style=min(vals)
        return best,self.greedy(s,style)

    def strong(self,s):
        best,act=self.incumbent(s)
        for a in self.options(s):
            t,sub=self.trans(s,a)
            value=t+sum(p*(0 if self.model.complete(x) else self.incumbent(x)[0]) for p,x in sub)
            if value<best-1e-8 or (abs(value-best)<1e-8 and order(a)<order(act)):
                best,act=value,a
        return best,act

    def solve(self,s,budget=10000,seconds=None):
        start=perf_counter();limit=self.expanded+budget
        until=start+seconds if seconds is not None else inf
        def remaining():
            return self.expanded<limit and perf_counter()<until
        # Any incomplete search returns a feasible continuation. The strongest one-step
        # incumbent is available to both planners before refining other legal schedules.
        fallback,first=self.strong(s) if not self.model.complete(s) else (0.,None)
        def visit(x,root=False):
            if self.model.complete(x):return 0.,None,True
            key=self.key(x)
            if key in self.solutions:
                self.hit+=1
                return self.solutions[key]
            best,chosen=(fallback,first) if root else self.incumbent(x)
            if not remaining():return best,chosen,False
            self.expanded+=1
            options=[]
            for a in self.options(x):
                t,sub=self.trans(x,a)
                lb=t+sum(p*self.cost_lb(y) for p,y in sub)
                options.append((lb,order(a),a,t,sub))
            options.sort(key=lambda it:(it[0],it[1]))
            unresolved_lower=inf
            for lower,_,a,t,sub in options:
                if lower>=best-1e-8:
                    break
                if not remaining():
                    unresolved_lower=min(unresolved_lower,lower);break
                self.action_count+=1
                val=t;cert=True
                for p,y in sub:
                    child,_,ok=visit(y)
                    val+=p*child
                    cert&=ok
                if not cert:unresolved_lower=min(unresolved_lower,lower)
                if val<best-1e-8 or (abs(val-best)<1e-8 and order(a)<order(chosen)):
                    best,chosen=val,a
            exact=unresolved_lower>=best-1e-8
            ans=(best,chosen,exact)
            if exact:self.solutions[key]=ans
            return ans
        value,action,proved=visit(s,True)
        return dict(value_ticks=value,action=action,proved=proved,expanded=self.expanded,cache_hits=self.hit,
                    candidate_evaluations=self.action_count,rollout_states=self.rollout_nodes,
                    cpu_wall_s=perf_counter()-start,root_lower_ticks=self.cost_lb(s),strong_ticks=fallback)
