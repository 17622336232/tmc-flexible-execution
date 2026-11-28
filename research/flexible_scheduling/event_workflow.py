"""Finite event model for scoped evidence collection, concurrent judgments, and asynchronous replanning.

All costs are integer ticks; effects on a branch are exposed only when the corresponding service
actually finishes. A site is a public view/position. A raw capture is static and reusable only in
this mission. The separate report ledger must retain the values and sources on each actual path.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from math import isfinite

UNKNOWN=-1
ALWAYS=((),)
NO_MOTION=(-1,False,0)


@dataclass(frozen=True)
class Query:
    label:str
    sites:tuple
    when:tuple=ALWAYS  # OR of AND clauses (parent index, bool); ALWAYS contains one empty clause.
    early_read:bool=False
    prior:float=.5


@dataclass(frozen=True)
class Mission:
    names:tuple
    queries:tuple
    travel:tuple
    capture:tuple
    early_capture:tuple
    mandatory_raw:int
    on_hand:int
    service:tuple  # table indexed by group size, starting at zero
    home:int=0
    finish:int=0
    model_slots:int=1

    def __post_init__(self):
        n=len(self.names)
        if (not n or len(set(self.names))!=n or len(self.travel)!=n or any(len(row)!=n for row in self.travel)
            or len(self.capture)!=n or len(self.early_capture)!=n or self.model_slots not in (1,2)):
            raise ValueError("Invalid worksite or service capacity")
        if len(self.service)<2 or self.service[0]!=0 or any(type(t) is not int or t<1 for t in self.service[1:]+self.capture):
            raise ValueError("Positive integer service and capture times required")
        if not (0<=self.home<n and 0<=self.finish<n and 0<=self.mandatory_raw<1<<n and 0<=self.on_hand<1<<n):
            raise ValueError("Invalid location or initial evidence")
        for i,row in enumerate(self.travel):
            if any(type(t) is not int or (t!=0 if i==j else t<1) for j,t in enumerate(row)):
                raise ValueError("Invalid directed travel time")
        if len({q.label for q in self.queries})!=len(self.queries):
            raise ValueError("Duplicate query identity")
        for i,q in enumerate(self.queries):
            if (not q.sites or any(s<0 or s>=n for s in q.sites) or not isfinite(q.prior) or not 0<=q.prior<=1
                    or any(j<0 or j>=i or type(target) is not bool for clause in q.when for j,target in clause)):
                raise ValueError("Invalid source, probability, or non-causal query guard")


@dataclass(frozen=True)
class State:
    values:tuple
    raw:int
    site:int
    motion:tuple=NO_MOTION         # destination, take a photo on arrival, time remaining
    jobs:tuple=()                  # (source, requested query bit mask, time remaining)


@dataclass(frozen=True)
class Dispatch:
    motion:tuple=()                # (site, capture)
    reads:tuple=()                 # (source, query bit mask) per free service slot


def indices(mask):
    while mask:
        b=mask & -mask
        yield b.bit_length()-1
        mask^=b


def residuals(queries,values):
    result=[]
    for query in queries:
        clauses=[]
        for clause in query.when:
            rem=[]
            for parent,want in clause:
                active=result[parent]
                if not active or (active==ALWAYS and values[parent]!=UNKNOWN and values[parent]!=int(want)):
                    break
                if active!=ALWAYS or values[parent]==UNKNOWN:
                    rem.append((parent,want))
            else:
                if not rem:
                    clauses=[()];break
                clauses.append(tuple(rem))
        result.append(tuple(sorted(set(clauses))))
    return tuple(result)


class EventModel:
    def __init__(self,mission):
        self.m=mission
        self.guards=lru_cache(None)(lambda v:residuals(mission.queries,v))
        self.dispatches=lru_cache(None)(self._dispatches)

    def initial(self):
        return State((UNKNOWN,)*len(self.m.queries),self.m.on_hand,self.m.home)

    def complete(self,s):
        return (s.motion==NO_MOTION and not s.jobs and s.site==self.m.finish
                and s.raw & self.m.mandatory_raw==self.m.mandatory_raw
                and all(not g or (g==ALWAYS and val!=UNKNOWN) for g,val in zip(self.guards(s.values),s.values)))

    def signature(self,s):
        guards=self.guards(s.values)
        # A result obtained early still has to be remembered until its own reporting branch resolves.
        # Activated old values that cannot distinguish any future guard remain in the caller's ledger.
        values=tuple(UNKNOWN if v==UNKNOWN else v if g and g!=ALWAYS else 2 for g,v in zip(guards,s.values))
        return (guards,values,s.raw,s.site,s.motion,s.jobs)

    def _dispatches(self,s):
        if self.complete(s):
            return ()
        m=self.m;guards=self.guards(s.values)
        in_service=sum(b for _,b,_ in s.jobs)  # jobs have disjoint question masks
        available=[i for i,q in enumerate(m.queries) if s.values[i]==UNKNOWN and not in_service&(1<<i)
                   and guards[i] and (guards[i]==ALWAYS or q.early_read)]
        groups=[]
        for site in indices(s.raw):
            qs=[i for i in available if site in m.queries[i].sites]
            for n in range(1,min(len(m.service)-1,len(qs))+1):
                for subset in combinations(qs,n):
                    groups.append((site,sum(1<<j for j in subset)))
        issues=[()]
        for n in range(1,m.model_slots-len(s.jobs)+1):
            for batch in combinations(groups,n):
                if n==1 or not (batch[0][1]&batch[1][1]):
                    issues.append(batch)
        out=[]
        for issue in issues:
            pending=bool(s.jobs or issue)
            motions=[()]
            if s.motion==NO_MOTION:
                for site in range(len(m.names)):
                    needed=[i for i,q in enumerate(m.queries) if guards[i] and s.values[i]==UNKNOWN and site in q.sites]
                    raw_needed=bool(m.mandatory_raw & (1<<site))
                    if not s.raw&(1<<site) and (needed or raw_needed):
                        if m.early_capture[site] or raw_needed or any(guards[i]==ALWAYS for i in needed):
                            motions.append((site,True))
                        elif pending and site!=s.site:
                            # Public transit is permitted even when the camera must remain closed.
                            motions.append((site,False))
                if s.site!=m.finish and (pending or all(not g or v!=UNKNOWN for g,v in zip(guards,s.values))):
                    motions.append((m.finish,False))
            for move in dict.fromkeys(motions):
                if move or issue or s.motion!=NO_MOTION or s.jobs:
                    out.append(Dispatch(move,issue))
        if not out:
            raise ValueError("No way to discharge the public work order")
        return tuple(out)

    def step(self,s,action):
        motion=s.motion;jobs=list(s.jobs)
        if action.motion:
            dest,cap=action.motion
            motion=(dest,cap,self.m.travel[s.site][dest]+(self.m.capture[dest] if cap else 0))
        for site,mask in action.reads:
            jobs.append((site,mask,self.m.service[bin(mask).count("1")]))
        elapsed=min([remaining for _,_,remaining in jobs]+([motion[2]] if motion!=NO_MOTION else []))
        at=s.site;raw=s.raw
        if motion!=NO_MOTION:
            dest,cap,remain=motion
            if remain==elapsed:
                at=dest;raw|=(1<<dest) if cap else 0
                motion=NO_MOTION
            else:
                motion=(dest,cap,remain-elapsed)
        done=tuple((site,b) for site,b,t in jobs if t==elapsed)
        jobs=tuple(sorted((site,b,t-elapsed) for site,b,t in jobs if t>elapsed))
        return elapsed,State(s.values,raw,at,motion,jobs),done

    def successors(self,s,action):
        elapsed,base,arrivals=self.step(s,action)
        revealed=tuple(i for _,mask in arrivals for i in indices(mask))
        if not revealed:
            return elapsed,((1.,base),)
        states=[]
        for outcomes in product((0,1),repeat=len(revealed)):
            vals=list(base.values);mass=1.
            for idx,value in zip(revealed,outcomes):
                vals[idx]=value
                p=self.m.queries[idx].prior
                mass*=p if value else 1-p
            if mass:
                states.append((mass,State(tuple(vals),base.raw,base.site,base.motion,base.jobs)))
        return elapsed,tuple(states)
