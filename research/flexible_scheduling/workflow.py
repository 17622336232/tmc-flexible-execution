"""Finite remaining-task scheduler for a static inspection section with a required return pose.

All policies receive the same public demand factors, prior observations, admissible views and
measured cost estimates. Each view's physical cost includes returning to the required anchor.
The current semantic request can read nominated views only when their early interpretation is authorized.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import isclose, isfinite
from time import perf_counter


@dataclass(frozen=True)
class View:
    name: str
    fields: int
    trip: float
    capture_early: bool
    read_early: bool
    on_hand: bool = False


@dataclass(frozen=True)
class Workflow:
    # Each independent factor has (outstanding requirements, probability) outcomes.
    # A jointly distributed set of predicates can be supplied as one categorical factor.
    factors: tuple
    views: tuple
    verified: int
    primary_seconds: tuple   # indexed by extra images interpreted in this request
    followup_seconds: tuple  # indexed by images; zero requires no request
    report_seconds: float

    def __post_init__(self):
        if len({v.name for v in self.views}) != len(self.views):
            raise ValueError("A view was entered twice")
        if any(v.fields<=0 or not isfinite(v.trip) or v.trip<0 for v in self.views):
            raise ValueError("All views require a coverage set and finite acquisition cost")
        if self.verified<0 or len(self.primary_seconds)<1 or len(self.followup_seconds)<2 or self.followup_seconds[0]!=0:
            raise ValueError("Invalid evidence or service table")
        if any(not isfinite(v) or v<0 for v in (*self.primary_seconds,*self.followup_seconds,self.report_seconds)):
            raise ValueError("Service estimates must be finite and nonnegative")
        cover=self.verified
        for view in self.views:
            cover |= view.fields
        for factor in self.factors:
            if not factor or not isclose(sum(p for _,p in factor),1,abs_tol=1e-10):
                raise ValueError("Each factor must describe a probability distribution")
            if any(mask<0 or mask & ~cover or not isfinite(p) or not 0<=p<=1 for mask,p in factor):
                raise ValueError("Unsupported demand or invalid probability")


def subsets(items,limit=None):
    items=tuple(items)
    for size in range(1+min(len(items),len(items) if limit is None else limit)):
        yield from combinations(items,size)


def merged_demands(factors,verified):
    states={0:1.}
    for outcomes in factors:
        following={}
        for previous,mass in states.items():
            for mask,prob in outcomes:
                if prob:
                    key=(previous|mask) & ~verified
                    following[key]=following.get(key,0.)+mass*prob
        states=following
    return states


class DecisionProblem:
    def __init__(self,task):
        self.task=task
        self.sources=range(len(task.views))
        self.raw=frozenset(i for i in self.sources if task.views[i].on_hand)
        # Instance-local caches also apply to the generic search, and expire between inspections.
        self.read_duration=lru_cache(None)(self._read_duration)
        self.finish=lru_cache(None)(self._finish)

    def _read_duration(self,count):
        # All group sizes in the measured response-size support are available to every solver.
        if count==0:
            return 0.
        costs=self.task.followup_seconds
        return min(costs[n]+self.read_duration(count-n) for n in range(1,min(count,len(costs)-1)+1))

    def coverage(self,views):
        fields=0
        for i in views:
            fields |= self.task.views[i].fields
        return fields

    def _finish(self,demand,raw):
        if not demand:
            return 0.
        eligible=[i for i in self.sources if self.task.views[i].fields & demand]
        best=float("inf")
        for group in subsets(eligible):
            if demand & ~self.coverage(group):
                continue
            travel=sum(self.task.views[i].trip for i in group if i not in raw)
            best=min(best,travel+self.read_duration(len(group)))
        return best

    def actions(self):
        early=[i for i in self.sources if self.task.views[i].capture_early and i not in self.raw]
        joint=[i for i in self.sources if self.task.views[i].read_early and (i in early or i in self.raw)]
        for batch in subsets(joint,len(self.task.primary_seconds)-1):
            for overlap in subsets(i for i in early if i not in batch):
                yield batch,overlap

    def prefix(self,batch,overlap):
        v=self.task.views
        pre=sum(v[i].trip for i in batch if i not in self.raw)
        during=sum(v[i].trip for i in overlap)
        primary=self.task.primary_seconds[len(batch)]
        done=self.task.verified | self.coverage(batch)
        return pre+max(primary,during)+self.task.report_seconds, self.raw.union(batch,overlap), done

    def solve(self,method):
        start=perf_counter()
        fullset=tuple(self.actions())
        # The generic comparator is a memoized exact backward induction over observed demand.
        # It has the same terminal optimization, dominance from already-covered requirements,
        # and complete choice of physical and semantic actions as the sparse forward calculation.
        if method not in ("generic","factored","serial","async","prefetch","fixed-joint"):
            raise ValueError(method)
        if method=="generic":
            @lru_cache(None)
            def transitions(i,needed,done):
                following={}
                for mask,prob in self.task.factors[i]:
                    key=(needed|mask) & ~done
                    following[key]=following.get(key,0.)+prob
                return tuple((k,p) for k,p in following.items() if p)

            @lru_cache(None)
            def distribution(i,needed,raw,done):
                if i==len(self.task.factors):
                    return self.finish(needed,raw)
                return sum(p*distribution(i+1,k,raw,done) for k,p in transitions(i,needed,done))
        else:
            marginal=merged_demands(self.task.factors,self.task.verified)
        if method=="serial":
            fullset=(((),()),)
        if method=="async":
            always=None
            for mask,p in marginal.items():
                if p:
                    always=mask if always is None else always & mask
            # Complete rolling work only when it is unconditionally needed.
            fullset=tuple((b,o) for b,o in fullset if not b and all(self.task.views[i].fields & (always or 0) for i in o))
        if method=="prefetch":
            # Strong physical prefetch baseline: exactly optimizes every eligible early capture
            # subset, with complete contingent continuation and shared evidence, no early semantic bundling.
            fullset=tuple((b,o) for b,o in fullset if not b)
        if method=="fixed-joint":
            # Largest authorized batch; ties use the exact expected task cost.
            largest=max(len(b) for b,_ in fullset)
            fullset=tuple((b,o) for b,o in fullset if len(b)==largest)
        candidates=[]
        for batch,overlap in fullset:
            prefix,raw,done=self.prefix(batch,overlap)
            if method=="generic":
                future=distribution(0,0,raw,done)
            else:
                future=sum(p*self.finish(mask & ~done,raw) for mask,p in marginal.items())
            pre_travel=sum(self.task.views[i].trip for i in (*batch,*overlap) if i not in self.raw)
            candidates.append((prefix+future,pre_travel,batch,overlap))
        value=min(c[0] for c in candidates)
        # A deterministic common tie break removes floating addition order from the chosen action.
        best=min((c for c in candidates if c[0]<=value+1e-8),key=lambda c:c[1:])
        return dict(expected_seconds=best[0],joint=best[2],overlap=best[3],early_motion_s=best[1],
                    evaluated=len(candidates),solve_seconds=perf_counter()-start,
                    terminal_states=self.finish.cache_info().currsize)
