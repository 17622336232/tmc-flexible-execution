"""Single-slot event planning with one or two sources in the same permitted request."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from math import isfinite
from statistics import median

from .event_lookahead import LookaheadPlanner
from .event_policy import bits
from .event_workflow import Dispatch, EventModel, Mission, NO_MOTION, UNKNOWN


class PermissionCache:
    """Reuse legal dispatches when motion/service prices change, including in the generic control."""
    def __init__(self, mission, cross_enabled, *, residual):
        self.scope = self.permission_scope(mission,cross_enabled)
        self.residual = residual
        self.entries = {}
        self.hits = 0

    @staticmethod
    def permission_scope(mission,cross_enabled):
        return (mission.names,mission.queries,mission.early_capture,mission.mandatory_raw,
                mission.finish,mission.model_slots,cross_enabled)

    def key(self, model, state):
        visual = state.values
        if self.residual:
            visual = (model.guards(state.values),tuple(v != UNKNOWN for v in state.values))
        return visual,state.raw,state.site,state.motion!=NO_MOTION,tuple(mask for _,mask,_ in state.jobs)


class CrossSourceWorkflow(EventModel):
    def __init__(self, mission: Mission, cross_ticks: int | None = None, *, permission_cache=None):
        if mission.model_slots != 1 or len(mission.service) != 3:
            raise ValueError("Cross-source planning uses one slot and up to two fields per request")
        if any(len(q.sites) != 1 or q.early_read for q in mission.queries):
            raise ValueError("Each field must have a single source and an explicit released grant")
        if cross_ticks is not None and (type(cross_ticks) is not int or cross_ticks <= 0):
            raise ValueError("Cross-source service must have a positive cost")
        self.cross_source = len(mission.names)
        self.same_pair_ticks = mission.service[2]
        self.cross_ticks = cross_ticks
        self.permission_cache = permission_cache
        if permission_cache is not None and permission_cache.scope != PermissionCache.permission_scope(mission,cross_ticks is not None):
            raise ValueError("Cannot reuse permissions across different task contracts")
        # Common search bound must use the cheapest enabled batch per field.
        best_pair = min(self.same_pair_ticks, cross_ticks) if cross_ticks is not None else self.same_pair_ticks
        super().__init__(replace(mission, service=(*mission.service[:2], best_pair)))

    def _dispatches(self, state):
        cache = self.permission_cache
        if cache is None:
            return self._build_dispatches(state)
        key = cache.key(self,state)
        if key not in cache.entries:
            cache.entries[key] = self._build_dispatches(state)
        else:
            cache.hits += 1
        return cache.entries[key]

    def _build_dispatches(self, state):
        ordinary = super()._dispatches(state)
        if self.cross_ticks is None:
            return ordinary
        by_motion = defaultdict(list)
        for action in ordinary:
            if action.reads and bits(action.reads[0][1]) == 1:
                by_motion[action.motion].append(action.reads[0])
        combined = []
        for motion, reads in by_motion.items():
            for (left, a), (right, b) in combinations(reads, 2):
                if left != right:
                    combined.append(Dispatch(motion, ((self.cross_source, a | b),)))
        return ordinary + tuple(combined)

    def read_ticks(self, source, mask):
        if bits(mask) == 1:
            return self.m.service[1]
        return self.cross_ticks if source == self.cross_source else self.same_pair_ticks

    def step(self, state, action):
        if action.reads:
            source, questions = action.reads[0]
            state = replace(state, jobs=((source, questions, self.read_ticks(source, questions)),))
        return super().step(state, Dispatch(action.motion))


class CrossSourcePlanner(LookaheadPlanner):
    def _greedy(self, state, style):
        if style != "cost":
            return super()._greedy(state, style)

        def priority(dispatch):
            elapsed, branches = self.trans(state, dispatch)
            bound = elapsed + sum(p * self.cost_lb(child) for p, child in branches)
            count = sum(bits(mask) for _, mask in dispatch.reads)
            premium = sum(max(0, self.model.read_ticks(source, mask) - bits(mask) * self.m.service[1])
                          for source, mask in dispatch.reads)
            return premium, bound, -count, dispatch.motion, dispatch.reads

        return min(self.options(state), key=priority)


class CausalMeasurements:
    """Clock samples are visible only when their originating action or request has settled."""
    def __init__(self):
        self.samples = defaultdict(list)

    def record(self, key, *, available_at, seconds, delivered):
        if (not isfinite(available_at) or not isfinite(seconds) or seconds <= 0 or type(delivered) is not bool):
            raise ValueError("Expected a settled positive measurement and an explicit delivery flag")
        self.samples[key].append((available_at, seconds, delivered))

    def describe(self, key, now, default):
        visible = [(seconds, delivered) for date, seconds, delivered in self.samples[key] if date <= now]
        successes = [seconds for seconds, delivered in visible if delivered]
        return dict(samples=len(visible), undelivered=len(visible)-len(successes),
                    successful_median_s=median(successes) if successes else default,
                    capped_median_s=median([seconds for seconds, _ in visible]) if visible else default)

    def conditional_remaining(self, key, now, elapsed, cutoff):
        """Measure the tail among past calls still pending at this elapsed time.

        Capped durations describe time until either delivery or timeout. A censored
        sample is not a plausible successful answer at its cutoff.
        """
        if not 0 <= elapsed < cutoff:
            raise ValueError("Call must still have time before its deadline")
        tails = [(min(seconds,cutoff)-elapsed, delivered and seconds<=cutoff)
                 for date, seconds, delivered in self.samples[key] if date<=now and seconds>elapsed]
        successes = [remaining for remaining, delivered in tails if delivered]
        return dict(samples=len(tails), deliveries=len(successes),
                    median_until_settled_s=median(t for t,_ in tails) if tails else None,
                    median_until_reply_s=median(successes) if successes else None,
                    maximum_wait_s=cutoff-elapsed)
