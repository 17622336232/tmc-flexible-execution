"""An interruptible search with a previously completed incumbent under each new cost profile."""
import signal
from time import perf_counter

from .event_policy import order


class SearchDeadline(Exception):
    pass


def limited_replan(planner, initial, milliseconds, *, prefetch_only=False, absolute_deadline=None):
    """Stop selection at a wall-clock budget and preserve the best complete evaluated policy."""
    model = planner.model
    candidate = None
    source = "common-collect"
    certificate = False
    began = perf_counter()
    if model.complete(initial):
        return (0.,None),"complete",True,(perf_counter()-began)*1000

    def timeout(*_):
        raise SearchDeadline()

    def improve(value, action, label):
        nonlocal candidate, source
        if absolute_deadline is not None and perf_counter()>=absolute_deadline:
            return
        if candidate is None or (round(value,8),order(action)) < (round(candidate[0],8),order(candidate[1])):
            candidate, source = (value,action), label

    # Install the timer before enumerating actions or scoring any incumbent.
    old = signal.signal(signal.SIGALRM,timeout)
    values = {}
    try:
        remaining = milliseconds/1000. if absolute_deadline is None else absolute_deadline-perf_counter()
        if remaining<=0:
            raise SearchDeadline()
        signal.setitimer(signal.ITIMER_REAL,remaining)
        for style in ("collect","prepare","cost"):
            improve(planner.roll(initial,style),planner.greedy(initial,style),"rollout")

        def shortlist(state):
            candidates = []
            for action in planner.options(state):
                elapsed, branches = planner.trans(state,action)
                lower = elapsed + sum(prob * planner.cost_lb(child) for prob,child in branches)
                candidates.append((lower,order(action),action,elapsed,branches))
            return sorted(candidates,key=lambda row:row[:2])

        options = shortlist(initial)
        for _,_,action,elapsed,branches in options:
            future = sum(prob * planner.one_step(child)[0] for prob,child in branches)
            improve(elapsed+future,action,"two-dispatch")
        if not prefetch_only:
            def exact(state):
                if model.complete(state):
                    return 0.
                k = planner.key(state)
                if k not in values:
                    feasible = planner.incumbent(state)[0]
                    for lower,_,_,elapsed,branches in shortlist(state):
                        if lower>=feasible-1e-8:
                            break
                        feasible = min(feasible,elapsed+sum(prob*exact(child) for prob,child in branches))
                    values[k] = feasible
                return values[k]

            for lower,_,action,elapsed,branches in options:
                if lower>=candidate[0]-1e-8:
                    break
                improve(elapsed+sum(prob*exact(child) for prob,child in branches),action,"complete-recourse")
            certificate = absolute_deadline is None or perf_counter()<absolute_deadline
    except SearchDeadline:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)
    return candidate, source, certificate, (perf_counter()-began)*1000
