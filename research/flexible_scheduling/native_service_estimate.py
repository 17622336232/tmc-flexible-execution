"""Request-shape service estimates using only settled observations."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean, median


@dataclass(frozen=True)
class RequestObservation:
    available_at: float
    duration_s: float
    delivered: bool
    independently_correct: bool | None = None


def restricted_request_estimate(samples, *, now, elapsed_s, limit_s, prior_s):
    """Estimate delay and delivery risk conditional on the request still being pending.

    Each past timeout contributes its observed censoring duration and no success.
    A verified wrong result also contributes no success. Unverified results inform
    delivery and time, with their verification coverage reported separately.
    The bounded selection proxy charges failures the remaining decision horizon.
    """
    if not all(isfinite(x) for x in (now,elapsed_s,limit_s,prior_s)) or not 0<=elapsed_s<limit_s or prior_s<=0:
        raise ValueError("Invalid service clock or prior")
    visible = [r for r in samples if r.available_at<=now and r.duration_s>elapsed_s]
    remaining = limit_s-elapsed_s
    delivered = [r for r in visible if r.delivered and r.duration_s<=limit_s]
    successful = [r for r in delivered if r.independently_correct is not False]
    known = [r for r in delivered if r.independently_correct is not None]
    fall_back = min(remaining,max(.01,prior_s-elapsed_s)) if not elapsed_s else remaining
    completed = [r.duration_s-elapsed_s for r in successful]
    unusable = len(visible)-len(successful)
    # A nominal prior contributes one sample before dispatch. During a request the
    # answer times have already survived the elapsed-time filter.
    priors = [fall_back] if not elapsed_s or not visible else []
    return dict(
        samples=len(visible),successful=len(successful),nondelivery=len(visible)-len(delivered),
        independently_checked=len(known),incorrect=sum(r.independently_correct is False for r in known),
        median_until_usable_s=median(completed) if completed else None,
        estimated_restricted_s=mean([*completed,*([remaining]*unusable),*priors]),
    )
