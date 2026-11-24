"""Fit READ processing times with explicit right censoring.

Only proxy logs are used offline. A strict transport rejection is a failure
branch, not a secretly successful read. No global host time is changed.
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm
from scipy.stats import norm


def fit_lognormal(times,censored):
    times=np.asarray(times,dtype=float);censored=np.asarray(censored,dtype=bool)
    if len(times)<3 or np.any(times<=0) or np.all(censored):raise ValueError('Insufficient duration support')
    logs=np.log(times)
    def objective(theta):
        sigma=math.exp(float(np.clip(theta[1],-12,6)));mu=theta[0]
        observed=lognorm.logpdf(times[~censored],sigma,scale=math.exp(mu)).sum()
        survival=lognorm.logsf(times[censored],sigma,scale=math.exp(mu)).sum()
        return -float(observed+survival)/len(times)
    def gradient(theta):
        mu=theta[0];sigma=math.exp(theta[1]);z=(logs-mu)/sigma
        mills=np.exp(norm.logpdf(z[censored])-norm.logsf(z[censored]))
        dmu=np.sum(z[~censored])/sigma+np.sum(mills)/sigma
        dsigma=np.sum(z[~censored]**2-1)+np.sum(mills*z[censored])
        return -np.array([dmu,dsigma])/len(times)
    result=minimize(objective,[np.mean(logs[~censored]),math.log(max(.05,np.std(logs[~censored])))],jac=gradient,method='BFGS')
    if not np.isfinite(result.fun):raise ValueError('Nonfinite censored likelihood')
    return dict(mu=float(result.x[0]),sigma=float(math.exp(result.x[1])),
        completed=int((~censored).sum()),right_censored=int(censored.sum()),
        negative_log_likelihood=float(result.fun)*len(times),optimizer_success=bool(result.success),optimizer_message=str(result.message),
        gradient_norm=float(np.linalg.norm(gradient(result.x))))


def sample(model,rng,size=None):return rng.lognormal(model['mu'],model['sigma'],size=size)


def sample_surviving(model,age,rng):
    """Exact D | D>age Gibbs update for a still-active pooled READ."""
    cdf=lognorm.cdf(max(0.,age),model['sigma'],scale=math.exp(model['mu']))
    probability=min(np.nextafter(1.,0.),cdf+(1-cdf)*rng.rand())
    return float(lognorm.ppf(probability,model['sigma'],scale=math.exp(model['mu'])))


def collect(episodes):
    import json
    from pathlib import Path
    times=[];censored=[];failures=[];unknown=[]
    for ep in episodes:
        log=Path(ep['directory'])/'private/robot/model_calls.jsonl'
        groups={}
        for step,line in enumerate(log.read_text().splitlines(),1):
            row=json.loads(line);image=row.get('input',{}).get('current_image_id')
            if image is None:unknown.append(dict(episode=ep['key'],step=step));continue
            groups.setdefault(image,[]).append((step,row))
        for image,attempts in groups.items():
            step,row=attempts[-1];request=row.get('input',{});camera=request.get('geometry',{}).get('camera',{})
            duration=row.get('wall_seconds');stamp=camera.get('ros_stamp')
            if not isinstance(duration,(int,float)) or duration<=0 or stamp is None:unknown.append(dict(episode=ep['key'],step=step));continue
            # Same-image retries are one latched READ operation. Counting a
            # failed120s attempt as motion and a later20s attempt as the whole
            # READ would corrupt both duration and cache timing labels.
            duration=row['started_at']+duration-stamp
            if duration<=0:raise ValueError('Incoherent proxy clocks')
            if row.get('output') is not None:times.append(duration);censored.append(False)
            elif (row.get('error') or {}).get('code') in ('timeout','deadline_exceeded'):
                times.append(duration);censored.append(True)
            else:failures.append(dict(episode=ep['key'],step=step,image=image,error=row.get('error'),duration=duration))
    return dict(model=fit_lognormal(times,censored),strict_failure_branches=failures,
                unknown_duration_labels=unknown,raw_times=times,censored=censored)
