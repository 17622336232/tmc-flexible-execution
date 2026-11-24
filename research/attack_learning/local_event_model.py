"""Shared external-event E law for actual filtering and imagined windows.

Residuals are conditional on the previous observed residual. Repeated hover
packets are not independent absolute-pose votes. Teacher geometry is not used
by this event model: its six channels are actual external pixel/RGB features.
"""
import math
import numpy as np

from build_local_operator_dataset import canonical


def fit(episodes):
    samples=[];launch=[];request_lags=[]
    for ep in episodes:
        events=ep['events'];examples=ep['examples'];perception=ep['perception']
        first=examples[0]
        launch.append(dict(t=first['t'],delta=(np.array(first['canonical_event'])-canonical(events[0]['value'],perception)).tolist()))
        for ex in examples:
            start=ex['t'];processing=ex['call_seconds']
            if processing is None:continue
            # Capture-to-request and final settle lags remain in the empirical
            # local segment residual, rather than becoming task-specific time.
            end=start+processing
            if ex['next_t'] is not None:end=min(end,ex['next_t'])
            prior=None
            for e in [e for e in events if start<=e['t']<=end]:
                residual=np.array(canonical(e['value'],perception))-ex['canonical_event']
                mask=np.array(e['mask'])&np.array(ex['event_mask'])
                if prior is not None:samples.append(dict(dt=e['t']-prior[0],before=prior[1],after=residual,kind='read',mask=mask&prior[2]))
                prior=(e['t'],residual,mask)
            if ex['next_t'] is not None and ex['next_t']>end+.1:
                prior=None;duration=ex['next_t']-end
                for e in [e for e in events if end<e['t']<=ex['next_t']]:
                    ratio=(e['t']-end)/duration
                    mean=(1-ratio)*np.array(ex['canonical_event'])+ratio*np.array(ex['next_canonical_event'])
                    residual=np.array(canonical(e['value'],perception))-mean
                    next_mask=min(events,key=lambda z:abs(z['t']-ex['next_t']))['mask']
                    mask=np.array(e['mask'])&np.array(ex['event_mask'])&np.array(next_mask)
                    if prior is not None:samples.append(dict(dt=e['t']-prior[0],before=prior[1],after=residual,kind='move',mask=mask&prior[2]))
                    prior=(e['t'],residual,mask)
    candidates=[]
    for tau in (0.,1.,3.,10.,30.):
        innovations=[];masks=[];jacobians=[]
        for s in samples:
            rho=math.exp(-s['dt']/tau) if tau else 0.
            factor=math.sqrt(max(1e-6,1-rho*rho))
            innovations.append((s['after']-rho*s['before'])/factor);masks.append(s['mask']);jacobians.append(2*math.log(factor))
        innovations=np.array(innovations);masks=np.array(masks)
        sigma=np.maximum(np.array([.001,.001,.001,.001,.01,.01]),np.sqrt(np.sum(innovations**2*masks,axis=0)/np.maximum(1,masks.sum(axis=0))))
        value=float(np.sum(((innovations/sigma)**2+2*np.log(sigma)+np.array(jacobians)[:,None])*masks)/masks.sum())
        candidates.append(dict(tau=tau,sigma=sigma.tolist(),training_objective=value))
    chosen=min(candidates,key=lambda c:c['training_objective'])
    return dict(**chosen,fit_candidates=candidates,samples=len(samples),launch=launch,
        distribution='Conditional diagonal Gaussian OU residual; same real/hypothetical law, with an exact residual semigroup. Sigma fitted on OWN local residuals.',
        limitations='Time constant chosen on training conditional residual likelihood; not yet independently calibrated against complete unknown-goal belief. No arbitrary-task OOD probability claim.')


def conditional(model,mean,previous_mean,previous_observed,dt,mask,previous_mask=None):
    tau=model['tau'];rho=math.exp(-max(dt,1e-8)/tau) if tau else 0.
    mask=np.array(mask,dtype=bool)
    if previous_observed is None:
        return np.asarray(mean)[mask],np.array(model['sigma'])[mask]
    valid=np.asarray(previous_mask,dtype=bool) if previous_mask is not None else np.ones(6,dtype=bool)
    center=np.where(valid,np.asarray(mean)+rho*(np.asarray(previous_observed)-np.asarray(previous_mean)),np.asarray(mean))
    stationary=np.array(model['sigma'])
    scale=np.where(valid,stationary*math.sqrt(max(1e-6,1-rho*rho)),stationary)
    return center[mask],scale[mask]


def log_density(model,event,mask,mean,previous_mean,previous_observed,dt,previous_mask=None):
    center,scale=conditional(model,mean,previous_mean,previous_observed,dt,mask,previous_mask)
    residual=(np.asarray(event)[np.asarray(mask,dtype=bool)]-center)/scale
    return float(np.sum(-.5*math.log(2*math.pi)-np.log(scale)-.5*residual*residual))


def sample(model,rng,mean,previous_mean,previous_observed,dt,previous_mask=None):
    center,scale=conditional(model,mean,previous_mean,previous_observed,dt,[True]*6,previous_mask)
    return center+scale*rng.normal(size=6)
