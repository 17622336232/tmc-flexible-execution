"""Learn local motion shape and mode-dependent pixel uncertainty from OWN data.

No task labels enter this fit. A turning/translation segment and a processing
hover need different event uncertainty. Native motion/control is unchanged.
"""
import math
import numpy as np
from build_local_operator_dataset import canonical


def basis(s,duration):
    z=2*s-1;v=s*(1-s);d=math.log1p(duration)
    return np.array([v,v*z,v*z*z,v*z*z*z,v*d,v*z*d])


def factors(model,mode,s,duration):
    if mode not in model.get('mean_curves',{}):return np.full(6,s)
    return s+np.asarray(model['mean_curves'][mode]).dot(basis(s,duration))


def fit(episodes):
    segments=[];launch=[]
    for ep in episodes:
        events=ep['events'];examples=ep['examples'];perception=ep['perception'];first=examples[0]
        initial=canonical(events[0]['value'],perception)
        launch.append(dict(t=first['t'],delta=(np.array(first['canonical_event'])-initial).tolist()))
        candidates=[('launch',0.,first['t'],initial,first['canonical_event'],events[0]['mask'],first['event_mask'])]
        for ex in examples:
            start=ex['t'];end=start+ex['call_seconds']
            if ex['next_t'] is not None:end=min(end,ex['next_t'])
            candidates.append(('read',start,end,ex['canonical_event'],ex['canonical_event'],ex['event_mask'],ex['event_mask']))
            if ex['next_t'] is not None and ex['next_t']>end+.1:
                next_mask=min(events,key=lambda e:abs(e['t']-ex['next_t']))['mask']
                candidates.append(('move',end,ex['next_t'],ex['canonical_event'],ex['next_canonical_event'],ex['event_mask'],next_mask))
        for mode,start,end,a,b,mask_a,mask_b in candidates:
            if end<=start:continue
            points=[dict(t=e['t'],value=np.array(canonical(e['value'],perception)),mask=np.array(e['mask'])&mask_a&mask_b)
                for e in events if start<=e['t']<=end]
            segments.append(dict(mode=mode,start=start,end=end,a=np.array(a),b=np.array(b),points=points))
    curves={}
    for mode in ('launch','move'):
        betas=[]
        for c in range(6):
            xx=[];yy=[]
            for segment in [s for s in segments if s['mode']==mode]:
                delta=segment['b'][c]-segment['a'][c];duration=segment['end']-segment['start']
                for point in segment['points']:
                    if not point['mask'][c]:continue
                    ratio=(point['t']-segment['start'])/duration
                    xx.append(delta*basis(ratio,duration));yy.append(point['value'][c]-segment['a'][c]-ratio*delta)
            x=np.array(xx);y=np.array(yy);ridge=max(1e-8,.001*np.trace(x.T.dot(x))/len(x[0]))
            betas.append(np.linalg.solve(x.T.dot(x)+ridge*np.eye(x.shape[1]),x.T.dot(y)).tolist())
        curves[mode]=betas
    provisional=dict(mean_curves=curves);pairs=[];raw=[]
    for segment in segments:
        previous=None;duration=segment['end']-segment['start']
        for point in segment['points']:
            ratio=(point['t']-segment['start'])/duration
            f=factors(provisional,segment['mode'],ratio,duration)
            mean=segment['a']+f*(segment['b']-segment['a']);residual=point['value']-mean
            raw.append(dict(mode=segment['mode'],r=residual,mask=point['mask']))
            if previous is not None:pairs.append(dict(mode=segment['mode'],dt=point['t']-previous[0],old=previous[1],new=residual,mask=point['mask']&previous[2]))
            previous=(point['t'],residual,point['mask'])
    candidates=[];floors=np.array([.001,.001,.001,.001,.01,.01])
    for tau in (0.,1.,3.,10.,30.):
        sigmas={};nll=0.;count=0
        for mode in ('read','move','launch'):
            rows=[p for p in pairs if p['mode']==mode];innovations=[];masks=[];logf=[]
            for p in rows:
                rho=math.exp(-p['dt']/tau) if tau else 0.;factor=math.sqrt(max(1e-8,1-rho*rho))
                innovations.append((p['new']-rho*p['old'])/factor);masks.append(p['mask']);logf.append(2*math.log(factor))
            if not rows:sigmas[mode]=floors.tolist();continue
            innovations=np.array(innovations);masks=np.array(masks)
            sigma=np.maximum(floors,np.sqrt(np.sum(innovations**2*masks,axis=0)/np.maximum(1,masks.sum(axis=0))))
            sigmas[mode]=sigma.tolist()
            nll+=float(np.sum(((innovations/sigma)**2+2*np.log(sigma)+np.array(logf)[:,None])*masks));count+=int(masks.sum())
        candidates.append(dict(tau=tau,mode_sigma=sigmas,training_objective=nll/count))
    chosen=min(candidates,key=lambda r:r['training_objective'])
    std={}
    for mode in ('read','move','launch'):
        rows=[r for r in raw if r['mode']==mode];vals=np.array([r['r'] for r in rows]);masks=np.array([r['mask'] for r in rows])
        std[mode]=np.sqrt(np.sum(vals**2*masks,axis=0)/np.maximum(1,masks.sum(axis=0))).tolist()
    return dict(**chosen,sigma=chosen['mode_sigma']['read'],mean_curves=curves,fit_candidates=candidates,launch=launch,
        samples=len(pairs),mode_residual_rms=std,source='OWN external RGB local segments only; task labels never used',
        residual_reset='Independent residual at each local operation boundary; correlated OU residual within an operation.',
        distribution='Mode-conditioned Gaussian OU residual with learned boundary-preserving motion mean curves',
        limitations='Training residual likelihood only. Complete unknown-goal prefix calibration still required; endpoint-model misspecification remains possible.')
