"""Local operation values from OWN physical/program counterfactuals.

The unknown policy never receives a current target instruction/world/onboard
state. The known-task diagnostic adds authorized natural-language semantics.
"""
import math
import numpy as np

from experiments.task_attack.event_timing import features as motion_features
from .forest import predict
from .programs import payload
from .value_features import action_features,source_features
from .vision import candidate_features


def informative(model,history,elapsed):
    x=motion_features(history,elapsed)
    if x is None:return 0.
    head=model['informative_prefix'];z=np.r_[1.,(x[:-1]-np.array(head['mean']))/np.array(head['scale'])]
    logits=z.dot(head['beta']);weights=np.exp(logits-logits.max());return float(weights[1]/weights.sum())


class Policy:
    def __init__(self,model,perception,actions,dynamic,known=None):
        self.model=model;self.perception=perception;self.actions=actions;self.dynamic=dynamic;self.known=known
        self.action_by_id={a['action_id']:a for a in actions};self.initial=None;self.information_weight=0.
        self.forecast=model.get('trajectory_library',[]);self.metric_scale=np.array(model['metric_scale'])
        self.library=[(e,np.array(e['descriptor']),np.array(e.get('static_descriptor',e['descriptor']))) for e in self.forecast if e.get('descriptor')]

    def descriptor(self,x):
        return np.concatenate([source_features(x,self.perception,source) for source in range(len(x))])

    def score(self,x,dynamic=None,groups=None,static_x=None,information_weight=None):
        use=self.dynamic if dynamic is None else dynamic;heads=groups or self.model['value_models']
        static_x=x if static_x is None else static_x;features,actions=action_features(x,self.perception,self.actions,self.known)
        static_features,_=action_features(static_x,self.perception,self.actions,self.known);prefix='known_' if self.known else ''
        static_value=predict(heads[prefix+'static'],static_features)
        weight=self.information_weight if information_weight is None else information_weight
        enabled=self.model.get('use_known_dynamic_value',True) if self.known else self.model.get('use_dynamic_value',True)
        if use and enabled:
            dynamic_value=predict(heads[prefix+'dynamic'],features)
            alpha=self.model.get('information_blend',1.)
            mixture=1. if self.known else (1-alpha)+alpha*weight
            values=(1-mixture)*static_value+mixture*dynamic_value
        else:values=static_value
        ranks=[]
        for action,value in zip(actions,values):
            change=payload(action)
            if self.known:
                program,target=self.known
                if program.get('literal') or change['role'] not in program['roles'] or program['scope']=='target' and target!=change['source']:value=0.
            ranks.append(dict(action_id=action['action_id'],expected_wrong_accepted=float(value),
                source=change['source'],field=change['role'],value=change['value'],
                training_target='Actual-outcome-calibrated whole-query effect and source acquisition opportunity; not an observed attack outcome'))
        ranks.sort(key=lambda item:(-item['expected_wrong_accepted'],item['action_id']));return ranks

    def neighbors(self,x):
        descriptor=self.descriptor(x);choices=[]
        for episode,dynamic_values,static_values in self.library:
            if self.known and 'authorized_program' in episode:
                if episode['authorized_program']!=self.known[0] or episode.get('authorized_target')!=self.known[1]:continue
            values=dynamic_values if self.dynamic else static_values
            if values.shape[1]!=len(descriptor):continue
            distance=np.mean(np.minimum(8.,np.abs((values-descriptor)/self.metric_scale))**2,axis=1);index=int(np.argmin(distance))
            choices.append((float(distance[index]),index,episode))
        choices.sort(key=lambda item:item[0]);limit=self.model.get('forecast_neighbors',len(choices));choices=choices[:min(limit,len(choices))]
        sigma=self.model.get('forecast_kernel_scale',.5)
        weights=np.array([math.exp(-.5*c[0]/sigma**2)*c[2].get('prior_weight',1.) for c in choices]);weights/=max(1e-300,weights.sum())
        return [(float(w),index,episode) for w,(_,index,episode) in zip(weights,choices)]

    @staticmethod
    def retime(x,elapsed):
        x=np.array(x,copy=True);clock=math.log1p(max(0.,elapsed));x[:,15]=clock
        x[:,16]=x[:,0]*clock;x[:,17]=x[:,1]*clock;x[:,18]=x[:,9]*clock
        return x

    def score_requests(self,requests):
        """Batch all hypothetical branches by fold; same values, less latency."""
        batches={};result=[None]*len(requests);prefix='known_' if self.known else ''
        for index,request in enumerate(requests):
            heads=request['groups'];key=id(heads)
            if key not in batches:batches[key]=dict(heads=heads,requests=[])
            dynamic,actions=action_features(request['x'],self.perception,self.actions,self.known)
            static,_=action_features(request['static'],self.perception,self.actions,self.known)
            batches[key]['requests'].append((index,request,dynamic,static,actions))
        for batch in batches.values():
            rows=batch['requests'];heads=batch['heads']
            static_values=predict(heads[prefix+'static'],np.concatenate([r[3] for r in rows]))
            enabled=self.model.get('use_known_dynamic_value',True) if self.known else self.model.get('use_dynamic_value',True)
            dynamic_values=predict(heads[prefix+'dynamic'],np.concatenate([r[2] for r in rows])) if self.dynamic and enabled else static_values
            offset=0
            for index,request,dynamic,static,actions in rows:
                count=len(actions);s=static_values[offset:offset+count];d=dynamic_values[offset:offset+count];offset+=count
                alpha=self.model.get('information_blend',1.);weight=1. if self.known else (1-alpha)+alpha*request['information']
                values=(1-weight)*s+weight*d if self.dynamic and enabled else s
                if self.known:
                    program,target=self.known
                    for j,action in enumerate(actions):
                        change=payload(action)
                        if program.get('literal') or change['role'] not in program['roles'] or program['scope']=='target' and target!=change['source']:values[j]=0.
                result[index]=float(np.max(values)) if len(values) else 0.
        return result

    def decide(self,history,elapsed,budget):
        if not history:return dict(kind='wait',reason='No actual external observations yet')
        if self.initial is None:self.initial=history[0]
        x=candidate_features(history if self.dynamic else [self.initial],elapsed,self.perception['anchors'])
        static=candidate_features([self.initial],elapsed,self.perception['anchors'])
        if x is None:return dict(kind='wait',reason='No actual robot image track')
        information=informative(self.model,history,elapsed) if self.dynamic else 0.
        return self.decide_features(x,static,elapsed,budget,information)

    def decide_features(self,x,static,elapsed,budget,information):
        """Same controller entry for actual-feature development replay."""
        self.information_weight=information if self.dynamic else 0.
        ranked=self.score(x,static_x=static);now=ranked[0]['expected_wrong_accepted'] if ranked else 0.;branches=[]
        neighbors=self.neighbors(x if self.dynamic else static);requests=[];delays=[]
        for delay in (3.,6.,12.,24.,48.):
            if elapsed+delay>=budget:continue
            delays.append(delay)
            for weight,index,episode in neighbors:
                target=episode['samples'][index]['t']+delay
                future=next((s for s in episode['samples'][index:] if s['t']>=target),None)
                if future is None:continue
                # The branch's own world was excluded from these fold weights.
                # A terminated analogue remains a zero-valued branch.
                groups=self.model['crossfit_value_models'][str(episode['fold'])]
                future_x=self.retime(future['x' if self.dynamic else 'static'],elapsed+delay)
                future_static=self.retime(future['static'],elapsed+delay)
                requests.append(dict(delay=delay,weight=weight,groups=groups,x=future_x,static=future_static,
                    information=future['informative_probability'] if self.dynamic else 0.))
        forecast_values=self.score_requests(requests) if requests else []
        for delay in delays:
            value=sum(request['weight']*score for request,score in zip(requests,forecast_values) if request['delay']==delay)
            if neighbors:branches.append(dict(delay_seconds=delay,expected_wrong_accepted=value,branches=len(neighbors)))
        best=max(branches,key=lambda item:item['expected_wrong_accepted']) if branches else None
        if best and best['expected_wrong_accepted']>now+1e-9:kind='wait';reason='Possible future observations have higher learned terminal value'
        elif now>0:kind='edit';reason='Immediate learned terminal value is at least as high as waiting'
        else:kind='abstain';reason='Allowed edits have no learned factual value'
        return dict(kind=kind,reason=reason,ranked_candidates=ranked,immediate_value=now,wait_branches=branches,
            learned_task_dependent_evidence_probability=self.information_weight,
            selected_delay_seconds=best['delay_seconds'] if best and kind=='wait' else None,
            information='actual external JPEG and motion prefix' if self.dynamic else 'initial external JPEG and public clock')
