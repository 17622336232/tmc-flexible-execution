"""Frozen local dependency model, compositional effects and learned lookahead."""
import math
import numpy as np

from .programs import payload,semantic_effect
from .vision import candidate_features
from experiments.task_attack.event_timing import features as motion_features


def softmax(value):
    value=value-np.max(value,axis=-1,keepdims=True);p=np.exp(value)
    return p/p.sum(axis=-1,keepdims=True)


class Policy:
    def __init__(self,model,perception,actions,dynamic,known=None):
        self.model=model;self.perception=perception;self.actions=actions
        self.dynamic=dynamic;self.known=known;self.initial=None
        self.effects={a['action_id']:semantic_effect(model['programs'],model['world_particles'],
            model['parameters'],a,perception['stations'],known) for a in actions}
        self.action_by_id={a['action_id']:a for a in actions}
        self.groups=model['dependency']['groups'];self.layout=model['layout']
        self.forecast=model.get('trajectory_library',[])
        self.static_x=None;self.information_weight=0.

    def dependency(self,x,role,dynamic=None,groups=None,static_x=None,information_weight=None):
        group=(groups or self.groups)[self.layout+':'+role];use=self.dynamic if dynamic is None else dynamic
        use=use and group['use_dynamic']
        static=x if static_x is None else static_x
        def evaluate(values,name):
            standardized=(values-np.array(group['mean']))/np.array(group['scale'])
            return softmax(np.column_stack([np.ones(len(values)),standardized]).dot(group[name]['beta']))
        prior=evaluate(static,'clock_only')
        if not use:return prior
        conditional=evaluate(x,'dynamic')
        if 'informative_prefix' not in self.model:return conditional
        weight=self.information_weight if information_weight is None else information_weight
        # Common ascent can change future physical visibility, but does not
        # reveal which user requirement or referent is active. Update semantic
        # relevance only in proportion to learned task-dependent evidence.
        prior_need=prior[:,1:].sum(axis=1);dynamic_need=conditional[:,1:].sum(axis=1)
        need=(1-weight)*prior_need+weight*dynamic_need
        result=np.empty_like(conditional);result[:,0]=1-need
        result[:,1:]=need[:,None]*conditional[:,1:]/np.maximum(1e-300,dynamic_need[:,None])
        return result

    def score(self,x,dynamic=None,groups=None,static_x=None,information_weight=None):
        probabilities={r:self.dependency(x,r,dynamic,groups,static_x,information_weight) for r in ('cracked','temperature')};rank=[]
        for action in self.actions:
            change=payload(action)
            if change is None:continue
            role=change['role'];source=change['source'];p=probabilities[role][source]
            fresh,cached=float(p[2]),float(p[3])
            if self.known:
                program,target=self.known;relevant=role in program['roles'] and (program['scope']!='target' or source==target)
                need=max(1e-300,float(np.sum(p[1:])));fresh=fresh/need if relevant else 0.;cached=cached/need if relevant else 0.
            endpoint=self.model['endpoint']['by_role'][role]
            effect=self.effects[action['action_id']]
            value=effect*(fresh*endpoint['fresh_adoption']+cached*endpoint['cached_adoption'])
            rank.append(dict(action_id=action['action_id'],expected_wrong_accepted=value,
                semantic_effect_given_relevance=effect,needed_fresh_read=fresh,needed_cached_read=cached,
                learned_fresh_adoption=endpoint['fresh_adoption'],learned_cached_adoption=endpoint['cached_adoption']))
        rank.sort(key=lambda item:(-item['expected_wrong_accepted'],item['action_id']))
        return rank

    def future_branches(self,x,elapsed,delay):
        # The branch library contains OWN development trajectories only. It
        # forecasts possible observations; no current episode future is opened.
        group=self.groups[self.layout+':temperature'];scale=np.array(group['scale'])
        choices=[]
        for episode in self.forecast:
            samples=episode['samples'];best=None
            for index,sample in enumerate(samples):
                current=np.array(sample['x' if self.dynamic else 'static'])
                if current.shape!=x.shape:continue
                distance=float(np.mean(np.minimum(8.,np.abs((current-x)/scale))**2))
                if best is None or distance<best[0]:best=(distance,index)
            if best is None:continue
            distance,index=best;wanted=samples[index]['t']+delay
            future=next((s for s in samples[index:] if s['t']>=wanted),None)
            # A terminated analogue is an actual no-op terminal branch, not
            # silently removed because it would lower the value of waiting.
            choices.append((distance,None if future is None else np.array(future['x' if self.dynamic else 'static']),episode.get('dependency'),
                None if future is None else np.array(future['static']),0. if future is None else future.get('informative_probability',1.)))
        choices.sort(key=lambda item:item[0]);choices=choices[:min(8,len(choices))]
        if not choices:return []
        weights=np.array([math.exp(-.5*item[0]) for item in choices]);weights/=max(1e-20,weights.sum())
        return [(float(w),value,heads,static,information) for w,(_,value,heads,static,information) in zip(weights,choices)]

    def decide(self,history,elapsed,budget):
        if not history:return dict(kind='wait',reason='No actual external RGB track yet')
        if self.initial is None:self.initial=history[0]
        x=candidate_features(history if self.dynamic else [self.initial],elapsed,self.perception['anchors'])
        if x is None:return dict(kind='wait',reason='No measured robot pixels; no world-state fallback')
        static_x=candidate_features([self.initial],elapsed,self.perception['anchors'])
        component=self.model.get('informative_prefix');information=0.
        if self.dynamic and component:
            external=motion_features(history,elapsed)
            if external is not None:
                design=np.r_[1.,(external[:-1]-np.array(component['mean']))/np.array(component['scale'])]
                information=float(softmax(design.dot(component['beta']))[1])
        self.information_weight=information
        rank=self.score(x,static_x=static_x);now=rank[0]['expected_wrong_accepted'] if rank else 0.;wait=[]
        for delay in (5.,15.,30.,60.):
            if elapsed+delay>=budget:continue
            branches=self.future_branches(x,elapsed,delay);value=0.
            for weight,future,heads,static,future_information in branches:
                if future is not None:
                    future_rank=self.score(future,groups=heads,static_x=static,information_weight=future_information);value+=weight*(future_rank[0]['expected_wrong_accepted'] if future_rank else 0.)
            if branches:wait.append(dict(delay_seconds=delay,expected_wrong_accepted=value,branches=len(branches)))
        best=max(wait,key=lambda item:item['expected_wrong_accepted']) if wait else None
        # Zero edit cost, one irreversible physical edit: use it now whenever
        # forecast waiting cannot improve the SAME terminal objective.
        if best and best['expected_wrong_accepted']>now+1e-9:
            kind='wait';reason='Learned possible future observations improve terminal value'
        elif now>0:kind='edit';reason='Immediate terminal value is at least as good as waiting'
        else:kind='abstain';reason='No modeled factual task effect from allowed carriers'
        return dict(kind=kind,reason=reason,ranked_candidates=rank,wait_branches=wait,
                    immediate_value=now,selected_delay_seconds=best['delay_seconds'] if best and kind=='wait' else None,
                    learned_task_dependent_evidence_probability=information,
                    information='external RGB prefix' if self.dynamic else 'initial external RGB and public clock')
