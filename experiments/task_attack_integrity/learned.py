"""Shared execution belief; independently strengthened prior/direct controls."""
from experiments.task_attack_operation.learned import Policy as OperationPolicy
from experiments.task_attack_value.learned import Policy as ValuePolicy,informative
from experiments.task_attack_value.vision import candidate_features
from .execution import CorrelatedBelief


class DispatchValuePolicy(ValuePolicy):
    def decide(self,history,elapsed,budget):
        if not history:return dict(kind='wait',reason='No actual external observation')
        if self.initial is None:self.initial=history[0]
        # The direct control is retrained with the actual common dispatch
        # delay in its labels. Do not shift the feature clock a second time.
        dispatch_time=elapsed if self.model.get('effect_label_dispatch_delay_seconds') else elapsed+self.model['operation_model']['fixed_compute_dispatch_seconds']
        x=candidate_features(history if self.dynamic else [self.initial],dispatch_time,self.perception['anchors'])
        static=candidate_features([self.initial],dispatch_time,self.perception['anchors'])
        if x is None:return dict(kind='wait',reason='No actual external body track')
        information=informative(self.model,history,elapsed) if self.dynamic else 0.
        return self.decide_features(x,static,dispatch_time,budget,information)


class CorrelatedPolicy(OperationPolicy):
    def __init__(self,model,perception,actions,dynamic,known=None):
        self.model=model;self.perception=perception;self.actions=actions;self.dynamic=dynamic;self.known=known
        self.execution=CorrelatedBelief(model['operation_model'],perception,actions,known)
        self.initial=None;self.first_stamp=None;self.last_id=None;self.last_sample=None


class Policy:
    def __init__(self,model,perception,actions,dynamic,known=None):
        # The prior is allowed to use the better-performing independently
        # trained STATIC controller. Do not weaken it to share our architecture.
        self.delegate=(CorrelatedPolicy if dynamic else DispatchValuePolicy)(model,perception,actions,dynamic,known)

    def decide(self,*args,**kwargs):return self.delegate.decide(*args,**kwargs)
    def decide_features(self,*args,**kwargs):return self.delegate.decide_features(*args,**kwargs)
