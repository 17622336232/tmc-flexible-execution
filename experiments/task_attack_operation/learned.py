"""Expose the shared execution process to actual and OWN replay controllers."""
import time
import numpy as np

from .execution import ExecutionBelief,observed_event
from experiments.task_attack_value.vision import candidate_features


class Policy:
    def __init__(self,model,perception,actions,dynamic,known=None):
        self.model=model;self.perception=perception;self.actions=actions;self.dynamic=dynamic;self.known=known
        self.execution=ExecutionBelief(model['operation_model'],perception,actions,known)
        self.initial=None;self.first_stamp=None;self.last_id=None;self.last_sample=None

    def decide_features(self,x,static,elapsed,budget,information=None):
        if self.dynamic:
            event,mask=observed_event(x,self.perception,self.execution.pose_heads)
            self.execution.observe(event,mask,elapsed)
        elif not self.execution.initialized:
            event,mask=observed_event(static,self.perception,self.execution.pose_heads)
            self.execution.observe(event,mask,elapsed)
        else:
            self.execution.p=self.execution.advance(self.execution.p,elapsed-self.execution.clock);self.execution.clock=elapsed
        return self.execution.decide(budget,elapsed,self.dynamic)

    def decide(self,history,elapsed,budget):
        if not history:return dict(kind='wait',reason='No actually observed external image')
        if self.initial is None:
            self.initial=history[0];self.first_stamp=self.initial['capture_ros']
            initial=candidate_features([self.initial],0.,self.perception['anchors'])
            if initial is None:return dict(kind='wait',reason='Initial external image has no reliable body track')
            self.execution.observe(*observed_event(initial,self.perception,self.execution.pose_heads),0.)
            self.last_id=self.initial['id']
        if self.dynamic:
            latest=history[-1]
            if latest['id']!=self.last_id:
                x=candidate_features(history,elapsed,self.perception['anchors'])
                if x is not None:self.execution.observe(*observed_event(x,self.perception,self.execution.pose_heads),latest['capture_ros']-self.first_stamp)
                self.last_id=latest['id']
            # Arrival age is public; no ROS pose/time interface is queried.
            age=max(0.,time.time()-latest['arrival_wall'])
            current=self.execution.p.copy();self.execution.p=self.execution.advance(current,age)
            decision=self.execution.decide(budget,elapsed,True);self.execution.p=current
            decision['external_arrival_age_seconds']=age
            return decision
        target=max(0.,time.time()-self.initial['arrival_wall'])
        self.execution.p=self.execution.advance(self.execution.p,max(0.,target-self.execution.clock));self.execution.clock=target
        return self.execution.decide(budget,elapsed,False)
