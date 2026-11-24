"""Shared conditional event model: correlated-hover residuals as an implementation nuisance.

Observation context is the last actual external event. Each hypothesis retains
a residual relative to its phase posterior, shared by real filtering and future
event generation. Repeated stationary packets need not be independent evidence
for a particular full proxy route. No current task/pose/log enters this update.
"""
import numpy as np

from experiments.task_attack_operation.execution import ExecutionBelief


class CorrelatedBelief(ExecutionBelief):
    def __init__(self,model,perception,actions,known=None):
        super().__init__(model,perception,actions,known)
        self.unadjusted_event=self.event.copy();self.residuals=np.zeros_like(self.event)
        self.strength=model['event_residual_strength']

    def observe(self,event,mask,t):
        super().observe(event,mask,t)
        event=np.asarray(event);mask=np.asarray(mask)
        for section in self.template_slices:
            mass=self.p[section].sum()
            if mass<=1e-300:continue
            mean=self.p[section].dot(self.unadjusted_event[section])/mass
            residual=event-mean
            self.residuals[section,mask]=self.strength*residual[mask]
        self.event=self.unadjusted_event+self.residuals


