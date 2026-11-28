"""Shared evidence recording and measured sign views in the actual original-style control loop."""
import re
from time import perf_counter

from experiments.flexible_execution.adapters.native_views import ViewScheduler,close_pose
from .front_facing import sign_goals
from .workflow import Workflow,View,DecisionProblem


EVIDENCE_INSTRUCTIONS="""
For every in-scope object that you actually inspect, put the observed fields and the exact supplied
evidence_image_id into top-level records in the SAME response, including objects that fail a filter.
A final filtered report may list only qualifying objects. Keep their excluded neighbors' observed
facts in top-level records; this is how your coverage can be audited. Never assign a count or identity
from an unseen surface. If the image does not resolve an item, obtain another source or leave it open.
A large top display can differ from a small body identity plate. Inspect the source indicated by the task.
With an observe or record action you can request one or two offered views using use_views, a list
of their ids. Each use refers only to a view you need for the remaining inspection. A previous image
that already supports this field on this object can be reused. Do not request that view again.
"""
_METHODS={"evidence-off":"serial","evidence-async":"async","evidence-prefetch":"strong-prefetch",
          "evidence-planner":"generic","evidence-deferred":"factored"}


class EvidenceScheduler(ViewScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.instructions+=EVIDENCE_INSTRUCTIONS

    def nominate(self,decision,obs,geometry,image_id,step):
        # Unparsed single-target or restricted tasks do not license searching other objects.
        # Known named-task grounding is checked again against the current image in the base loop.
        if self.program is None and not re.match(r"^(?:请)?检查(?:眼前)?这一排(?:货位|仓储货位|设备|车位|通道)",self.task):
            self.trace.event("native_nomination_refused",step,reason="object scope unsupported by local tool")
            return []
        return super().nominate(decision,obs,geometry,image_id,step)

    def camera_goals(self,obs,geometry,target,field):
        if field=="identity":
            for proposal in sign_goals(obs["data"],geometry,target.bbox):
                details={k:v for k,v in proposal.items() if k!="goal"}
                yield proposal["goal"],dict(method="observed_rectangular_surface",**details)
        yield from super().camera_goals(obs,geometry,target,field)

    def requested_views(self,answer,offers):
        values=answer.get("use_views")
        if values is None:
            return super().requested_views(answer,offers)
        if not isinstance(values,list) or len(values)>2 or any(not isinstance(v,str) for v in values):
            return []
        return [o for o in offers if o.id in values]

    def select_many(self,offers,anchor,window,remaining,step):
        begin=perf_counter()
        allowed=[]
        costs=[]
        try:
            self.backend.check_goal(anchor)
        except (ValueError,RuntimeError):
            return []
        for offer in offers:
            trip=self.costs.move_seconds(anchor,offer.goal)+self.costs.move_seconds(offer.goal,anchor)+self.costs.observation_seconds
            if trip+window>=remaining or close_pose(anchor,offer.goal):
                continue
            try:
                self.backend.check_goal(offer.goal)
            except (ValueError,RuntimeError):
                continue
            allowed.append(offer)
            costs.append(trip)
        if not allowed:
            return []
        # A normal controller reply settles latent demand for each nominated view. Reading any
        # released photos is piggybacked on the next ordinary decision for every comparison arm.
        # Semantic service is not shortened by assumption; only marginal extra RPCs are zero.
        factors=tuple(((1<<i,1.),) if o.independent else ((0,.5),(1<<i,.5)) for i,o in enumerate(allowed))
        views=tuple(View(o.id,1<<i,costs[i],True,False) for i,o in enumerate(allowed))
        task=Workflow(factors,views,0,(window,),(0.,0.),0.,max_early_motion=max(0.,remaining-window))
        result=DecisionProblem(task).solve(self.solver_method())
        self.trace.event("native_policy_checked",step,policy=self.method,request_window_s=window,
             options=[dict(id=o.id,roundtrip_s=c,independent=o.independent) for o,c in zip(allowed,costs)],
             selected=[allowed[i].id for i in result["overlap"]],expected_section_s=result["expected_seconds"],
             candidate_plans=result["evaluated"],planning_seconds=perf_counter()-begin,
             cost_scope="remaining camera work plus current ordinary request, identical later semantic service")
        return [allowed[i] for i in sorted(result["overlap"],key=lambda j:(not allowed[j].independent,costs[j],allowed[j].id))]

    def solver_method(self):
        return _METHODS[self.method]
