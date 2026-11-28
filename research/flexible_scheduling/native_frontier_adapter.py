"""Shadow integration with the hash-pinned original controller and its ordinary requests."""
from functools import wraps
from hashlib import sha256
from unittest.mock import patch

from experiments.flexible_execution.adapters import original_controller
from .event_workflow import Mission,Query,State,UNKNOWN
from .natural_frontier import discover, read_ordinary_records, released, source_map


class NativeFrontierObserver:
    def __init__(self, task):
        self.task = task
        self.plan = discover(task)
        self.anchor = None
        self.version = None
        self.prior = {}

    def consume(self, context, image_hashes, reply):
        """Use only the photo/pose pairs in the actual ordinary request and earlier accepted records."""
        if self.plan is None or context["authorized_task"] != self.task:
            return dict(status="unsupported")
        version = context.get("scene_conditions", {}).get("state_version")
        if self.anchor is None:
            self.anchor, self.version = context["geometry"], version
        if version != self.version:
            self.plan = None
            return dict(status="scope-changed")
        geometry = context["geometry"]
        current = dict(image_id=context["current_image_id"],position=geometry["position"],yaw=geometry["yaw"])
        supplied = [*context.get("recent_images",[]),current]
        if len(supplied) != len(image_hashes) or len({p["image_id"] for p in supplied})!=len(supplied):
            return dict(status="unmatched-sources")
        views = source_map(self.plan,self.anchor,supplied)
        hashes = dict(zip((p["image_id"] for p in supplied), image_hashes))
        new = read_ordinary_records(self.plan,reply,self.prior,views)
        self.prior.update(new)
        ready = released(self.plan,self.prior)
        return dict(status="observed", newly_bound=[self.plan.fields[i].name for i in sorted(new)],
                    available_sources={i:dict(view=views[i],sha256=hashes[i]) for i in views},
                    authorized_next=[dict(field=self.plan.fields[i].name,view=self.plan.fields[i].view,
                                          photos=[ref for ref,v in views.items() if v==self.plan.fields[i].view])
                                     for i in sorted(ready)],
                    initial_side_capture_grant=self.plan.early_capture)


def run_shadow(backend, task, directory, event, timeout, *, model, trace):
    """Log the discovered frontier while running the original transport and executing its reply unchanged."""
    observe = NativeFrontierObserver(task)
    original_request = model.request

    @wraps(original_request)
    def request(instructions, context, images=(), **kwargs):
        reply = original_request(instructions,context,images,**kwargs)
        if observe.plan is not None:
            try:
                discovery = observe.consume(context,[sha256(v["data"]).hexdigest() for v in images],reply)
            except (KeyError,TypeError,ValueError,AttributeError):
                discovery = dict(status="unmatched-native-format")
            trace.event("natural_frontier_observed",context.get("current_image_id"),**discovery)
        return reply

    with patch.object(model,"request",request):
        return original_controller.run(backend,task,directory,event,timeout,model=model,trace=trace)


def scheduling_state(observer, available_views, measured_view, travel_ticks, capture_ticks, service_ticks):
    """Compile the discovered user conditions and current evidence into the existing planner's state."""
    plan = observer.plan
    if plan is None:
        raise ValueError("No supported, anchored task")
    names = ("front","side")
    finish = names.index(plan.finish)
    queries = tuple(Query(f"{field.view}:{field.name}",(names.index(field.view),),
                          tuple(tuple((i,True) for i in joint) for joint in field.any_of))
                    for field in plan.fields)
    raw = sum(1<<names.index(view) for view in set(available_views))
    mission = Mission(names,queries,travel_ticks,capture_ticks,(True,plan.early_capture),0,raw,service_ticks,0,finish,1)
    values = tuple(int(observer.prior[i]) if i in observer.prior else UNKNOWN for i in range(len(queries)))
    return mission,State(values,raw,names.index(measured_view))
