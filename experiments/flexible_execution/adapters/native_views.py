"""Defer interpretation of optional, actual photos to an ordinary controller turn."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
from math import atan2, cos, dist, sin
from pathlib import Path
from statistics import median
from time import monotonic, sleep
from types import SimpleNamespace

from experiments.task_attack.agent import run_agent
from experiments.task_attack.runtime import write_json

from ..gpt_vision import Material, OUTCOMES, _bbox
from ..live_planning import permission_for
from ..material_schema import INSPECTION_FIELDS
from ..text_requirements import compile_text
from .original_controller import upstream
from .viewpoints import _candidates, xyz


MODES = ("native-off","native-async","native-prefetch","native-planner","native-deferred")
EVIDENCE_MODES = ("evidence-off","evidence-async","evidence-prefetch","evidence-planner","evidence-deferred")
COORDINATED_MODES = ("coordinated-off","coordinated-async","coordinated-prefetch","coordinated-planner",
                     "coordinated-deferred","coordinated-observe-only","coordinated-no-ocr")
INSTRUCTIONS = """
The optional camera tool is a real view of a previously visible subject. If an extra view may be useful
after the next normal decision, you may include at most two next_views in your existing answer:
[{"field":"identity","subject":"which visible object and surface","bbox":[left,top,right,bottom]}].
Use normalized coordinates in CURRENT RGB. Nominate only a subject in the authorized task. Leave the list
empty if existing evidence already suffices. A candidate has no reading or fact attached.
If offered_views are supplied in the next input, use your current observations to decide whether one
is actually needed. To request it, return action {"type":"observe"} and use_view equal to its id.
Otherwise omit use_view. Do not assume whether a view has been taken, or invent its contents.
A delivered image has its own real camera pose, capture time and image_id. Inspect it in the same
decision as the current image. Associate it with the intended object visually before using it.
Keep the original report and coverage requirements. Report only what the supplied images support.
"""


@dataclass(frozen=True)
class Offer:
    id: str
    field: str
    subject: str
    bbox: tuple
    reference_image: str
    reference_hash: str
    goal: dict
    independent: bool
    grant: str

    def visible(self):
        return dict(id=self.id,field=self.field,subject=self.subject,source=self.reference_image,
                    source_bbox=self.bbox,goal=self.goal)


def close_pose(a,b):
    delta=a["yaw"]-b["yaw"]
    return dist(xyz(a),xyz(b))<=.35 and abs(atan2(sin(delta),cos(delta)))<=.15


class ViewScheduler:
    def __init__(self,backend,task,directory,trace,method,costs):
        self.backend=backend
        self.task=task
        self.root=Path(directory)
        self.trace=trace
        self.method=method
        self.costs=costs
        self.program=compile_text(task)
        self.instructions=upstream().SYSTEM+INSTRUCTIONS
        self.offers=[]
        self.ready=[]
        self.intervals=[]
        self.unrestored=False

    def release(self):
        pending,self.ready=self.ready,[]
        return pending

    def camera_goals(self,obs,geometry,target,field):
        return _candidates(self.backend,obs,geometry,target,self.trace)

    def remember_source(self,offer,obs,geometry,details):
        pass

    def remember_capture(self,offer,tag,geometry,digest):
        pass

    def accepted_uses(self,answer,offers,context):
        if (answer.get("action") or {}).get("type") not in ("observe","record"):
            return []
        return self.requested_views(answer,offers)

    def released_photo(self,photo,offer,answer,context,deadline):
        return photo

    def nominate(self, decision, obs, geometry, image_id, step):
        plans=decision.get("next_views",[])
        if decision.get("action",{}).get("type")=="submit" or not isinstance(plans,list):
            return []
        result=[]
        for i,row in enumerate(plans[:2]):
            if not isinstance(row,dict) or set(row)!={"field","subject","bbox"}:
                continue
            field=row["field"]
            if not isinstance(field,str) or field not in {"identity",*INSPECTION_FIELDS} or not isinstance(row["subject"],str) or not 1<=len(row["subject"])<=240:
                continue
            independent=False
            if self.program:
                program,_=self.program
                if field!="identity" and field not in program.fields:
                    continue
                independent=program.filter=={"op":"keep"} and (field in program.report or field=="identity")
            gate=permission_for(self.task, Material(field, OUTCOMES if independent else ("positive",),"",None))
            if not gate.early:
                self.trace.event("native_nomination_refused",step,field=field,reason=gate.source)
                continue
            try:
                box=_bbox(row["bbox"])
            except (ValueError,TypeError):
                continue
            tag=f"v{step}-{i+1}"
            for goal,details in islice(self.camera_goals(obs,geometry,SimpleNamespace(id=tag,bbox=box),field),8):
                try:
                    self.backend.check_goal(goal)
                except (ValueError,RuntimeError):
                    continue
                if any(close_pose(o.goal,goal) and o.field==field and o.bbox==tuple(box) for o in result):
                    break
                offer=Offer(tag,field,row["subject"],tuple(box),image_id,sha256(obs["data"]).hexdigest(),goal,independent,gate.source)
                self.remember_source(offer,obs,geometry,details)
                self.trace.event("native_offer",step,**offer.visible(),independent=independent,
                                 permission=gate.source,reference_sha256=offer.reference_hash,view_geometry=details)
                result.append(offer)
                break
        return result

    def select(self, offers, anchor, window, remaining, step):
        # One excursion per frozen decision in this first live interface. Planner enumerates the same choices.
        options=[]
        for o in offers:
            if close_pose(o.goal,anchor):
                continue
            cost=self.costs.move_seconds(anchor,o.goal)+self.costs.move_seconds(o.goal,anchor)+self.costs.observation_seconds
            if cost+window>=remaining:
                continue
            try:
                self.backend.check_goal(o.goal)
                self.backend.check_goal(anchor)
            except (ValueError,RuntimeError):
                continue
            p=1. if o.independent else .5
            gain=p*cost-max(0.,cost-window)
            options.append((gain,o,cost,p))
        self.trace.event("native_policy_checked",step,policy=self.method,request_window_s=window,
                         options=[dict(id=o.id,expected_gain_s=g,roundtrip_s=c,need_probability=p) for g,o,c,p in options])
        if self.method=="native-off":
            return None
        if self.method=="native-async":
            options=[item for item in options if item[1].independent]
        if self.method=="native-prefetch":
            options.sort(key=lambda a:(-a[3],a[2],a[1].id))
        else:
            options.sort(key=lambda a:(-a[0],a[2],a[1].id))
        return next((o for g,o,_,_ in options if g>0),None)

    def select_many(self,offers,anchor,window,remaining,step):
        choice=self.select(offers,anchor,window,remaining,step)
        return [choice] if choice else []

    def requested_views(self,answer,offers):
        return [o for o in offers if o.id==answer.get("use_view")]

    def excursion(self, offer, anchor, step, early, deadline):
        tag="onboard_side_%04d_%s_%s"%(step,offer.id,"early" if early else "required")
        observed=None
        departed=False
        try:
            self.backend.check_goal(anchor)
            self.backend.check_goal(offer.goal)
            with self.trace.measure("action",f"{step}-side",candidate=offer.id,early=early):
                departed=True
                self.backend.move(offer.goal)
            with self.trace.measure("observation",f"{step}-side",candidate=offer.id):
                frame,geometry=self.backend.observe()
            actual={**frame["position"],"yaw":frame["yaw"]}
            if not close_pose(actual,offer.goal) or monotonic()>=deadline:
                return None
            meta={k:v for k,v in frame.items() if k!="data"}
            h=sha256(frame["data"]).hexdigest()
            meta.update(image_id=tag,sha256=h,source_reference=offer.reference_image,
                        intended_subject=offer.subject,candidate=offer.id,field=offer.field)
            (self.root/(tag+".jpg")).write_bytes(frame["data"])
            write_json(self.root/(tag+".json"),meta)
            self.remember_capture(offer,tag,geometry,h)
            observed=dict(image_id=tag,position=frame["position"],yaw=frame["yaw"],data=frame["data"],
                          capture_ros=frame["ros_stamp"],source_reference=offer.reference_image,
                          intended_subject=offer.subject,requested_field=offer.field)
            self.trace.event("native_raw_photo",step,candidate=offer.id,image_id=tag,image_sha256=h,
                             reference_sha256=offer.reference_hash,reading=None)
        except (RuntimeError,ValueError) as error:
            self.trace.event("native_excursion_failed",step,candidate=offer.id,reason=str(error)[:200])
        finally:
            if departed:
                try:
                    with self.trace.measure("action",f"{step}-return",candidate=offer.id):
                        reached=self.backend.move(anchor)
                    self.unrestored=not close_pose({**reached["position"],"yaw":reached["yaw"]},anchor)
                except (RuntimeError,ValueError):
                    self.unrestored=True
                self.trace.event("native_anchor_checked",step,candidate=offer.id,restored=not self.unrestored)
        return observed

    def decide(self, model, context, images, obs, geometry, step):
        offers,self.offers=self.offers,[]
        self.unrestored=False
        supplied={context["current_image_id"],*(r["image_id"] for r in context["recent_images"])}
        offers=[o for o in offers if o.reference_image in supplied]
        if (context.get("scene_conditions") or {}).get("dynamic"):
            offers=[]
        if self.program and self.program[0].scope["mode"]=="target":
            named=self.program[0].scope["identity"]
            supported={r.get("evidence_image_id") for r in context["records"] if r.get("object_id")==named}
            offers=[o for o in offers if o.reference_image in supported and named in o.subject]
        if offers:
            context={**context,"offered_views":[o.visible() for o in offers]}
        anchor={**obs["position"],"yaw":obs["yaw"]}
        deadline=model.deadline_monotonic
        window=median(self.intervals[-5:]) if self.intervals else self.costs.frozen_query_seconds
        choices=self.select_many(offers,anchor,window,deadline-monotonic(),step) if offers else []
        def infer():
            t=monotonic()
            with self.trace.measure("model_request",step,image_id=context["current_image_id"],purpose="native_control"):
                answer=model.request(self.instructions,context,images)
            return answer,monotonic()-t
        photos={}
        if choices:
            before=model.calls
            with ThreadPoolExecutor(max_workers=1) as executor:
                request=executor.submit(infer)
                while not request.done() and before==model.calls and monotonic()<deadline:
                    sleep(.01)
                for choice in choices:
                    if request.done() or model.calls<=before or monotonic()>=deadline or self.unrestored:
                        break
                    self.trace.event("native_early_start",step,candidate=choice.id,field=choice.field,
                                     grant=choice.grant,reference_sha256=choice.reference_hash)
                    photo=self.excursion(choice,anchor,step,True,deadline)
                    if photo:
                        photos[choice.id]=photo
                answer,duration=request.result()
        else:
            answer,duration=infer()
        self.intervals.append(duration)
        if self.unrestored or monotonic()>=deadline or not isinstance(answer,dict):
            return answer
        uses=self.accepted_uses(answer,offers,context)
        for use in uses:
            if self.unrestored or monotonic()>=deadline:
                break
            photo=photos.get(use.id)
            if photo is None and not close_pose(anchor,use.goal):
                photo=self.excursion(use,anchor,step,False,deadline)
            if photo and not self.unrestored:
                self.ready.append(self.released_photo(photo,use,answer,context,deadline))
        if self.unrestored:
            self.ready=[]
        self.trace.event("native_material_release",step,selected=choices[0].id if choices else None,
                         requested=uses[0].id if uses else None,selected_ids=[o.id for o in choices],
                         requested_ids=[o.id for o in uses],released=[f["image_id"] for f in self.ready])
        if not self.unrestored and monotonic()<deadline:
            self.offers=self.nominate(answer,obs,geometry,context["current_image_id"],step)
        return answer


def run(backend,task,directory,event,timeout,*,model,trace,method,costs):
    cls=ViewScheduler
    if method in EVIDENCE_MODES:
        from research.flexible_scheduling.native_controller import EvidenceScheduler
        cls=EvidenceScheduler
    elif method in COORDINATED_MODES:
        from research.flexible_scheduling.coordinated_controller import CoordinatedScheduler
        cls=CoordinatedScheduler
    scheduler=cls(backend,task,directory,trace,method,costs)
    return run_agent(backend,task,directory,event,timeout,model=model,trace=trace,view_scheduler=scheduler)
