"""Carry legally activated photos into a normal navigation turn, with an optional source-bound local reading."""
from collections import deque
from hashlib import sha256
from math import isfinite
import re
from subprocess import SubprocessError
from time import monotonic

from experiments.flexible_execution.adapters.native_views import ViewScheduler
from experiments.flexible_execution.adapters.original_controller import upstream
from experiments.task_attack.runtime import write_json
from .front_facing import sign_goals
from .literal_ocr import decode_rgb,engine_verified
from .material_binding import bind_identity
from .native_controller import EvidenceScheduler


INSTRUCTIONS="""
An optional camera tool supports a close view of an already visible in-scope subject. Nominate up to
two next_views in the current RGB, each with field, subject, bbox (normalized left, top, right, bottom).
Leave this list empty if the current evidence suffices. A nomination is not evidence about its contents.
When offered_views are present, request relevant ids with use_views. You cannot assume a view exists.
If your mission has a conditional filter, provide view_judgments for each requested conditional view:
[{"id":"offered id","source":"the offer's source image id","field":"tested filter field","value":<observed value>}].
Inspect the actual supplied source image at the offer's box. Omit unsupported judgments. Report the
observed value, including negative values when applicable. A conditional photo can be delivered only
when the observed value activates the user's stated branch. Never claim evidence from an unseen photo.
You can request a view with observe or record. If material_handoff permits it, a move or approach
can ALSO request a view in the same answer. Do this only if the motion serves the authorized task
regardless of what the extra photo shows; include motion_independent_of_view=true. If you need to see
that photo before choosing where to move, observe instead. A submission ends the task.
A delivered history image may contain an optional local measurement of an ASSET ID. It is tied to the
source and capture image hashes and the specified object region. Check the actual surface and image
before using a measurement, especially when a header display and the body identity plate differ.
Record what you actually observed for EVERY inspected object in top-level records, with the precise
supplied evidence_image_id, even if it fails the reporting filter. Submit the task's requested subset.
"""


def activation_matches(offer,decision,condition,supplied):
    if offer.reference_image not in supplied:
        return False
    if offer.independent:
        return True
    if not isinstance(condition,dict) or condition.get("op") not in ("eq","gt"):
        return False
    rows=decision.get("view_judgments")
    if not isinstance(rows,list):
        return False
    observed=[r for r in rows if isinstance(r,dict) and r.get("id")==offer.id]
    if len(observed)!=1:
        return False
    row=observed[0]
    if row.get("source")!=offer.reference_image or row.get("field")!=condition.get("field"):
        return False
    actual=row.get("value")
    expected=(condition.get("right") or {}).get("literal")
    if condition["op"]=="eq":
        return type(actual) is type(expected) and actual is not None and actual==expected
    return (type(actual) in (int,float) and type(expected) in (int,float)
            and isfinite(actual) and isfinite(expected) and actual>expected)


class CoordinatedScheduler(EvidenceScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.instructions=upstream().SYSTEM+INSTRUCTIONS
        self.candidate_planes={}
        self.sources=deque(maxlen=8)
        self.captures=deque(maxlen=8)
        self.read_local=self.method!="coordinated-no-ocr" and engine_verified()
        self.condition=self.program[0].filter if self.program else None
        if self.condition is None and isinstance(self.task,str):
            public_count=re.fullmatch(
                r"检查眼前这一排货位，报告实际箱数严格多于([0-9]+)箱的货位编号和箱数，保留图像证据。\s*"
                r"报告中以 object_id 表示对象编号，facts 使用字段：box_count（实际箱数）。",
                self.task.splitlines()[0])
            if public_count:
                self.condition={"op":"gt","field":"box_count","right":{"literal":int(public_count[1])}}

    def solver_method(self):
        return {"off":"serial","async":"async","prefetch":"strong-prefetch","planner":"generic",
                "deferred":"factored","observe-only":"factored","no-ocr":"factored"}[self.method[len("coordinated-"):]]

    def camera_goals(self,obs,geometry,target,field):
        if field=="identity":
            proposals=sign_goals(obs["data"],geometry,target.bbox)
            if proposals:
                self.candidate_planes[target.id]=proposals[0]
            for p in proposals:
                yield p["goal"],dict(method="observed_rectangular_surface",**{k:v for k,v in p.items() if k!="goal"})
        yield from ViewScheduler.camera_goals(self,obs,geometry,target,field)

    def nominate(self,decision,obs,geometry,image_id,step):
        self.candidate_planes.clear()
        return super().nominate(decision,obs,geometry,image_id,step)

    def remember_source(self,offer,obs,geometry,details):
        plane=self.candidate_planes.get(offer.id)
        if plane:
            self.sources.append((offer.id,offer.reference_hash,geometry["camera"],plane))
            write_json(self.root/("source_measurement_"+offer.id+".json"),
                       dict(source_image=offer.reference_image,source_hash=offer.reference_hash,
                            source_bbox=list(offer.bbox),plane=plane))

    def remember_capture(self,offer,tag,geometry,digest):
        self.captures.append((tag,digest,geometry["camera"]))
        write_json(self.root/(tag+"_geometry.json"),dict(camera=geometry["camera"],image_sha256=digest))

    def decide(self,model,context,images,obs,geometry,step):
        names=[r["image_id"] for r in context["recent_images"]]+[context["current_image_id"]]
        hashes={name:sha256(item["data"]).hexdigest() for name,item in zip(names,images)} if len(names)==len(images) else {}
        self.offers=[o for o in self.offers if hashes.get(o.reference_image)==o.reference_hash]
        mode=dict(motion_with_view=self.method!="coordinated-observe-only",activation=self.condition)
        return super().decide(model,{**context,"material_handoff":mode},images,obs,geometry,step)

    def accepted_uses(self,answer,offers,context):
        kind=(answer.get("action") or {}).get("type")
        moving=kind in ("move","approach")
        if kind not in ("record","observe") and not (moving and self.method!="coordinated-observe-only"
                                                     and answer.get("motion_independent_of_view") is True):
            return []
        supplied={context["current_image_id"],*(v["image_id"] for v in context["recent_images"])}
        return [o for o in self.requested_views(answer,offers) if activation_matches(o,answer,self.condition,supplied)]

    def released_photo(self,photo,offer,answer,context,deadline):
        self.trace.event("coordinated_delivery",photo["image_id"],candidate=offer.id,mode=self.method,
                         controller_action=answer["action"]["type"],source_frame=offer.reference_image,
                         source_digest=offer.reference_hash,activated_by_actual_reply=True)
        if not self.read_local or offer.field!="identity" or monotonic()+.1>=deadline:
            return photo
        source=next((s for s in reversed(self.sources) if s[:2]==(offer.id,offer.reference_hash)),None)
        photo_hash=sha256(photo["data"]).hexdigest()
        capture=next((s for s in reversed(self.captures) if s[:2]==(photo["image_id"],photo_hash)),None)
        if source is None or capture is None:
            return photo
        start=monotonic()
        try:
            readings=decode_rgb(photo["data"],timeout_s=max(.05,min(2.,deadline-start)))
            reading=bind_identity(source[2],capture[2],source[3],offer.bbox,readings)
        except (OSError,SubprocessError,ValueError) as ex:
            self.trace.event("local_material_unavailable",photo["image_id"],error=type(ex).__name__)
            return photo
        self.trace.event("local_material_measured",photo["image_id"],candidate=offer.id,
                         seconds=monotonic()-start,assigned=reading is not None,image_sha256=photo_hash)
        if reading is None or monotonic()>=deadline:
            return photo
        hint=dict(field="identity",value=reading["value"],raw_image_sha256=photo_hash,
                  reference_image=offer.reference_image,reference_sha256=offer.reference_hash,
                  reference_bbox=list(offer.bbox),center=reading["center"],confidence=reading["confidence"],
                  interpretation="printed ASSET ID with calibrated source association; verify against RGB")
        return {**photo,"local_measurement":hint}
