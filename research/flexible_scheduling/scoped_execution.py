"""Execute a user-enumerated static inspection against a genuine grouped response and actual camera poses."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from subprocess import SubprocessError
from time import monotonic, sleep

import cv2
import numpy as np
from PIL import Image

from experiments.flexible_execution.adapters.native_views import Offer, close_pose
from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.revision_audit import read,save
from .closed_scope import ClosedInspection, QUESTION
from .front_facing import sign_goals
from .grouped_regions import Region, dispatch
from .literal_ocr import decode_rgb, engine_verified
from .material_binding import bind_identity
from .native_controller import EvidenceScheduler


def match_source(old,new,bbox):
    src=np.asarray(Image.open(BytesIO(old)).convert("L"))
    rgb=np.asarray(Image.open(BytesIO(new)).convert("L"))
    if src.shape!=rgb.shape:
        return None
    h,w=src.shape
    x0,y0,x1,y1=[round(x*y) for x,y in zip(bbox,(w,h,w,h))]
    roi=src[y0:y1,x0:x1]
    if min(roi.shape)<6 or roi.std()<4:
        return None
    left,top=max(0,x0-16),max(0,y0-16)
    region=rgb[top:min(h,y1+16),left:min(w,x1+16)]
    _,peak,_,(x,y)=cv2.minMaxLoc(cv2.matchTemplate(region,roi,cv2.TM_CCOEFF_NORMED))
    return dict(ncc=float(peak),delta_pixels=[left+x-x0,top+y-y0])


def prepare(mission,template,obs,geo):
    if sha256(template).hexdigest()!=mission["reference_sha256"] or mission.get("static") is not True:
        raise ValueError("A matching public static mission reference is required")
    photo=Frame("scope-overview","inspection-roster",1,obs["ros_stamp"],obs["data"],"fresh-onboard-camera")
    w,h=geo["camera"]["width"],geo["camera"]["height"]
    regions=[];measurements=[]
    for row in mission["regions"]:
        source=Region(row["id"],row["bbox"],QUESTION)
        reg=match_source(template,photo.payload,source.bbox)
        if not reg or reg["ncc"]<.90 or max(abs(v) for v in reg["delta_pixels"])>8:
            return None,[],[dict(id=source.id,reference_unregistered=reg)]
        dx,dy=reg["delta_pixels"]
        l,t,r,b=source.bbox
        try:
            aligned=Region(source.id,(l+dx/w,t+dy/h,r+dx/w,b+dy/h),QUESTION)
        except ValueError:
            return None,[],[dict(id=source.id,box_out_of_frame=True)]
        regions.append(aligned)
        proposals=sign_goals(photo.payload,geo,aligned.bbox)
        measurements.append(dict(region=source.id,registration=reg,
                     early_authorized=row.get("may_photograph_identity_before_result") is True,proposals=proposals))
    roster=ClosedInspection(mission["id"],photo,[(r.id,r.bbox) for r in regions])
    return roster,regions,measurements


class CameraRunner(EvidenceScheduler):
    def remember_capture(self,offer,tag,geometry,digest):
        save(self.root/(tag+"_geometry.json"),dict(camera=geometry["camera"],image_sha256=digest))


def execute(backend,model,mission,template,root,trace,costs,method,deadline):
    if method not in ("evidence-off","evidence-async","evidence-prefetch","evidence-planner","evidence-deferred"):
        raise ValueError("Unknown scoped camera policy")
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    obs,geo=backend.observe()
    anchor={**obs["position"],"yaw":obs["yaw"]}
    roster,regions,measurements=prepare(mission,template,obs,geo)
    save(root/"source_alignment.json",measurements)
    if roster is None:
        return dict(report=None,gaps=[["mission","registered-public-overview"]],sends=0)
    (root/"scope-overview.jpg").write_bytes(roster.reference.payload)
    save(root/"scope-overview.json",{**{k:v for k,v in obs.items() if k!="data"},"sha256":roster.reference.digest})
    save(root/"scope-overview_geometry.json",geo)
    save(root/"live_roster.json",dict(scope=roster.scope_hash,reference=roster.reference.digest,regions=[r.public() for r in regions]))
    runner=CameraRunner(backend,"用户圈出的设备，逐一检查正面裂纹；有裂纹的设备需实拍本体编号。",root,trace,method,costs)
    model.deadline_monotonic=backend.execution_deadline_monotonic=deadline
    offers={}; planes={}; permissions=set()
    for i,(region,m) in enumerate(zip(regions,measurements),1):
        for p in m["proposals"]:
            try:
                backend.check_goal(anchor)
                backend.check_goal(p["goal"])
            except (ValueError,RuntimeError):
                continue
            if close_pose(anchor,p["goal"]):
                continue
            # File ids come from the local enumeration, never from external text.
            offer=Offer("scope-%02d"%i,"identity",region.id,region.bbox,roster.reference.id,roster.reference.digest,
                        p["goal"],False,"explicit-roster-identity" if m["early_authorized"] else "after-positive")
            offers[offer.id]=offer
            planes[offer.id]=p
            if m["early_authorized"]:
                permissions.add(offer.id)
            break
    allowed=[o for k,o in offers.items() if k in permissions]
    remaining=deadline-monotonic()
    if remaining<=0:
        return dict(report=None,gaps=[["mission","deadline"]],sends=0)
    choices=runner.select_many(allowed,anchor,costs.frozen_query_seconds,remaining,1) if allowed else []
    before=model.calls
    photos={}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(dispatch,model,roster.reference.id,roster.reference.digest,roster.reference.payload,regions)
        while choices and not future.done() and model.calls==before and monotonic()<deadline:
            sleep(.01)
        for o in choices:
            if future.done() or model.calls<=before or runner.unrestored or monotonic()>=deadline:
                break
            trace.event("scoped_raw_capture_started",1,candidate=o.id,region=o.subject,permission=o.grant)
            photo=runner.excursion(o,anchor,1,True,deadline)
            if photo and not runner.unrestored:
                photos[o.id]=photo
        binding,reply=future.result()
    if not roster.record_group(binding,[roster.reference.digest],reply):
        trace.event("scoped_invalid_reply",1)
    for k,o in offers.items():
        if runner.unrestored or monotonic()>=deadline:
            break
        if not roster.needs_identity(o.subject):
            continue
        photo=photos.get(k)
        if photo is None:
            photo=runner.excursion(o,anchor,2,False,deadline)
        if not photo or runner.unrestored or monotonic()>=deadline or not engine_verified():
            continue
        f=Frame(photo["image_id"],o.subject,2,photo["capture_ros"],photo["data"],"fresh-onboard-camera")
        g=read(root/(f.id+"_geometry.json"))
        if g["image_sha256"]!=f.digest:
            raise ValueError("Capture geometry belongs to a different image")
        try:
            readings=decode_rgb(f.payload,timeout_s=max(.05,min(2.,deadline-monotonic())))
            measured=bind_identity(geo["camera"],g["camera"],planes[k],o.bbox,readings)
        except (SubprocessError,OSError,ValueError):
            measured=None
        if measured:
            hint=dict(field="identity",value=measured["value"],raw_image_sha256=f.digest,
                      reference_sha256=roster.reference.digest,reference_bbox=list(o.bbox),scope_hash=roster.scope_hash)
            roster.record_identity(o.subject,f,hint,photo["capture_ros"])
            trace.event("scoped_local_read",f.id,region=o.subject,measurement=measured,source_sha256=f.digest)
    if monotonic()>=deadline:
        return dict(report=None,gaps=[["mission","deadline"]],sends=model.calls-before)
    live,_=backend.observe()
    returned=not runner.unrestored and close_pose({**live["position"],"yaw":live["yaw"]},anchor)
    save(root/"measured_return.json",dict(initial=anchor,final={**live["position"],"yaw":live["yaw"]},
                                          ros_stamp=live["ros_stamp"],returned=returned))
    report,gaps=roster.assemble(at_anchor=returned)
    if monotonic()>=deadline:
        report=None;gaps=[("mission","deadline")]
    return dict(report=report,gaps=gaps,sends=model.calls-before,anchor_checked=returned,
                source_sha256=roster.reference.digest,source_scope=roster.scope_hash)
