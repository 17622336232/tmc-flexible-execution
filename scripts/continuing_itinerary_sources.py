"""Register public camera itineraries and independently grade photographed physical components."""
from hashlib import sha256
import json
from math import dist
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from experiments.flexible_execution.revision_audit import digest, read
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_crossworld_views import assert_snapshot, copied_renderer
from scripts.continuing_navigation_assay import geometry_hash
from scripts.continuing_sensor_visibility import bodies, camera


CFG = Path("config/inspection_travel_dev01.json")
ROOT = Path("outputs/continuing/inspection-travel-dev01")
METHOD_FILE = Path("research/flexible_scheduling/inspection_travel.py")


def labelled_frames(ep, field):
    robot = ep/"private/robot"
    path = robot/"model_calls.jsonl"
    if not path.is_file():
        return []
    out = {}
    for c in read_trace_rows(path, allow_truncated=True)[0]:
        ctx = c["input"]
        current = ctx.get("current_image_id", "")
        if re.fullmatch(r"onboard_[0-9]{4}", current) is None:
            continue
        supplied = [v["image_id"] for v in ctx.get("recent_images", [])] + [current]
        if len(supplied) != len(c["image_sha256"]):
            continue
        visible = dict(zip(supplied, c["image_sha256"]))
        decision = c.get("output") or {}
        rows = (decision.get("records") or []) + ((decision.get("report") or {}).get("records") or [])
        for row in rows:
            name = row.get("evidence_image_id","")
            target = row.get("object_id","")
            if (not re.fullmatch(r"onboard_[0-9]{4}",name) or name not in visible or
                    not re.fullmatch("EQ-[0-9]+" if field=="cracked" else "BAY-[0-9]+",target)
                    or field not in (row.get("facts") or {})):
                continue
            photo = robot/(name+".jpg")
            meta = read(robot/(name+".json"))
            if digest(photo) != visible[name] or meta["sha256"] != visible[name]:
                raise ValueError("Historical source was not the supplied camera")
            out[(name,target)] = dict(frame=name,target=target,reference=str(photo),sha256=visible[name],
                goal={**meta["position"],"yaw":meta["yaw"]})
    return list(out.values())


def register():
    cfg = read(CFG)
    cases = []
    for c in cfg["cases"]:
        source = ROOT.parent/c["cohort"]/c["case"]
        candidates = []
        for rc in sorted(source.glob("*.receipt.json")):
            method = read(rc)["method"]
            ep = source/method
            items = labelled_frames(ep,c["field"])
            for a in items:
                for b in items:
                    if a["frame"] >= b["frame"] or a["target"] == b["target"]:
                        continue
                    size = dist([a["goal"][k] for k in ("x","y","z")],[b["goal"][k] for k in ("x","y","z")])
                    if size < 1:
                        continue
                    key = [cfg["salt"],c["case"],str(ep),a["frame"],b["frame"],a["target"],b["target"]]
                    rank = sha256(json.dumps(key,separators=(",",":")).encode()).hexdigest()
                    candidates.append((rank,ep,a,b,size))
        if not candidates:
            raise ValueError("No evidenced two-stop route for "+c["id"])
        rank,ep,a,b,size = min(candidates,key=lambda v:v[0])
        s = read(ep/"clean_preflight.json")
        cases.append(dict(id=c["id"],stops=[a,b],direct_distance_m=size,source_episode=str(ep),
            candidates=len(candidates),selection_digest=rank,matching_world=geometry_hash(ep),
            source_log_sha256=digest(ep/"private/robot/model_calls.jsonl"),
            settings={k:s[k] for k in ("family","seed","layout_seed","equipment_count","source_profile")}))
    reg = dict(config_sha256=digest(CFG),policy_sha256=digest(METHOD_FILE),source_grading_sha256=digest(Path(__file__)),
               cases=cases)
    ROOT.mkdir(parents=True,exist_ok=True)
    assert_snapshot(ROOT/"registration.json",reg)
    return cfg,reg


def image_agreement(reference, fresh):
    # A fixed interior window constrains both spatial displacement and source-wide similarity.
    src = cv2.imread(str(reference),cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(str(fresh),cv2.IMREAD_GRAYSCALE)
    if src is None or dst is None or src.shape != dst.shape or min(src.shape)<=64:
        return 0., []
    _,best,_,(x,y) = cv2.minMaxLoc(cv2.matchTemplate(dst,src[16:-16,16:-16],cv2.TM_CCOEFF_NORMED))
    return float(best), [x-16,y-16]


def score_frame(ep,scene,renderer,registry,devices,field,stop,key,claimed):
    photo = ep/(key+".jpg")
    meta = read(ep/(key+".json"))
    if digest(photo)!=meta["sha256"] or digest(Path(stop["reference"]))!=stop["sha256"]:
        raise ValueError("Actual or reference photo was modified")
    agreement,delta = image_agreement(stop["reference"],photo)
    rgb,mask = camera(renderer,meta)
    with Image.open(photo) as observed:
        rgb_match = float((np.abs(rgb.astype(float)-np.asarray(observed.convert("RGB"),dtype=float)).max(2)<=18).mean())
    pairs = [(i,d) for i,d in enumerate(devices,1) if d["identity"]==stop["target"]]
    if len(pairs)!=1:
        return dict(verified=False,correct=False,reference_ncc=agreement,reason="identity-unassigned")
    i,device = pairs[0]
    station = ("equipment" if field=="cracked" else "warehouse")+"_station_%02d"%i
    label = (station,"reference_identity") if field=="cracked" else ("panel_%02d"%i,"front")
    # Both inspectors look at the same parts, regardless of the queried branch or model response.
    parts = ["door_surface"] if field=="cracked" else [
        "rack_post_-0.67","rack_post_0.67","rack_shelf_0.23","rack_shelf_0.82","rack_shelf_1.42",
        *["cargo_%d"%n for n in range(device["box_count"])]]
    label_mask = mask == registry[label]
    counts = {k:int((mask==registry[(station,k)]).sum()) for k in parts}
    label_px = int(label_mask.sum())
    enough = label_px>=144 and min(counts.values()) >= (144 if field=="cracked" else 36)
    source_ok = agreement>=.85 and rgb_match>=.90 and enough
    localized = False
    if claimed and label_px:
        h,w = mask.shape
        l,t,r,b = claimed["label_box"]
        yy,xx = np.where(label_mask)
        center = l<=xx.mean()/w<=r and t<=yy.mean()/h<=b
        fraction = float(((xx>=l*w)&(xx<=r*w)&(yy>=t*h)&(yy<=b*h)).mean())
        localized = bool(center and fraction>=.5 and (r-l)*(b-t)<=.20)
    actual = device[field]
    correct = bool(source_ok and claimed and claimed["id"]==stop["target"] and claimed["source"]==key
                   and claimed["sha256"]==meta["sha256"] and type(claimed["value"]) is type(actual)
                   and claimed["value"]==actual and localized)
    return dict(verified=source_ok,label_pixels=label_px,part_pixels=counts,label_box_verified=localized,
                reference_ncc=agreement,delta_px=delta,rgb_agreement=rgb_match,actual=actual,
                correct=correct,source_sha256=meta["sha256"])


def grade_frames(ep,reg_item,field,findings):
    devices = read(ep/"private/truth.json")["devices"]
    with copied_renderer(ep) as (scene,renderer):
        registry = bodies(scene,renderer)
        return {key:score_frame(ep,scene,renderer,registry,devices,field,stop,key,findings.get(key))
                for key,stop in zip(("a","b"),reg_item["stops"]) if (ep/(key+".jpg")).is_file()}

