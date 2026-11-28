"""Bind a second viewpoint to a genuine pre-motion visual-grounding refusal."""
from math import atan2, cos, dist, isfinite, sin

import numpy as np

from experiments.flexible_execution.gpt_vision import _bbox


REFUSALS={"unproven_foreground","split_depth","insufficient_core_points","insufficient_center_coverage"}
INSTRUCTION="""Your ordinary first action remains your choice. If it is an approach and a different
side view would help inspection even when the physical target cannot be located from these pixels,
you may include one optional top-level on_unresolved object in this SAME answer. Format:
{"image_id":"current image id","when":"unresolved_before_motion","independent_observation":true,
"goal":{"x":number,"y":number,"z":number,"yaw":number}}. Use a genuine lateral, backward or
in-place camera viewpoint useful for the authorized inspection. Do not advance toward the
unverified visual target, use a map refusal as permission to move, or assume a new field value.
The option is considered only if the original resolver cannot associate the target before any
movement, and only at this same pose. Original mapping and motion checks still apply.
Omit this field if there is no independently useful permitted view."""


def option(reply, *, image, current, original, camera, code):
    raw=reply.get("on_unresolved")
    action=reply.get("action") or {}
    if (code not in REFUSALS or action.get("type")!="approach" or not isinstance(raw,dict)
            or set(raw)!={"image_id","when","independent_observation","goal"} or raw["image_id"]!=image
            or raw["when"]!="unresolved_before_motion" or raw["independent_observation"] is not True):
        return None,"no-bound-contingency"
    axes=("x","y","z")
    if dist([original[k] for k in axes],[current[k] for k in axes])>.12 or abs(
            atan2(sin(original["yaw"]-current["yaw"]),cos(original["yaw"]-current["yaw"])))>.10:
        return None,"source-pose-changed"
    g=raw["goal"]
    if (not isinstance(g,dict) or set(g)!={"x","y","z","yaw"} or
            not all(type(v) in (float,int) and isfinite(v) for v in g.values())):
        return None,"invalid-explicit-goal"
    try:
        b=_bbox(action.get("bbox"))
        k=camera["intrinsics"]
        cx=(b[0]+b[2])*camera["width"]/2
        cy=(b[1]+b[3])*camera["height"]/2
        ray=np.array([(cx-k["cx"])/k["fx"],(cy-k["cy"])/k["fy"],1.])
        world=np.asarray(camera["world_T_camera"],dtype=float)[:3,:3]@ray
        world=world/np.linalg.norm(world)
        advance=sum((g[axis]-original[axis])*component for axis,component in zip(axes,world))
        if not isfinite(advance) or advance>.15:
            return None,"would-advance-on-unresolved-target"
    except (KeyError,ValueError,TypeError,ZeroDivisionError):
        return None,"missing-source-geometry"
    return g,"independent-view"


def perform(backend,obs,geo,reply,image,enable,log):
    action=reply.get("action") or {}
    kind=action.get("type")
    at={**obs["position"],"yaw":obs["yaw"]}
    if kind=="move":
        reached=backend.move(action["goal"])
        return dict(kind=kind,reached={**reached["position"],"yaw":reached["yaw"]},fallback=False)
    if kind!="approach":
        return dict(kind=kind,reached=None,fallback=False)
    # Resolve and move are separate so a map or in-motion failure cannot activate a second movement.
    try:
        target=backend.adapter.resolve_visual_target(obs,action["bbox"],action.get("standoff_m",2.))
    except ValueError as ex:
        code=getattr(ex,"code",None)
        log("primary_refused_before_motion",code=code)
        if not enable:
            return dict(kind=kind,primary_refusal=code,reached=None,fallback=False)
        actual=backend.adapter.snapshot()
        g,why=option(reply,image=image,current={**actual["position"],"yaw":actual["yaw"]},
                     original=at,camera=geo["camera"],code=code)
        log("contingency_check",decision=why)
        if g is None:
            return dict(kind=kind,primary_refusal=code,reached=None,fallback=False,reason=why)
        backend.check_goal(g)
        log("contingency_admitted",goal=g)
        s=backend.move(g)
        return dict(kind=kind,primary_refusal=code,reached={**s["position"],"yaw":s["yaw"]},fallback=True,goal=g)
    reached=backend.move(target["goal"])
    return dict(kind=kind,reached={**reached["position"],"yaw":reached["yaw"]},fallback=False)
