"""Sealed RGB-only proposals and independent matched endpoint scoring across exposed worlds."""
from collections import Counter,defaultdict
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from experiments.flexible_execution.revision_audit import read,save,write_table,digest
from experiments.task_attack.cpu_world import SceneRenderer
from research.flexible_scheduling.front_facing import sign_goals
from research.flexible_scheduling.label_geometry import rectangular_surfaces
from scripts.continuing_ocr import decode,projected_label_boxes,inside
from scripts.continuing_sensor_visibility import bodies,camera


CONFIG=Path("config/archived_views_crossworld01.json")
OUT=Path("outputs/continuing/archived-views-crossworld01")
ROLES=("previous","radial_same_distance","frontal_same_distance")


def assert_snapshot(path,data):
    if path.exists() and read(path)!=data:
        raise ValueError("Frozen offline input or predictions changed: "+str(path))
    if not path.exists():
        save(path,data)


def plan():
    cfg=read(CONFIG)
    source=Path(cfg["inputs"])
    inputs=read(source)["rows"]
    if len(inputs)!=cfg["prespecified_inputs"]:
        raise ValueError("Historical input set differs from registration")
    plans=[]
    for i,item in enumerate(inputs):
        ep=Path(item["robot"]).parents[1]
        robot=Path(item["robot"])
        ref=robot/(item["ref"]+".jpg")
        g=read(robot/(item["ref"]+"_geometry.json"))
        cam=g["camera"]
        box=item["ref_bbox"]
        img=ref.read_bytes()
        if digest(robot/(item["photo"]+".jpg"))!=item["digest"]:
            raise ValueError("Recorded comparison source changed")
        measured=sign_goals(img,g,box)
        width,height=cam["width"],cam["height"]
        K=cam["intrinsics"]
        rgb=np.asarray(Image.open(BytesIO(img)).convert("RGB"))
        quads=sum(inside((v["center"][0]/width,v["center"][1]/height),box) for v in rectangular_surfaces(rgb,K))
        inv=np.linalg.inv(cam["world_T_camera"])
        rays=0
        for c in g["measured_surface_grid"]["cells"]:
            x,y,z,_=inv @ np.r_[c["world_xyz"],1.]
            if z>0:
                rays+=inside(((K["fx"]*x/z+K["cx"])/width,(K["fy"]*y/z+K["cy"])/height),box)
        rec=dict(i=i,case=item["case"],ep=str(ep),ref=item["ref"],ref_sha256=sha256(img).hexdigest(),
                 historical=item["photo"],historical_sha256=item["digest"],bbox=box,quads=int(quads),rays=int(rays),
                 design_layout=item["case"]=="dev-cracks-prim",goals={})
        if measured:
            proposal=measured[0]
            front=proposal["goal"]
            distance=proposal["standoff_m"]
            target=np.array([front["x"]+distance*math.cos(front["yaw"]),front["y"]+distance*math.sin(front["yaw"])])
            origin=np.array([g["position"][k] for k in ("x","y")])
            unit=target-origin
            if np.linalg.norm(unit)<=1e-6:
                raise ValueError("Original sight line undefined")
            unit=unit/np.linalg.norm(unit)
            radial=target-distance*unit
            old=read(robot/(item["photo"]+".json"))
            rec["goals"]=dict(previous={**old["position"],"yaw":old["yaw"]},
                              radial_same_distance=dict(x=float(radial[0]),y=float(radial[1]),z=front["z"],yaw=math.atan2(unit[1],unit[0])),
                              frontal_same_distance=front)
        plans.append(rec)
    doc=dict(config_sha256=digest(CONFIG),inputs_sha256=digest(source),plans=plans)
    assert_snapshot(OUT/"proposals.json",doc)
    return plans


@contextmanager
def copied_renderer(ep):
    with TemporaryDirectory(dir=OUT) as temp:
        scene=Path(temp)/"inspection.world"
        shutil.copy2(ep/"inspection.world",scene)
        shutil.copytree(ep/"assets",Path(temp)/"assets")
        renderer=SceneRenderer(scene,robot=True)
        try:
            yield scene,renderer
        finally:
            renderer.close()


def viewpoint(renderer,goal,reference):
    quat=renderer.p.getQuaternionFromEuler([0,0,goal["yaw"]])
    return dict(position={k:goal[k] for k in ("x","y","z")},width=reference["width"],height=reference["height"],
                orientation=dict(zip(("x","y","z","w"),quat)))


def capture(plans):
    grouped=defaultdict(list)
    for p in plans:
        if p["goals"]:
            grouped[p["ep"]].append(p)
    predictions=[]
    for loc,items in grouped.items():
        ep=Path(loc)
        with copied_renderer(ep) as (_,renderer):
            for item in items:
                g=read(ep/"private/robot"/(item["ref"]+"_geometry.json"))["camera"]
                for role in ROLES:
                    goal=item["goals"][role]
                    pose=viewpoint(renderer,goal,g)
                    renderer.set_pose("vla_uav",list(pose["position"].values()),list(pose["orientation"].values()))
                    collision=bool(renderer.contacts())
                    h=None
                    answers=[]
                    if not collision:
                        rgb,_=camera(renderer,pose)
                        path=OUT/("%03d-%s.jpg"%(item["i"],role))
                        Image.fromarray(rgb).save(path,quality=95)
                        h=digest(path)
                        answers=decode(str(path),())
                    predictions.append(dict(i=item["i"],role=role,contact=collision,photo_sha256=h,answers=answers))
    assert_snapshot(OUT/"predictions.json",predictions)
    return predictions


def grade(plans,predictions):
    byid={p["i"]:p for p in plans}
    predicted={(r["i"],r["role"]):r for r in predictions}
    scored=[]
    for ep in sorted({Path(p["ep"]) for p in plans if p["goals"]}):
        items=[p for p in plans if Path(p["ep"])==ep and p["goals"]]
        result=read(ep/"result.json")
        domain=result["scene_name"]
        truth=read(ep/"private/truth.json")["devices"]
        indices={v["identity"]:i for i,v in enumerate(truth,1)}
        with copied_renderer(ep) as (scene,renderer):
            names=bodies(scene,renderer)
            for item in items:
                g=read(ep/"private/robot"/(item["ref"]+"_geometry.json"))["camera"]
                boxes=projected_label_boxes(ep,g,domain)
                assigned=[key for key,(a,b,c,d) in boxes.items() if inside(((a+c)/2,(b+d)/2),item["bbox"])]
                target=assigned[0] if len(assigned)==1 else None
                body=names.get(("%s_station_%02d"%(domain,indices[target]),"reference_identity")) if target else None
                for role in ROLES:
                    rec=predicted[item["i"],role]
                    answers=rec["answers"]
                    grounded=[]
                    pixels=None
                    if not rec["contact"] and body is not None:
                        pose=viewpoint(renderer,item["goals"][role],g)
                        _,mask=camera(renderer,pose)
                        pixels=int((mask==body).sum())
                        h,w=mask.shape
                        for a in answers:
                            x,y=a["center"]
                            if 0<=x<1 and 0<=y<1 and mask[min(h-1,int(y*h)),min(w-1,int(x*w))]==body:
                                grounded.append(a["value"])
                    status=("unassigned" if target is None else "collision" if rec["contact"] else
                            "abstain" if not grounded else "correct" if set(grounded)=={target} else "wrong_or_conflicting")
                    scored.append(dict(i=item["i"],case=item["case"],domain=domain,design_layout=item["design_layout"],
                            role=role,target=target,outcome=status,body_visible_pixels=pixels,answers="|".join(a["value"] for a in answers),
                            grounded="|".join(grounded),photo_sha256=rec["photo_sha256"]))
    if scored:
        write_table(OUT/"scored.csv",scored)
    strata={}
    for partition in ("all","outside_design_layout"):
        candidates=[p for p in plans if partition=="all" or not p["design_layout"]]
        selected={p["i"] for p in candidates}
        members=[r for r in scored if r["i"] in selected]
        strata[partition]=dict(questions=len(candidates),scenes=len({p["case"] for p in candidates}),
          unique_rgb_regions=len({(p["ref_sha256"],tuple(p["bbox"])) for p in candidates}),
          can_propose=sum(bool(p["goals"]) for p in candidates),has_rect=int(sum(bool(p["quads"]) for p in candidates)),
          has_depth=int(sum(bool(p["rays"]) for p in candidates)),
          roles={role:dict(Counter(r["outcome"] for r in members if r["role"]==role)) for role in ROLES})
    summary=dict(new_model_requests=0,config_sha256=digest(CONFIG),proposals_sha256=digest(OUT/"proposals.json"),
                 predictions_sha256=digest(OUT/"predictions.json"),strata=strata)
    save(OUT/"summary.json",summary)
    print(strata)


if __name__=="__main__":
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=plan()
    grade(frozen,capture(frozen))
