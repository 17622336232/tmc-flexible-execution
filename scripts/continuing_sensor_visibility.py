"""Post-task referee for masks of the cited physical surfaces. Does not run in the robot process."""
import argparse
from collections import Counter,defaultdict
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from experiments.task_attack.cpu_world import SceneRenderer
from experiments.flexible_execution.revision_audit import read,save,write_table,digest
from experiments.flexible_execution.clean_coverage import audit_run


def bodies(scene,renderer):
    names={}
    for model in ET.parse(scene).findall("world/model"):
        key=model.get("name")
        parts=[item for link in model.findall("link") for item in [*link.findall("visual"),*link.findall("collision")]
               if item.find("geometry") is not None]
        built=renderer.groups[key]
        if len(parts)!=len(built):
            raise ValueError("Renderer part mapping differs: "+key)
        for item,(body,_,_) in zip(parts,built):
            if item.tag=="visual":
                names[(key,item.get("name"))]=body
    return names


def camera(renderer,meta):
    pos=np.array([meta["position"][k] for k in ("x","y","z")])
    quat=[meta["orientation"][k] for k in ("x","y","z","w")]
    renderer.set_pose("vla_uav",pos,quat)
    r=np.array(renderer.p.getMatrixFromQuaternion(quat)).reshape((3,3))
    origin=pos+r.dot([.17,0,.02])
    view=renderer.p.computeViewMatrix(origin.tolist(),(origin+r[:,0]).tolist(),r[:,2].tolist())
    w,h=meta["width"],meta["height"]
    vfov=math.degrees(2*math.atan(math.tan(1.3962634016/2)*h/w))
    proj=renderer.p.computeProjectionMatrixFOV(vfov,w/h,.05,30.)
    observed=renderer.p.getCameraImage(w,h,view,proj,shadow=0,lightDirection=[-1,-2,4],lightAmbientCoeff=.7,
                lightDiffuseCoeff=.3,renderer=renderer.p.ER_TINY_RENDERER,physicsClientId=renderer.client)
    return np.asarray(observed[2],dtype=np.uint8).reshape(h,w,4)[:,:,:3],np.array(observed[4]).reshape(h,w)


def nominated_positive_visible(ep,proof):
    """Post-seal witness for a cited positive on the body nominated in the reference RGB."""
    from scripts.continuing_crossworld_views import copied_renderer
    robot=ep/"private/robot"
    checked=audit_run(ep,recheck_legacy_fragments=True)["supported_by_frame"]
    devices=read(ep/"private/truth.json")["devices"]
    family=read(ep/"manifest.json")["family"]
    field={"cracks":"cracked","blocked_exit":"blocked"}.get(family)
    if field is None:
        return None
    stem="equipment" if family=="cracks" else "campus"
    def observed(renderer,frame):
        path=robot/(frame+".jpg")
        meta=read(path.with_suffix(".json"))
        if meta["sha256"]!=digest(path):
            raise ValueError("Claim frame bytes differ from receipt")
        generated,mask=camera(renderer,meta)
        with Image.open(path) as img:
            error=np.abs(np.asarray(img.convert("RGB"),dtype=float)-generated)
        return mask,float((error.max(axis=2)<=18).mean())
    with copied_renderer(ep) as (scene,renderer):
        names=bodies(scene,renderer)
        ref,a=observed(renderer,proof["reference_image"])
        captured,b=observed(renderer,proof["image"])
        box=proof["reference_bbox"]
        candidates=[]
        for i,dev in enumerate(devices,1):
            station=stem+"_station_%02d"%i
            panel=names.get((station,"reference_identity"))
            if panel is None:
                continue
            yy,xx=np.where(ref==panel)
            if len(xx)>=36 and box[0]<=np.mean(xx)/ref.shape[1]<=box[2] and box[1]<=np.mean(yy)/ref.shape[0]<=box[3]:
                candidates.append((station,dev,panel))
        if len(candidates)!=1 or min(a,b)<.90:
            return None
        station,dev,panel=candidates[0]
        if dev[field] is not True or proof["image"] not in checked.get(dev["identity"],{}).get(field,[]):
            return None
        relevant=["door_surface"] if field=="cracked" else dev["approach_obstructions"]
        surface=max((int((captured==names[(station,v)]).sum()) for v in relevant),default=0)
        identity=int((captured==panel).sum())
        if min(surface,identity)<144:
            return None
        return dict(object_id=dev["identity"],field=field,identity_pixels=identity,field_surface_pixels=surface,
                    reference_alignment=a,capture_alignment=b,image_sha256=proof["sha256"])


def grade_episode(ep,renderer,names,case):
    robot=ep/"private/robot"
    result=read(ep/"result.json")
    if not isinstance(result.get("report"),dict):
        return []
    verified=audit_run(ep,recheck_legacy_fragments=True)
    refs=verified["supported_by_frame"]
    ids={r["identity"]:(idx,r) for idx,r in enumerate(read(ep/"private/truth.json")["devices"],1)}
    photos={frame for fields in refs.values() for frames in fields.values() for frame in frames}
    views={}
    for frame in sorted(photos):
        meta=read(robot/(frame+".json"))
        raw=robot/(frame+".jpg")
        if digest(raw)!=meta["sha256"]:
            raise ValueError("Evidence image changed")
        rgb,mask=camera(renderer,meta)
        original=np.array(Image.open(raw).convert("RGB"))
        # Record exact differences as JPEG compression, OS timing and odometry may matter.
        difference=np.abs(original.astype(float)-rgb.astype(float))
        views[frame]=(Counter(dict(zip(*np.unique(mask,return_counts=True)))),float(difference.mean()),
                      float((difference.max(axis=2)<=18).mean()))
    records=[]
    for ident,fields in refs.items():
        idx,dev=ids[ident]
        for field,frames in fields.items():
            for frame in frames:
                count,mae,rate=views[frame]
                domain={"cracked":"equipment","box_count":"warehouse","blocked":"campus"}.get(field)
                if domain is None:
                    continue
                station=domain+"_station_%02d"%idx
                if field=="cracked":
                    parts=[("door",station,"door_surface"),("identity",station,"reference_identity")]
                elif field=="box_count":
                    parts=[("identity","panel_%02d"%idx,"front")]
                    # If there are no boxes, visibility of the whole shelf cannot be certified by a count of zero.
                    parts.extend(("box_%d"%i,station,"cargo_%d"%i) for i in range(dev["box_count"]))
                elif field=="blocked":
                    identity=(station,"reference_identity") if (station,"reference_identity") in names else ("panel_%02d"%idx,"front")
                    parts=[("identity",*identity),("left_post",station,"exit_post_-0.5"),
                           ("right_post",station,"exit_post_0.5"),("lintel",station,"exit_lintel")]
                    parts.extend(("obstruction_"+v,station,v) for v in dev["approach_obstructions"])
                pixels={role:int(count[names[obj,part]]) for role,obj,part in parts}
                records.append(dict(case=case,method=ep.name,object_id=ident,field=field,source=frame,
                    mean_rgb_absolute_difference=mae,fraction_pixels_within_18=rate,
                    source_pixels=pixels,minimum_source_pixels=min(pixels.values()),
                    zero_item_requires_separate_coverage=(field=="box_count" and dev["box_count"]==0 or
                                                         field=="blocked" and not dev["blocked"])))
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path("config/native_views_dev01.json"))
    conf=read(parser.parse_args().config)
    root=Path("outputs/continuing")/conf["id"]
    dest=root/"audit"
    dest.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for case in conf["cases"]:
        receipts=[r for r in sorted((root/case["id"]).glob("*.receipt.json")) if read(r).get("state")=="scored"]
        if not receipts:
            continue
        base=receipts[0].with_name(read(receipts[0])["method"])
        with TemporaryDirectory(dir=dest) as temp:
            scene=Path(temp)/"inspection.world"
            shutil.copy2(base/"inspection.world",scene)
            shutil.copytree(base/"assets",Path(temp)/"assets")
            renderer=SceneRenderer(scene,robot=True)
            try:
                names=bodies(scene,renderer)
                for r in receipts:
                    ep=r.with_name(read(r)["method"])
                    allrows.extend(grade_episode(ep,renderer,names,case["id"]))
            finally:
                renderer.close()
    save(dest/"cited_sensor_visibility.json",allrows)
    summary=[]
    for case in conf["cases"]:
        for m in case["primary"]+case["comparators"]:
            rr=[r for r in allrows if (r["case"],r["method"])==(case["id"],m)]
            if not rr:
                continue
            # Thresholds are a displayed sensitivity diagnostic, not a replacement for the study grade.
            summary.append(dict(case=case["id"],method=m,bound_claim_frames=len(rr),
                 min_image_alignment_rate=min(r["fraction_pixels_within_18"] for r in rr),
                 max_rgb_mae=max(r["mean_rgb_absolute_difference"] for r in rr),
                 visible_at_1=sum(r["minimum_source_pixels"]>=1 for r in rr),
                 visible_at_36=sum(r["minimum_source_pixels"]>=36 for r in rr),
                 visible_at_144=sum(r["minimum_source_pixels"]>=144 for r in rr)))
    if summary:
        write_table(dest/"cited_sensor_visibility.csv",summary)
    print("Actual bound claim-frame records",len(allrows),"arm summaries",len(summary))


if __name__=="__main__":
    main()
