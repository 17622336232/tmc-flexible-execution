"""Scoring-only geometry audit. No projected label location or identifier is fed to a policy or reader."""
from pathlib import Path
from statistics import median
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.continuing_ocr import projected_label_boxes, inside
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.flexible_execution.revision_audit import digest,read,save,write_table


def world_body(meta):
    pose=np.eye(4)
    pose[:3,:3]=Rotation.from_quat([meta["orientation"][k] for k in ("x","y","z","w")]).as_matrix()
    pose[:3,3]=[meta["position"][k] for k in ("x","y","z")]
    return pose


def visible_geometry(box,camera):
    x0,y0,x1,y1=box
    if x1<=x0 or y1<=y0:
        return 0.,0.,0.
    w,h=camera["width"],camera["height"]
    visible_width=max(0,min(x1,1)-max(x0,0))*w
    visible_height=max(0,min(y1,1)-max(y0,0))*h
    fraction=visible_width*visible_height/((x1-x0)*w*(y1-y0)*h)
    return visible_width,visible_height,fraction


def run():
    cohort=Path("outputs/continuing/native-views-dev01")
    dest=cohort/"audit"
    photos=[]
    for proof in read(dest/"photo_chains.json"):
        if not proof.get("raw_bound") or proof["case"]!="cracked-row" or proof["field"]!="identity":
            continue
        ep=cohort/proof["case"]/proof["method"]
        robot=ep/"private/robot"
        calls=read_trace_rows(robot/"model_calls.jsonl")[0]
        cameras={c["input"]["current_image_id"]:c["input"]["geometry"]["camera"] for c in calls}
        offer=next(e for e in read_trace_rows(robot/"timeline.jsonl")[0] if e["event"]=="native_offer" and e["id"]==proof["candidate"])
        ref=offer["source"]
        geometry=cameras[ref]
        src=projected_label_boxes(ep,geometry)
        # Grader identity is established by unique plaque center within the originally nominated object.
        candidates=[k for k,(a,b,c,d) in src.items() if inside(((a+c)/2,(b+d)/2),offer["source_bbox"])]
        transforms=[np.linalg.inv(world_body(read(robot/(name+".json")))) @ np.array(cam["world_T_camera"])
                    for name,cam in cameras.items()]
        # A camera rigidly mounted to the robot should give the same extrinsics in independently saved photos.
        T=transforms[0]
        max_translation=max(np.linalg.norm(v[:3,3]-T[:3,3]) for v in transforms)
        max_angle=max(Rotation.from_matrix(v[:3,:3] @ T[:3,:3].T).magnitude() for v in transforms)
        # Fail closed if the side frame cannot be localized from independently checked calibration.
        valid=max_translation<.07 and max_angle<.04
        shot=read(robot/(proof["image"]+".json"))
        camera={**geometry,"world_T_camera":(world_body(shot) @ T).tolist()}
        seen=projected_label_boxes(ep,camera) if valid else {}
        reference={k:visible_geometry(box,geometry) for k,box in src.items()}
        projected={k:visible_geometry(box,camera) for k,box in seen.items()}
        photos.append(dict(house=proof["case"],method=proof["method"],image=proof["image"],sha256=proof["sha256"],
                    early=proof["early"],target_plate_candidates=candidates,focused_plate=len(candidates)==1,
                    reference_size_and_fraction=reference,acquired_size_and_fraction=projected,
                    mount_max_translation_m=float(max_translation),mount_max_angle_rad=float(max_angle),
                    mount_verified=bool(valid),direct_final_citation=proof["final_report_cites"]))
    save(dest/"equipment_view_geometry.json",photos)
    predicted=read(Path("outputs/continuing/ocr/predictions.json"))
    heights=[]
    for row in predicted["rows"]:
        robot=Path(row["robot"])
        ref=robot/(row["ref"]+"_geometry.json")
        current=robot/(row["photo"]+"_geometry.json")
        if not ref.exists() or not current.exists():
            continue
        earlier=projected_label_boxes(robot.parents[1],read(ref)["camera"])
        candidates=[k for k,(a,b,c,d) in earlier.items() if inside(((a+c)/2,(b+d)/2),row["ref_bbox"])]
        if len(candidates)!=1:
            continue
        camera=read(current)["camera"]
        box=projected_label_boxes(robot.parents[1],camera).get(candidates[0])
        if box:
            w,h,f=visible_geometry(box,camera)
            heights.append(dict(sha256=row["digest"],width_px=w,height_px=h,in_frame_box_fraction=f))
    if heights:
        write_table(dest/"historical_label_geometry.csv",heights)
    print("Archived projection associations:",len(heights),"median visible plaque height:",median(r["height_px"] for r in heights) if heights else None)
    for p in photos:
        print(p["image"],p["target_plate_candidates"],p["mount_verified"],
              {k:tuple(round(x,2) for x in v) for k,v in p["acquired_size_and_fraction"].items()})


if __name__=="__main__":
    run()
