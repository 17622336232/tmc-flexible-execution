"""Matched counterfactual sensor views in a copy of an exposed world. No simulated trajectory or online success."""
import csv
from math import cos,sin
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
import cv2
import numpy as np
from PIL import Image
from experiments.task_attack.cpu_world import SceneRenderer
from experiments.flexible_execution.revision_audit import read,save,write_table,digest
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.front_facing import sign_goals
from scripts.continuing_ocr import decode,projected_label_boxes,inside


COHORT=Path("outputs/continuing/native-views-dev01")
OUT=Path("outputs/continuing/camera-pairs")


def camera_pose(goal,intr):
    c,s=cos(goal["yaw"]),sin(goal["yaw"])
    T=np.array([[s,0,c,goal["x"]+.17*c],[-c,0,s,goal["y"]+.17*s],[0,-1,0,goal["z"]+.02],[0,0,0,1.]])
    return dict(intr,world_T_camera=T.tolist())


def pixel_extent(ep,identity,cam):
    # Referee's projected mesh footprint. RGB visibility and readability are measured separately.
    devices=read(ep/"private/truth.json")["devices"]
    index=next(i for i,row in enumerate(devices,1) if row["identity"]==identity)
    node=ET.parse(ep/"inspection.world").find(".//model[@name='equipment_station_%02d']"%index)
    plate=node.find("./link/visual[@name='reference_identity']")
    pose=np.fromstring(node.findtext("pose"),sep=" ")
    sub=np.fromstring(plate.findtext("pose"),sep=" ")
    if max(abs(pose[3:]))>1e-6 or max(abs(sub[3:]))>1e-6:
        return {}
    dx,_,dz=np.fromstring(plate.findtext("geometry/mesh/scale"),sep=" ")
    center=pose[:3]+sub[:3]
    T=np.linalg.inv(cam["world_T_camera"])
    K=cam["intrinsics"]
    corners=[]
    for a,b in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)):
        x,y,z,_=T @ np.r_[center+[a*dx,0,b*dz],1.]
        if z<=0:
            return {}
        corners.append([x/z*K["fx"]+K["cx"],y/z*K["fy"]+K["cy"]])
    poly=np.float32(corners)
    w,h=cam["width"],cam["height"]
    area,_=cv2.intersectConvexConvex(poly,np.float32([[0,0],[w,0],[w,h],[0,h]]))
    camera=np.array(cam["world_T_camera"])[:3,3]
    ray=camera-center
    incidence=float(np.rad2deg(np.arccos(np.clip(-ray[1]/np.linalg.norm(ray),-1,1))))
    return dict(projected_area_px=float(area),incidence_deg=incidence)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[]
    for receipt in sorted((COHORT/"cracked-row").glob("*.receipt.json")):
        if read(receipt)["state"]!="scored":
            continue
        ep=receipt.with_name(receipt.name[:-len(".receipt.json")])
        robot=ep/"private/robot"
        calls={c["input"]["current_image_id"]:c for c in read_trace_rows(robot/"model_calls.jsonl")[0]}
        for offer in read_trace_rows(robot/"timeline.jsonl")[0]:
            if offer["event"]!="native_offer" or offer["field"]!="identity":
                continue
            obs=robot/(offer["source"]+".jpg")
            assert digest(obs)==offer["reference_sha256"]
            geom=calls[offer["source"]]["input"]["geometry"]
            goals=sign_goals(obs.read_bytes(),geom,offer["source_bbox"])
            inputs.append(dict(method=ep.name,offer=offer["id"],source=offer["source"],rgb=offer["reference_sha256"],
                       ep=str(ep),geometry=geom["camera"],bbox=offer["source_bbox"],ray_goal=offer["goal"],measured=goals))
    save(OUT/"frozen_proposals.json",inputs)
    measures=[]
    prototype=COHORT/"cracked-row/native-deferred"
    with TemporaryDirectory(dir=OUT) as tmp:
        scene=Path(tmp)/"inspection.world"
        shutil.copy2(prototype/"inspection.world",scene)
        shutil.copytree(prototype/"assets",Path(tmp)/"assets")
        renderer=SceneRenderer(scene,robot=True)
        try:
            for i,item in enumerate(inputs):
                if not item["measured"]:
                    continue
                # Same first distance in both algorithms' recorded order; no post-scoring selection.
                proposals={"old":item["ray_goal"],"frontal":item["measured"][0]["goal"]}
                ep=Path(item["ep"])
                projected=projected_label_boxes(ep,item["geometry"])
                targets=[k for k,(a,b,c,d) in projected.items() if inside(((a+c)/2,(b+d)/2),item["bbox"])]
                for role,goal in proposals.items():
                    position=[goal[k] for k in ("x","y","z")]
                    quat=renderer.p.getQuaternionFromEuler([0,0,goal["yaw"]])
                    renderer.set_pose("vla_uav",position,quat)
                    contact=renderer.contacts()
                    cam=camera_pose(goal,item["geometry"])
                    path=OUT/("%02d-%s.jpg"%(i,role))
                    area=pixel_extent(ep,targets[0],cam) if len(targets)==1 else {}
                    literals=[]
                    associated=[]
                    if not contact:
                        rgb=renderer.camera(np.array(cam["world_T_camera"])[:3,3],goal["yaw"],orientation=quat)
                        Image.fromarray(rgb).save(path,quality=95)
                        readings=decode(str(path),())
                        literals=[v["value"] for v in readings]
                        if len(targets)==1:
                            plate=projected_label_boxes(ep,cam).get(targets[0])
                            associated=[v["value"] for v in readings if inside(v["center"],plate)]
                    measures.append(dict(item=i,originating_arm=item["method"],offer=item["offer"],role=role,
                         unique_target=targets[0] if len(targets)==1 else None,map_route_tested=False,
                         endpoint_contact=contact or "",sha256=digest(path) if not contact else None,
                         literal_readings="|".join(literals),source_bound_readings="|".join(associated),
                         target_correct=bool(len(targets)==1 and associated==targets),projected_area_px=area.get("projected_area_px"),
                         incidence_deg=area.get("incidence_deg")))
        finally:
            renderer.close()
    if measures:
        write_table(OUT/"measured.csv",measures)
    results=dict(nominations=len(inputs),eligible_pairs=sum(bool(i["measured"]) for i in inputs),
         scored_endpoints=len(measures),contact_endpoints=sum(bool(m["endpoint_contact"]) for m in measures),
         literal_endpoint_reads=sum(bool(m["literal_readings"]) for m in measures),
         unique_sources=len({(r["rgb"],tuple(r["bbox"])) for r in inputs}),
         scope="Static endpoint RGB in copied scene only. Projected area excludes frame clipping but does not account for occlusion. An actual navigation trajectory is required for online use.")
    save(OUT/"summary.json",results)
    print(results)


if __name__=="__main__":
    run()
