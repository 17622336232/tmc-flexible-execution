"""Source-only near-frontal views of a previously nominated rectangular inspection sign."""
from io import BytesIO
from math import atan2
import numpy as np
from PIL import Image
from .label_geometry import rectangular_surfaces


def sign_goals(rgb,geometry,bbox):
    cam=geometry["camera"]
    x0,y0,x1,y1=bbox
    wh=np.array([cam["width"],cam["height"]])
    img=np.asarray(Image.open(BytesIO(rgb)).convert("RGB"))
    if tuple(img.shape[:2])!=(wh[1],wh[0]):
        return []
    K=cam["intrinsics"]
    T=np.asarray(cam["world_T_camera"])
    grid=geometry["measured_surface_grid"]
    world_to_camera=np.linalg.inv(T)
    available=[]
    for cell in grid["cells"]:
        x,y,z,_=world_to_camera @ np.r_[cell["world_xyz"],1.]
        if z<=0:
            continue
        px=(K["fx"]*x/z+K["cx"])/wh[0]
        py=(K["fy"]*y/z+K["cy"])/wh[1]
        if x0<=px<=x1 and y0<=py<=y1:
            available.append((px,py,z))
    if not available:
        return []
    proposals=[]
    for surface in rectangular_surfaces(img,K):
        px,py=surface["center"]/wh
        if not (x0<=px<=x1 and y0<=py<=y1):
            continue
        proximity=sorted((np.hypot((x-px)/(x1-x0),(y-py)/(y1-y0)),z) for x,y,z in available)
        if proximity[0][0]>.65:
            continue
        depths=[z for d,z in proximity if d<=proximity[0][0]+.22]
        if max(depths)-min(depths)>.55:
            continue
        depth=float(np.median(depths))
        target=T @ np.array([(surface["center"][0]-K["cx"])/K["fx"]*depth,
                            (surface["center"][1]-K["cy"])/K["fy"]*depth,depth,1.])
        normal=T[:3,:3] @ surface["optical_normal"]
        if abs(normal[2])>.35 or np.linalg.norm(normal[:2])<.8:
            continue
        direction=normal[:2]/np.linalg.norm(normal[:2])
        # Static inspection tool only. The live adapter separately checks this endpoint
        # and the return endpoint in the robot's current navigation map.
        for standoff in (1.8,2.4):
            xy=target[:2]-standoff*direction
            goal=dict(x=float(xy[0]),y=float(xy[1]),z=float(geometry["position"]["z"]),
                      yaw=atan2(direction[1],direction[0]))
            proposals.append(dict(goal=goal,standoff_m=standoff,reference_quad=surface["quad"].tolist(),
                                  angular_fit=float(surface["score"]),depth_m=depth,depth_cells=len(depths)))
        break
    return proposals
