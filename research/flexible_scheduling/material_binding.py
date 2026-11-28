"""Associate an observed body-ID literal with a previously nominated static sign using calibrated rays."""
import re
import numpy as np

ID=re.compile(r"[A-Z]{1,6}-[0-9]{2,6}\Z")


def rays(camera,points):
    intrinsic=camera["intrinsics"]
    xy=np.asarray(points,dtype=float)
    return np.column_stack(((xy[:,0]-intrinsic["cx"])/intrinsic["fx"],
                            (xy[:,1]-intrinsic["cy"])/intrinsic["fy"],np.ones(len(xy))))


def measured_plane(camera,quad,optical_depth):
    if not np.isfinite(optical_depth) or optical_depth<=0 or np.shape(quad)!=(4,2):
        return None
    directions=rays(camera,quad)
    edges=np.cross(directions,np.roll(directions,-1,axis=0))
    a,b=np.cross(edges[0],edges[2]),np.cross(edges[1],edges[3])
    lengths=(np.linalg.norm(a),np.linalg.norm(b))
    if min(lengths)<=1e-12:
        return None
    a,b=a/lengths[0],b/lengths[1]
    if abs(a@b)>.4:
        return None
    normal=np.cross(a,b)
    normal/=np.linalg.norm(normal)
    tf=np.asarray(camera["world_T_camera"])
    point=tf[:3,:3] @ (directions.mean(axis=0)*optical_depth)+tf[:3,3]
    return point,tf[:3,:3] @ normal


def inside_source_cone(reference,current,plane,box,center,depth_uncertainty=.30):
    if len(center)!=2 or not all(np.isfinite(v) and 0<=v<1 for v in center):
        return False
    xyz,normal=plane
    cur=np.asarray(current["world_T_camera"])
    ref=np.linalg.inv(reference["world_T_camera"])
    K=reference["intrinsics"]
    ray=cur[:3,:3] @ rays(current,[[center[0]*current["width"],center[1]*current["height"]]])[0]
    denom=float(normal@ray)
    if abs(denom)<.12:
        return False
    left,top,right,bottom=box
    for offset in (-depth_uncertainty,0.,depth_uncertainty):
        distance=(normal@(xyz-cur[:3,3])+offset)/denom
        if distance<=0:
            return False
        point=cur[:3,3]+ray*distance
        x,y,z,_=ref @ np.r_[point,1.]
        if z<=0:
            return False
        px=(K["fx"]*x/z+K["cx"])/reference["width"]
        py=(K["fy"]*y/z+K["cy"])/reference["height"]
        if not (left<=px<=right and top<=py<=bottom):
            return False
    return True


def bind_identity(reference,current,proposal,box,readings):
    if not proposal:
        return None
    plane=measured_plane(reference,proposal["reference_quad"],proposal["depth_m"])
    if plane is None:
        return None
    candidates=[r for r in readings if isinstance(r.get("value"),str) and ID.fullmatch(r["value"])
                and r.get("confidence",0)>=75 and inside_source_cone(reference,current,plane,box,r["center"])]
    if len({r["value"] for r in candidates})!=1:
        return None
    return max(candidates,key=lambda r:r["confidence"])
