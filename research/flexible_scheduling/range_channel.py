"""A texture-free evidence channel derived only from actual calibrated onboard range returns."""
from hashlib import sha256
import json
from math import isfinite

import numpy as np

DEPTH_DESCRIPTION="optical depth in metres from reference camera; absent cells unknown"


def project(camera,world):
    T=np.asarray(camera["world_T_camera"],dtype=float)
    K=camera["intrinsics"];w=camera["width"];h=camera["height"]
    if T.shape!=(4,4) or not np.isfinite(T).all() or min(w,h,K["fx"],K["fy"])<=0:
        raise ValueError("Invalid measured camera calibration")
    inv=np.linalg.inv(T)
    xyz=world@inv[:3,:3].T+inv[:3,3]
    z=xyz[:,2]
    valid=z>.05
    uv=np.full((len(z),2),np.nan)
    uv[valid,0]=K["fx"]*xyz[valid,0]/z[valid]+K["cx"]
    uv[valid,1]=K["fy"]*xyz[valid,1]/z[valid]+K["cy"]
    return uv,z,valid&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)


def packet(reference,camera,box,world,*,reference_digest,capture_digest,source_kind,grid=(12,8)):
    """Returns are expressed in the original source region. Each empty output bin stays unmeasured."""
    if len(box)!=4 or not all(type(a) in (int,float) and isfinite(a) for a in box) or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):
        raise ValueError("Invalid public source region")
    if source_kind not in ("onboard-projection","archived-cell-medians"):
        raise ValueError("Unsupported range source")
    for value in (reference_digest,capture_digest):
        if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing observation fingerprint")
    world=np.asarray(world,dtype="<f8")
    if world.size==0:world=np.empty((0,3),dtype="<f8")
    if world.ndim!=2 or world.shape[1]!=3 or not np.isfinite(world).all():
        raise ValueError("Expected immutable finite measured world returns")
    nx,ny=grid
    if type(nx) is not int or type(ny) is not int or not (1<=nx<=64 and 1<=ny<=64):
        raise ValueError("Invalid packet bin count")
    ref,z,ok=project(reference,world)
    cur,_,visible=project(camera,world)
    l,t,r,b=box
    u=ref[:,0]/reference["width"];v=ref[:,1]/reference["height"]
    ids=np.flatnonzero(ok&visible&(u>=l)&(u<r)&(v>=t)&(v<b))
    points={}
    for i in ids:
        key=(int((u[i]-l)*nx/(r-l)),int((v[i]-t)*ny/(b-t)))
        points.setdefault(key,[]).append(int(i))
    bins=[]
    for (x,y),hits in sorted(points.items(),key=lambda k:(k[0][1],k[0][0])):
        dist=z[hits]
        bins.append(dict(col=x,row=y,n=len(hits),min_m=round(float(dist.min()),3),
                         median_m=round(float(np.median(dist)),3),max_m=round(float(dist.max()),3)))
    fields=dict(source_kind=source_kind,reference_sha256=reference_digest,capture_sha256=capture_digest,
        source_region=list(box),projection=DEPTH_DESCRIPTION,
        columns=nx,rows=ny,sample_count=int(len(ids)),bins=bins,
        measured_xyz_sha256=sha256(np.ascontiguousarray(world).tobytes()).hexdigest())
    fields["packet_sha256"]=sha256(json.dumps(fields,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    # Referee-only indices let it compare actual sensor-hit rays to actual photographed physical surfaces;
    # consumers receive fields only, never simulated segmentation or hidden station labels.
    return fields,ids,cur


def release_for_primary(*,explicit_range_grant,actual_request,branch,source_digest,allowed_region,measurement):
    if (explicit_range_grant is not True or actual_request is not True or branch!="unresolved" or
            measurement.get("reference_sha256")!=source_digest or measurement.get("source_region")!=list(allowed_region) or
            not measurement.get("sample_count")):
        return None
    fields={"source_kind","reference_sha256","capture_sha256","source_region","projection","columns","rows","sample_count",
            "bins","measured_xyz_sha256","packet_sha256"}
    if set(measurement)!=fields:
        return None
    if (measurement["projection"]!=DEPTH_DESCRIPTION or measurement["source_kind"] not in ("onboard-projection","archived-cell-medians")
            or not isinstance(measurement["bins"],list) or type(measurement["columns"]) is not int
            or type(measurement["rows"]) is not int or not 1<=measurement["columns"]<=64 or not 1<=measurement["rows"]<=64):
        return None
    if any(not isinstance(measurement[k],str) or len(measurement[k])!=64 or any(c not in "0123456789abcdef" for c in measurement[k])
           for k in ("reference_sha256","capture_sha256","measured_xyz_sha256","packet_sha256")):
        return None
    cells=set();hits=0
    for cell in measurement["bins"]:
        if not isinstance(cell,dict) or set(cell)!={"col","row","n","min_m","median_m","max_m"}:
            return None
        x,y,n=cell["col"],cell["row"],cell["n"]
        limits=[cell[k] for k in ("min_m","median_m","max_m")]
        if (any(type(v) is not int for v in (x,y,n)) or not (0<=x<measurement["columns"] and 0<=y<measurement["rows"] and n>=1)
                or (x,y) in cells or not all(type(v) in (float,int) and isfinite(v) for v in limits)
                or not 0<limits[0]<=limits[1]<=limits[2]):
            return None
        cells.add((x,y));hits+=n
    if type(measurement["sample_count"]) is not int or hits!=measurement["sample_count"]:
        return None
    fingerprint=sha256(json.dumps({k:v for k,v in measurement.items() if k!="packet_sha256"},
                       sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return measurement if fingerprint==measurement["packet_sha256"] else None
