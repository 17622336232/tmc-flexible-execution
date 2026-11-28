"""Return to a public departure pose using only reached positions and the current native map."""
from math import dist, isfinite

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def distance(a, b):
    return dist([a[k] for k in ("x","y","z")], [b[k] for k in ("x","y","z")])


class ObservedTrail:
    def __init__(self, start):
        self.points = [dict(start)]
        self.saturated = False

    def record(self, pose):
        if (set(pose)!={"x","y","z","yaw"} or not all(isfinite(v) for v in pose.values())
                or distance(self.points[-1],pose)<.75):
            return
        if len(self.points)>=30:
            self.saturated = True
            return
        self.points.append(dict(pose))

    def return_home(self, current, check, move, log):
        # A previously visited position is only a candidate. The current mapper decides admission.
        limit=len(self.points)
        while limit:
            chosen=None
            for i in range(limit):
                candidate=self.points[i]
                if i and distance(candidate,current)<.35:
                    continue
                try:
                    check(candidate)
                except ValueError as ex:
                    reason=(getattr(ex,"evidence",None) or {}).get("map_query_reason")
                    log("return_admission",index=i,accepted=False,code=getattr(ex,"code",None),map_reason=reason)
                    if getattr(ex,"code",None)!="goal_unobserved" or reason!="not_known_free":
                        raise
                    continue
                log("return_admission",index=i,accepted=True)
                chosen=i
                break
            if chosen is None:
                raise RuntimeError("No earlier measured position has current map admission")
            # The normal mover repeats the actual admission and reports the reached pose.
            current=move(chosen,self.points[chosen])
            if chosen==0:
                return
            limit=chosen


def rotation_score(original, capture, source_meta, capture_meta, calibration):
    if original is None or capture is None or original.shape!=capture.shape or original.ndim!=2:
        raise ValueError("Measured grayscale cameras must share resolution")
    h,w=original.shape
    if (w,h)!=(calibration["width"],calibration["height"]):
        raise ValueError("Camera calibration and images differ")
    def orientation(meta):
        q=meta["orientation"]
        return Rotation.from_quat([q[k] for k in ("x","y","z","w")])
    sr,cr=orientation(source_meta),orientation(capture_meta)
    # Native optical axes are right, down, forward in a forward-left-up body frame.
    axes=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    rot=axes.T@(cr.inv()*sr).as_matrix()@axes
    k=calibration["intrinsics"]
    intrinsic=np.array([[k["fx"],0,k["cx"]],[0,k["fy"],k["cy"]],[0,0,1.]])
    homography=intrinsic@rot@np.linalg.inv(intrinsic)
    warped=cv2.warpPerspective(original,homography,(w,h),flags=cv2.INTER_LINEAR)
    valid=cv2.warpPerspective(np.ones_like(original),homography,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
    ratio=float(valid.mean())
    x=warped[valid].astype(float);y=capture[valid].astype(float)
    corr=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and min(x.std(),y.std())>0 else None
    separation=distance(source_meta["position"],capture_meta["position"])
    degrees=float(np.degrees((cr.inv()*sr).magnitude()))
    return dict(registered=bool(separation<=.10 and degrees<=8 and ratio>=.90 and corr is not None and corr>=.90),
                translation_m=separation,rotation_deg=degrees,coverage=ratio,measured_correlation=corr)
