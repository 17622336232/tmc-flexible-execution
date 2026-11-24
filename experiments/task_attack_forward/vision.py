"""External image-only relationships; no private state or task inputs."""
import math
import cv2
import numpy as np
from experiments.task_attack.event_timing import features as motion_features

def anchors_from_rgb(data,count):
    image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:return None
    height,width=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    for threshold in (185,175,195,150):
        mask=cv2.inRange(hsv,np.array([0,0,threshold]),np.array([180,55,255]))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,_,stats,centers=cv2.connectedComponentsWithStats(mask);candidates=[]
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if .018*width<=w<=.12*width and .008*height<=h<=.065*height and 2<=w/h<=8 and area/(w*h)>.40:
                candidates.append(dict(u=float((x+w/2)/width),v=float((y+h/2)/height),
                    w=float(w/width),h=float(h/height),y=int(y)))
        candidates.sort(key=lambda a:a['y'])
        for first in candidates:
            row=[c for c in candidates if abs(c['y']-first['y'])<=height*.015]
            if len(row)==count:
                row.sort(key=lambda a:a['u'])
                if min(b['u']-a['u'] for a,b in zip(row,row[1:]))>width*.00001:
                    return [{k:v for k,v in c.items() if k!='y'} for c in row]
    return None


def candidate_features(history,elapsed,anchors):
    entries=[x for x in history if x.get('robot_pixels')]
    base=motion_features(history,elapsed)
    if base is None:return None
    points=np.array([e['robot_pixels']['centroid'] for e in entries]);ts=np.array([e['capture_ros'] for e in entries])
    queries=np.linspace(ts[0],ts[-1],5);trail=np.column_stack([np.interp(queries,ts,points[:,axis]) for axis in (0,1)])
    old=np.array([np.interp(max(ts[0],ts[-1]-15),ts,points[:,axis]) for axis in (0,1)])
    values=[]
    for anchor in anchors:
        # Normalize by the observed label width, not a true world scale.
        scale=max(.005,anchor['w']);center=np.array([anchor['u'],anchor['v']]);delta=(points[-1]-center)/scale
        velocity=np.array(base[4:6])/scale;distance=np.linalg.norm(delta);previous=np.linalg.norm((old-center)/scale)
        local=((trail-center)/scale).flatten()
        values.append([*delta,*np.abs(delta),*(delta**2),*velocity,float(delta.dot(velocity)),distance,
            previous-distance,base[2]/scale,base[3]/scale,base[7]/scale,base[9],math.log1p(elapsed),
            delta[0]*math.log1p(elapsed),delta[1]*math.log1p(elapsed),distance*math.log1p(elapsed),*local])
    return np.array(values)


def context_features(candidate):
    # Keep two likely local source relations without identifying the task.
    order=np.argsort(candidate[:,9]);return np.concatenate([candidate[order[0]],candidate[order[1]],candidate.mean(axis=0)])

