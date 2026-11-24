"""External image-only relationships; no private state or task inputs."""
import math
import base64
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


def geometric_features(history,elapsed,anchors):
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



def rgb_features(raw,pixels):
    if not pixels:return None
    image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:return None
    height,width=image.shape[:2];box=pixels['bbox'];u,v=pixels['centroid'];bw=max(8.,(box[2]-box[0])*width);bh=max(8.,(box[3]-box[1])*height)
    left=max(0,int(u*width-1.4*bw));right=min(width,int(u*width+1.4*bw));top=max(0,int(v*height-1.4*bh));bottom=min(height,int(v*height+1.4*bh))
    crop=image[top:bottom,left:right]
    if min(crop.shape[:2])<5:return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);h,s,value=cv2.split(hsv)
    cyan=(h>=75)&(h<=115)&(s>=90)&(value>=35)
    if cyan.sum()<5:return None
    border=np.r_[value[0],value[-1],value[:,0],value[:,-1]];background=float(np.median(border))
    contrast=np.clip((background-value.astype(float)-8)/60,0,1)
    gray=contrast*(s<100);fg=contrast+cyan.astype(float)
    yy,xx=np.indices(value.shape);xx=(xx+left-u*width)/bw;yy=(yy+top-v*height)/bh
    def moments(weight):
        total=max(1e-6,float(weight.sum()));mx=float((weight*xx).sum()/total);my=float((weight*yy).sum()/total)
        return np.array([mx,my,float((weight*(xx-mx)**2).sum()/total),float((weight*(yy-my)**2).sum()/total),float((weight*(xx-mx)*(yy-my)).sum()/total)])
    front=moments(cyan);body=moments(fg);rear=moments(gray)
    return np.r_[front-body,front-rear,gray.sum()/max(1.,cyan.sum()),bw/bh].tolist()



def candidate_features(history,elapsed,anchors):
    geometric=geometric_features(history,elapsed,anchors)
    if geometric is None:return None
    observed=[f for f in history if f.get("rgb_orientation_features") is not None]
    latest=history[-1].get("rgb_orientation_features") if history else None
    first=history[0].get("rgb_orientation_features") if history else None
    now=np.array(latest if latest is not None else [0.]*12)
    delta=now-np.array(first) if latest is not None and first is not None else np.zeros(12)
    extra=np.r_[now,delta,float(latest is not None),float(first is not None)]
    return np.column_stack([geometric,np.tile(extra,(len(geometric),1))])


def enrich_window(window,history):
    """Descriptors exist only for JPEGs actually returned by the gateway."""
    for frame in window.get('external_pixel_history',[]):
        history.setdefault(frame['id'],{}).update(frame)
    for frame in window.get('frames',[]):
        item=history.setdefault(frame['id'],{})
        item.update({k:frame[k] for k in ('id','capture_ros','arrival_wall','robot_pixels')})
        if 'rgb_orientation_features' not in item and frame.get('jpeg_base64'):
            item['rgb_orientation_features']=rgb_features(base64.b64decode(frame['jpeg_base64']),frame['robot_pixels'])
    return history
