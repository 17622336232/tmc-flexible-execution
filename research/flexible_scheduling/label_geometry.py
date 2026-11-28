"""Measured planar sign proposals from RGB edges and calibrated camera intrinsics."""
import cv2
import numpy as np


def rectangular_surfaces(rgb,intrinsics):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    _,bright=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    camera=np.array([[intrinsics["fx"],0,intrinsics["cx"]],[0,intrinsics["fy"],intrinsics["cy"]],[0,0,1.]])
    inverse=np.linalg.inv(camera)
    proposals=[]
    for mask in (bright,cv2.Canny(gray,45,125)):
        contours=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
        for contour in contours:
            poly=cv2.approxPolyDP(contour,.028*cv2.arcLength(contour,True),True)
            if len(poly)!=4 or not cv2.isContourConvex(poly):
                continue
            area=cv2.contourArea(poly)
            if not .00045*w*h <= area <= .70*w*h:
                continue
            pts=poly.reshape(4,2).astype(float)
            left,right=np.array_split(pts[np.argsort(pts[:,0])],2)
            tl,bl=left[np.argsort(left[:,1])]
            tr,br=right[np.argsort(right[:,1])]
            quad=np.array([tl,tr,br,bl])
            if not cv2.isContourConvex(quad.astype(np.float32)) or min(np.linalg.norm(quad-np.roll(quad,1,axis=0),axis=1))<5:
                continue
            rays=(inverse @ np.column_stack([quad,np.ones(4)]).T).T
            edges=np.cross(rays,np.roll(rays,-1,axis=0))
            u,v=np.cross(edges[0],edges[2]),np.cross(edges[1],edges[3])
            norm_u,norm_v=np.linalg.norm(u),np.linalg.norm(v)
            if min(norm_u,norm_v)<1e-12:
                continue
            u,v=u/norm_u,v/norm_v
            # A rectangle's two world directions must be approximately orthogonal.
            if abs(u@v)>.4:
                continue
            normal=np.cross(u,v)
            normal/=np.linalg.norm(normal)
            if normal@rays.mean(axis=0)<0:
                normal=-normal
            depth=rays @ normal
            if min(depth)<.045:
                continue
            surface=rays/depth[:,None]
            dims=np.linalg.norm(surface-np.roll(surface,-1,axis=0),axis=1)
            aspect=(dims[0]+dims[2])/(dims[1]+dims[3])
            if not 1.25<=aspect<=15.:
                continue
            center=quad.mean(axis=0)
            score=area/(1+abs(u@v))
            if any(np.linalg.norm(p["center"]-center)<max(3.,.10*np.sqrt(area)) and
                   abs(p["area"]-area)<.30*area for p in proposals):
                continue
            proposals.append(dict(quad=quad.astype(np.float32),area=area,aspect=aspect,center=center,
                                  optical_normal=normal,score=score))
    return sorted(proposals,key=lambda p:-p["score"])


def rectify(rgb,surface):
    height=76
    width=round(height*surface["aspect"])
    corners=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    mapping=cv2.getPerspectiveTransform(surface["quad"],corners)
    return cv2.warpPerspective(rgb,mapping,(width,height),flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
