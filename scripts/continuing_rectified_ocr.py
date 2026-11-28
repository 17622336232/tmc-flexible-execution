"""Propose flat signs from RGB only, preserve each reading; join grader geometry after predictions."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from experiments.flexible_execution.revision_audit import read,save,digest
from research.flexible_scheduling.label_geometry import rectangular_surfaces,rectify
from scripts.continuing_ocr import decode,projected_label_boxes,inside


OUT=Path("outputs/continuing/rectified-ocr")
LIMIT=8


def evaluate_pixels(path,intrinsics,parent):
    rgb=np.asarray(Image.open(path).convert("RGB"))
    h,w=rgb.shape[:2]
    found=[]
    surfaces=rectangular_surfaces(rgb,intrinsics)
    for i,s in enumerate(surfaces[:LIMIT]):
        im=rectify(rgb,s)
        fp=parent/("candidate-%02d.png"%i)
        Image.fromarray(im).save(fp)
        for d in decode(str(fp),(),True):
            found.append(dict(value=d["value"],confidence=d["confidence"],
                              center=(s["center"]/[w,h]).tolist(),candidate=i))
    return dict(photo_sha256=digest(path),available_rectangles=len(surfaces),attempted=min(LIMIT,len(surfaces)),readings=found)


def run():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=read(Path("outputs/continuing/ocr/predictions.json"))["rows"]
    unique={}
    for item in inputs:
        robot=Path(item["robot"])
        photo=robot/(item["photo"]+".jpg")
        geometry=read(robot/(item["photo"]+"_geometry.json"))["camera"]
        unique.setdefault(item["digest"],(photo,geometry["intrinsics"]))
    # Positive/negative pipeline controls. Each contains a single bordered planar sign.
    from PIL import ImageDraw,ImageFont
    source=Image.new("RGB",(600,240),"#303a42")
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",40)
    controls={}
    for label,phrase in (("positive","ASSET ID XR-73"),("distractor","STATION EQ-10")):
        sample=source.copy()
        draw=ImageDraw.Draw(sample)
        draw.rectangle((45,82,540,160),fill="white",outline="black",width=3)
        draw.text((120,94),phrase,font=font,fill="black")
        folder=OUT/label
        folder.mkdir(exist_ok=True)
        path=folder/"photo.png"
        sample.save(path)
        controls[label]=evaluate_pixels(path,dict(fx=400,fy=400,cx=300,cy=120),folder)
    save(OUT/"controls.json",controls)
    started=monotonic()
    predictions={}
    for h,(path,intr) in unique.items():
        folder=OUT/h
        folder.mkdir(exist_ok=True)
        r=evaluate_pixels(path,intr,folder)
        if h!=r["photo_sha256"]:
            raise ValueError("Input RGB fingerprint changed")
        predictions[h]=r
    save(OUT/"predictions.json",predictions)
    # Grounded assignment is evaluator-only and cannot affect the above crops or OCR.
    counts=Counter()
    for item in inputs:
        robot=Path(item["robot"]); ep=robot.parents[1]
        ref=projected_label_boxes(ep,read(robot/(item["ref"]+"_geometry.json"))["camera"])
        target=[k for k,(a,b,c,d) in ref.items() if inside(((a+c)/2,(b+d)/2),item["ref_bbox"])]
        actual=projected_label_boxes(ep,read(robot/(item["photo"]+"_geometry.json"))["camera"])
        if len(target)!=1 or target[0] not in actual:
            counts["target_unassigned"]+=1
            continue
        ident=target[0]
        answers={a["value"] for a in predictions[item["digest"]]["readings"] if inside(a["center"],actual[ident])}
        status=("abstain" if len(answers)!=1 else "correct" if answers=={ident} else "wrong")
        counts[status]+=1
    data=dict(input_questions=len(inputs),unique_photos=len(predictions),counts=dict(counts),
        attempted_rectifications=sum(v["attempted"] for v in predictions.values()),
        rgb_images_with_any_literal=sum(bool(v["readings"]) for v in predictions.values()),
        controls={k:[x["value"] for x in d["readings"]] for k,d in controls.items()},
        cpu_assay_wall_s=monotonic()-started,per_image_cap=LIMIT,confidence_floor=75)
    save(OUT/"summary.json",data)
    print(data)


if __name__=="__main__":
    run()
