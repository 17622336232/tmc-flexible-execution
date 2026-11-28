"""Offline assay. Predict from actual RGB; only the separate scoring stage opens the world."""
import csv
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import time
from xml.etree import ElementTree

import numpy as np
from PIL import Image, ImageOps

from experiments.flexible_execution.revision_audit import BASE, OUT as PRIOR, digest, read, save, write_table

OUT = Path("outputs/continuing/ocr")
INSTALL = Path("outputs/continuing/research-tools/ocr/root").resolve()
TEXT = re.compile(r"(?<![A-Z0-9])([A-Z]{1,6}-[0-9]{2,6})(?![A-Z0-9])")
SOURCE = re.compile(r"\bASSET\s*ID\s*[:#]?\s*([A-Z]{1,6}-[0-9]{2,6})\b")


def inputs():
    jobs = sorted(BASE.glob("custom-final/custom-*/*/attempt-*/episode/private/robot/model_calls.jsonl"))
    jobs += sorted(PRIOR.glob("**/dev-*/*/private/robot/model_calls.jsonl"))
    available = {}
    total = 0
    for path in jobs:
        frames = {p.stem:digest(p) for p in path.parent.glob("probe_*.jpg")}
        photos = {v:k for k,v in frames.items()}
        for line in path.read_text().splitlines():
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            d, out = c.get("input",{}), c.get("output") or {}
            if d.get("field") != "identity" or "reference_frame_id" not in d or len(c["image_sha256"]) != 2:
                continue
            total += 1
            rgb = photos.get(c["image_sha256"][1])
            ref = photos.get(c["image_sha256"][0])
            if rgb and ref:
                bbox = out.get("material_bbox") if out.get("source_role") == "body_identity" else None
                key = (str(path.parent), rgb, str(d.get("reference_bbox")))
                if key not in available:
                    available[key] = dict(robot=str(path.parent),photo=rgb,ref=ref,ref_bbox=d.get("reference_bbox"),bbox=bbox,
                                          digest=frames[rgb], model_answer=out.get("value"),role=out.get("source_role"))
    # Allocate one first item per exposed scene before adding more from that scene.
    groups = defaultdict(list)
    for x in available.values():
        robot=Path(x["robot"])
        case=next((p.name for p in robot.parents if p.name.startswith(("custom-","dev-")) and p.name!="custom-final"),None)
        x["case"]=case
        groups[case].append(x)
    order = []
    for group in groups.values():
        group.sort(key=lambda x: (x["digest"],x["robot"]))
    for i in range(max(map(len,groups.values()),default=0)):
        order.extend(g[i] for _,g in sorted(groups.items()) if i<len(g))
    chosen=order[:64]
    # Report the known diagnostic separately if the deterministic coverage cap missed it.
    sentinels=[x for x in available.values() if "/round3/" in x["robot"] and "/conditional-a/" in x["robot"]]
    return total, [*chosen, *(x for x in sentinels if x not in chosen)]


@lru_cache(maxsize=256)
def decode(jpeg, bbox, single_line_plaque=False):
    binary=INSTALL/"usr/bin/tesseract"
    image=Image.open(jpeg).convert("L")
    w,h=image.size
    left,top,right,bottom=bbox or (0.,0.,1.,1.)
    box=(max(0,int(left*w)),max(0,int(top*h)),min(w,int(np.ceil(right*w))),min(h,int(np.ceil(bottom*h))))
    if box[2]<=box[0] or box[3]<=box[1]:
        return []
    region=image.crop(box)
    scale=4 if bbox else 2
    region=ImageOps.expand(ImageOps.autocontrast(region.resize((region.width*scale,region.height*scale),
                                                              Image.Resampling.LANCZOS if hasattr(Image,"Resampling") else Image.LANCZOS)),8,fill=255)
    stream=BytesIO()
    region.save(stream,format="PNG")
    cmd=[str(binary),"stdin","stdout","--tessdata-dir",str(INSTALL/"usr/share/tesseract-ocr/4.00/tessdata"),
         "-l","eng","--psm","7" if bbox or single_line_plaque else "11","-c","tessedit_create_tsv=1"]
    p=subprocess.run(cmd,input=stream.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     env={**os.environ,"OMP_THREAD_LIMIT":"1"},timeout=15,check=True)
    lines=defaultdict(list)
    reader=csv.DictReader(StringIO(p.stdout.decode("utf8")),delimiter="\t")
    if not {"level","text","conf","line_num"} <= set(reader.fieldnames or ()):
        raise ValueError("OCR did not emit word-level TSV")
    for r in reader:
        if r.get("level")=="5" and (r.get("text") or "").strip():
            lines[(r["block_num"],r["par_num"],r["line_num"])].append(r)
    detections=[]
    for words in lines.values():
        line=" ".join(r["text"] for r in words).upper()
        pattern=TEXT if bbox else SOURCE
        for m in pattern.finditer(line):
            conf=min(float(r["conf"]) for r in words)
            if conf<75:
                continue
            x0=min(int(r["left"]) for r in words); x1=max(int(r["left"])+int(r["width"]) for r in words)
            y0=min(int(r["top"]) for r in words); y1=max(int(r["top"])+int(r["height"]) for r in words)
            detections.append(dict(value=m[1],confidence=conf,
                center=[(box[0]+((x0+x1)/2-8)/scale)/w,(box[1]+((y0+y1)/2-8)/scale)/h]))
    return detections


def predict():
    total, selected=inputs()
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    start=time.monotonic()
    for item in selected:
        rgb=str(Path(item["robot"])/(item["photo"]+".jpg"))
        crop=item["bbox"]
        valid=isinstance(crop,list) and len(crop)==4 and all(isinstance(x,(int,float)) and 0<=x<=1 for x in crop)
        rows.append({**item,"image_only":decode(rgb,()),"assisted_crop":decode(rgb,tuple(crop)) if valid else None})
    save(OUT/"predictions.json",dict(engine=digest(INSTALL/"usr/bin/tesseract"),language=digest(INSTALL/"usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata"),
         input_calls=total,selected=len(selected),confidence_floor=75,wall_s=time.monotonic()-start,rows=rows))


def controls():
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 45)
    records = []
    for case, phrase, expected in (("positive", "ASSET ID XR-73", "XR-73"),
                                    ("no_id", "INSPECT SURFACE", None)):
        frame = Image.new("RGB", (600, 150), "white")
        ImageDraw.Draw(frame).text((36, 45), phrase, fill="black", font=font)
        path = OUT / ("synthetic_control_" + case + ".jpg")
        frame.save(path, quality=98)
        for mode, box in (("image_only", ()), ("assisted_crop", (.02,.16,.96,.84))):
            values = [item["value"] for item in decode(str(path), box)]
            records.append(dict(case=case, mode=mode, expected=expected, observed=values,
                                agrees=values == ([expected] if expected else [])))
    save(OUT/"synthetic_controls.json", records)
    print(records)


def positions(ep, name):
    # These inputs exist only in this offline referee; no OCR call receives them.
    g=read(ep/"private/robot"/(name+"_geometry.json"))["camera"]
    return projected_label_boxes(ep,g)


def projected_label_boxes(ep,g):
    tf=np.linalg.inv(np.asarray(g["world_T_camera"]))
    intr=g["intrinsics"]
    truth=read(ep/"private/truth.json")
    scene=ElementTree.parse(ep/"inspection.world")
    result={}
    for i,d in enumerate(truth["devices"],1):
        node=scene.find(".//model[@name='equipment_station_%02d']"%i)
        if node is None:
            continue
        plate=node.find("./link/visual[@name='reference_identity']")
        if plate is None:
            continue
        base=np.array([float(v) for v in node.findtext("pose").split()])
        off=np.array([float(v) for v in plate.findtext("pose").split()])
        if max(abs(base[3:]))>1e-6 or max(abs(off[3:]))>1e-6:
            continue
        size=list(map(float,plate.findtext("geometry/mesh/scale").split()))
        center=base[:3]+off[:3]
        points=[]
        for dx,dz in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)):
            x,y,z,_=tf@np.r_[center+[dx*size[0],0,dz*size[2]],1.]
            if z<=0:break
            points.append(((x/z*intr["fx"]+intr["cx"])/g["width"],(y/z*intr["fy"]+intr["cy"])/g["height"]))
        if len(points)==4:
            xs,ys=zip(*points)
            result[d["identity"]]=(min(xs),min(ys),max(xs),max(ys))
    return result


def inside(pt,box):
    return box and box[0]<=pt[0]<=box[2] and box[1]<=pt[1]<=box[3]


def score():
    document=read(OUT/"predictions.json")
    counts=defaultdict(Counter);rows=[]
    for item in document["rows"]:
        robot=Path(item["robot"]);ep=robot.parents[1]
        if not (robot/(item["ref"]+"_geometry.json")).exists() or not (robot/(item["photo"]+"_geometry.json")).exists():
            continue
        ref=positions(ep,item["ref"]); current=positions(ep,item["photo"])
        target=[k for k,(a,b,c,d) in ref.items() if inside(((a+c)/2,(b+d)/2),item["ref_bbox"])]
        for mode in ("image_only","assisted_crop"):
            n=counts[mode];n["sampled"]+=1
            if len(target)!=1 or target[0] not in current:
                n["ungraded_target"]+=1;continue
            identity=target[0]
            n["graded"]+=1
            if item[mode] is None:
                n["unavailable_roi"]+=1;continue
            matched=[d["value"] for d in item[mode] if inside(d["center"],current[identity])]
            # Automatic detections for other objects are not assertions about the target.
            outcome=("abstain" if len(matched)!=1 else "correct" if matched[0]==identity else "wrong")
            n[outcome]+=1
            rows.append(dict(case=item["case"],photo=item["digest"],mode=mode,visible_target=identity,
                        detected="|".join(matched),outcome=outcome,all_detections=len(item[mode]),model_read=item["model_answer"]))
    write_table(OUT/"graded.csv",rows)
    all_samples=document["rows"]
    unique={(r["digest"],tuple(r.get("bbox") or [])) for r in all_samples}
    summary=dict(counts=counts,attempted_identity_requests=document["input_calls"],sampled=len(all_samples),
                 scenes=len({r["case"] for r in all_samples}),photo_roi_pairs=len(unique),
                 diagnostic=[r for r in rows if r["photo"].startswith("a483d5a8")],
                 note="No task outcome is inferred. Crop mode requires a previous visual localization; image-only scoring grants an offline target association and is an optimistic upper bound on deployable target-bound coverage.")
    save(OUT/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    import sys
    {"predict":predict,"score":score,"controls":controls}[sys.argv[1]]()
