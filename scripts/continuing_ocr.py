"""Offline assay. Predict from actual RGB; only the separate scoring stage opens the world."""
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import time
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from research.flexible_scheduling.literal_ocr import INSTALL, decode_rgb

from experiments.flexible_execution.revision_audit import BASE, OUT as PRIOR, digest, read, save, write_table

OUT = Path("outputs/continuing/ocr")


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
    return decode_rgb(Path(jpeg).read_bytes(), bbox, single_line_plaque)


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


def projected_label_boxes(ep,g,domain="equipment"):
    tf=np.linalg.inv(np.asarray(g["world_T_camera"]))
    intr=g["intrinsics"]
    truth=read(ep/"private/truth.json")
    scene=ElementTree.parse(ep/"inspection.world")
    result={}
    for i,d in enumerate(truth["devices"],1):
        node=scene.find(".//model[@name='%s_station_%02d']"%(domain,i))
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
