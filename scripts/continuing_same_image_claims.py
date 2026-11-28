"""Audit what authentic multi-object decisions can establish on exactly matched frozen RGBs."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.flexible_execution.revision_audit import digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.closed_scope import is_front_crack_question
from scripts.continuing_crossworld_views import assert_snapshot, copied_renderer, viewpoint
from scripts.continuing_sensor_visibility import bodies, camera


ROOT=Path("outputs/protocol-v2-optimization/custom-final")
OUT=Path("outputs/continuing/same-image-claims")


def extract():
    grouped=defaultdict(list)
    questions=defaultdict(list)
    for path in sorted(ROOT.glob("custom-*/*/attempt-*/episode/private/robot/model_calls.jsonl")):
        checksum=digest(path)
        for r in read_trace_rows(path,allow_truncated=True)[0]:
            inp=r["input"]
            out=r.get("output") or {}
            stamp=dict(source=str(path),source_sha256=checksum,attempt=r["logical_job_attempt"])
            box=inp.get("bbox_in_attached_photo")
            if box and len(r["image_sha256"])==1:
                questions[r["image_sha256"][0]].append(dict(**stamp,bbox=box,question=inp["question"],
                                           outcome=out.get("outcome"),error=(r.get("error") or {}).get("code")))
            if not inp.get("current_image_id"):
                continue
            frames=[v["image_id"] for v in inp.get("recent_images",[])]+[inp["current_image_id"]]
            if len(frames)!=len(r["image_sha256"]):
                raise ValueError("Input metadata and actually sent images differ")
            supported=defaultdict(list)
            for v in list(out.get("records") or [])+list((out.get("report") or {}).get("records") or []):
                src=v.get("evidence_image_id")
                if src in frames and v.get("object_id") and isinstance(v.get("facts"),dict) and v["facts"]:
                    supported[src].append({"object":v["object_id"],"facts":v["facts"]})
            for src,observations in supported.items():
                if len({v["object"] for v in observations})<2:
                    continue
                h=r["image_sha256"][frames.index(src)]
                grouped[h].append(dict(**stamp,image=src,images_in_input=len(frames),prior_records=len(inp.get("records") or []),
                         history_has_text=bool(inp.get("memory")),observations=observations))
    matched={h:dict(ordinary=grouped[h],specialized=questions[h]) for h in sorted(grouped.keys() & questions.keys())}
    for h,entry in matched.items():
        for o in entry["ordinary"]:
            source=Path(o["source"]).parent/(o["image"]+".jpg")
            if digest(source)!=h:
                raise ValueError("Archived source differs from transmitted bytes")
    return dict(unique_multi_object_photos=len(grouped),actual_reply_photo_pairs=sum(map(len,grouped.values())),
                unique_explicit_box_photos=len(questions),matched=matched)


def predicate(question,fields):
    if is_front_crack_question(question) or question=="该设备是否有裂纹？":
        required={"cracked":True}
    elif question=="该通道是否被实物堵住？":
        required={"blocked":True}
    elif question=="该点位是否同时配置了灭火器且通道没有被实物堵住？":
        required={"extinguisher_present":True,"blocked":False}
    else:
        return None
    # A conjunction can be disproved by one observed false literal.
    pairs=[(fields.get(k),v) for k,v in required.items()]
    if any(type(x) is bool and x!=y for x,y in pairs):
        return "negative"
    if all(type(x) is bool and x==y for x,y in pairs):
        return "positive"
    return None


def reference_centers(robot,frame):
    ep=robot.parents[1]
    rgb=robot/(frame+".jpg")
    geom=read(rgb.with_name(rgb.stem+"_geometry.json"))
    meta=read(ep/"manifest.json")
    devices=read(ep/"private/truth.json")["devices"]
    with copied_renderer(ep) as (scene,renderer):
        names=bodies(scene,renderer)
        pose=viewpoint(renderer,{**geom["position"],"yaw":geom["yaw"]},geom["camera"])
        reconstructed,mask=camera(renderer,pose)
        with Image.open(rgb) as photo:
            alignment=float((np.abs(np.asarray(photo.convert("RGB"),dtype=float)-reconstructed).max(axis=2)<=18).mean())
        centers={}
        for i,d in enumerate(devices,1):
            key=(meta["scene_name"]+"_station_%02d"%i,"reference_identity") if meta["source_profile"]!="single" else ("panel_%02d"%i,"front")
            body=names.get(key)
            if body is None:
                continue
            yy,xx=np.where(mask==body)
            if len(xx)>=36:
                centers[d["identity"]]=(float(xx.mean()/mask.shape[1]),float(yy.mean()/mask.shape[0]))
    return centers,alignment,devices


def score(doc):
    rows=[]
    for h,data in doc["matched"].items():
        for normal in data["ordinary"]:
            path=Path(normal["source"])
            ep=path.parents[2]
            centers,alignment,devices=reference_centers(path.parent,normal["image"])
            for q in data["specialized"]:
                if q["source"]!=normal["source"]:
                    continue
                box=q["bbox"]
                ids=[i for i,(x,y) in centers.items() if box[0]<=x<=box[2] and box[1]<=y<=box[3]]
                obj=ids[0] if len(ids)==1 and alignment>=.90 else None
                truth=next((d for d in devices if d["identity"]==obj),{})
                fields={}
                for v in normal["observations"]:
                    if v["object"]==obj:
                        for k,val in v["facts"].items():
                            # Conflicting claims about a single image stay unresolved.
                            if k not in fields or fields[k]==val:
                                fields[k]=val
                            else:
                                fields[k]=None
                gold=predicate(q["question"],truth)
                predicted=predicate(q["question"],fields)
                rows.append(dict(rgb=h,case=ep.parents[2].name,method=ep.parents[1].name,object=obj,
                      ordinary_attempt=normal["attempt"],single_attempt=q["attempt"],bbox=box,image_alignment=alignment,
                      question=q["question"],truth=gold,ordinary=predicted,solo=q["outcome"],single_error=q["error"],
                      ordinary_supplied_images=normal["images_in_input"],ordinary_prior_records=normal["prior_records"],
                      ordinary_memory=normal["history_has_text"],single_earlier=q["attempt"]<normal["attempt"]))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    doc=extract()
    assert_snapshot(OUT/"actual_inputs_and_replies.json",doc)
    rows=score(doc)
    write_table(OUT/"matched_attempts.csv",rows)
    per_source={}
    for h,g in doc["matched"].items():
        t=[r for r in rows if r["rgb"]==h]
        per_source[h]=dict(ordinary_replies=len(g["ordinary"]),single_attempts=len(g["specialized"]),
            registered_boxes=len({tuple(q["bbox"]) for q in g["specialized"]}),matched_response_pairs=len(t),
            spatially_assigned_pairs=sum(r["truth"] is not None for r in t),
            extra_image_in_all=all(r["ordinary_supplied_images"]>1 for r in t),
            specialized_preceded_ordinary=all(r["single_earlier"] for r in t))
    graded=[r for r in rows if r["truth"] is not None]
    ordinary={(r["rgb"],r["ordinary_attempt"],r["object"],r["question"]):(r["ordinary"],r["truth"]) for r in graded}
    specialized={(r["rgb"],r["single_attempt"]):(r["solo"],r["truth"]) for r in graded}
    def counts(pairs):
        return dict(Counter("unresolved" if p not in ("positive","negative") else "correct" if p==g else "incorrect"
                            for p,g in pairs.values()))
    summary=dict(unique_multi_object_photos=doc["unique_multi_object_photos"],actual_reply_photo_pairs=doc["actual_reply_photo_pairs"],
       unique_explicit_box_photos=doc["unique_explicit_box_photos"],identical_photo_overlaps=len(per_source),
       matched_layouts=len({r["case"] for r in rows}),response_comparisons=len(rows),identifiable_pairs=len(graded),
       ordinary_unique_case_region_judgments=counts(ordinary),specialized_unique_attempts=counts(specialized),
       sources=per_source,new_model_requests=0,
       interpretation="Selected actual multi-object citations matched to specialized queries, not a controlled joint/single comparison. Ordinary replies may see earlier material, memory and additional photographs. No timing or one-image accuracy effect can be attributed to batching from these samples.")
    save(OUT/"summary.json",summary)
    print({k:v for k,v in summary.items() if k not in ("sources","interpretation")})

if __name__=="__main__":
    main()
