"""Freeze source-only identity assignments before regrading physical object membership."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path

from experiments.flexible_execution.revision_audit import read,save,digest,write_table
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.front_facing import sign_goals
from research.flexible_scheduling.material_binding import bind_identity
from scripts.continuing_camera_pairs import camera_pose
from scripts.continuing_crossworld_views import assert_snapshot,copied_renderer,viewpoint
from scripts.continuing_ocr import projected_label_boxes,inside
from scripts.continuing_sensor_visibility import bodies,camera


CFG=Path("config/local_material_binding01.json")
OUT=Path("outputs/continuing/local-material-binding01")
VIEWS=Path("outputs/continuing/archived-views-crossworld01")
NAV=Path("outputs/continuing/local-view-navigation01")


def disjoint(a,b):
    return a[2]<=b[0] or b[2]<=a[0] or a[3]<=b[1] or b[3]<=a[1]


def actual_boxes(p):
    robot=Path(p["ep"])/"private/robot"
    a=tuple(p["bbox"])
    boxes={a:"requested"}
    for call in read_trace_rows(robot/"model_calls.jsonl")[0]:
        if call["image_sha256"]!=[p["ref_sha256"]] or "object_scope" not in call["input"]:
            continue
        for target in (call.get("output") or {}).get("targets",[]):
            box=target.get("bbox")
            if isinstance(box,list) and len(box)==4 and all(isinstance(v,(int,float)) and 0<=v<=1 for v in box):
                if box[0]<box[2] and box[1]<box[3] and disjoint(box,a):
                    boxes[tuple(box)]="other_observed_object"
    return sorted(boxes.items())


def collected(mode):
    plans={p["i"]:p for p in read(VIEWS/"proposals.json")["plans"]}
    rows=[]
    if mode=="static":
        for d in read(VIEWS/"predictions.json"):
            if d["contact"]:continue
            p=plans[d["i"]]
            ref=read(Path(p["ep"])/"private/robot"/(p["ref"]+"_geometry.json"))["camera"]
            cur=camera_pose(p["goals"][d["role"]],ref)
            pose=dict(position={k:p["goals"][d["role"]][k] for k in ("x","y","z")},yaw=p["goals"][d["role"]]["yaw"])
            jpg=VIEWS/("%03d-%s.jpg"%(p["i"],d["role"]))
            if digest(jpg)!=d["photo_sha256"]:raise ValueError("Static image mismatch")
            rows.append((p,d["role"],cur,pose,d["photo_sha256"],d["answers"]))
    else:
        if not (NAV/"sealed_navigation.json").exists():
            raise ValueError("Finish and seal all fresh calibration captures first")
        reg=read(NAV/"registration.json")
        for case in reg["cases"]:
            p=plans[case["selected_i"]]
            for arm in case["order"]:
                root=NAV/case["source_case"]/arm
                d=read(root/"navigation.json")
                if not d["outward_arrived"]:continue
                if digest(root/"calibration_camera.jpg")!=d["captured_sha256"]:raise ValueError("Fresh image mismatch")
                rows.append((p,arm,read(root/"calibration_geometry.json")["camera"],read(root/"calibration_camera.json"),
                             d["captured_sha256"],d["readings"]))
    return rows


def predict(inputs):
    records=[]
    @lru_cache(None)
    def proposal(loc,frame,h,box):
        robot=Path(loc)/"private/robot"
        file=robot/(frame+".jpg")
        if digest(file)!=h:raise ValueError("Reference image mismatch")
        g=read(robot/(frame+"_geometry.json"))
        choices=sign_goals(file.read_bytes(),g,box)
        return g["camera"],choices[0] if choices else None
    for p,arm,current,pose,h,answers in inputs:
        vals={r["value"] for r in answers}
        naive=max(answers,key=lambda r:r["confidence"]) if len(vals)==1 else None
        for box,question in actual_boxes(p):
            ref,s=proposal(p["ep"],p["ref"],p["ref_sha256"],box)
            bound=bind_identity(ref,current,s,box,answers)
            for mode,pred in (("unique-literal",naive),("source-cone",bound)):
                records.append(dict(i=p["i"],arm=arm,image=h,method=mode,box=list(box),question=question,
                                    proposed=s is not None,value=pred["value"] if pred else None,
                                    center=pred["center"] if pred else None,available_literal=bool(answers)))
    return records


def score(inputs,records,mode):
    groups=defaultdict(list)
    for p,arm,_,pose,_,_ in inputs:
        groups[p["ep"]].append((p,arm,pose))
    bypose=defaultdict(list)
    for row in records:bypose[(row["i"],row["arm"])].append(row)
    rows=[]
    for loc,items in groups.items():
        ep=Path(loc)
        domain=read(ep/"result.json")["scene_name"]
        devices=read(ep/"private/truth.json")["devices"]
        indices={d["identity"]:i for i,d in enumerate(devices,1)}
        with copied_renderer(ep) as (scene,renderer):
            names=bodies(scene,renderer)
            for p,arm,pose in items:
                g=read(ep/"private/robot"/(p["ref"]+"_geometry.json"))["camera"]
                boxes=projected_label_boxes(ep,g,domain)
                if "orientation" not in pose:
                    pose=viewpoint(renderer,p["goals"][arm],g)
                _,mask=camera(renderer,pose)
                for row in bypose[(p["i"],arm)]:
                    labels=[k for k,(a,b,c,d) in boxes.items() if inside(((a+c)/2,(b+d)/2),row["box"])]
                    value=row["value"]
                    key=labels[0] if len(labels)==1 else None
                    outcome="no-unique-reference"
                    if key is not None:
                        if value is None:outcome="abstain"
                        else:
                            height,width=mask.shape
                            px,py=row["center"]
                            surface=int(mask[int(py*height),int(px*width)])
                            expected=names[("%s_station_%02d"%(domain,indices[key]),"reference_identity")]
                            outcome="correct-source" if value==key and surface==expected else "wrong-source-or-value"
                    rows.append(dict(**row,case=p["case"],truth_target=key,outcome=outcome))
    if rows:write_table(OUT/(mode+"-graded.csv"),rows)
    groups={}
    for sub,items in (("all",rows),("visible-literal",[r for r in rows if r["available_literal"]])):
        groups[sub]={}
        for method in ("unique-literal","source-cone"):
            subset=[r for r in items if r["method"]==method]
            groups[sub][method]=dict(Counter(r["outcome"] for r in subset))
            groups[sub][method]["positive_without_unique_reference"]=sum(r["value"] is not None and r["truth_target"] is None for r in subset)
    summary=dict(mode=mode,config_sha256=digest(CFG),images=len(inputs),source_layouts=len({p["case"] for p,*_ in inputs}),
                 outcomes=groups,perception_only=True,new_model_requests=0)
    save(OUT/(mode+"-summary.json"),summary)
    print(summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("static","navigation"))
    mode=parser.parse_args().mode
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=collected(mode)
    predictions=predict(inputs)
    assert_snapshot(OUT/(mode+"-predictions.json"),predictions)
    score(inputs,predictions,mode)
