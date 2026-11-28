"""Test whether genuine supported judgments and captured photos can close an explicit public roster."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from experiments.flexible_execution.evidence import Frame
from experiments.flexible_execution.gpt_vision import parse_targets
from experiments.flexible_execution.revision_audit import read,save,write_table,digest
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.closed_scope import ClosedInspection,is_front_crack_question
from research.flexible_scheduling.front_facing import sign_goals
from research.flexible_scheduling.material_binding import bind_identity
from research.flexible_scheduling.literal_ocr import decode_rgb
from scripts.continuing_crossworld_views import assert_snapshot,copied_renderer
from scripts.continuing_ocr import projected_label_boxes,inside
from scripts.continuing_sensor_visibility import bodies,camera


CFG=Path("config/scope_closure_dev01.json")
OUT=Path("outputs/continuing/scope-closure-dev01-parserfix")
NAV=Path("outputs/continuing/local-view-navigation01")
ROLES=("radial_same_distance","frontal_same_distance","without-local-reading")


def register():
    cases=[]
    for item in read(NAV/"registration.json")["cases"]:
        ep=Path(item["origin"]); robot=ep/"private/robot"
        rgb=robot/(item["source_frame"]+".jpg")
        if digest(rgb)!=item["ref_sha256"]:raise ValueError("Historical source mismatch")
        meta=read(rgb.with_suffix(".json"))
        records=read_trace_rows(robot/"model_calls.jsonl")[0]
        locators=[c for c in records if c["image_sha256"]==[item["ref_sha256"]] and
                 c["input"].get("onboard_image_id")==meta["frame_id"] and "object_scope" in c["input"] and
                 isinstance((c.get("output") or {}).get("targets"),list)]
        if not locators:raise ValueError("No genuine source-matched public localization")
        regions=[dict(id=t.id,bbox=list(t.bbox)) for t in parse_targets(locators[0]["output"])]
        case=dict(**item,regions=regions,source_backend_id=meta["frame_id"],locator_attempt=locators[0].get("logical_job_attempt"),
                  raw_log_sha256=digest(robot/"model_calls.jsonl"))
        cases.append(case)
    doc=dict(protocol=digest(CFG),role="explicit-roster evidence composition",cases=cases)
    assert_snapshot(OUT/"rosters.json",doc)
    return cases


def assemble(case,role):
    ep=Path(case["origin"]);robot=ep/"private/robot"
    path=robot/(case["source_frame"]+".jpg")
    meta=read(path.with_suffix(".json"))
    src=Frame(meta["frame_id"],"public-scope",1,meta["capture_ros"],path.read_bytes(),"historical-physical-camera")
    roster=ClosedInspection("exposed-composition/"+case["source_case"],src,[(r["id"],r["bbox"]) for r in case["regions"]])
    trace=[]
    for c in read_trace_rows(robot/"model_calls.jsonl")[0]:
        d=c["input"]; reply=c.get("output")
        if not is_front_crack_question(d.get("question")) or d.get("bbox_in_attached_photo") is None or not reply:
            continue
        if roster.record_judgment(d.get("target_id"),d,c["image_sha256"],reply):
            trace.append(dict(region=d["target_id"],outcome=reply["outcome"],attempt=c["logical_job_attempt"]))
    arm="frontal_same_distance" if role=="without-local-reading" else role
    nav=NAV/case["source_case"]/arm
    d=read(nav/"navigation.json")
    at_anchor=(d["return_arrived"] or d["outcome"]=="map-rejected" and d["matching_world"])
    release=[]
    reads=0
    if role!="without-local-reading" and at_anchor and d["outward_arrived"]:
        jpg=nav/"calibration_camera.jpg"
        capture=read(nav/"calibration_camera.json")
        geom=read(robot/(case["source_frame"]+"_geometry.json"))
        current=read(nav/"calibration_geometry.json")["camera"]
        positives=[r for r in case["regions"] if roster.needs_identity(r["id"])]
        literals=decode_rgb(jpg.read_bytes()) if positives else []
        reads=int(bool(positives))
        for r in positives:
            options=sign_goals(src.payload,geom,r["bbox"])
            bound=bind_identity(geom["camera"],current,options[0] if options else None,r["bbox"],literals)
            if not bound:continue
            frame=Frame("calibration_camera",r["id"],2,capture["ros_stamp"],jpg.read_bytes(),"fresh-same-world-simulator-route")
            hint=dict(field="identity",value=bound["value"],raw_image_sha256=frame.digest,
                      reference_sha256=src.digest,reference_bbox=r["bbox"],scope_hash=roster.scope_hash)
            if roster.record_identity(r["id"],frame,hint,capture["ros_stamp"]):
                release.append(dict(region=r["id"],center=bound["center"],source=frame.digest))
    report,missing=roster.assemble(at_anchor=at_anchor)
    return dict(case=case["source_case"],role=role,reference_sha256=src.digest,judgments=trace,
                real_source_log=case["raw_log_sha256"],second_source_navigation_sha256=digest(nav/"navigation.json"),
                local_read_calls=reads,releases=release,report=report,missing=[list(x) for x in missing])


def grade(cases,attempts):
    result=[]
    for case in cases:
        ep=Path(case["origin"]);robot=ep/"private/robot"
        physical=read(ep/"private/truth.json")["devices"]
        g=read(robot/(case["source_frame"]+"_geometry.json"))["camera"]
        boxes=projected_label_boxes(ep,g)
        assignments={}
        for r in case["regions"]:
            ids=[k for k,(a,b,c,d) in boxes.items() if inside(((a+c)/2,(b+d)/2),r["bbox"])]
            assignments[r["id"]]=ids[0] if len(ids)==1 else None
        devices={d["identity"]:(i,d) for i,d in enumerate(physical,1)}
        with copied_renderer(ep) as (scene,renderer):
            names=bodies(scene,renderer)
            for a in attempts:
                if a["case"]!=case["source_case"]:continue
                report=a["report"]
                strict=False
                if report is not None and all(assignments.values()) and len(set(assignments.values()))==len(assignments):
                    covered={r["region"]:r for r in report["coverage"]}
                    strict=set(covered)==set(assignments) and all(
                        covered[k]["cracked"]==devices[ident][1]["cracked"] and covered[k]["image_sha256"]==case["ref_sha256"]
                        for k,ident in assignments.items())
                    goal={k:v for k,v in assignments.items() if devices[v][1]["cracked"]}
                    found={r["region"]:r["asset_id"] for r in report["findings"]}
                    strict &= goal==found
                    if found:
                        nav=NAV/a["case"]/a["role"]
                        _,mask=camera(renderer,read(nav/"calibration_camera.json"))
                        h,w=mask.shape
                        for rb in a["releases"]:
                            k=rb["region"];x,y=rb["center"]
                            if k not in goal:strict=False;continue
                            part=names[("equipment_station_%02d"%devices[goal[k]][0],"reference_identity")]
                            strict &= int(mask[int(y*h),int(x*w)])==part
                result.append(dict(case=a["case"],role=a["role"],scope=len(case["regions"]),
                        grounded_public_regions=sum(bool(v) for v in assignments.values()),judged=len(a["judgments"]),
                        positives=sum(j["outcome"]=="positive" for j in a["judgments"]),
                        negative=sum(j["outcome"]=="negative" for j in a["judgments"]),
                        identity_reads=a["local_read_calls"],bound=len(a["releases"]),terminal=report is not None,
                        strict_composed=bool(strict),remaining="|".join(k+"/"+f for k,f in a["missing"])))
    write_table(OUT/"graded.csv",result)
    summary=dict(role="cross-reset evidence sufficiency only",layouts=len(cases),new_model_requests=0,
                 previous_literal_match_summary=digest(OUT.with_name("scope-closure-dev01")/"summary.json") if (OUT.with_name("scope-closure-dev01")/"summary.json").exists() else None,
                 question_rule="Same affirmative front-crack predicate, optional image reference and determiner variation. No new source or model result.",
                 estimated_online_time=None,roles={m:dict(cases=sum(r["role"]==m for r in result),
                     terminal=sum(r["terminal"] for r in result if r["role"]==m),
                     strictly_supported=sum(r["strict_composed"] for r in result if r["role"]==m)) for m in ROLES})
    save(OUT/"summary.json",summary)
    print(summary)


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    fixed=register()
    outcomes=[assemble(c,role) for c in fixed for role in ROLES]
    assert_snapshot(OUT/"certificates_and_gaps.json",outcomes)
    grade(fixed,outcomes)
