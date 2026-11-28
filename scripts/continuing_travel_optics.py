"""Frozen RGB-only registration diagnostics on prior sources and deliberately mismatched cameras."""
from collections import defaultdict
from pathlib import Path

import cv2

from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.observed_return import rotation_score
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_itinerary_sources import image_agreement, register as registered_sources


CFG=Path("config/observed_return_dev02.json")
ROOT=Path("outputs/continuing/observed-return-dev02")
PARENT=ROOT.parent/"inspection-travel-dev01"


def measure(ref,fresh,cal):
    result=rotation_score(cv2.imread(str(ref),0),cv2.imread(str(fresh),0),
                          read(ref.with_suffix(".json")),read(fresh.with_suffix(".json")),cal)
    raw,_=image_agreement(ref,fresh)
    return {**result,"raw_ncc":raw,"reference_sha256":digest(ref),"capture_sha256":digest(fresh)}


def register():
    config=read(CFG)
    parent,roster=registered_sources()
    historical=defaultdict(list)
    for path in sorted(ROOT.parent.glob("*/*/*.receipt.json")):
        d=read(path)
        if not (d.get("method") and d.get("case")):
            continue
        ep=path.parent/d["method"]
        robot=ep/"private/robot"
        raw=robot/"onboard_0001.jpg"
        pre=ep/"clean_preflight.json"
        if not raw.is_file() or not pre.is_file() or not (robot/"model_calls.jsonl").exists():
            continue
        info=read(pre)
        if info["family"] not in ("cracks","large_inventory"):
            continue
        log=read_trace_rows(robot/"model_calls.jsonl",allow_truncated=True)[0]
        cam=next((c["input"]["geometry"]["camera"] for c in log if c["input"].get("current_image_id")=="onboard_0001"),None)
        if cam:
            historical[info["family"]].append((info["layout_seed"],raw,cam))
    tests=[]
    def record(kind,correct,reference,fresh,cal):
        tests.append(dict(kind=kind,matching=correct,reference=str(reference),fresh=str(fresh),
                          reference_sha256=digest(reference),fresh_sha256=digest(fresh),
                          calibration={k:cal[k] for k in ("width","height","intrinsics")}))
    for c in roster["cases"]:
        ep=PARENT/"calibration"/c["id"]/"calibration"
        cal=read(ep/"sensor_geometry.json")["camera"]
        for i,stop in enumerate(c["stops"]):
            for j,name in enumerate(("a","b")):
                record("same-stop" if i==j else "wrong-stop",i==j,Path(stop["reference"]),ep/(name+".jpg"),cal)
    for family,cams in historical.items():
        for index,(layout,ref,cam) in enumerate(cams):
            for same in (True,False):
                # Choose one deterministic distinct photo per anchor and layout relationship.
                others=[v for v in cams if (v[0]==layout)==same and digest(v[1])!=digest(ref)]
                if others:
                    item=others[index%len(others)]
                    record("same-layout-start" if same else "wrong-layout-start",same,ref,item[1],cam)
    desc=dict(config_sha256=digest(CFG),public_source_registration=digest(PARENT/"registration.json"),
              core_sha256=digest(Path("research/flexible_scheduling/observed_return.py")),optics_code_sha256=digest(Path(__file__)),
              prior_task_policy_sha256=roster["policy_sha256"],tests=tests)
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/"registration.json").is_file():
        # The frozen source list and measurement code bind registration. Keep the original
        # grader fingerprint in that record and record the independently run grader below.
        desc["optics_code_sha256"]=read(ROOT/"registration.json")["optics_code_sha256"]
    assert_snapshot(ROOT/"registration.json",desc)
    return config,parent,roster,desc


def grade():
    _,_,_,d=register()
    rows=[]
    for t in d["tests"]:
        v=measure(Path(t["reference"]),Path(t["fresh"]),t["calibration"])
        if v["reference_sha256"]!=t["reference_sha256"] or v["capture_sha256"]!=t["fresh_sha256"]:
            raise ValueError("A frozen optical comparison changed")
        rows.append(dict(kind=t["kind"],matching=t["matching"],**v))
    groups={}
    for k in sorted({r["kind"] for r in rows}):
        g=[r for r in rows if r["kind"]==k]
        groups[k]=dict(total=len(g),raw_pass=sum(r["raw_ncc"]>=.85 for r in g),
                       corrected_pass=sum(r["registered"] for r in g))
    out=dict(groups=groups,allow_rotated=any(r["matching"] and r["registered"] for r in rows)
             and any(not r["matching"] for r in rows) and not any(not r["matching"] and r["registered"] for r in rows),
             cases=rows,model_requests=0,source_registration_sha256=digest(ROOT/"registration.json"),
             grader_sha256=digest(Path(__file__)))
    save(ROOT/"optics.json",out)
    return out


if __name__=="__main__":
    print(grade()["groups"])
