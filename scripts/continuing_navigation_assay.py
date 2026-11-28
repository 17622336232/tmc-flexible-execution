"""Source-bound, zero-model route calibration on independent simulated equipment layouts."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
from time import monotonic,time
from urllib.parse import unquote
import fcntl

from experiments.flexible_execution.adapters.native_views import close_pose
from research.flexible_scheduling.scoped_execution import match_source
from experiments.flexible_execution.revision_audit import digest,read,save,write_table
from scripts.continuing_crossworld_views import assert_snapshot,copied_renderer
from scripts.continuing_ocr import decode,projected_label_boxes,inside
from scripts.continuing_sensor_visibility import bodies,camera


CFG=Path("config/local_view_navigation01.json")
OUT=Path("outputs/continuing/local-view-navigation01")


def sort_hash(*parts):
    return sha256(json.dumps(parts,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def geometry_hash(ep):
    raw=unquote((ep/"inspection.world").read_text()).replace(str(ep.resolve()),"{episode}")
    textures=[(p.name,digest(p)) for p in sorted((ep/"assets/materials/textures").glob("*.png"))]
    return sort_hash(raw,textures)


def register():
    cfg=read(CFG)
    path=Path(cfg["input"])
    records=read(path)["plans"]
    chosen={}
    for p in records:
        if p["design_layout"] or not p["goals"] or p["ref"]!="probe_0001":
            continue
        ep=Path(p["ep"])
        d=read(ep/"clean_preflight.json"); meta=read(ep/"manifest.json")
        if (meta.get("scene_name"),meta.get("family"),meta.get("source_profile"))!=("equipment","cracks","independent"):
            continue
        signature=sort_hash(cfg["salt"],p["case"],p["ref_sha256"],p["bbox"])
        row=dict(origin=p["ep"],source_case=p["case"],selected_i=p["i"],source_frame=p["ref"],
                 ref_sha256=p["ref_sha256"],bbox=p["bbox"],goals={k:p["goals"][k] for k in cfg["arms"]},
                 settings={k:d[k] for k in ("family","seed","equipment_count","source_profile","layout_seed")})
        group=d["layout_seed"]
        if group not in chosen or signature<chosen[group][0]:
            chosen[group]=(signature,row)
    ids=sorted(chosen,key=lambda i:sort_hash(cfg["salt"],chosen[i][1]["source_case"]))[:cfg["layout_limit"]]
    cases=[]
    for index,i in enumerate(ids):
        row=chosen[i][1]
        row["world_and_textures_sha256"]=geometry_hash(Path(row["origin"]))
        row["order"]=cfg["arms"][::1 if index%2==0 else -1]
        cases.append(row)
    reg=dict(config_sha256=digest(CFG),proposals_sha256=digest(path),cases=cases)
    assert_snapshot(OUT/"registration.json",reg)
    return reg


def execute(case,arm,ep):
    from experiments.task_attack_integrity.run import main as simulate
    settings=case["settings"]
    args=["--policy","clean","--backend","cpu","--headless","--no-rviz","--sensor-smoke","--ros-port","11854",
          "--gazebo-port","11855","--episode-timeout","95","--output",str(ep)]
    for flag in ("family","seed","layout_seed","equipment_count","source_profile"):
        args.extend(("--"+flag.replace("_","-"),str(settings[flag])))
    def probe(backend,obs,root,event):
        ref=Path(case["origin"])/"private/robot"
        old=ref/(case["source_frame"]+".jpg")
        g=read(ref/(case["source_frame"]+"_geometry.json"))
        if digest(old)!=case["ref_sha256"]:
            raise ValueError("Archived calibration camera image changed")
        # Bind an existing image/region to the actual starting RGB. No scene
        # labels, object identities or segmentation are available to this choice.
        identity=geometry_hash(root)==case["world_and_textures_sha256"]
        alignment=match_source(old.read_bytes(),obs["data"],case["bbox"])
        pose={**obs["position"],"yaw":obs["yaw"]}
        delta=math.dist(tuple(pose[k] for k in ("x","y","z")),tuple(g["position"][k] for k in ("x","y","z")))
        yaw=abs(math.atan2(math.sin(pose["yaw"]-g["yaw"]),math.cos(pose["yaw"]-g["yaw"])))
        result=dict(arm=arm,case=case["source_case"],matching_world=identity,
                    initial_delta_m=delta,initial_yaw_delta_rad=yaw,alignment=alignment,
                    calibration_rgb=sha256(obs["data"]).hexdigest(),outward_arrived=False,
                    return_arrived=False,map_admitted=False,readings=[],model_requests=0)
        goal=case["goals"][arm]
        if not identity or delta>.25 or yaw>.12 or not alignment or alignment["ncc"]<.75:
            result["outcome"]="source-alignment-rejected"
            save(root/"navigation.json",result)
            return result
        departed=False
        try:
            backend.check_goal(goal)
            backend.check_goal(pose)
            result["map_admitted"]=True
            departed=True
            start=monotonic()
            backend.execution_deadline_monotonic=start+45
            backend.move(goal)
            capture,geo=backend.observe()
            result["outward_s"]=monotonic()-start
            if not close_pose({**capture["position"],"yaw":capture["yaw"]},goal):
                raise ValueError("Measured pose differs from checked destination")
            result["outward_arrived"]=True
            save(root/"calibration_geometry.json",geo)
            save(root/"calibration_camera.json",{k:v for k,v in capture.items() if k!="data"})
            photo=root/"calibration_camera.jpg"
            photo.write_bytes(capture["data"])
            result["captured_sha256"]=digest(photo)
            result["outcome"]="arrived"
        except (RuntimeError,ValueError) as ex:
            result["outcome"]="motion-failed" if departed else "map-rejected"
            result["error"]=str(ex)[:300]
        finally:
            if departed:
                start=monotonic()
                try:
                    backend.execution_deadline_monotonic=start+45
                    actual=backend.move(pose)
                    result["return_arrived"]=close_pose({**actual["position"],"yaw":actual["yaw"]},pose)
                except (ValueError,RuntimeError) as ex:
                    result["return_error"]=str(ex)[:300]
                result["return_s"]=monotonic()-start
            backend.execution_deadline_monotonic=None
        if result["outward_arrived"]:
            result["readings"]=decode(str(root/"calibration_camera.jpg"),())
        save(root/"navigation.json",result)
        return result
    return simulate(args,sensor_exercise=probe,archive_external_rgb=False)


def status(root):
    records=[]
    for f in sorted(root.glob("*/**/navigation.json")):
        v=read(f)
        records.append(dict(case=v["case"],arm=v["arm"],outcome=v["outcome"],arrived=v["outward_arrived"],
                            returned=v["return_arrived"],readings=len(v["readings"])))
    return records


def simulate_all(reg):
    with (OUT/"navigation.lock").open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case in reg["cases"]:
            for arm in case["order"]:
                ep=OUT/case["source_case"]/arm
                if (ep/"navigation.json").is_file() and (ep/"result.json").exists():
                    continue
                if ep.exists():
                    raise SystemExit("Preserve and inspect unfinished local route in "+str(ep))
                ep.parent.mkdir(exist_ok=True)
                args=[sys.executable,"scripts/continuing_navigation_assay.py","worker","--case",case["source_case"],"--arm",arm]
                with ep.with_name(arm+".log").open("a") as logfile:
                    worker=subprocess.Popen(args,stdout=logfile,stderr=subprocess.STDOUT)
                    save(OUT/"heartbeat.json",dict(state="running",wall=time(),case=case["source_case"],arm=arm,pid=worker.pid))
                    rc=worker.wait()
                if rc or not (ep/"navigation.json").exists():
                    raise SystemExit("Local route worker failed: "+str(ep))
                print(case["source_case"],arm,read(ep/"navigation.json")["outcome"],flush=True)
        save(OUT/"heartbeat.json",dict(state="idle",wall=time()))


def score(reg):
    files=[OUT/c["source_case"]/a/"navigation.json" for c in reg["cases"] for a in c["order"]]
    assert_snapshot(OUT/"sealed_navigation.json",{str(p.relative_to(OUT)):digest(p) for p in files})
    scored=[]
    for case in reg["cases"]:
        source=Path(case["origin"])
        g=read(source/"private/robot"/(case["source_frame"]+"_geometry.json"))["camera"]
        boxes=projected_label_boxes(source,g)
        targets=[k for k,(a,b,c,d) in boxes.items() if inside(((a+c)/2,(b+d)/2),case["bbox"])]
        truth=read(source/"private/truth.json")["devices"]
        with copied_renderer(source) as (scene,renderer):
            names=bodies(scene,renderer)
            for arm in case["order"]:
                ep=OUT/case["source_case"]/arm
                d=read(ep/"navigation.json")
                readings=[]
                pixels=None
                if len(targets)==1 and d["outward_arrived"]:
                    if digest(ep/"calibration_camera.jpg")!=d["captured_sha256"]:
                        raise ValueError("Onboard frame changed")
                    idx=next(i for i,r in enumerate(truth,1) if r["identity"]==targets[0])
                    body=names[("equipment_station_%02d"%idx,"reference_identity")]
                    _,mask=camera(renderer,read(ep/"calibration_camera.json"))
                    h,w=mask.shape
                    pixels=int((mask==body).sum())
                    for a in d["readings"]:
                        x,y=a["center"]
                        if 0<=x<1 and 0<=y<1 and mask[int(y*h),int(x*w)]==body:
                            readings.append(a["value"])
                scored.append(dict(case=case["source_case"],arm=arm,alignment=d["alignment"]["ncc"] if d["alignment"] else None,
                    outcome=d["outcome"],map_admitted=d["map_admitted"],arrived=d["outward_arrived"],returned=d["return_arrived"],
                    outward_s=d.get("outward_s"),return_s=d.get("return_s"),target=targets[0] if len(targets)==1 else None,
                    pixels=pixels,all_readings="|".join(r["value"] for r in d["readings"]),source_bound="|".join(readings),
                    success=bool(d["return_arrived"] and len(targets)==1 and set(readings)==set(targets)),
                    source_sha256=d.get("captured_sha256")))
    write_table(OUT/"scored.csv",scored)
    summary=dict(registered=len(files),started=len(scored),model_requests=0,real_vehicle_motion=0,
                 arms={a:dict(trials=sum(r["arm"]==a for r in scored),arrived=sum(r["arrived"] for r in scored if r["arm"]==a),
                     returned=sum(r["returned"] for r in scored if r["arm"]==a),grounded_read_and_return=sum(r["success"] for r in scored if r["arm"]==a))
                     for a in read(CFG)["arms"]})
    save(OUT/"summary.json",summary)
    print(summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("register","run","worker","score","status"))
    parser.add_argument("--case");parser.add_argument("--arm")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    reg=register() if args.action=="register" or not (OUT/"registration.json").is_file() else read(OUT/"registration.json")
    if reg["config_sha256"]!=digest(CFG):
        raise ValueError("Navigation assay configuration changed")
    if args.action=="worker":
        c=next(x for x in reg["cases"] if x["source_case"]==args.case)
        if args.arm not in c["order"]:raise ValueError("Unregistered camera role")
        raise SystemExit(execute(c,args.arm,OUT/c["source_case"]/args.arm))
    elif args.action=="run":simulate_all(reg)
    elif args.action=="score":score(reg)
    elif args.action=="status": print(status(OUT))
    else:print([(c["source_case"],c["selected_i"],c["order"]) for c in reg["cases"]])


if __name__=="__main__":
    main()
