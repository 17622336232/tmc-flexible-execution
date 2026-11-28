"""Exercise a field-specific range channel on both actual requested campus viewpoints, no GPT or real aircraft."""
import argparse
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np
from PIL import Image

from experiments.flexible_execution.adapters.native_views import close_pose
from experiments.flexible_execution.revision_audit import read,save,digest,write_table
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.range_channel import packet,release_for_primary
from research.flexible_scheduling.scoped_execution import match_source
from scripts.continuing_crossworld_views import assert_snapshot,copied_renderer
from scripts.continuing_navigation_assay import geometry_hash
from scripts.continuing_sensor_visibility import bodies,camera


CFG=Path("config/condition_range_dev01.json")
ROOT=Path("outputs/continuing/condition-range-dev01")


def register():
    c=read(CFG);ep=Path(c["source"]);robot=ep/"private/robot"
    log=robot/"model_calls.jsonl";calls=read_trace_rows(log)[0]
    for call in calls:
        opts=call["input"].get("offered_views",[])
        raw=call.get("output") or {};action=raw.get("action") or {}
        if (len(opts)==2 and action.get("type")=="observe" and set(action.get("use_views",[]))=={o["id"] for o in opts}
            and not raw.get("view_judgments") and not action.get("view_judgments")):
            break
    else:raise ValueError("No motivating paired request")
    template=robot/(opts[0]["source"]+".jpg")
    if len({(o["source"],o["field"]) for o in opts})!=1 or opts[0]["field"]!="identity":
        raise ValueError("Sources and material types must match")
    names=[x["image_id"] for x in call["input"]["recent_images"]]+[call["input"]["current_image_id"]]
    if dict(zip(names,call["image_sha256"])).get(template.stem)!=digest(template):
        raise ValueError("Source template was not in the actual model request")
    settings=read(ep/"clean_preflight.json")
    rows=[dict(id=o["id"],bbox=o["source_bbox"],goal=o["goal"]) for o in opts]
    actual=[]
    for p in robot.glob("onboard_side*_geometry.json"):
        actual.append(dict(file=p.name,stored_fields=sorted(read(p)),has_range="measured_surface_grid" in read(p)))
    doc=dict(config_sha256=digest(CFG),range_module_sha256=digest(Path("research/flexible_scheduling/range_channel.py")),
        source_log_sha256=digest(log),template=str(template),source_hash=digest(template),settings={
            k:settings[k] for k in ("family","seed","layout_seed","equipment_count","source_profile")},
        matching_world=geometry_hash(ep),regions=rows,original_side_geometry=actual,original_request_attempt=call["logical_job_attempt"])
    ROOT.mkdir(parents=True,exist_ok=True)
    assert_snapshot(ROOT/"registration.json",doc)
    return doc


def measure(backend,observation,geometry,reference,box,refhash,where,kind):
    adapter=backend.adapter._visual
    with adapter.lock:
        projection=adapter.registry[observation["observation_id"]]["projection"]
        xyz=np.asarray(projection["points_world"],dtype="<f8").copy()
    where.mkdir()
    (where/"camera.jpg").write_bytes(observation["data"])
    h=digest(where/"camera.jpg")
    save(where/"camera.json",{**{k:v for k,v in observation.items() if k!="data"},"sha256":h})
    save(where/"geometry.json",geometry)
    np.save(where/"native_world_points.npy",xyz,allow_pickle=False)
    m,_,_=packet(reference,geometry["camera"],box,xyz,reference_digest=refhash,capture_digest=h,source_kind="onboard-projection")
    save(where/"range.json",m)
    coarse=np.asarray([p["world_xyz"] for p in geometry["measured_surface_grid"]["cells"]],dtype=float)
    q,_,_=packet(reference,geometry["camera"],box,coarse,reference_digest=refhash,capture_digest=h,source_kind="archived-cell-medians")
    save(where/"coarse_range.json",q)
    authorized=release_for_primary(explicit_range_grant=True,actual_request=True,branch="unresolved",
                                source_digest=refhash,allowed_region=box,measurement=m)
    return dict(view=kind,native_returns=len(xyz),scoped_returns=m["sample_count"],occupied=len(m["bins"]),
        coarse_returns=q["sample_count"],coarse_occupied=len(q["bins"]),released_numeric_only=authorized is not None,
        packet_sha256=m["packet_sha256"],raw_image_sha256=h)


def worker(reg,identifier):
    from experiments.task_attack_integrity.run import main as simulate
    row=next(x for x in reg["regions"] if x["id"]==identifier)
    ep=ROOT/identifier
    if ep.exists():raise SystemExit("Preserve existing local attempt: "+str(ep))
    args=["--policy","clean","--backend","cpu","--headless","--no-rviz","--sensor-smoke","--ros-port","11874","--gazebo-port",
          "11875","--episode-timeout","95","--output",str(ep)]
    for k,v in reg["settings"].items():args+=["--"+k.replace("_","-"),str(v)]
    def probe(backend,obs,root,event):
        # Keep camera calibration and measured returns bound to the same observation.
        obs,geom=backend.observe()
        h=sha256(obs["data"]).hexdigest();ref=Path(reg["template"])
        if digest(ref)!=reg["source_hash"]:raise ValueError("Registered template changed")
        alignment=match_source(ref.read_bytes(),obs["data"],row["bbox"])
        out=dict(region=identifier,anchor_measured=False,close_measured=False,returned=False,local_model_requests=0,
                 matching_world=geometry_hash(root)==reg["matching_world"],alignment=alignment)
        if not out["matching_world"] or not alignment or alignment["ncc"]<.90 or max(abs(v) for v in alignment["delta_pixels"])>8:
            out["status"]="public-reference-not-aligned";save(root/"route.json",out);return
        w,hh=geom["camera"]["width"],geom["camera"]["height"];dx,dy=alignment["delta_pixels"]
        l,t,r,b=row["bbox"];box=[l+dx/w,t+dy/hh,r+dx/w,b+dy/hh]
        out["registered_box"]=box;out["fresh_reference_sha256"]=h
        out["anchor"]=measure(backend,obs,geom,geom["camera"],box,h,root/"anchor","anchor")
        out["anchor_measured"]=True
        start_pose={**obs["position"],"yaw":obs["yaw"]};departed=False
        try:
            backend.check_goal(row["goal"]);backend.check_goal(start_pose)
            begin=monotonic();backend.execution_deadline_monotonic=begin+42
            departed=True;backend.move(row["goal"])
            current,g=backend.observe()
            if not close_pose({**current["position"],"yaw":current["yaw"]},row["goal"]):
                raise ValueError("Requested viewpoint was not achieved")
            out["close"]=measure(backend,current,g,geom["camera"],box,h,root/"close","close")
            out["outward_capture_s"]=monotonic()-begin;out["close_measured"]=True;out["status"]="measured"
        except (ValueError,RuntimeError) as ex:
            out["status"]="route-failed" if departed else "map-rejected";out["error"]=str(ex)[:240]
        finally:
            if departed:
                begin=monotonic()
                try:
                    backend.execution_deadline_monotonic=begin+42
                    reached=backend.move(start_pose)
                    out["returned"]=close_pose({**reached["position"],"yaw":reached["yaw"]},start_pose)
                except (ValueError,RuntimeError) as ex:
                    out["return_error"]=str(ex)[:200]
                out["return_s"]=monotonic()-begin
            backend.execution_deadline_monotonic=None
        save(root/"route.json",out)
    simulate(args,sensor_exercise=probe,archive_external_rgb=False)


def grade(reg):
    paths=[ROOT/v["id"]/"route.json" for v in reg["regions"]]
    if not all(p.is_file() and (p.parent/"result.json").is_file() for p in paths):
        raise SystemExit("All registered simulator attempts must settle before scoring")
    seal={str(p.relative_to(ROOT)):digest(p) for p in paths}
    for p in paths:
        for s in ("anchor","close"):
            folder=p.parent/s
            if folder.exists():
                for f in folder.iterdir():seal[str(f.relative_to(ROOT))]=digest(f)
    assert_snapshot(ROOT/"sealed_measurements.json",seal)
    rows=[]
    for p in paths:
        ep=p.parent;route=read(p)
        if not route["anchor_measured"]:continue
        box=route["registered_box"];anchor=ep/"anchor"
        rc=read(anchor/"geometry.json")["camera"];h=route["fresh_reference_sha256"]
        devices=read(ep/"private/truth.json")["devices"]
        with copied_renderer(ep) as (scene,renderer):
            names=bodies(scene,renderer)
            def rendered(folder):
                meta=read(folder/"camera.json")
                if digest(folder/"camera.jpg")!=meta["sha256"]:raise ValueError("Fresh camera changed after measurement")
                generated,mask=camera(renderer,meta)
                with Image.open(folder/"camera.jpg") as photo:
                    quality=float((np.abs(np.asarray(photo.convert("RGB"),dtype=float)-generated).max(axis=2)<=18).mean())
                return mask,quality
            refmask,refquality=rendered(anchor)
            assigned=[]
            for i,d in enumerate(devices,1):
                station="campus_station_%02d"%i
                plate=names[(station,"reference_identity")];yy,xx=np.where(refmask==plate)
                if len(xx)>=36 and box[0]<=xx.mean()/refmask.shape[1]<=box[2] and box[1]<=yy.mean()/refmask.shape[0]<=box[3]:
                    assigned.append((station,d))
            for kind in ("anchor","close"):
                folder=ep/kind
                if not folder.exists():continue
                mask,quality=rendered(folder)
                xyz=np.load(folder/"native_world_points.npy",allow_pickle=False)
                m,ids,uv=packet(rc,read(folder/"geometry.json")["camera"],box,xyz,reference_digest=h,
                         capture_digest=digest(folder/"camera.jpg"),source_kind="onboard-projection")
                if m!=read(folder/"range.json"):raise ValueError("Independent reconstruction of output bins failed")
                observed=mask[uv[ids,1].astype(int),uv[ids,0].astype(int)]
                frames=[];obstacles=0;target=None
                if len(assigned)==1:
                    station,d=assigned[0];target=d["identity"]
                    frames=[int((observed==names[(station,part)]).sum()) for part in ("exit_post_-0.5","exit_post_0.5","exit_lintel")]
                    occluders=set(d["approach_obstructions"])
                    if "vehicle" in occluders:occluders.update(part for obj,part in names if obj==station and (part.startswith("wheel_") or part in ("windscreen","vehicle_cabin")))
                    if "exit_obstruction" in occluders:occluders.add("crate_band")
                    obstacles=sum(int((observed==names[(station,part)]).sum()) for part in occluders)
                spatial=len({v["col"] for v in m["bins"]})>=2 and len({v["row"] for v in m["bins"]})>=2 and len(m["bins"])>=4
                supporting=bool(target and min(refquality,quality)>=.90 and spatial and sum(frames)>=3 and sum(n>0 for n in frames)>=2 and obstacles>=3)
                rows.append(dict(region=ep.name,view=kind,target=target,raw_hits=m["sample_count"],occupied_bins=len(m["bins"]),
                   portal_hit_counts=frames,obstacle_hits=obstacles,source_rgb_alignment=refquality,capture_rgb_alignment=quality,
                   positive_geometry_support=supporting,verified_return=route["returned"],packet_sha256=m["packet_sha256"]))
    if rows:write_table(ROOT/"scored_range.csv",rows)
    out=dict(routes=[read(p) for p in paths],scored=rows,registered=len(paths),model_requests=0,real_vehicle_motion=0,
        interpretation="A scoped numerical range packet can show actual physical surfaces. Correct branch inference is untested, an absent hit does not establish a clear portal, and these fresh resets do not supply the old run's missing range scan.")
    save(ROOT/"summary.json",out)
    print([(r["region"],r["view"],r["raw_hits"],r["portal_hit_counts"],r["obstacle_hits"],r["positive_geometry_support"]) for r in rows])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("register","worker","run","grade"))
    parser.add_argument("--region")
    a=parser.parse_args();reg=register()
    if a.mode=="register":print([(r["id"],r["bbox"]) for r in reg["regions"]],reg["original_side_geometry"])
    elif a.mode=="worker":worker(reg,a.region)
    elif a.mode=="run":
        for r in reg["regions"]:
            ep=ROOT/r["id"]
            if (ep/"route.json").is_file() and (ep/"result.json").is_file():continue
            if ep.exists():raise SystemExit("Preserve and inspect unsettled route "+str(ep))
            with (ROOT/(r["id"]+".log")).open("a") as log:
                subprocess.run([sys.executable,"scripts/continuing_condition_range.py","worker","--region",r["id"]],check=True,stdout=log,stderr=subprocess.STDOUT)
        grade(reg)
    else:grade(reg)


if __name__=="__main__":
    main()
