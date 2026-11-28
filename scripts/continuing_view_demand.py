"""Train a bounded diagnostic model of expressed camera demand on genuine, source-verified decisions."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from math import log,log1p
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

from experiments.flexible_execution.revision_audit import digest,read,save,write_table
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_exposures import before_restoration


ROOT=Path("outputs/continuing")
OUT=ROOT/"view-demand-dev01"
CFG=Path("config/view_demand_dev01.json")
CAMPUS=ROOT/"coordinated-handoff-dev03/campus-access-03"
VARIANTS=("fixed-half","prevalence","task-only","visual-and-cost")


def stated(raw,offered):
    if not isinstance(raw,dict) or not isinstance(raw.get("action"),dict):
        return None
    action=raw["action"]
    picks=[]
    for place in (raw,action):
        for field in ("use_view","use_views"):
            if field not in place or place[field] is None:
                continue
            vals=place[field] if field=="use_views" else [place[field]]
            if not isinstance(vals,list) or len(vals)>2 or any(not isinstance(i,str) or i not in offered for i in vals):
                return None
            picks.append(set(vals))
    if any(v!=picks[0] for v in picks[1:]):
        return None
    if action.get("type") not in ("move","approach","record","observe"):
        return None if any(picks) else set()
    return picks[0] if picks else set()


def features(offer,trip,window,task,path):
    if digest(path)!=offer["reference_sha256"]:
        raise ValueError("Nominated source pixels changed")
    with Image.open(path) as img:
        rgb=np.asarray(img.convert("L"),dtype=float)
    h,w=rgb.shape
    l,t,r,b=offer["source_bbox"]
    patch=rgb[int(t*h):int(np.ceil(b*h)),int(l*w):int(np.ceil(r*w))]
    if min(patch.shape)<2 or window<=0:
        raise ValueError("Candidate lacks a measurable source region or forecast")
    gradient=(np.abs(np.diff(patch,axis=0)).mean()+np.abs(np.diff(patch,axis=1)).mean())*.5
    return [float("box_count" in task),log((r-l)*(b-t)),log1p(gradient),log1p(trip/window)]


def candidates(files):
    data=[]
    for f in files:
        r=read(f);ep=f.with_name(r["method"]);robot=ep/"private/robot"
        log=robot/"model_calls.jsonl"
        calls=read_trace_rows(log,allow_truncated=True)[0]
        events=read_trace_rows(robot/"timeline.jsonl",allow_truncated=True)[0]
        offers={e["id"]:e for e in events if e["event"]=="native_offer"}
        for e in events:
            if e["event"]!="native_policy_checked" or not e["options"]:
                continue
            frame="onboard_%04d"%e["step"]
            attempts=[v for v in calls if v["input"].get("current_image_id")==frame]
            if not attempts:
                continue
            c=attempts[-1];inp=c["input"]
            offered={o["id"] for o in inp.get("offered_views",[])}
            requested=stated(c.get("output"),offered)
            names=[i["image_id"] for i in inp["recent_images"]]+[inp["current_image_id"]]
            actual=dict(zip(names,c["image_sha256"]))
            group=ep.parent.name;job=r["method"];turn=f"{group}/{job}/{frame}"
            for opt in e["options"]:
                k=opt["id"];o=offers[k]
                if k not in offered or actual.get(o["source"])!=o["reference_sha256"]:
                    raise ValueError("Offer is missing from the actual frozen decision")
                vector=features(o,opt["roundtrip_s"],e["request_window_s"],inp["authorized_task"],robot/(o["source"]+".jpg"))
                data.append(dict(layout=group,method=job,turn=turn,candidate=k,source_sha256=o["reference_sha256"],
                      actual_log_sha256=digest(log),features=vector,y=None if requested is None else int(k in requested)))
    return data


def weights(rows):
    group_turns=defaultdict(set)
    sizes=Counter(r["turn"] for r in rows)
    for r in rows:group_turns[r["layout"]].add(r["turn"])
    raw=np.array([1./len(group_turns[r["layout"]])/sizes[r["turn"]] for r in rows])
    return raw*len(rows)/raw.sum()


def fit(rows,variant):
    rows=[r for r in rows if r["y"] is not None]
    if not rows:
        raise ValueError("No genuine training labels")
    w=weights(rows);y=np.array([r["y"] for r in rows])
    if variant=="fixed-half":return dict(p=.5)
    prior=float((np.dot(w,y)+1)/(w.sum()+2))
    if variant=="prevalence" or len(set(y))==1:return dict(p=prior)
    dims=[0] if variant=="task-only" else [0,1,2,3]
    x=np.array([r["features"] for r in rows])[:,dims]
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    scale[scale<1e-8]=1.
    estimator=LogisticRegression(C=1.,solver="lbfgs",max_iter=200,random_state=17)
    estimator.fit((x-mean)/scale,y,sample_weight=w)
    return dict(dims=dims,mean=mean.tolist(),scale=scale.tolist(),weights=estimator.coef_[0].tolist(),intercept=float(estimator.intercept_[0]))


def predict(m,r):
    if "p" in m:return m["p"]
    x=np.array(r["features"])[m["dims"]]
    linear=np.dot((x-np.array(m["mean"]))/m["scale"],m["weights"])+m["intercept"]
    return float(1/(1+np.exp(-np.clip(linear,-50,50))))


def metrics(rows):
    valid=[r for r in rows if r["y"] is not None]
    if not valid:return dict(labelled=0,brier=None,log_loss=None,two_way_choices=0,top1_hits=None)
    w=weights(valid);p=np.clip([r["p"] for r in valid],1e-10,1-1e-10);y=np.array([r["y"] for r in valid])
    byturn=defaultdict(list)
    for r in valid:byturn[r["turn"]].append(r)
    pairs=[g for g in byturn.values() if len(g)==2 and sum(v["y"] for v in g)==1]
    return dict(labelled=len(valid),positives=int(y.sum()),turns=len(byturn),layouts=len({r["layout"] for r in valid}),
          brier=float(np.average((p-y)**2,weights=w)),log_loss=float(np.average(-y*np.log(p)-(1-y)*np.log(1-p),weights=w)),
          two_way_choices=len(pairs),top1_hits=sum(min(g,key=lambda a:(-a["p"],a["candidate"]))["y"] for g in pairs))


def learn():
    files=before_restoration(include_interrupted=True)
    data=candidates(files);out=[];models={}
    for variant in VARIANTS:
        models[variant]=fit(data,variant)
        for layout in sorted({r["layout"] for r in data}):
            m=fit([r for r in data if r["layout"]!=layout],variant)
            out.extend(dict(**r,variant=variant,p=predict(m,r)) for r in data if r["layout"]==layout)
    artifact=dict(config_sha256=digest(CFG),program_sha256=digest(Path(__file__)),train_manifest_sha256=digest(Path("config/pre_recovery_native_exposures.json")),
       training_labelled=len([r for r in data if r["y"] is not None]),models=models)
    path=OUT/"frozen_weights.json"
    if path.exists() and read(path)!=artifact:
        raise ValueError("Frozen diagnostic model changed. Do not overwrite the registered weights.")
    save(path,artifact)
    stamp=OUT/"freeze_receipt.json"
    if not stamp.exists():
        any_campus=any(p.read_bytes().strip() for p in CAMPUS.glob("*/request_attempts.jsonl"))
        save(stamp,dict(utc=datetime.now(timezone.utc).isoformat(),weights_sha256=digest(path),before_any_campus_send=not any_campus))
    elif read(stamp)["weights_sha256"]!=digest(path):raise ValueError("Freeze receipt mismatch")
    write_table(OUT/"cross_fitted.csv",out)
    summary=dict(training_candidates=len(data),distinct_layouts=len({r["layout"] for r in data}),
       source_cutoff="before plan restoration",model_sha256=digest(path),campus_freeze=read(stamp),deployed=False,
       cross_fitted={v:metrics([r for r in out if r["variant"]==v]) for v in VARIANTS},
       by_layout={k:{v:metrics([r for r in out if r["layout"]==k and r["variant"]==v]) for v in VARIANTS}
                  for k in sorted({r["layout"] for r in out})},new_model_requests=0)
    save(OUT/"training_summary.json",summary)
    print(summary["training_candidates"],summary["campus_freeze"],summary["cross_fitted"])


def campus():
    reg=read(OUT/"freeze_receipt.json")
    path=OUT/"frozen_weights.json";m=read(path)
    if digest(path)!=reg["weights_sha256"] or m["program_sha256"]!=digest(Path(__file__)) or m["config_sha256"]!=digest(CFG):
        raise ValueError("Frozen scoring path changed")
    receipts=sorted(CAMPUS.glob("*.receipt.json"))
    closed=[p for p in receipts if read(p)["state"] in ("scored","interrupted-after-dispatch","closed-failure")]
    data=candidates(p for p in closed if read(p)["method"]!="original-gpt")
    out=[dict(**r,variant=v,p=predict(mm,r)) for v,mm in m["models"].items() for r in data]
    if out:write_table(OUT/"campus_predictions.csv",out)
    report=dict(freeze=reg,exposed_layouts=1 if closed else 0,terminal_methods=[read(p)["method"] for p in closed],
        actual_candidate_count=len(data),prospective_by_calendar=reg["before_any_campus_send"],deployed=False,
        metrics={v:metrics([r for r in out if r["variant"]==v]) for v in VARIANTS},new_model_requests=0)
    save(OUT/"campus_summary.json",report)
    print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("learn","campus"))
    OUT.mkdir(parents=True,exist_ok=True)
    if p.parse_args().mode=="learn":learn()
    else:campus()
