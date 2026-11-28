"""Derive paper tables from recorded development and historical data. No dispatch."""
import csv
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from math import dist
from statistics import mean, median
from urllib.parse import unquote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.flexible_execution.clean_coverage import audit_run
from experiments.flexible_execution.revision_audit import BASE, OUT, digest, read, save, write_table
from experiments.flexible_execution.revision_jobs import jobs, observed
from experiments.flexible_execution.timeline import read_trace_rows


def percent(values, p):
    if not values:
        return None
    xs = sorted(values)
    i = (len(xs)-1)*p
    j = int(i)
    return xs[j]+(i-j)*(xs[min(j+1,len(xs)-1)]-xs[j])


def sampled_route(robot):
    observations=[]
    for photo in robot.glob("*.jpg"):
        meta=photo.with_suffix(".json")
        if not meta.exists():
            continue
        d=read(meta)
        stamp=d.get("capture_ros",d.get("ros_stamp"))
        if stamp is not None and d.get("sha256")==digest(photo) and "position" in d:
            observations.append((stamp,tuple(d["position"][k] for k in ("x","y","z"))))
    poses=[xyz for _,xyz in sorted(set(observations))]
    if not poses:
        return None,None
    length=sum(dist(a,b) for a,b in zip(poses,poses[1:]))
    returns=0
    for i,p in enumerate(poses[1:],1):
        if dist(p,poses[i-1])>.8 and any(dist(p,q)<=.4 for q in poses[:i-1]):
            returns+=1
    return length,returns


def history_tables():
    totals, subgroups = [], []
    for domain in ("custom", "public"):
        with (BASE/"reports"/(domain+"-final")/"episodes.csv").open() as f:
            rows = list(csv.DictReader(f))
        groups = defaultdict(list)
        for r in rows:
            groups[r["method"]].append(r)
        for method, subset in groups.items():
            def values(k):
                return [float(r[k]) for r in subset if r.get(k) not in (None, "")]
            d = dict(domain=domain, method=method, scenes=len(subset), groups=len({r["group"] for r in subset}),
                     passed=sum(r["success"]=="True" for r in subset), timeouts=sum(r["state"]=="TIMEOUT" for r in subset))
            for k in ("restricted_seconds", "total_action_seconds", "model_action_overlap_seconds",
                      "first_action_from_first_logged_event_seconds","total_local_worker_cpu_seconds","total_search_seconds",
                      "total_model_transport_wall_seconds", "actual_dispatches"):
                v=values(k)
                d[k+"_n"]=len(v)
                d[k+"_mean"]=mean(v) if v else None
                d[k+"_p50"]=percent(v,.5)
                d[k+"_p95"]=percent(v,.95)
            totals.append(d)
            for name in ("family","stratum"):
                for label in sorted({r.get(name) or domain for r in subset}):
                    rs=[r for r in subset if (r.get(name) or domain)==label]
                    subgroups.append(dict(domain=domain, method=method, split=name, label=label, scenes=len(rs),
                                          passes=sum(r["success"]=="True" for r in rs), timeouts=sum(r["state"]=="TIMEOUT" for r in rs),
                                          restricted_s=mean(float(r["restricted_seconds"]) for r in rs)))
    write_table(OUT/"historical_costs.csv", totals)
    write_table(OUT/"historical_subgroups.csv", subgroups)


def graph():
    with (OUT/"symbolic_development.csv").open() as f:
        rows=list(csv.DictReader(f))
    fig, axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
    names=[("conditional-async","Ordinary async"),("conditional-planner","Planner"),("adaptive-progress","Adaptive")]
    for ax,move in zip(axes,(".5","1.5")):
        for i,(method,label) in enumerate(names):
            latency=[2,7,13]
            ys=[mean(float(r["restricted_completion_seconds"]) for r in rows if r["method"]==method and
                     float(r["unit_travel"])==float(move) and int(r["latency"])==t) for t in latency]
            ax.plot(latency,ys,marker=["o","s","x"][i],linestyle=["-","--",":"][i],label=label)
        ax.set(xlabel="Modeled reply latency (s)",title=f"Travel cost {float(move):.1f} s/unit")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Modeled whole-task time (s)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/"modeled_latency_cost.png",dpi=170)
    plt.close(fig)


def development(*, round3=False):
    rows=[]
    config=read(Path("config/revision_development.json"))
    if round3:
        extra=read(Path("config/revision_round3.json"))
        config=dict(config,cases=[dict(next(c for c in config["cases"] if c["id"]==extra["case"]),order=extra["online_order"])])
    for case,arm,p,_ in jobs(config):
        if round3:
            p=OUT/"round3"/case["id"]/arm
        receipt=p.parent/(arm+".receipt.json")
        o=observed(p)
        r=read(receipt) if receipt.exists() else {}
        if o["state"]!="scored" and r.get("state") not in ("closed-failure","endpoint-blocked"):
            continue
        data=read(p/"clean_timing.json") if (p/"clean_timing.json").exists() else {}
        result=read(p/"result.json") if (p/"result.json").exists() else {}
        coverage=read(p/"clean_coverage.json") if (p/"clean_coverage.json").exists() else {}
        calls_path=p/"private/robot/model_calls.jsonl"
        calls=read_trace_rows(calls_path,allow_truncated=True)[0] if calls_path.exists() else []
        trace=p/"private/robot/timeline.jsonl"
        events=read_trace_rows(trace,allow_truncated=True)[0] if trace.exists() else []
        ctimes=[x["wall_seconds"] for x in calls if isinstance(x.get("wall_seconds"),(int,float))]
        valid=bool(o.get("strict") and o["seconds"]<=config["timeout_s"])
        restricted=o["seconds"] if valid else config["timeout_s"]
        status=o.get("old") if valid else ("deadline" if (o["seconds"] or 0)>=config["timeout_s"] else "failed")
        clauses=p/"clean_task_scope.json"
        task=read(clauses)["authorized_task"] if clauses.exists() else read(p/"private/truth.json")["task"] if (p/"private/truth.json").exists() else None
        starts=[e["monotonic_seconds"] for e in events if e["event"]=="action_started"]
        first=starts[0]-events[0]["monotonic_seconds"] if starts else None
        compiles=[x["wall_seconds"] for x in calls if "supported_fields" in x.get("input",{}) and "wall_seconds" in x]
        guard_calls=[x for x in calls if any(k in x.get("input",{}) for k in ("proposed_report","verified_groups"))]
        intervals=data.get("intervals",{})
        compile_events={e["event"]:e["monotonic_seconds"] for e in events if e["event"] in ("local_compile_started","local_compile_finished")}
        local_compile_s=(compile_events["local_compile_finished"]-compile_events["local_compile_started"]
                         if len(compile_events)==2 else None)
        verified_starts=[a["start"] for a in intervals.get("action",[]) if a.get("ok") is True]
        first_ok=min(verified_starts)-events[0]["monotonic_seconds"] if events and verified_starts else None
        polyline,returns=sampled_route(p/"private/robot")
        def interval_seconds(name):
            return sum(v["seconds"] for v in intervals.get(name,[]) if v.get("seconds") is not None)
        scene=p/"inspection.world"
        normalized=unquote(scene.read_text()).replace(str(p.resolve()),"{episode}") if scene.exists() else None
        tex=sorted((p/"assets/materials/textures").glob("*.png"))
        normalized_assets=sha256(json.dumps([(f.name,digest(f)) for f in tex]).encode()).hexdigest() if tex else None
        strict=audit_run(p,recheck_legacy_fragments=True) if isinstance(result.get("report"),dict) else {}
        rows.append(dict(case=case["id"],method=arm,outcome=status,strict_success=valid,
                         whole_s=o["seconds"],restricted_s=restricted,sends=o["sends"],calls_with_duration=len(ctimes),
                         call_p50_s=percent(ctimes,.5),call_p95_s=percent(ctimes,.95),compile_s=sum(compiles),
                         model_s=data.get("model_request_seconds"),actions_s=interval_seconds("action"),
                         overlap_s=data.get("model_action_overlap_seconds"),first_logged_action_s=first,
                         first_successfully_finished_action_start_s=first_ok,
                         camera_sampled_polyline_m=polyline,camera_pose_returns=returns,
                         independent_online_guard_calls=len(guard_calls),ranking_search_s=interval_seconds("search"),
                         shared_fragment_regrade=strict.get("complete"),local_compile_s=local_compile_s,
                         missing_fields=str(coverage.get("missing_fields_by_object")),
                         raw_result_sha256=digest(p/"result.json") if result else None,
                         world_sha256=digest(scene) if scene.exists() else None,
                         canonical_world_sha256=sha256(normalized.encode()).hexdigest() if normalized else None,
                         texture_set_sha256=normalized_assets,
                         task_sha256=sha256(task.encode()).hexdigest() if task else None))
    if rows:
        for c in {r["case"] for r in rows}:
            batch=[r for r in rows if r["case"]==c]
            for field in ("canonical_world_sha256","texture_set_sha256","task_sha256"):
                if len({r[field] for r in batch if r[field] is not None})>1:
                    raise ValueError("Mismatched comparison input: "+field)
        write_table((OUT/"round3" if round3 else OUT)/"development_results.csv",rows)
    return rows


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    history_tables()
    graph()
    rows=development()
    if Path("config/revision_round3.json").is_file():
        rows+=development(round3=True)
    print([(r["case"],r["method"],r["outcome"]) for r in rows])
