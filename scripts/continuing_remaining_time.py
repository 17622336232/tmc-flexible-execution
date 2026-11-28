"""A descriptive remaining-time estimate for the registered queue using real elapsed job clocks."""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from statistics import median

from experiments.flexible_execution.revision_audit import read,save


ROOT=Path("outputs/continuing")
CONFIGS=[Path("config")/(s+".json") for s in ("native_views_dev01","native_evidence_dev02","coordinated_handoff_dev03")]


def instant(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))


def main():
    asof=datetime.now(timezone.utc)
    reference=defaultdict(list);all_times=[];rows=[]
    for path in CONFIGS:
        cfg=read(path);root=ROOT/cfg["id"]
        for case in cfg["cases"]:
            for block in ("primary","comparators"):
                closed=(root/"comparator_gate_closed.json").exists()
                admitted=(root/"comparator_gate.json").exists()
                for arm in case[block]:
                    p=root/case["id"]/(arm+".receipt.json");d=read(p) if p.is_file() else {}
                    state=d.get("state","queued")
                    if state in ("scored","closed-failure","interrupted-after-dispatch"):
                        if not d.get("endpoint_block"):
                            t=(instant(d["end_utc"])-instant(d["start_utc"])).total_seconds()
                            if t>0:reference[case["family"]].append(t);all_times.append(t)
                        continue
                    if block=="comparators" and closed:continue
                    extra=block=="comparators" and not admitted
                    age=max(0.,(asof-instant(d["start_utc"])).total_seconds()) if d.get("start_utc") else 0.
                    rows.append(dict(cohort=cfg["id"],case=case["id"],method=arm,family=case["family"],
                         elapsed_s=age,state=state,conditional=extra,task_deadline_s=cfg["episode_s"]))
    for r in rows:
        observations=reference.get(r["family"],[]) or all_times
        eligible=[t for t in observations if t>r["elapsed_s"]]
        r["conditioning_runs"]=len(eligible)
        r["estimated_remaining_s"]=median(eligible)-r["elapsed_s"] if eligible else max(0.,r["task_deadline_s"]-r["elapsed_s"])
        r["same_family_evidence"]=bool(reference.get(r["family"]))
    def total(extra):
        group=[r for r in rows if r["conditional"]==extra]
        return dict(slots=len(group),descriptive_minutes=sum(r["estimated_remaining_s"] for r in group)/60)
    reg=ROOT/"same-frame-region-assay01/registration.json";area=read(reg) if reg.is_file() else {}
    registered=sum(len(c["order"]) for c in area.get("cases",[]))
    used=len(list((ROOT/"same-frame-region-assay01").glob("*/*.receipt.json")))
    closed=ROOT/"scoped-mission-dev01/registration.json"
    cap=len(read(closed)["bounded_online_order"]) if closed.is_file() else 0
    completed=len(list((ROOT/"scoped-mission-dev01").glob("*/*.receipt.json")))
    gate=ROOT/"same-frame-region-assay01/summary.json"
    if gate.is_file() and not all(c["group_all_correct"] is True and c["single_all_correct"] is True for c in read(gate)["cases"]):
        cap=0
    value=dict(utc=asof.isoformat(),native_registered=total(False),native_conditional_after_mechanism_gate=total(True),
               remaining_standalone_image_slots=max(0,registered-used),image_service_limit_s_each=120,
               possible_remaining_scoped_slots=max(0,cap-completed),scoped_task_limit_s_each=180,open_native=rows,
               reference_closed_nonquota_jobs=len(all_times),method="Elapsed worker durations include simulator startup and teardown. Median conditional on outlasting the active slot's observed age, using the same family when available. New-family fallback uses all closed non-quota jobs. Small samples, no guaranteed finish time. An infrastructure pause or a new research protocol has unbounded calendar delay.")
    save(ROOT/"results/remaining_registered_time.json",value)
    print({k:v for k,v in value.items() if k not in ("open_native","method")})


if __name__=="__main__":
    main()
