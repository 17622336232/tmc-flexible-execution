"""Check whether earlier matched camera service results justify batching unseen view pairs."""
import csv
from statistics import median

from experiments.flexible_execution.revision_audit import digest, read, save
from scripts.continuing_online_cost_replay import BASE,CROSS,OUT

OLD = BASE.parent/"paired-rgb-assay01"/"paired.csv"


def load_old():
    rows = []
    for case in csv.DictReader(OLD.open()):
        times = [float(case[k]) for k in ("joint_s","front_s","side_s")]
        rows.append(dict(id=case["house"], joint_ok=case["joint_both_correct"]=="True",
            separate_ok=case["separate_both_correct"]=="True",
            joint_s=times[0], separate_s=sum(times[1:]),
            matched_service_delta_s=float(case["separate_minus_joint_service_s"]) if case["separate_minus_joint_service_s"] else None))
    return rows


def choose(rows):
    deltas = [c["matched_service_delta_s"] for c in rows if c["matched_service_delta_s"] is not None]
    both = [c for c in rows if c["joint_ok"] and c["separate_ok"]]
    return {
        "all-group":"joint",
        "all-separate":"separate",
        "median-savings":"joint" if deltas and median(deltas)>0 else "separate",
        "no-worse-observed-success-and-time":"joint" if len(both)>=2
               and all(not c["separate_ok"] or c["joint_ok"] for c in rows)
               and all(c["matched_service_delta_s"] is not None and c["matched_service_delta_s"]>0 for c in both)
               else "separate",
    }


def main():
    prior = load_old()
    if len(prior)!=6:
        raise ValueError("Calibration anchor changed")
    live = read(CROSS)
    rows = []
    for case in live["cases"]:
        modes = case["observations"]
        baseline = choose(prior)
        row = dict(id=case["house"],joint_ok=case["both_correct_joint"],separate_ok=case["both_correct_separate"],
                   joint_s=modes["joint"]["service_s"],separate_s=modes["front"]["service_s"]+modes["side"]["service_s"],
                   matched_service_delta_s=case["service_delta_s"])
        rows.append(dict(**row,training_pairs=len(prior),selected=baseline))
        prior.append(row)
    results = {}
    for policy in rows[0]["selected"]:
        selected = [row["selected"][policy] for row in rows]
        deliveries = [row[mode+"_ok"] for row,mode in zip(rows,selected)]
        measured = [row[mode+"_s"] if result else 240. for row,mode,result in zip(rows,selected,deliveries)]
        results[policy] = dict(choices=selected,correct_pairs=sum(deliveries),
                             restricted_service_s=measured,restricted_mean_s=sum(measured)/len(measured))
    out=dict(prior_sha256=digest(OLD),current_sha256=digest(CROSS),results=results,rows=rows,new_model_requests=0,
             scope="Calibration-only. Matched already-completed responses stand in for choices to expose transfer risk. "
                   "A 240s penalty represents incorrect or missing pairs, not their observed service time. "
                   "Two prospective input pairs; clock reconstruction excludes movement and accepted downstream task completion.")
    save(OUT/"cross_service_transfer.json",out)
    print(results)


if __name__=="__main__":
    main()
