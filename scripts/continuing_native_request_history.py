"""Build a request-shape prior from intact development receipts, including timeouts."""
from collections import defaultdict
from pathlib import Path

from experiments.flexible_execution.revision_audit import digest,read,save
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.native_service_estimate import RequestObservation,restricted_request_estimate


ROOT=Path("outputs/continuing")
NATIVE=("native-joint-execution-dev01","native-pending-photo-dev02")
CROSS="cross-source-assay-dev02"


def build():
    history=defaultdict(list)
    for name in NATIVE:
        study=ROOT/name
        graded=read(study/"summary.json")
        sealed=read(study/"sealed.json")
        if digest(study/"sealed.json")!=graded["sealed_sha256"]:
            raise ValueError("A timing cohort does not match its frozen score")
        for row in graded["rows"]:
            key=row["case"]+"/"+row["method"]
            ep=study/key
            receipt=ep.with_suffix(".receipt.json")
            calls=ep/"private/robot/model_calls.jsonl"
            if (digest(receipt)!=sealed[key]
                    or digest(calls)!=read(receipt)["files"]["private/robot/model_calls.jsonl"]):
                raise ValueError("A historical request ledger changed")
            for turn,request in enumerate(read_trace_rows(calls)[0]):
                scope=request["input"].get("this_request_scope",{}).get("open_for_reading",[])
                if not scope:
                    continue
                views={f["evidence_image_id"] for f in scope}
                shape="single" if len(scope)==1 else "same" if len(views)==1 else "cross"
                took=request["return_monotonic_seconds"]-request["dispatched_monotonic_seconds"]
                delivered="output" in request
                history[shape].append(dict(seconds=took,delivered=delivered,correct=True if row["strict"] else None,
                                           phase="initial" if turn==0 else "continuation",
                                           cohort=name,source_sha256=digest(calls)))
    cross_summary=ROOT/CROSS/"summary.json"
    for row in read(cross_summary)["cases"]:
        for variant,reply in row["observations"].items():
            shape="cross" if variant=="joint" else "single"
            history[shape].append(dict(seconds=reply["service_s"],delivered=reply["valid"],
                                     correct=all(row["correct"][variant].values()),phase="calibration",
                                     cohort=CROSS,source_sha256=digest(cross_summary)))
    return dict(history)


def main():
    rows=build()
    profile={}
    for key,prior in (("single",15.72),("same",21.07),("cross",101.40)):
        samples=[RequestObservation(0,row["seconds"],row["delivered"],row["correct"]) for row in rows[key]]
        profile[key]={str((wait,cutoff)):restricted_request_estimate(samples,now=0,elapsed_s=wait,
                      limit_s=cutoff,prior_s=prior) for wait,cutoff in ((0,120),(0,30),(10,120))}
    output=ROOT/"native-service-history-dev01.json"
    save(output,dict(prior_sources=list(NATIVE)+[CROSS],new_requests=0,samples=rows,profile=profile))
    for key,estimates in profile.items():
        print(key,{k:round(v["estimated_restricted_s"],2) for k,v in estimates.items()})


if __name__=="__main__":
    main()
