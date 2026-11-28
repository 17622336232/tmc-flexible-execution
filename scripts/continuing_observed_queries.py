"""Census of genuine simultaneous sends and multiple conditions on the same actually supplied RGB."""
from collections import Counter,defaultdict
from pathlib import Path

from experiments.flexible_execution.revision_audit import read,save,write_table
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.flexible_execution.pilot_schedule import FAMILIES
from scripts.continuing_exposures import before_restoration


OUT=Path("outputs/continuing/observed-queries")
BASE=Path("outputs/protocol-v2-optimization")
NAT=Path("outputs/natural-pilot")


def active(intervals):
    points=[]
    for a,b in intervals:
        if b<a:raise ValueError("Negative service interval")
        if b>a:points.extend(((a,1),(b,-1)))
    running=peak=0
    overlap=0.
    last=None
    for t,delta in sorted(points):
        if last is not None and running>=2:overlap+=t-last
        running+=delta
        peak=max(peak,running);last=t
    return peak,overlap


def summarize(path,domain,case,arm):
    calls=read_trace_rows(path,allow_truncated=True)[0]
    intervals=[(r["dispatched_monotonic_seconds"],r["return_monotonic_seconds"]) for r in calls
               if r.get("dispatched_monotonic_seconds") is not None and r.get("return_monotonic_seconds") is not None]
    perframe=defaultdict(list)
    multimember_decisions=0
    for r in calls:
        d=r.get("input") or {}
        o=r.get("output") or {}
        if len(r.get("image_sha256",[]))==1 and d.get("bbox_in_attached_photo") is not None:
            perframe[(r["image_sha256"][0],d.get("question"))].append(r)
        if d.get("current_image_id") and isinstance(o,dict):
            claims=list(o.get("records") or [])+list((o.get("report") or {}).get("records",[]) or [])
            byframe=defaultdict(set)
            supplied={d["current_image_id"],*(i["image_id"] for i in d.get("recent_images",[]))}
            for row in claims:
                f=row.get("evidence_image_id")
                if f in supplied and row.get("object_id") and row.get("facts"):
                    byframe[f].add(row["object_id"])
            multimember_decisions+=int(any(len(s)>1 for s in byframe.values()))
    support=[]
    for (h,q),rows in perframe.items():
        unique=defaultdict(list)
        for r in rows:unique[tuple(r["input"]["bbox_in_attached_photo"])].append(r)
        if len(unique)<2:continue
        success=sum(any((r.get("output") or {}).get("outcome") in ("positive","negative") for r in rs)
                    for rs in unique.values())
        replied=sum(any((r.get("output") or {}).get("outcome") in ("positive","negative","uncertain") for r in rs)
                    for rs in unique.values())
        support.append(dict(domain=domain,case=case,method=arm,source_file=str(path),photo=h,question=q,
                            distinct_regions=len(unique),answered_regions=success,attempts=len(rows),
                            regions_with_declared_reply=replied,uncertain_only=replied-success,
                            no_valid_reply=len(unique)-replied,all_regions_answered=(success==len(unique))))
    peak,overlap=active(intervals)
    return dict(domain=domain,case=case,method=arm,calls=len(calls),timed=len(intervals),
                peak_simultaneous_transports=peak,parallel_service_wall_s=overlap,
                multiple_region_groups=len(support),multi_object_claim_calls=multimember_decisions),support


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[]
    for f in sorted((BASE/"custom-final").glob("custom-*/*/attempt-*/episode/private/robot/model_calls.jsonl")):
        ep=f.parents[2];paths.append((f,"historical-custom",ep.parents[2].name,ep.parents[1].name))
    for f in sorted((BASE/"public-final").glob("procthor-*/*/attempt-*/episode/model_calls.jsonl")):
        ep=f.parent;paths.append((f,"historical-houses",ep.parents[2].name,ep.parents[1].name))
    for receipt in before_restoration(include_interrupted=True):
        f=receipt.with_name(read(receipt)["method"])/"private/robot/model_calls.jsonl"
        ep=f.parents[2];paths.append((f,ep.parents[1].name,ep.parent.name,ep.name))
    rows=[];groups=[]
    for p,domain,c,m in paths:
        summary,one=summarize(p,domain,c,m);rows.append(summary);groups.extend(one)
    write_table(OUT/"attempts.csv",rows)
    if groups:write_table(OUT/"same_frame_conditions.csv",groups)
    bydomain={}
    for domain in sorted({r["domain"] for r in rows}):
        rr=[r for r in rows if r["domain"]==domain]
        gg=[g for g in groups if g["domain"]==domain]
        bydomain[domain]=dict(attempts_with_log=len(rr),sends=sum(r["calls"] for r in rr),
            timed_sends=sum(r["timed"] for r in rr),attempts_with_simultaneous_transports=sum(r["peak_simultaneous_transports"]>=2 for r in rr),
            peak_real_transports=max(r["peak_simultaneous_transports"] for r in rr),
            multi_region_frame_question_groups=len(gg),all_regions_answered=sum(g["all_regions_answered"] for g in gg),
            grouped_region_questions=sum(g["distinct_regions"] for g in gg),
            definite_region_answers=sum(g["answered_regions"] for g in gg),uncertain_only=sum(g["uncertain_only"] for g in gg),
            no_valid_reply=sum(g["no_valid_reply"] for g in gg),explicit_multi_object_claim_replies=sum(r["multi_object_claim_calls"] for r in rr))
    jobs=[read(f) for f in NAT.glob("pair-*/*/job.json")]
    represented={r["case"]["family"] for r in jobs}
    save(OUT/"summary.json",dict(by_domain=bydomain,legacy_code_task_options=list(FAMILIES),
         clean_pilot_registered_families=sorted(represented),clean_pilot_family_count=len(represented),
         note="A group shares the image and question but has separate measured object regions. Multiple condition questions do not prove simultaneous remote transports or equal accuracy after batching. Raw attempts, not logical episode count."))
    print(bydomain,sorted(represented))


if __name__=="__main__":
    main()
