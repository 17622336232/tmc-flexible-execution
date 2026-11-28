"""Judge archived rendered views using post-episode segmentation and raw hashes."""
import csv
from collections import Counter, defaultdict

from experiments.flexible_execution.public_audit import _grade_segmentation
from experiments.flexible_execution.revision_audit import BASE, OUT, digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows


def main():
    rows=[]
    with (BASE/"reports/public-final/episodes.csv").open() as stream:
        episodes=list(csv.DictReader(stream))
    for episode in episodes:
        arm=BASE/"public-final"/episode["case"]/episode["method"]
        for path in sorted(arm.glob("attempt-*/episode/model_calls.jsonl")):
            ep=path.parent
            calls,_=read_trace_rows(path,allow_truncated=True)
            for c in calls:
                q=c.get("input") or {}
                view=q.get("frame")
                if view not in ("front","side") or not q.get("target_class"):
                    continue
                rgb=ep/(view+".jpg")
                label=ep/"scoring"/(view+".json")
                hashed=rgb.is_file() and digest(rgb)==q.get("sha256") and c.get("image_sha256")==[q.get("sha256")]
                expected=(_grade_segmentation(ep/"scoring",view) if hashed and label.is_file() and
                          read(label)["category"]==q["target_class"] else None)
                result=(c.get("output") or {}).get("outcome")
                rows.append(dict(case=episode["case"],group=episode["group"],method=episode["method"],frame=view,
                                 category=q["target_class"],photo_sha256=q.get("sha256"),photo_verified=hashed,
                                 expected=expected,answer=result,correct=(expected==result) if expected and result else None,
                                 error=c.get("error_code"),dispatch=c.get("logical_job_attempt")))
    write_table(OUT/"archived_vision_decisions.csv",rows)
    groups=defaultdict(list)
    for r in rows:
        groups[r["frame"]].append(r)
    summary={}
    for name,rs in groups.items():
        graded=[r for r in rs if r["correct"] is not None]
        summary[name]=dict(dispatched=len(rs),graded=len(graded),correct=sum(r["correct"] for r in graded),
                           skipped=len(rs)-len(graded),distinct_photo_questions=len({(r["category"],r["photo_sha256"]) for r in graded}),
                           groups=len({r["group"] for r in graded}),
                           false_positive=sum(r["expected"]=="negative" and r["answer"]=="positive" for r in graded),
                           false_negative=sum(r["expected"]=="positive" and r["answer"]=="negative" for r in graded),
                           uncertain=sum(r["answer"]=="uncertain" for r in graded))
    save(OUT/"archived_vision_summary.json",summary)
    logged=read(BASE/"formal-progress/registered-early-results.json")["cases"]
    overlap=[]
    for c in logged:
        for ep in c["episodes"]:
            for w in ep["early_action_witnesses"]:
                remain=w["reply_wait_seconds"]-w["motion_started_after_dispatch_seconds"]
                end=w["motion_finished_seconds_before_reply"]
                overlap.append(dict(case=c["case"],method=ep["method"],target=w["target"],reply=w["outcome"],
                                    reply_s=w["reply_wait_seconds"],actual_motion_reply_overlap_s=(
                                    max(0,remain-max(0,end)) if end is not None else None),
                                    photo_offset_before_reply_s=w["photo_seconds_before_reply"],
                                    verified=w["photo_material_verified"],whole_task_success=ep["success"],
                                    per_action_counterfactual_wait_saved="unidentified"))
    write_table(OUT/"observed_early_motion.csv",overlap)
    print(summary)


if __name__=="__main__":
    main()
