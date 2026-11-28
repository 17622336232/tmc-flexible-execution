"""Verify that the precise in-flight RGB exposure traveled from camera to final report."""
from pathlib import Path

from experiments.flexible_execution.revision_audit import digest,read,save
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_native_joint_execution import signature


ROOT=Path("outputs/continuing/native-pending-photo-dev02")


def main():
    summary=read(ROOT/"summary.json")
    sealed=read(ROOT/"sealed.json")
    registered=read(ROOT/"registration.json")
    if digest(ROOT/"sealed.json")!=summary["sealed_sha256"]:
        raise ValueError("The episode seal does not match the score")
    rows=[]
    for score in summary["rows"]:
        case_id,method=score["case"],score["method"]
        key=case_id+"/"+method
        episode=ROOT/key
        receipt=episode.with_suffix(".receipt.json")
        if digest(receipt)!=sealed[key] or read(receipt)["files"]!=signature(episode):
            raise ValueError("The original observation or report changed")
        robot=episode/"private/robot"
        calls=[row for row in read_trace_rows(robot/"model_calls.jsonl")[0]
               if "dispatched_monotonic_seconds" in row]
        trace=read_trace_rows(robot/"timeline.jsonl")[0]
        side={"蓝色货箱","白色货箱"}
        adopted=[v for v in score.get("submission_sources",()) if v[0] in side]
        candidates=[]
        for event in trace:
            if event["event"]!="joint_pending_capture":
                continue
            photo_sha=event["image_sha256"]
            captured_at=event["monotonic_seconds"]
            start,end=calls[0]["dispatched_monotonic_seconds"],calls[0]["return_monotonic_seconds"]
            image=next((p.stem for p in sorted(robot.glob("onboard_*.json"))
                        if read(p).get("ros_stamp")==event["capture_ros"]
                        and read(p).get("sha256")==photo_sha
                        and digest(p.with_suffix(".jpg"))==photo_sha),None)
            continued=[
                call for call in calls[1:]
                if call["dispatched_monotonic_seconds"]>=end and photo_sha in call["image_sha256"]
                and {q["field"] for q in call["input"].get("this_request_scope",{}).get("open_for_reading",())
                     if q["evidence_image_id"]==image}==side
            ]
            first_fields={q["field"] for q in calls[0]["input"]["this_request_scope"]["open_for_reading"]}
            case=next(c for c in registered["cases"] if c["id"]==case_id)
            candidates.append(dict(image_id=image,sha256=photo_sha,first_request_s=end-start,
                start_after_dispatch_s=event["started_monotonic_seconds"]-start,
                captured_before_return_s=end-captured_at,
                early_allowance_in_registered_task="可以在等待时提前拍摄侧面照片" in case["instruction"],
                first_request_was_only_red=first_fields=={"红色货箱"},
                both_side_questions_using_exact_photo=len(continued)==1,
                both_final_side_answers_cite_exact_photo=bool(image) and len(adopted)==2
                    and {f for f,_,_ in adopted}==side and {p for _,p,_ in adopted}=={image}))
        verified=score["strict"] and len(candidates)==1 and any(
            c["image_id"] and c["start_after_dispatch_s"]>=0 and c["captured_before_return_s"]>0
            and c["early_allowance_in_registered_task"] and c["first_request_was_only_red"]
            and c["both_side_questions_using_exact_photo"]
            and c["both_final_side_answers_cite_exact_photo"] for c in candidates)
        rows.append(dict(case=case_id,method=method,verified=verified,captures=candidates))
    result=dict(sealed_sha256=summary["sealed_sha256"],new_requests=0,rows=rows)
    save(ROOT/"capture_audit.json",result)
    print([(r["method"],r["verified"]) for r in rows])


if __name__=="__main__":
    main()
