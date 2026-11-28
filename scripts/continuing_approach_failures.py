"""Audit native visual approaches, real tool feedback, the subsequent normal decision and observed clocks."""
from collections import Counter
from pathlib import Path
from statistics import median

from experiments.flexible_execution.revision_audit import read,save,write_table,digest
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_exposures import before_restoration


OLD=Path("outputs/protocol-v2-optimization/custom-final")
ROOT=Path("outputs/continuing")
KNOWN={"目标两侧缺少更远背景回波，不能排除把后墙当作目标":"unproven_foreground",
       "目标框中心的独立雷达回波不足":"sparse_center",
       "目标中心存在多个深度表面，无法可靠关联物体":"ambiguous_depth",
       "雷达回波没有充分覆盖目标中心":"insufficient_coverage",
       "视觉目标需要移动超过单次安全上限，请先缩短距离后重新观测":"move_too_far",
       "目标框内存在更近的回波，中心深度可能属于背景":"nearer_obstruction"}


def stream(p):
    return read_trace_rows(p,allow_truncated=True)[0] if p.is_file() else []


def audit(path,cohort,case,method):
    robot=path.parent
    normal=[c for c in stream(path) if c["input"].get("current_image_id") and
            "last_tool_feedback" in c["input"] and "segmented_navigation" not in c["input"]]
    decisions={v["image_id"]:v for v in stream(robot/"decisions.jsonl") if v.get("image_id") and not v.get("late_not_executed")}
    tasks={};events=stream(robot/"timeline.jsonl")
    for c in normal:
        fid=c["input"]["current_image_id"]
        decision=decisions.get(fid,{})
        if c.get("output")==decision.get("raw_model_decision",decision.get("decision")) or fid not in tasks:
            tasks[fid]=c
    final={(e.get("image_id"),e["step"],e.get("action")):e for e in events if e["event"]=="action_finished"}
    output=[]
    for c in tasks.values():
        act=(c.get("output") or {}).get("action") or {}
        if act.get("type")!="approach":
            continue
        source=c["input"]["current_image_id"]
        i=int(source.rsplit("_",1)[1])
        if digest(robot/(source+".jpg"))!=c["image_sha256"][-1]:
            raise ValueError("Approach did not refer to the actual current RGB")
        after=tasks.get("onboard_%04d"%(i+1)) or {}
        nxt=after.get("input") or {}
        following=(after.get("output") or {}).get("action") or {}
        after_next=tasks.get("onboard_%04d"%(i+2)) or {}
        e=final.get((source,i,"approach"),{})
        feedback=nxt.get("last_tool_feedback") or {}
        if e.get("ok") is not None and feedback.get("ok") is not None and e["ok"]!=feedback["ok"]:
            raise ValueError("Native tool result and actual next input disagree")
        decision=decisions.get(source,{})
        full_match=decision.get("raw_model_decision",decision.get("decision"))==c["output"]
        raw_finished=c.get("return_monotonic_seconds")
        exact_action=(raw_finished is not None and e.get("monotonic_seconds",-1)>=raw_finished and
                      decision.get("decision",{}).get("action")==act)
        bound=full_match or exact_action
        ok=e.get("ok",feedback.get("ok")) if bound else None
        message=feedback.get("message") or ""
        retry_outcome=final.get((nxt.get("current_image_id"),i+1,following.get("type")),{})
        next_feedback=(after_next.get("input") or {}).get("last_tool_feedback") or {}
        next_ok=retry_outcome.get("ok",next_feedback.get("ok"))
        service=None
        if after.get("dispatched_monotonic_seconds") is not None and after.get("return_monotonic_seconds") is not None:
            service=after["return_monotonic_seconds"]-after["dispatched_monotonic_seconds"]
        output.append(dict(cohort=cohort,case=case,method=method,attempt=c["logical_job_attempt"],step=i,
            source_log_sha256=digest(path),source_rgb=c["image_sha256"][-1],box=act.get("bbox"),requested_standoff=act.get("standoff_m"),
            full_reply_matches=full_match,exact_executed_action_and_clock=exact_action,
            decision_bound=bound,actual_approach_ok=ok,error_type=e.get("error_type",feedback.get("error_type")),
            reason=KNOWN.get(message,message or None),next_ordinary_action=following.get("type"),next_ordinary_ok=next_ok,
            next_request_service_s=service,rejected_then_validated_move=ok is False and following.get("type")=="move" and next_ok is True,
            later_request_also_records=bool((after.get("output") or {}).get("records") or (after.get("output") or {}).get("report"))))
    return output


def report(rows):
    subset=[r for r in rows if r["rejected_then_validated_move"]]
    return dict(approach_replies=len(rows),actual_successes=sum(r["actual_approach_ok"] is True for r in rows),
        actual_rejections=sum(r["actual_approach_ok"] is False for r in rows),unverified=sum(r["actual_approach_ok"] is None for r in rows),
        rejection_reasons=dict(Counter(r["reason"] or "no recorded description" for r in rows if r["actual_approach_ok"] is False)),
        next_actions=dict(Counter(r["next_ordinary_action"] or "no usable response" for r in rows if r["actual_approach_ok"] is False)),
        followed_by_successful_direct_move=len(subset),confirmed_move_also_returned_material=sum(r["later_request_also_records"] for r in subset),
        median_next_service_s_on_confirmed_move=median(r["next_request_service_s"] for r in subset) if subset else None)


def main():
    rows=[]
    for p in sorted(OLD.glob("custom-*/*/attempt-*/episode/private/robot/model_calls.jsonl")):
        ep=p.parents[2];rows.extend(audit(p,"historical-custom",ep.parents[2].name,ep.parents[1].name))
    frozen=set(before_restoration(include_interrupted=True))
    for n in ("native-views-dev01","native-evidence-dev02","coordinated-handoff-dev03"):
        for p in sorted((ROOT/n).glob("*/*.receipt.json")):
            r=read(p)
            if r["state"] not in ("scored","interrupted-after-dispatch","closed-failure"):continue
            label="pre-restoration-native" if p in frozen else n+"-later"
            rows.extend(audit(p.with_name(r["method"])/"private/robot/model_calls.jsonl",label,r["case"],r["method"]))
    dest=ROOT/"results"
    write_table(dest/"actual_approach_failures.csv",rows)
    out=dict(domains={k:report([r for r in rows if r["cohort"]==k]) for k in sorted({r["cohort"] for r in rows})},
         note="A normal decision is matched to its execution record. Segmented-navigation calls with the same frame id are separate and cannot replace the original action. A later ordinary response can incorporate a fresh camera or additional evidence even after rejection; its service time cannot all be credited to an unobserved alternative. Original map and target-grounding refusals remain intact.",
         new_model_requests=0)
    save(dest/"approach_audit.json",out)
    print(out["domains"])


if __name__=="__main__":
    main()
