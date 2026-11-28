"""Count where a proposed photograph exits the genuine ordinary request/decision sequence."""
from collections import defaultdict
from pathlib import Path

from experiments.flexible_execution.revision_audit import read,save,write_table,digest
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_native_report import audited_activation


ROOT=Path("outputs/continuing")
GROUPS=("native-views-dev01","native-evidence-dev02","coordinated-handoff-dev03")


def chosen(container,offered):
    values=container.get("use_views")
    if values is None:
        value=container.get("use_view")
        values=[] if value is None else [value]
    if not isinstance(values,list) or len(values)>2 or any(not isinstance(x,str) for x in values):
        return [],False
    return [x for x in values if x in offered],all(x in offered for x in values)


def episode(ep,cohort):
    robot=ep/"private/robot"
    log=robot/"model_calls.jsonl"
    ev=read_trace_rows(robot/"timeline.jsonl",allow_truncated=True)[0]
    costs=defaultdict(list);shots=defaultdict(list);emitted=defaultdict(list);releases={};cards={}
    for e in ev:
        if e["event"]=="native_offer":
            emitted[e["step"]].append(e["id"]);cards[e["id"]]=e
        elif e["event"]=="native_early_start":costs[e["step"]].append(e["candidate"])
        elif e["event"]=="native_raw_photo":shots[e["step"]].append(e["image_id"])
        elif e["event"]=="native_material_release":releases[e["step"]]=e
    result=[]
    calls=read_trace_rows(log,allow_truncated=True)[0]
    last={c["input"]["current_image_id"]:c["logical_job_attempt"] for c in calls}
    for c in calls:
        data=c["input"];raw=c.get("output") or {}
        step=int(data["current_image_id"].rsplit("_",1)[1])
        offered={o["id"] for o in data.get("offered_views",[])}
        action=raw.get("action") or {}
        top,top_valid=chosen(raw,offered);nested,nested_valid=chosen(action,offered)
        nom=raw.get("next_views")
        typed=raw.get("view_judgments")
        final_attempt=last[data["current_image_id"]]==c["logical_job_attempt"]
        actual=releases.get(step,{}) if final_attempt else {}
        moving=action.get("type") in ("move","approach")
        active=[]
        if ep.name.startswith("coordinated-"):
            active=[k for k in top if k in cards and audited_activation(cards[k],raw,c)]
        image_ids=[x["image_id"] for x in data["recent_images"]]+[data["current_image_id"]]
        if [digest(robot/(x+".jpg")) for x in image_ids]!=c["image_sha256"]:
            raise ValueError("Model input differs from camera archive")
        result.append(dict(cohort=cohort,case=ep.parent.name,method=ep.name,step=step,attempt=c["logical_job_attempt"],
            call_sha256=digest(log),returned=bool(raw),error=(c.get("error") or {}).get("code"),
            ordinary_action=action.get("type"),top_level_nominations=len(nom) if isinstance(nom,list) else 0,
            published_next_offers=emitted[step] if final_attempt else [],offers_before_reply=sorted(offered),top_selected=top,top_shape_valid=top_valid,
            nested_only_selected=nested if nested_valid and not top else [],conflicting_explicit_selections=bool(top and nested and top!=nested),
            declared_independent_move=moving and raw.get("motion_independent_of_view") is True,
            nested_only_independence=moving and "motion_independent_of_view" not in raw and action.get("motion_independent_of_view") is True,
            top_judgments=len(typed) if isinstance(typed,list) else 0,semantically_activated_ids=active,
            actual_early_starts=costs[step] if final_attempt else [],acquired_ids=shots[step] if final_attempt else [],
            accepted_camera_ids=actual.get("requested_ids",[actual["requested"]] if actual.get("requested") else []),
            released_photos=actual.get("released",[])))
    return result


def summary(rows):
    return dict(sends=len(rows),valid_replies=sum(r["returned"] for r in rows),
        replies_nominating=sum(bool(r["top_level_nominations"]) for r in rows),
        published_next_offers=sum(len(r["published_next_offers"]) for r in rows),
        turns_offering_a_photo=sum(bool(r["offers_before_reply"]) for r in rows),
        turns_selecting_top_level=sum(bool(r["top_selected"]) for r in rows),
        turns_selecting_only_inside_action=sum(bool(r["nested_only_selected"]) for r in rows),
        selected_and_moving=sum(bool(r["top_selected"]) and r["ordinary_action"] in ("move","approach") for r in rows),
        explicit_independent_move_with_selection=sum(bool(r["top_selected"]) and r["declared_independent_move"] for r in rows),
        turns_with_valid_positive_or_independent_grant=sum(bool(r["semantically_activated_ids"]) for r in rows),
        executor_accepted_turns=sum(bool(r["accepted_camera_ids"]) for r in rows),
        raw_early_starts=sum(len(r["actual_early_starts"]) for r in rows),released=sum(len(r["released_photos"]) for r in rows))


def main():
    rows=[]
    for cohort in GROUPS:
        for receipt in sorted((ROOT/cohort).glob("*/*.receipt.json")):
            r=read(receipt)
            if r["state"] not in ("scored","closed-failure","interrupted-after-dispatch") or r["method"]=="original-gpt":
                continue
            rows.extend(episode(receipt.with_name(r["method"]),cohort))
    dest=ROOT/"results"
    write_table(dest/"actual_camera_interface.csv",rows)
    out=dict(by_cohort={g:summary([r for r in rows if r["cohort"]==g]) for g in GROUPS},
             by_arm={g:{m:summary([r for r in rows if r["cohort"]==g and r["method"]==m])
                    for m in sorted({r["method"] for r in rows if r["cohort"]==g})} for g in GROUPS},
             note="Terminal enhanced episodes only. A nested valid id identifies an unrecognized expression of demand, not an actual release. On a retry, logical-step physical events are counted once on its terminal attempt; photo_chains separately checks actual transport overlap. Accepted camera requests and independently grounded downstream uses are different events.")
    save(dest/"actual_camera_interface.json",out)
    print(out["by_cohort"])

if __name__=="__main__":
    main()
