"""When did the actually recorded, source-bound fields suffice for delivery?
Exposure-only retrospective upper bound. The policy never sees evaluator membership labels.
"""
from pathlib import Path
import re
from collections import Counter
from hashlib import sha256

from experiments.flexible_execution.revision_audit import read,write_table,save,digest
from experiments.flexible_execution.timeline import read_trace_rows


def public_filter(task):
    first=task.splitlines()[0]
    if "报告存在裂纹的设备编号" in first:
        return "cracked",lambda v:type(v) is bool,lambda v:v
    match=re.search(r"报告实际箱数严格多于([0-9]+)箱的货位编号",first)
    if match:
        threshold=int(match[1])
        return "box_count",lambda v:type(v) is int and v>=0,lambda v:v>threshold
    return None


def episode(ep):
    robot=ep/"private/robot"
    task=read(ep/"clean_task_scope.json")["authorized_task"]
    rule=public_filter(task)
    if not rule:
        return None
    key,is_type,include=rule
    calls=read_trace_rows(robot/"model_calls.jsonl")[0]
    decisions=read_trace_rows(robot/"decisions.jsonl")[0] if (robot/"decisions.jsonl").exists() else []
    executed={r["image_id"]:r for r in decisions if not r.get("late_not_executed")}
    records={}
    drafts=[]
    for i,c in enumerate(calls,1):
        inp=c["input"]
        current=inp["current_image_id"]
        supplied=[v["image_id"] for v in inp["recent_images"]]+[current]
        if [digest(robot/(f+".jpg")) for f in supplied]!=c["image_sha256"]:
            raise ValueError("Raw input cannot be bound")
        raw=c.get("output")
        accepted=executed.get(current,{})
        if not raw or accepted.get("raw_model_decision",accepted.get("decision"))!=raw:
            continue
        terminal=(raw.get("action") or {}).get("type")=="submit"
        asserted=list(raw.get("records",[]) or [])
        if terminal:
            asserted.extend((raw.get("report") or {}).get("records",[]) or [])
        # Every new claim must be explicitly present in a genuine response and refer to
        # an actually supplied photo. Later explicit values supersede an earlier one.
        for row in asserted:
            ident,src=row.get("object_id"),row.get("evidence_image_id")
            value=(row.get("facts") or {}).get(key)
            if isinstance(ident,str) and src in supplied and is_type(value):
                records[ident]=(value,src)
        drafts.append(dict(attempt=i,time=c["return_monotonic_seconds"],terminal=terminal,
                          records=dict(records),reported=sorted(k for k,(v,_) in records.items() if include(v))))
    # Only the offline referee goes below this line to see the full physical roster.
    truth=read(ep/"private/truth.json")
    expected={x["identity"]:x[key] for x in truth["devices"]}
    valid=[d for d in drafts if set(d["records"])==set(expected) and all(
           d["records"][k][0]==v for k,v in expected.items())]
    first=valid[0] if valid else None
    end=calls[-1]["return_monotonic_seconds"] if calls else None
    return dict(case=ep.parent.name,method=ep.name,field=key,requests=len(calls),
            first_retrospectively_complete=first["attempt"] if first else None,
            complete_before_last=bool(first and first["attempt"]<len(calls)),
            actual_later_requests=(len(calls)-first["attempt"]) if first else None,
            actual_later_request_service_s=(sum(c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"]
                 for c in calls[first["attempt"]:]) if first else None),
            observed_tail_to_last_reply_s=end-first["time"] if first else None,
            retained_records_at_last=len(drafts[-1]["records"]) if drafts else 0,scope_size=len(expected),
            source_log=digest(robot/"model_calls.jsonl"))


def main():
    dest=Path("outputs/continuing/results")
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for root in sorted(Path("outputs/continuing").glob("native-*-dev*")):
        for rec in sorted(root.glob("*/*.receipt.json")):
            d=read(rec)
            if d.get("state")!="scored":
                continue
            r=episode(rec.with_name(d["method"]))
            if r:
                rows.append(dict(cohort=root.name,**r))
    if rows:
        write_table(dest/"record_sufficiency_prefixes.csv",rows)
    result=dict(exposed_episodes=len(rows),ever_sufficient=sum(r["first_retrospectively_complete"] is not None for r in rows),
                sufficient_strictly_before_last=sum(r["complete_before_last"] for r in rows),
                zero_additional_requests=True,scope="Fixed actual decision outputs. Retrospective full roster is evaluator-only. No deployable stopping certificate is inferred.")
    save(dest/"record_sufficiency_prefixes.json",result)
    print(result)


if __name__=="__main__":
    main()
