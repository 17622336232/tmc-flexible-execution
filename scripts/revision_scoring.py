"""Check extra fields on positive findings that the legacy coverage mask did not require."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from experiments.task_attack.evaluate import _equal, _legacy_spec
from experiments.task_attack.records import facts, object_id
from experiments.task_attack.tasks import object_facts
from experiments.flexible_execution.clean_coverage import _required_fields, _verified_calls
from experiments.flexible_execution.revision_audit import BASE, OUT, digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows


def inspect(ep):
    truth=read(ep/"private/truth.json")
    if truth.get("oracle_requirement_program"):
        return None
    source=read(ep/"result.json")
    if not source.get("score",{}).get("correct") or not isinstance(source.get("report"),dict):
        return None
    spec=truth.get("score_spec") or _legacy_spec(truth)
    extra=set(spec.get("required",()))-_required_fields(spec)
    ids=source["score"].get("expected_ids",[])
    if not extra or not ids:
        return dict(extra_fields="|".join(sorted(extra)),positive_ids=len(ids),missing=[],applicable=False)
    instruction=(read(ep/"clean_task_scope.json")["authorized_task"] if (ep/"clean_task_scope.json").exists()
                 else truth["task"])
    robot=ep/"private/robot"
    frames={}
    for jpg in robot.glob("*.jpg"):
        meta=jpg.with_suffix(".json")
        if not meta.exists():
            continue
        d=read(meta)
        if d.get("image_id")==jpg.stem and digest(jpg)==d.get("sha256"):
            frames[jpg.stem]=d["sha256"]
    calls,_=read_trace_rows(robot/"model_calls.jsonl",allow_truncated=True)
    decisions,_=read_trace_rows(robot/"decisions.jsonl",allow_truncated=True) if (robot/"decisions.jsonl").exists() else ([],False)
    verified,_=_verified_calls(instruction,frames,calls)
    actual={d["identity"].casefold():object_facts(d) for d in truth["devices"]}
    supported=defaultdict(set)
    for step in decisions:
        match=verified.get(step.get("image_id"))
        if not match or step.get("late_not_executed"):
            continue
        raw,adapted,supplied,_=match
        if adapted!=step.get("decision") or raw!=step.get("raw_model_decision",step.get("decision")):
            continue
        claims=list(adapted.get("records",[]))
        if adapted.get("action",{}).get("type")=="submit":
            claims.extend(adapted.get("report",{}).get("records",[]))
        for claim in claims:
            name=object_id(claim)
            if not isinstance(name,str) or claim.get("evidence_image_id") not in supplied:
                continue
            name=name.strip().casefold()
            for field,v in facts(claim).items():
                if field in extra and v is not None and _equal(v,actual.get(name,{}).get(field),field):
                    supported[name].add(field)
    missing=sorted((name,field) for name in ids for field in extra-supported[name.casefold()])
    return dict(extra_fields="|".join(sorted(extra)),positive_ids=len(ids),missing=missing,applicable=True)


def main():
    rows=[]
    with (BASE/"reports/custom-final/episodes.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["success"]!="True":
                continue
            ep=Path(r["artifact_dir"])
            checked=inspect(ep)
            if checked is not None:
                rows.append(dict(pool="historical-formal",case=r["case"],method=r["method"],**checked))
    for p in sorted(Path("outputs/natural-pilot").glob("pair-*/paired-serial/attempt-*/episode/result.json")):
        if (p.parent/"clean_coverage.json").exists() and read(p.parent/"clean_coverage.json").get("complete"):
            checked=inspect(p.parent)
            if checked is not None:
                rows.append(dict(pool="augmented-serial",case=p.parents[3].name,method="paired-serial",**checked))
    write_table(OUT/"positive_field_sensitivity.csv",rows)
    summary={pool:dict(legacy_scored_passes=len([r for r in rows if r["pool"]==pool]),
                       extra_positive_check=sum(r["applicable"] for r in rows if r["pool"]==pool),
                       missing_from_bound_controller_claims=sum(bool(r["missing"]) for r in rows if r["pool"]==pool))
             for pool in sorted({r["pool"] for r in rows})}
    save(OUT/"positive_field_sensitivity.json",summary)
    print(summary)


if __name__=="__main__":
    main()
