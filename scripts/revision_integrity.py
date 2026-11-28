"""Check recorded symbolic traces, original source files and online request receipts."""
import csv
import json
from dataclasses import asdict
from pathlib import Path

from experiments.flexible_execution.evaluate import score
from experiments.flexible_execution.fixtures import Diagnostic, QueryFixture
from experiments.flexible_execution.revision_audit import BASE, OUT, digest, read, save, write_table
from experiments.flexible_execution.task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract
from scripts.revision_mechanisms import fixture


def from_stored(raw):
    t=raw["task"]
    def gate(c):
        return Condition(c["query_id"],tuple(c["outcomes"])) if c else None
    qs=tuple(QuerySpec(q["id"],q["object_id"],tuple(q["outcomes"]),q["estimated_ready"],q["latest_ready"]) for q in t["queries"])
    req=tuple(Requirement(**{**r,"when":gate(r["when"])}) for r in t["requirements"])
    skills=tuple(Skill(**{**s,"covers":tuple(s["covers"]),"permission":gate(s["permission"])}) for s in t["skills"])
    task=TaskContract(t["instruction"],t["start"],t["terminal"],t["deadline"],t["report_seconds"],
                      req,qs,skills,tuple(tuple(e) for e in t["travel"]))
    return Diagnostic(raw["id"],raw["mechanism"],raw["legacy_family"],task,
                      tuple(QueryFixture(**q) for q in raw["query_truth"]),tuple(tuple(v) for v in raw["initial_material"]),
                      raw["material_values"],tuple(raw["reject_first"]))


def verify_traces():
    with (OUT/"symbolic_development.csv").open() as f:
        rows={(r["case"],r["method"]):r for r in csv.DictReader(f)}
    seen=set()
    with (OUT/"symbolic_development.jsonl").open() as f:
        for line in f:
            entry=json.loads(line)
            key=entry["case"],entry["method"]
            r=rows[key]
            case=fixture(r["type"],int(r["latency"]),float(r["unit_travel"]),r["scope_grant"]=="True")
            assert json.loads(json.dumps(asdict(case.task)))==entry["contract"]
            verdict=score(case,entry["events"])
            assert verdict["success"]==(r["success"]=="True") and not verdict["trace_errors"]
            assert verdict["restricted_completion_seconds"]==float(r["restricted_completion_seconds"]) and key not in seen
            seen.add(key)
    assert set(rows)==seen and len(seen)==600
    a=OUT/"ablation-verified"
    with (a/"mechanism_ablations.csv").open() as f:
        ab={(r["case"],r["variant"]):r for r in csv.DictReader(f)}
    reviewed=set()
    with (a/"events.jsonl").open() as f:
        for line in f:
            e=json.loads(line)
            key=e["case"],e["variant"]
            verdict=score(from_stored(e["fixture"]),e["events"])
            assert verdict["success"]==(ab[key]["success"]=="True") and not verdict["trace_errors"]
            assert verdict["restricted_completion_seconds"]==float(ab[key]["time_s"]) and key not in reviewed
            reviewed.add(key)
    assert set(ab)==reviewed and len(ab)==96
    return len(seen),len(reviewed)


def main():
    symbolic,ablations=verify_traces()
    for d in ("custom","public"):
        base=BASE/"reports"/(d+"-final")
        hashes=read(BASE/"formal-progress/final-integrity.json")["domains"][d]["report_hashes"]
        assert all(digest(base/f)==h for f,h in hashes.items())
    receipts=[]
    sources=[]
    for root in (OUT,OUT/"round3"):
        for p in sorted(root.glob("dev-*/*.receipt.json")):
            rec=read(p)
            if rec["state"]!="scored":
                raise ValueError("Unsettled receipt: "+str(p))
            ep=p.with_name(p.name.replace(".receipt.json",""))
            raw=read(ep/"clean_preflight.json")
            assert raw["backend"]=="cpu" and raw["model"]=="gpt-6.1-terra" and raw["reasoning_effort"]=="max"
            assert digest(ep/"result.json")==rec["result_sha256"]
            ledger=[json.loads(l) for l in (ep/"request_attempts.jsonl").read_text().splitlines()]
            calls=[json.loads(l) for l in (ep/"private/robot/model_calls.jsonl").read_text().splitlines()]
            assert len(calls)==len(ledger)==rec["sends"]<=24
            assert all(c["model"]=="gpt-6.1-terra" and c["reasoning_effort"]=="max" and c["role"]=="robot"
                       for c in calls)
            assert all((r["reservation"],r["image_sha256"],r["model"],r["reasoning_effort"])==
                       (c["logical_job_attempt"],c["image_sha256"],c["model"],c["reasoning_effort"]) for r,c in zip(ledger,calls))
            receipts.append(dict(block=str(root.relative_to(OUT)),case=p.parent.name,method=ep.name,sends=rec["sends"],
                                 raw_result_sha256=rec["result_sha256"],request_ledger_sha256=digest(ep/"request_attempts.jsonl"),
                                 model_calls_sha256=digest(ep/"private/robot/model_calls.jsonl"),
                                 preflight_sha256=digest(ep/"clean_preflight.json")))
            for f in sorted(ep.rglob("*")):
                if f.is_file() and ("ros_home" not in f.parts) and f.suffix.lower() in (".json",".jsonl",".jpg",".png",".world",".gz"):
                    sources.append(dict(path=str(f.relative_to(OUT)),sha256=digest(f),size_bytes=f.stat().st_size))
    assert len(receipts)==11
    write_table(OUT/"development_receipts.csv",receipts)
    write_table(OUT/"development_raw_inventory.csv",sources)
    summary=dict(historical_episodes=456,historical_reports_hashes_match=True,
                 symbolic_regraded=symbolic,ablations_regraded=ablations,development_episodes=len(receipts),
                 natural_sends=sum(r["sends"] for r in receipts),raw_files=len(sources))
    save(OUT/"integrity.json",summary)
    print(summary)


if __name__=="__main__":
    main()
