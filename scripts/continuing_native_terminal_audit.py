"""Separate independent visual work from the final report-only model call."""
import argparse
from pathlib import Path

from experiments.flexible_execution.revision_audit import digest,read,save
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_native_joint_execution import evidence_claims,independent_observations,signature


def audit(root):
    summary=read(root/"summary.json")
    sealed=read(root/"sealed.json")
    registered=read(root/"registration.json")
    if summary["sealed_sha256"]!=digest(root/"sealed.json"):
        raise ValueError("The result and episode seal disagree")
    rows=[]
    for scored in summary["rows"]:
        case_id=scored["case"]
        method=scored["method"]
        case=next(v for v in registered["cases"] if v["id"]==case_id)
        key=case_id+"/"+method
        ep=root/key
        receipt=ep.with_suffix(".receipt.json")
        if digest(receipt)!=sealed[key] or read(receipt)["files"]!=signature(ep):
            raise ValueError("The native trace changed after grading")
        robot=ep/"private/robot"
        if not (robot/"timeline.jsonl").is_file() or not (robot/"model_calls.jsonl").is_file():
            rows.append(dict(case=case_id,method=method,verifiable=False))
            continue
        trace=read_trace_rows(robot/"timeline.jsonl")[0]
        calls=read_trace_rows(robot/"model_calls.jsonl")[0]
        facts=[v for v in trace if v["event"]=="joint_fact"]
        if not facts:
            rows.append(dict(case=case_id,method=method,verifiable=False))
            continue
        expected={v["field"] for v in scored["expected"]}
        results={}
        def cited(reply):
            return list(evidence_claims(reply.get("records") or []))+list(
                evidence_claims((reply.get("report") or {}).get("records") or []))
        with independent_observations(case,ep) as assess:
            for v in facts:
                field,image,value=v["field"],v["image_id"],v["value"]
                if field not in expected or digest(robot/(image+".jpg"))!=v["image_sha256"]:
                    raise ValueError("An unexpected accepted finding or an altered source")
                origins=[c for c in calls if c.get("output") and v["image_sha256"] in c["image_sha256"]
                         and c["return_monotonic_seconds"]<=v["monotonic_seconds"]
                         and (field,image,value) in cited(c["output"])]
                if not origins:
                    raise ValueError("A claimed finding did not return on its actual source")
                results[field]=bool(assess(image,field) is value)
        began=next(v["monotonic_seconds"] for v in trace if v["event"]=="observation_started")
        latest=max(v["monotonic_seconds"] for v in facts)
        report_only=[dict(service_s=c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"],
                          error=c.get("error")) for c in calls
                     if c["input"].get("this_request_scope",{}).get("open_for_reading")==[]
                     and "dispatched_monotonic_seconds" in c]
        final=calls[-1]
        merged_report=(final["input"].get("this_request_scope",{}).get("complete_after_boolean_results") is True
                       and (final.get("output") or {}).get("action",{}).get("type")=="submit"
                       and scored["strict"])
        rows.append(dict(case=case_id,method=method,verifiable=True,expected_fields=len(expected),
            independently_correct_fields=sum(results.values()),all_correct_before_report=bool(expected)
                and results==dict.fromkeys(expected,True),findings_ready_s=latest-began,
            final_pose_ok=scored.get("final_pose_ok"),report_only=report_only,
            submitted_with_last_reading=merged_report,strict=scored["strict"]))
    result=dict(new_requests=0,sealed_sha256=summary["sealed_sha256"],rows=rows)
    save(root/"terminal_audit.json",result)
    print([(r["case"],r["method"],r.get("all_correct_before_report"),r.get("report_only")) for r in rows])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("study",choices=("native-resolved-scope-dev05","native-shared-source-dev06"))
    audit(Path("outputs/continuing")/parser.parse_args().study)
