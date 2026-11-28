"""Relate fresh request windows and accepted artifacts to complete-task outcomes."""
from collections import Counter
from pathlib import Path

from experiments.flexible_execution.live_audit import audit_capture_permissions
from experiments.flexible_execution.pilot_report import _mechanism
from experiments.flexible_execution.revision_audit import OUT, read, write_table
from experiments.flexible_execution.timeline import read_trace_rows


def main():
    rows=[]
    for scope in (OUT, OUT/"round3"):
        for file in sorted(scope.glob("dev-*/*.receipt.json")):
            rec=read(file)
            if rec["state"]!="scored":
                continue
            ep=file.with_name(file.name.replace(".receipt.json",""))
            robot=ep/"private/robot"
            ev,_=read_trace_rows(robot/"timeline.jsonl",allow_truncated=True)
            calls,_=read_trace_rows(robot/"model_calls.jsonl",allow_truncated=True)
            report=read(ep/"result.json")["report"] if rec["strict"] else {}
            metrics=_mechanism(ep,ev,report,True)
            hand=next((r for r in ev if r["event"]=="full_inspection_started"),{})
            bound=hand.get("monotonic_seconds")
            later=[c for c in calls if bound is not None and c.get("dispatched_monotonic_seconds",0)>=bound]
            perm=audit_capture_permissions(ep,allow_truncated=True)
            rows.append(dict(block="round3" if scope!=OUT else "first",case=ep.parent.name,method=ep.name,
                pass_strict=rec["strict"],transport_calls=len(calls),
                controller_handoff_elapsed_s=bound-ev[0]["monotonic_seconds"] if bound is not None else None,
                post_handoff_calls=len(later) if bound is not None else None,
                post_handoff_transport_s=sum(c.get("wall_seconds",0) for c in later) if bound is not None else None,
                total_returned_transport_s=sum(c.get("wall_seconds",0) for c in calls),
                seeded_records=hand.get("seeded_records"),accepted_front_materials=hand.get("accepted_active_materials"),
                applicable_permission_actions=perm["audited"] if perm else None,
                permission_violations=perm["violations"] if perm else None,
                **{k:metrics[k] for k in ("conditional_moves_started_before_bound_reply",
                   "conditional_photos_completed_before_bound_reply","conditional_early_materials_used_in_report")}))
    write_table(OUT/"fresh_mechanism.csv",rows)
    print([(r["block"],r["case"],r["conditional_moves_started_before_bound_reply"],
            r["conditional_early_materials_used_in_report"]) for r in rows if r["method"]=="conditional-a"])


if __name__=="__main__":
    main()
