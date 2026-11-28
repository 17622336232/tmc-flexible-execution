"""Matched task clocks and frozen-output mechanism ablations for the third native development block."""
import csv
from pathlib import Path
from statistics import median

from experiments.flexible_execution.revision_audit import digest,read,save,write_table
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.flexible_execution.adapters.native_views import Offer
from research.flexible_scheduling.coordinated_controller import CoordinatedScheduler
from scripts.continuing_native_report import length

ROOT=Path("outputs/continuing/coordinated-handoff-dev03")
DEST=ROOT/"audit"


def interface_ablation():
    rows=[]
    for path in sorted(ROOT.glob("*/*/private/robot/model_calls.jsonl")):
        robot=path.parent;ep=robot.parents[1]
        if not ep.name.startswith("coordinated-"):continue
        events=read_trace_rows(robot/"timeline.jsonl")[0]
        registered={e["id"]:e for e in events if e["event"]=="native_offer"}
        actual={e["step"]:e.get("requested_ids",[]) for e in events if e["event"]=="native_material_release"}
        for c in read_trace_rows(path)[0]:
            reply=c.get("output");ctx=c["input"]
            if not isinstance(reply,dict):continue
            offers=[]
            for o in ctx.get("offered_views",[]):
                e=registered[o["id"]]
                if digest(robot/(o["source"]+".jpg"))!=e["reference_sha256"]:raise ValueError("Changed source")
                item=Offer(o["id"],o["field"],o["subject"],tuple(o["source_bbox"]),o["source"],e["reference_sha256"],
                           o["goal"],e["independent"],e["permission"])
                exported=item.visible()
                if {**exported,"source_bbox":list(exported["source_bbox"])}!=o:raise ValueError("Different actual offer")
                offers.append(item)
            accepted={}
            for mode in ("coordinated-deferred","coordinated-observe-only"):
                # Replay the two actual controller admission functions without creating a motion or issuing a request.
                policy=object.__new__(CoordinatedScheduler)
                policy.method=mode;policy.condition=ctx["material_handoff"]["activation"]
                accepted[mode]=[o.id for o in policy.accepted_uses(reply,offers,ctx)]
            step=int(ctx["current_image_id"].split("_")[-1])
            if step in actual and actual[step]!=accepted[ep.name]:raise ValueError("Original admission not reproduced")
            rows.append(dict(case=ep.parent.name,source_method=ep.name,step=step,offers=len(offers),accepted=accepted,
                             modes_differ=len({tuple(v) for v in accepted.values()})>1))
    return rows


def describe(row):
    ep=ROOT/row["case"]/row["method"]
    meta=read(ep.parent/(ep.name+".receipt.json"))
    if (meta["result_sha256"]!=row["receipt_sha256"] or digest(ep/"result.json")!=meta["result_sha256"]
            or meta["state"]!="scored"):
        raise ValueError("Missing original terminal receipt")
    robot=ep/"private/robot"
    calls=read_trace_rows(robot/"model_calls.jsonl")[0]
    clock=[(c["dispatched_monotonic_seconds"],c["return_monotonic_seconds"]) for c in calls]
    service=length(clock)
    if abs(service-float(row["request_union_s"]))>1e-5 or abs(service-sum(b-a for a,b in clock))>.01:
        raise ValueError("This decomposition requires serial, fully metered requests")
    fields=set();events=[]
    for c in calls:
        reply=c.get("output") or {};inp=c["input"]
        new=set()
        for record in reply.get("records") or []:
            key=record.get("evidence_image_id")
            for f,v in (record.get("facts") or {}).items():
                if v is not None:
                    new.add((record.get("object_id"),f,key))
        events.append(dict(case=row["case"],method=row["method"],step=inp["current_image_id"],
           action=(reply.get("action") or {}).get("type"),service_s=c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"],
           new_cited_claims=len(new-fields),nominated=len(reply.get("next_views") or []),
           source_sha256=c["image_sha256"][-1],response_error=(c.get("error") or {}).get("code")))
        fields|=new
    whole=float(row["whole_s"]);n=len(calls)
    return dict(case=row["case"],method=row["method"],strict=row["strict"]=="True",whole_s=whole,
        calls=n,transport_s=service,mean_service_s=service/n,residual_s=whole-service,
        overlap_s=float(row["transport_with_motion_s"]),offer_count=int(row["offers"]),
        early_starts=int(row["early_starts"]),early_deliveries=int(row["early_handoffs"]),
        supported_early_claims=int(row["early_evidence_support"]),final_early_citations=int(row["early_final_citations"]),
        raw_reply_log_sha256=digest(robot/"model_calls.jsonl"),receipt_sha256=digest(ep.parent/(ep.name+".receipt.json")),
        result_sha256=meta["result_sha256"]),events


def main():
    episodes=[];events=[]
    for row in csv.DictReader((DEST/"episodes.csv").open()):
        ep,ev=describe(row);episodes.append(ep);events+=ev
    changes=[]
    for scene in sorted({d["case"] for d in episodes}):
        group=[d for d in episodes if d["case"]==scene]
        local=[r["service_s"] for r in events if r["case"]==scene]
        common=median(local)
        a=next(d for d in group if d["method"]=="coordinated-deferred")
        for b in group:
            if a is b:continue
            dn=a["calls"]-b["calls"];dt=a["whole_s"]-b["whole_s"]
            count_term=dn*(a["mean_service_s"]+b["mean_service_s"])/2
            mixture=(a["mean_service_s"]-b["mean_service_s"])*(a["calls"]+b["calls"])/2
            residual=a["residual_s"]-b["residual_s"]
            if abs(dt-count_term-mixture-residual)>1e-5:
                raise ValueError("Clock decomposition does not close")
            changes.append(dict(case=scene,comparison=b["method"],both_delivered=a["strict"] and b["strict"],
                proposed_s=a["whole_s"],baseline_s=b["whole_s"],wall_delta_s=dt,observed_request_delta=dn,
                service_total_delta_s=a["transport_s"]-b["transport_s"],count_term_s=count_term,
                service_time_and_input_mix_term_s=mixture,other_time_delta_s=residual,
                center_service_s=common,fixed_count_equal_service_delta_s=dn*common+residual,
                causal_limit="Equal-service calculation holds the observed decisions, failures and residuals fixed; changing the service clock can change a live trajectory."))
    frozen=interface_ablation()
    out=dict(episodes=episodes,pairs=changes,new_model_requests=0,
             same_reply_ablation=dict(actual_replies=len(frozen),with_offers=sum(bool(r["offers"]) for r in frozen),
                different_admissions=sum(r["modes_differ"] for r in frozen),turns=frozen),
             controlled_trace_ablation=[dict(case=r["case"],nominations=r["offer_count"],early_starts=r["early_starts"],
                 delivered=r["early_deliveries"],correct_fields_from_early=r["supported_early_claims"],
                 actual_transport_motion_overlap_s=r["overlap_s"],
                 direct_delivered_material_difference_if_withheld=0 if not r["early_deliveries"] else None,
                 invariant="With no delivered photo, disabling its delivery leaves the actual per-turn model image list unchanged in a frozen-response replay. Live navigation, prompt, clock and future responses are not interchangeable.")
                 for r in episodes if r["method"]=="coordinated-deferred"],
             interpretation="Symmetric count/rate decomposition is an accounting identity, not a randomized effect. Prompt, inputs, output timing and endpoint variance cannot be distinguished by one natural run per arm. A timeout without delivery is reported as a failure plus its observation time.")
    write_table(DEST/"attribution_turns.csv",events)
    save(DEST/"causal_clocks.json",out)
    for p in changes:
        print(p["case"],p["comparison"],*[round(p[k],2) for k in ("wall_delta_s","count_term_s","service_time_and_input_mix_term_s","other_time_delta_s")])


if __name__=="__main__":
    main()
