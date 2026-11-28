"""Evaluate deadline-aware and mean-time policies on the same empirical motion scenarios without model calls."""
from dataclasses import asdict
from pathlib import Path
from statistics import median

from experiments.flexible_execution.revision_audit import read,save,digest,write_table
from research.flexible_scheduling.deadline_workflow import DeadlineWorkflow
from research.flexible_scheduling.event_workflow import EventModel
from research.flexible_scheduling.live_event_room import mission,time_reference
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_room_break_even import previous_turns
from scripts.continuing_room_event_calibration import ROOT as ROOM,CAL,prepare

CFG=Path("config/deadline_coalescing_dev01.json")
ROOT=ROOM.parent/"deadline-coalescing-dev01"


def metrics(score):
    return dict(restricted_s=score.restricted/10,delivery_probability=score.delivered,
                mean_calls_until_stop=score.sends,
                mean_success_time_s=score.delivered_time_mass/score.delivered/10 if score.delivered else None)


def action(a):
    return {"motion":a.motion,"reads":a.reads}


def main():
    cfg,_=prepare()
    cal=read(CAL/"summary.json")
    past=previous_turns()
    # Earlier eligible API measurements used the same subscribed model but a different inspection prompt.
    root=ROOM.parent/"paired-rgb-assay01"
    from experiments.flexible_execution.timeline import read_trace_rows
    old=[c for p in root.glob("*/*jsonl") if p.name=="model_calls.jsonl" for c in read_trace_rows(p)[0] if len(c["image_sha256"])==1]
    if len(old)!=12 or sum((c.get("error") or {}).get("code")=="timeout" for c in old)!=1:
        raise ValueError("Hazard sensitivity anchor differs from its original twelve sends")
    h=1/len(old)
    profiles=[("no-hazard",0.,0.),("shared-call-hazard",h,h),("group-as-two-exposures",h,1-(1-h)**2)]
    motion=[]
    for c in cfg["cases"]:
        nominal=time_reference(c,cal,ROOM.parent/"room-workflow-dev01/smoke")
        for name in ("old-single-smoke","site-history","other-site-history"):
            cost=dict(nominal);counts={}
            if name!="old-single-smoke":
                direction=c["direction"]
                t1,t2=(("RotateRight","RotateLeft") if direction=="right" else ("RotateLeft","RotateRight"))
                for key,turn in (("outward",t1),("return",t2)):
                    times=[v["s"] for v in past if v["turn"]==turn and
                           ((v["house"]==c["house"])==(name=="site-history"))]
                    counts[key]=len(times)
                    if not times:raise ValueError("Missing prior measured turn costs")
                    cost[key]=median(times)
            motion.append(dict(case=c["id"],history=name,cost=cost,counts=counts))
    ROOT.mkdir(exist_ok=True)
    reg=dict(study_sha256=digest(CFG),core_sha256=digest(Path("research/flexible_scheduling/deadline_workflow.py")),
             house_protocol=digest(ROOM/"registration.json"),sensor_clock_calibration=digest(CAL/"summary.json"),
             prior_clock_logs=sorted({v["sha256"] for v in past}),motion=motion,profiles=profiles)
    assert_snapshot(ROOT/"registration.json",reg)
    rows=[]
    for scenario in motion:
        c=next(v for v in cfg["cases"] if v["id"]==scenario["case"])
        m=mission(c,scenario["cost"])
        base=DeadlineWorkflow(EventModel(m),(0.,0.,0.),2400)
        initial=base.model.initial()
        no_hazard,old_act=base.solve(initial)
        for name,h1,h2 in profiles:
            p=DeadlineWorkflow(EventModel(m),(0.,h1,h2),2400)
            g=DeadlineWorkflow(EventModel(m),(0.,h1,h2),2400,quotient=False)
            optimized,cmd=p.solve(initial)
            generic,gcmd=g.solve(initial)
            if abs(generic.restricted-optimized.restricted)>1e-5:raise ValueError("Exact search disagreement")
            old=p.fixed(initial,lambda s,t:base.solve(s,t)[1])
            common=p.solve(initial,horizon=2)[0]
            if optimized.restricted>min(old.restricted,common.restricted)+1e-6:raise ValueError("Infeasible optimization bound")
            rows.append(dict(case=c["id"],history=scenario["history"],hazard=name,h1=h1,h2=h2,
                original_policy=metrics(old),deadline_policy=metrics(optimized),generic=metrics(generic),two_dispatch=metrics(common),
                original_first=action(old_act),deadline_first=action(cmd),changed_first=cmd!=old_act,
                mean_time_optimum_at_h0_s=no_hazard.restricted/10,joint_states=p.nodes,generic_states=g.nodes))
    assert_snapshot(ROOT/"sealed_policy_expectations.json",rows)
    # The branch actually observed is used only after sealing expectations and public first actions.
    outcomes=read(ROOM/"live/summary.json")
    scored=[]
    for r in rows:
        j=next(v for v in outcomes["cases"] if v["case"]==r["case"] and v["strict"])
        observed={k:v["value"] for k,v in j["expected_fields"].items()}
        scored.append(dict(case=r["case"],history=r["history"],hazard=r["hazard"],changed_first=r["changed_first"],
            expected_restricted_gain_s=r["original_policy"]["restricted_s"]-r["deadline_policy"]["restricted_s"],
            expected_delivery_gain=r["deadline_policy"]["delivery_probability"]-r["original_policy"]["delivery_probability"],
            two_dispatch_gap_s=r["two_dispatch"]["restricted_s"]-r["deadline_policy"]["restricted_s"],
            observed_active_fields=sorted(observed)))
    write_table(ROOT/"scenario_comparisons.csv",scored)
    out=dict(scenarios=len(rows),changed_first=sum(r["changed_first"] for r in rows),
      expected_restricted_improved=sum(r["expected_restricted_gain_s"]>1e-6 for r in scored),
      maximum_expected_gain_s=max(r["expected_restricted_gain_s"] for r in scored),
      joint_generic_identical_values=len(rows),comparisons=scored,actual_model_requests=0,
      note="All costs and probabilities are model-based sensitivity results. The two timeout rates for a two-question one-image call are not estimates from an adequate sample. Previously measured motion at a specific site is available on revisits only.")
    save(ROOT/"summary.json",out)
    print(out["scenarios"],out["changed_first"],out["expected_restricted_improved"],out["maximum_expected_gain_s"])


if __name__=="__main__":
    main()
