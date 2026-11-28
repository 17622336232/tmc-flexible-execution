"""Rebuild aggregate comparison, resource coverage and two-domain evidence figures without inference."""
import csv
from collections import defaultdict
from math import dist
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.flexible_execution.revision_audit import read,save,write_table,digest
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_recovery import interruptions


OUT=Path("outputs/continuing/results")
COHORTS=("native-views-dev01","native-evidence-dev02","coordinated-handoff-dev03")
NAMES={"original-gpt":"Original GPT","off":"Enhanced serial","async":"Ordinary async","prefetch":"Strong prefetch",
       "planner":"Generic planner","deferred":"Proposed","observe-only":"Observe-only ablation","no-ocr":"No local reading"}
ORDER=["Original GPT","Enhanced serial","Ordinary async","Strong prefetch","Generic planner","Proposed"]


def family(arm):
    return NAMES[arm if arm=="original-gpt" else arm.split("-",1)[1]]


def samples(root):
    groups=defaultdict(list)
    for path in root.glob("resources-*.jsonl"):
        for v in read_trace_rows(path,allow_truncated=True)[0]:
            groups[(v["case"],v["method"])].append(v)
    return groups


def timeline_summary(ep,sampled):
    robot=ep/"private/robot"
    calls=read_trace_rows(robot/"model_calls.jsonl")[0] if (robot/"model_calls.jsonl").is_file() else []
    durations=[c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"] for c in calls]
    records=read_trace_rows(ep/"private/events.jsonl")[0]
    trace=read_trace_rows(robot/"timeline.jsonl")[0]
    check=[e["planning_seconds"] for e in trace if "planning_seconds" in e]
    # Multiple legs in one step have separate action intervals and could share display labels.
    excursions=sum(e["event"]=="native_early_start" for e in trace)
    begin=next((r["wall"] for r in records if r["event"]=="takeoff_complete"),None)
    end=next((r["wall"] for r in reversed(records) if r["event"] in ("episode_teardown_started","episode_complete")),None)
    meta=read(ep/"independent_scoring_time.json") if (ep/"independent_scoring_time.json").is_file() else {}
    poses=[]
    for path in robot.glob("onboard_*.json"):
        v=read(path)
        if all(k in v for k in ("ros_stamp","position","sha256")):
            poses.append((v["ros_stamp"],tuple(v["position"][k] for k in ("x","y","z"))))
    points=[p for _,p in sorted(poses)]
    revisits=sum(dist(a,b)>.8 and any(dist(b,c)<=.4 for c in points[:i]) for i,(a,b) in enumerate(zip(points,points[1:])))
    return dict(request_p95_s=float(np.percentile(durations,95)) if durations else None,
         sampled_image_path_m=sum(dist(a,b) for a,b in zip(points,points[1:])),sampled_pose_revisits=revisits,
         early_excursions=excursions,planning_measured_calls=len(check),planning_logged_ms=sum(check)*1000 if check else None,
         scoring_s=meta["end_monotonic"]-meta["start_monotonic"] if meta else None,
         simulated_takeoff_complete_to_teardown_s=end-begin if end and begin else None,
         resource_samples=len(sampled),resource_window_s=(max(v["monotonic"] for v in sampled)-min(v["monotonic"] for v in sampled)) if sampled else None,
         sampled_peak_rss_sum_mib=max((v["rss_sum_bytes"]/2**20 for v in sampled),default=None),
         sampled_max_alive_cpu_s=max((v["alive_cpu_seconds_sum"] for v in sampled),default=None))


def collect():
    episodes=[];measures=[];paired=[]
    for cohort in COHORTS:
        root=OUT.parent/cohort
        path=root/"audit/episodes.csv"
        if not path.exists():
            continue
        groups=samples(root)
        for r in csv.DictReader(path.open()):
            method=family(r["method"])
            info=dict(cohort=cohort,case=r["case"],method=method)
            episodes.append(dict(**info,passed=r["strict"]=="True",whole_s=float(r["whole_s"]),
                   restricted_s=float(r["restricted_s"]),requests=int(r["requests"]),transport_s=float(r["request_union_s"]),
                   concurrent_motion_s=float(r["transport_with_motion_s"]),idle_transport_s=float(r["transport_without_motion_s"]),
                   first_valid_move_start_s=float(r["first_completed_motion_start_s"]) if r["first_completed_motion_start_s"] else None,
                   early_handoff=int(r["early_handoffs"]),early_field_support=int(r["early_evidence_support"]),
                   early_final_citation=int(r["early_final_citations"]),field_violations=int(r["field_grant_violation_count"])))
            ep=root/r["case"]/r["method"]
            measures.append(dict(**info,**timeline_summary(ep,groups[(r["case"],r["method"])])))
        for case in sorted({r["case"] for r in episodes if r["cohort"]==cohort}):
            g={r["method"]:r for r in episodes if r["cohort"]==cohort and r["case"]==case}
            if "Proposed" not in g:continue
            a=g["Proposed"]
            for m in (key for key in g if key!="Proposed"):
                if m in g:
                    b=g[m]
                    paired.append(dict(cohort=cohort,case=case,baseline=m,proposed_pass=a["passed"],baseline_pass=b["passed"],
                                delta_restricted_s=a["restricted_s"]-b["restricted_s"],
                                delta_restricted_pct=100*(a["restricted_s"]/b["restricted_s"]-1),
                                delta_transport_s=a["transport_s"]-b["transport_s"],delta_requests=a["requests"]-b["requests"]))
    return episodes,measures,paired


def figure(episodes):
    cases=list(dict.fromkeys((r["cohort"],r["case"]) for r in episodes))
    fig,axes=plt.subplots(len(cases),1,figsize=(8.3,max(2.8,2.65*len(cases))),squeeze=False)
    for ax,(group,case) in zip(axes[:,0],cases):
        g={r["method"]:r for r in episodes if r["cohort"]==group and r["case"]==case}
        order=ORDER+[m for m in g if m not in ORDER]
        for i,m in enumerate(order):
            if m not in g:continue
            r=g[m]
            ax.barh(i,r["whole_s"],height=.59,color="#367b74" if r["passed"] else "#b36d40")
            if not r["passed"]:
                ax.barh(i,r["restricted_s"]-min(r["whole_s"],r["restricted_s"]),left=min(r["whole_s"],r["restricted_s"]),
                        height=.59,facecolor="none",edgecolor="#b36d40",hatch="///",lw=.7)
            detail=" · "+r["failure_label"] if r.get("failure_label") else ""
            ax.text(r["whole_s"]+3,i,f"{r['whole_s']:.0f} s"+detail,va="center",fontsize=8)
        ax.set_yticks(range(len(order)),order)
        ax.invert_yaxis();ax.set_xlim(0,390)
        ax.set_title(case+"  |  "+group,loc="left",fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlabel("Task wall time (s). Hatching extends failed deliveries to the deadline.",fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/"paired_task_times.png",dpi=145)
    plt.close(fig)


def all_started_pairs(episodes,stopped):
    by_case=defaultdict(dict)
    for r in episodes:
        by_case[r["case"]][r["method"]]=dict(strict=r["passed"],restricted=r["restricted_s"],external=None)
    for f in stopped:
        by_case[f["case"]][family(f["method"])]=dict(strict=False,restricted=f["restricted_if_counted_s"],external=f["code"])
    rows=[]
    for case,methods in by_case.items():
        if "Proposed" not in methods:
            continue
        p=methods["Proposed"]
        for name,b in methods.items():
            if name=="Proposed":
                continue
            rows.append(dict(case=case,baseline=name,proposed_delivered=p["strict"],baseline_delivered=b["strict"],
                     delta_deadline_restricted_s=p["restricted"]-b["restricted"],proposed_interruption=p["external"],
                     baseline_interruption=b["external"],both_algorithm_outcomes_scored=not (p["external"] or b["external"])))
    return rows


def fixed_service_figure():
    path=OUT/"critical_path.json"
    if not path.is_file():
        return
    pairs=sorted(read(path)["paired_strict_device_tasks"],key=lambda r:r["case"])
    if not pairs:
        return
    fig,axes=plt.subplots(len(pairs),1,figsize=(7.2,2.18*len(pairs)),squeeze=False,layout="constrained")
    for index,(ax,r) in enumerate(zip(axes[:,0],pairs),1):
        target=r["target_wall_s"]
        ax.axvline(target,color="#994219",lw=1.35,ls="--",zorder=3)
        ax.text(target+.8,1.56,f"20% target  {target:.1f} s",color="#783716",fontsize=8)
        ax.set_title(f"Equipment layout {index}",loc="left",fontsize=10,fontweight="bold")
        for y,total,model in ((1,r["original_time_s"],r["original_fixed_transport_s"]),
                              (0,r["observed_proposed_time_s"],r["proposed_fixed_transport_s"])):
            ax.barh(y,model,height=.52,color="#276777",label="Model transport" if y==1 else None)
            ax.barh(y,total-model,height=.52,left=model,color="#bdd4da",label="Remaining task time" if y==1 else None)
            ax.text(model/2,y,f"{model:.1f} s",ha="center",va="center",color="white",fontsize=9)
            ax.text(total+1.6,y,f"{total:.1f} s total",va="center",fontsize=8)
        ax.set(yticks=[1,0],yticklabels=["Original GPT","Proposed"],xlim=(0,204),ylim=(-.55,1.90))
        ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0)
        ax.set_xlabel("Wall time (seconds)",fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False,fontsize=8)
    for ext in ("png","pdf"):
        fig.savefig(OUT/("fixed_transport_floors."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    episodes,measures,paired=collect()
    if episodes:write_table(OUT/"task_results.csv",episodes)
    if measures:write_table(OUT/"resources_latency.csv",measures)
    if paired:write_table(OUT/"paired_differences.csv",paired)
    fixed_service_figure()
    assay_path=OUT.parent/"paired-rgb-assay01/summary.json"
    scene_path=OUT.parent/"camera-pairs/summary.json"
    brief=dict(role="development-only",cohorts={},perception=read(assay_path) if assay_path.exists() else None,
               measured_viewpoints=read(scene_path) if scene_path.exists() else None,
               limitations=["No measured hardware motion or energy; simulator clock only","No usage-token telemetry from this transport",
                            "Repeated methods in one layout are paired, not independent houses",
                            "Partial process windows; summed RSS may double-count shared pages; alive CPU is a snapshot",
                            "Motion overlap is an interval observation, not causal credit"])
    sensitivity=[]
    plotted=list(episodes)
    for g in COHORTS:
        batch=[r for r in episodes if r["cohort"]==g]
        name={"native-views-dev01":"native_views_dev01","native-evidence-dev02":"native_evidence_dev02",
              "coordinated-handoff-dev03":"coordinated_handoff_dev03"}[g]
        config=read(Path("config")/(name+".json"))
        stopped=interruptions(OUT.parent/g,config,include_other=True)
        for f in stopped:
            plotted.append(dict(cohort=g,case=f["case"],method=family(f["method"]),whole_s=f["task_seconds"],
                passed=False,restricted_s=f["restricted_if_counted_s"],
                failure_label="service limit" if f["kind"]=="external-service-limit" else "run error"))
        all_pairs=all_started_pairs(batch,stopped)
        sensitivity.extend(dict(cohort=g,**p) for p in all_pairs)
        brief["cohorts"][g]=dict(scored=len(batch),strict=sum(r["passed"] for r in batch),sends=sum(r["requests"] for r in batch),
                                started=len(batch)+len(stopped),all_started_sends=sum(r["requests"] for r in batch)+sum(f["actual_sends"] for f in stopped),
                                service_interrupted=[s for s in stopped if s["kind"]=="external-service-limit"],
                                other_undelivered=[s for s in stopped if s["kind"]!="external-service-limit"],
                                episodes=batch,paired=[p for p in paired if p["cohort"]==g],all_started_delivery_pairs=all_pairs)
    if sensitivity:
        write_table(OUT/"all_started_delivery_pairs.csv",sensitivity)
    if plotted:
        figure(plotted)
    for name,file in {
        "prefix_audit":OUT/"record_sufficiency_prefixes.json",
        "static_crossworld_views":OUT.parent/"archived-views-crossworld01/summary.json",
        "closed_loop_local_navigation":OUT.parent/"local-view-navigation01/summary.json",
        "static_binding":OUT.parent/"local-material-binding01/static-summary.json",
        "real_simulator_camera_binding":OUT.parent/"local-material-binding01/navigation-summary.json",
        "finite_workflow":OUT.parent/"workflow/balanced_summary.json",
        "closed_roster_evidence_composition":OUT.parent/"scope-closure-dev01-parserfix/summary.json",
        "observed_query_concurrency":OUT.parent/"observed-queries/summary.json",
        "censored_specialized_service":OUT.parent/"observed-queries/wait_bounds.json",
        "real_offer_planners":OUT/"actual_offer_planners.json",
        "frozen_ranker_relative_order":OUT/"old_ranker_on_actual_offers.json",
        "learned_demand_on_real_training_decisions":OUT.parent/"view-demand-dev01/training_summary.json",
        "frozen_demand_on_future_campus":OUT.parent/"view-demand-dev01/campus_summary.json",
        "cross_fitted_demand_action_changes":OUT.parent/"view-demand-dev01/policy_summary.json",
        "observed_approach_refusals":OUT/"approach_audit.json",
        "same_frame_region_requests":OUT.parent/"same-frame-region-assay01/summary.json",
        "measured_condition_range":OUT.parent/"condition-range-dev01/summary.json",
        "scoped_live_evidence":OUT.parent/"scoped-mission-dev01/mechanism_results.json",
        "direct_inspection_travel_calibration":OUT.parent/"inspection-travel-dev01/calibration/summary.json",
        "measured_return_travel_calibration":OUT.parent/"observed-return-dev02/calibration/summary.json",
        "measured_return_travel_live":OUT.parent/"observed-return-dev02/live/summary.json",
        "reference_schedule_on_real_travel":OUT.parent/"observed-return-dev02/schedule_reference.json",
        "conditional_first_decision_milestones":OUT.parent/"first-view-contingency-dev01/replay_summary.json",
        "native_v3_clock_decomposition_and_frozen_reply_ablation":OUT.parent/"coordinated-handoff-dev03/audit/causal_clocks.json",
        "concurrent_workflow_benchmark":OUT.parent/"concurrent-workflow-dev01/summary.json",
        "matched_room_calibration":OUT.parent/"matched-event-rooms-dev01/calibration/summary.json",
        "matched_room_prior_time_bounds":OUT.parent/"matched-event-rooms-dev01/prior_bounds.json",
        "matched_room_live_tasks":OUT.parent/"matched-event-rooms-dev01/live/summary.json",
        "matched_room_same_source_break_even":OUT.parent/"matched-event-rooms-dev01/causal_branch.json",
        "stronger_equal_wall_clock_comparison":OUT.parent/"event-compute-dev02/summary.json",
        "native_semantic_frontier":OUT/"native_semantic_frontier.json",
        "deadline_sensitive_workflow":OUT.parent/"deadline-coalescing-dev01/summary.json",
        "ordinary_section_proposals":OUT.parent/"native-section-dev01/proposal_summary.json",
        "ordinary_action_and_source_binding":OUT.parent/"native-action-binding-dev01/summary.json",
        "ordinary_camera_acquisition_clocks":OUT.parent/"native-acquisition-clocks-dev01/summary.json",
        "house_56_task_critical_path":OUT.parent/"online-cost-dev01/clock_audit.json",
        "house_56_order_cost_envelope":OUT.parent/"online-cost-dev01/choice_envelope.json",
        "cross_source_calibration_development":OUT.parent/"cross-source-assay-dev02/summary.json",
        "cross_source_service_transfer":OUT.parent/"online-cost-dev01/cross_service_transfer.json",
        "native_unnumbered_task_discovery":OUT.parent/"native-natural-frontier-dev01/summary.json",
        "native_discovered_schedule_replay":OUT.parent/"online-cost-dev01/native_schedule.json",
        "dynamic_permission_reuse_node_budget":OUT.parent/"online-cost-dev01/permission-cache/summary.json",
        "dynamic_permission_reuse_wall_budget":OUT.parent/"online-cost-dev01/permission-cache/wall/summary.json",
        "dynamic_permission_reuse_ablation":OUT.parent/"online-cost-dev01/permission-cache/ablation/summary.json",
        "rechecked_dynamic_permission_budget":OUT.parent/"online-cost-dev01/permission-cache/budget-v2-guarded/summary.json",
        "three_question_order_switch":OUT.parent/"online-cost-dev01/switch-witness/summary.json",
        "native_bound_followup":OUT.parent/"native-natural-followup-dev01/summary.json",
        "fixed_transport_critical_path":OUT/"critical_path.json",
        "same_photo_claim_comparison":OUT.parent/"same-image-claims/summary.json",
        "observed_camera_interface":OUT/"actual_camera_interface.json",
        "public_roster_on_fresh_camera":OUT.parent/"scoped-mission-dev01/preflight.json",
    }.items():
        if file.is_file():
            brief[name]=read(file)
    optics=OUT.parent/"observed-return-dev02/optics.json"
    if optics.is_file():
        d=read(optics)
        brief["optical_registration_controls"]={k:d[k] for k in ("groups","allow_rotated","source_registration_sha256","grader_sha256")}
    save(OUT/"development_summary.json",brief)
    save(Path("docs/continuing_development_results.json"),brief)
    print("scored",len(episodes),"paired differences",len(paired),"simulator process resource windows",sum(r["resource_samples"]>0 for r in measures))


if __name__=="__main__":
    main()
