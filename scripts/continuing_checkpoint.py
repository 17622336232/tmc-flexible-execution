"""Persist counts from receipts, provenance checks, and the live controller heartbeat."""
import csv
from datetime import datetime, timezone
from pathlib import Path

from experiments.flexible_execution.continuing_jobs import snapshot
from experiments.flexible_execution.revision_audit import read,save
from scripts.continuing_recovery import interruptions,acknowledged


def upsert(path,title,section):
    start="<!-- continuing-snapshot -->"
    end="<!-- /continuing-snapshot -->"
    text=path.read_text()
    blob=start+"\n"+title+"\n\n"+section+"\n"+end
    if start in text:
        before,middle=text.split(start,1)
        _,after=middle.split(end,1)
        text=before+blob+after
    else:
        first,rest=text.split("\n",1)
        text=first+"\n\n"+blob+"\n"+rest
    path.write_text(text)


def main():
    data,checked,workers,cohorts=[],[],[],{}
    terminal_external=[]
    other_undelivered=[]
    unavailable=[]
    latest_beat=""
    versions={}
    planned_counts={}
    for name in ("native_views_dev01","native_evidence_dev02","coordinated_handoff_dev03"):
        path=Path("config")/(name+".json")
        if not path.exists():
            continue
        config=read(path)
        root=Path("outputs/continuing")/config["id"]
        subset=snapshot(root,config)
        terminal=interruptions(root,config,include_other=True)
        failed=[s for s in terminal if s["kind"]=="external-service-limit"]
        other_undelivered.extend(s for s in terminal if s["kind"]!="external-service-limit")
        terminal_external.extend(failed)
        if failed and not acknowledged(root,failed):
            unavailable.extend(failed)
        settled=[r for r in subset if r["state"]=="scored"]
        table=root/"audit/episodes.csv"
        details=list(csv.DictReader(table.open())) if table.exists() else []
        if len(details)!=len(settled):
            raise SystemExit("First run continuing_native_report.py --config "+str(path))
        checked.extend(details)
        data.extend(subset)
        cohorts[config["id"]]={b:sum(r["block"]==b for r in settled) for b in ("primary","comparators")}
        planned_counts[config["id"]]={b:sum(len(c[b]) for c in config["cases"]) for b in ("primary","comparators")}
        versions[config["id"]]=config.get("runtime_sha256","2e8e26cb8f0099bb68d02e5aa1c1e9569423706ec413227f9d9aa7714ffcef2b")
        if (root/"heartbeat.json").exists():
            beat=read(root/"heartbeat.json")
            latest_beat=max(latest_beat,beat.get("utc",""))
            controller=beat.get("controller_pid")
            if controller and Path("/proc/%s"%controller).exists() and beat.get("state")=="running":
                workers.append({**beat,"cohort":config["id"],"heartbeat":str(root/"heartbeat.json")})
    scored=[r for r in data if r["state"]=="scored"]
    doc=read(Path("RESEARCH_STATE.json"))
    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds")
    totals=sum(r["sends"] or 0 for r in data)
    closed=sum(r["sends"] for r in scored)
    stopped=sum(r["actual_sends"] for r in terminal_external+other_undelivered)
    live_sends=sum(r["sends"] or 0 for r in data if r["state"] in ("running","starting"))
    photos=sum(int(r["early_handoffs"]) for r in checked)
    cites=sum(int(r["early_final_citations"]) for r in checked)
    supported=sum(int(r["early_evidence_support"]) for r in checked)
    field_errors=sum(int(r["field_grant_violation_count"]) for r in checked)
    assay=Path("outputs/continuing/paired-rgb-assay01")
    reservations=assay/"sends.jsonl"
    assay_sends=len(reservations.read_text().splitlines()) if reservations.is_file() else 0
    assay_results=read(assay/"summary.json") if (assay/"summary.json").is_file() else None
    status="".join("总控 %s，%s / %s，心跳 %s。"%(w["controller_pid"],w["case"],w["method"],w["utc"])
                   for w in workers) or "当前完整任务总控已退出，最后心跳 "+latest_beat+"。"
    if unavailable:
        status+=f"有 {len(unavailable)} 回合因套餐限额中断，已发 {sum(f['actual_sends'] for f in unavailable)} 次。服务未返回重置时间。"
    if other_undelivered:
        status+=f"另有 {len(other_undelivered)} 个运行错误未交付，保留 {sum(f['actual_sends'] for f in other_undelivered)} 次发送。"
    diagnostics=Path("outputs/continuing/local-view-navigation01/summary.json")
    bindings=Path("outputs/continuing/local-material-binding01/navigation-summary.json")
    calibration=read(diagnostics) if diagnostics.exists() else None
    source_assay=read(bindings) if bindings.exists() else None
    if calibration:
        arms=calibration["arms"]
        status+=("已另做 %s 次零模型仿真路线检查，完成读号并回位为正面 %s/%s、同距离沿原方向 %s/%s。"%(
              calibration["registered"],arms["frontal_same_distance"]["grounded_read_and_return"],arms["frontal_same_distance"]["trials"],
              arms["radial_same_distance"]["grounded_read_and_return"],arms["radial_same_distance"]["trials"]))
    local=Path("outputs/continuing")
    def optional(relative):
        path=local/relative
        return read(path) if path.is_file() else {}
    timing=optional("observed-queries/wait_bounds.json")
    scope=optional("scope-closure-dev01-parserfix/summary.json")
    slots=timing.get("actual_offered_slots",{})
    if slots:
        status+=(f"已有 {slots['turns_with_selected_offer']} 次普通回复选了补图，其中 {slots['following_ordinary_actions'].get('move',0)} 次下一轮选移动。")
    new_ledger=local/"same-frame-region-assay01/sends.jsonl"
    regional_sends=len(new_ledger.read_text().splitlines()) if new_ledger.exists() else 0
    scoped_sends=sum(len(f.read_text().splitlines()) for f in (local/"scoped-mission-dev01").glob("*/*/sends.jsonl"))
    registration=optional("same-frame-region-assay01/registration.json")
    planned_assay=sum(len(c["order"]) for c in registration.get("cases",[]))
    mechanism=optional("scoped-mission-dev01/registration.json")
    if planned_assay:
        status+=f"另行登记的单图多区域感知比较实发 {regional_sends}/{planned_assay} 次。"
    if unavailable:
        next_step="等待现有 Diff-VLA 套餐显示可用，再从未发送且已登记的位置恢复。"
    elif workers:
        next_step="跟踪正在进行的 "+workers[0]["cohort"]+" 批次，逐一保留原始收据并复核完整评分。"
    else:
        next_step="从各批次收据与原有门槛检查下一项尚未发送的作业，继续当前版本的开发分析。"
    stage="paired native development and measured serial inference dependencies"
    doc.update(as_of_utc=timestamp,method_version=versions,
        new_dev=dict(scored_episodes=len(scored),strict_successes=sum(r["strict"] for r in scored),
        scored_sends=closed,in_flight_sends_at_snapshot=live_sends,closed_unscored_sends=stopped,
        started_episodes=sum(bool(r["sends"]) for r in data),blocks_completed=cohorts,
        early_source_verified_handoffs=photos,early_photos_supporting_correct_fields=supported,
        early_photos_cited_in_final=cites,early_field_violations=field_errors),
        closed_external_attempts=terminal_external,other_closed_undelivered_attempts=other_undelivered,
        unresolved_endpoint_limits=[f["code"] for f in unavailable],
        zero_model_navigation=calibration,source_binding_assay=source_assay,scoped_evidence_replay=scope,
        scoped_wait_censoring=timing.get("service"),actual_offered_action_slots=slots,
        critical_path=optional("results/critical_path.json"),matched_sparse_and_generic=optional("results/actual_offer_planners.json"),
        task_roster_on_fresh_camera=optional("scoped-mission-dev01/preflight.json"),
        optional_development=dict(region_assay_registered_sends=planned_assay,region_assay_actual_sends=regional_sends,
            closed_mechanism_registered_arms=mechanism.get("bounded_online_order",[]),closed_mechanism_actual_sends=scoped_sends),
        current_hypothesis="Test whether co-issuing a grounded photo with an ordinary move or judging several regions on one image can remove serial model service. Test public-scope assembly only with externally defined regions.",
        registered_slots=planned_counts,stage=stage,phase="coordinating actual material handoff, ordinary movement and source-bound completion",
        pending_workers=workers,paired_rgb_assay=dict(sends=assay_sends,results=assay_results),
        accounted_sends=dict(historical_formal=3842,historical_revision_development=93,
          continued_native_development=totals,archived_rgb_assay=assay_sends,one_image_regions=regional_sends,
          explicit_roster_mechanism=scoped_sends),
        next=next_step)
    save(Path("RESEARCH_STATE.json"),doc)
    tally="；".join(f"{k} 主对照 {v['primary']}/{planned_counts[k]['primary']}、强对照 {v['comparators']}/{planned_counts[k]['comparators']}" for k,v in cohorts.items())
    stat=f"UTC {timestamp[:19]}。{tally} 已取得完整评分。严格通过 {sum(r['strict'] for r in scored)}/{len(scored)}。新完整任务累计实发 {totals} 次，其中取得评分的回合 {closed} 次。独立核对提前交图 {photos} 次，其中 {supported} 张支持正确字段，最终报告直接引用提前图 {cites} 次。单独的历史照片感知试验实发 {assay_sends} 次。{status}"
    table=["| 场景 | 方法 | 原评分 | 完整通过 | 完整用时（秒） | 按期限计分（秒） | 真实请求 |","|---|---|---|---|---:|---:|---:|"]
    labels={"cracked-row":"设备外壳","warehouse-filter":"货位筛查","native-deferred":"提前采集、延迟判读",
          "native-off":"同优化串行","original-gpt":"原版 GPT","native-prefetch":"强条件预取",
          "native-async":"普通异步","native-planner":"通用规划"}
    labels.update({"cracked-row-02":"新种子设备外壳","warehouse-filter-02":"新种子货位筛查",
        "evidence-deferred":"证据调度","evidence-off":"同优化串行","evidence-async":"普通异步",
        "evidence-prefetch":"强条件预取","evidence-planner":"通用规划"})
    for r in checked:
        table.append("| %s | %s | %s | %s | %.2f | %.2f | %s |"%(labels.get(r["case"],r["case"]),labels.get(r["method"],r["method"]),
              r["legacy"],"是" if r["strict"]=="True" else "否",float(r["whole_s"]),float(r["restricted_s"]),r["requests"]))
    for f in terminal_external:
        table.append("| %s | %s | 套餐限额 | 未交付 | %.2f | %.2f | %d |"%(labels.get(f["case"],f["case"]),
              labels.get(f["method"],f["method"]),f["task_seconds"],f["restricted_if_counted_s"],f["actual_sends"]))
    for f in other_undelivered:
        table.append("| %s | %s | %s | 未交付 | %.2f | %.2f | %d |"%(labels.get(f["case"],f["case"]),
            labels.get(f["method"],f["method"]),f["code"],f["task_seconds"],f["restricted_if_counted_s"],f["actual_sends"]))
    upsert(Path("PROGRESS.md"),"## 持续研发最新实测",stat+"\n\n"+"\n".join(table))
    upsert(Path("NEXT_ACTION.md"),"## 最近一次恢复位置",stat+"\n\n"+next_step+"在仓库根目录使用项目 Python 环境运行 `scripts/continuing_recovery.py` 可零请求列出第二批主对照断点，追加 `--block comparators` 查看强对照。确认同一套餐恢复后可用 `--run --restoration-evidence '实际恢复凭据或确认时间'` 恢复首个未发送位置。用 `scripts/continuing_native_report.py --config config/native_evidence_dev02.json` 独立重查，再运行 `scripts/continuing_checkpoint.py` 更新统计。")
    manuscript=Path("docs/DEVELOPMENT_EVIDENCE_CN.md")
    if manuscript.exists():
        upsert(manuscript,"## 当前完整任务样本",f"截至 {timestamp[:19]} UTC，新的自然在线开发共有 {sum(bool(r['sends']) for r in data)} 个已启动回合，{len(scored)} 个已有完整评分，其中 {sum(r['strict'] for r in scored)} 个严格通过。全部已启动回合累计发送 {totals} 次。以下时间下限与决策重算使用恢复前封存的 16 个有评分回合。逐回合新结果和中断记录见[完整进度表](../PROGRESS.md)。")
    print(stat)


if __name__=="__main__":
    main()
