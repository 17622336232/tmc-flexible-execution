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
    unavailable=[]
    latest_beat=""
    versions={}
    for name in ("native_views_dev01","native_evidence_dev02"):
        path=Path("config")/(name+".json")
        if not path.exists():
            continue
        config=read(path)
        root=Path("outputs/continuing")/config["id"]
        subset=snapshot(root,config)
        failed=interruptions(root,config)
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
    stopped=sum(r["actual_sends"] for r in terminal_external)
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
        status+=f"有 {len(unavailable)} 回合因套餐限额中断，已发 {stopped} 次。服务未返回重置时间。"
    diagnostics=Path("outputs/continuing/local-view-navigation01/summary.json")
    bindings=Path("outputs/continuing/local-material-binding01/navigation-summary.json")
    calibration=read(diagnostics) if diagnostics.exists() else None
    source_assay=read(bindings) if bindings.exists() else None
    if calibration:
        arms=calibration["arms"]
        status+=("已另做 %s 次零模型仿真路线检查，完成读号并回位为正面 %s/%s、同距离沿原方向 %s/%s。"%(
              calibration["registered"],arms["frontal_same_distance"]["grounded_read_and_return"],arms["frontal_same_distance"]["trials"],
              arms["radial_same_distance"]["grounded_read_and_return"],arms["radial_same_distance"]["trials"]))
    doc.update(as_of_utc=timestamp,method_version=versions,
        new_dev=dict(scored_episodes=len(scored),strict_successes=sum(r["strict"] for r in scored),
        scored_sends=closed,in_flight_sends_at_snapshot=live_sends,closed_unscored_sends=stopped,
        started_episodes=sum(bool(r["sends"]) for r in data),blocks_completed=cohorts,
        early_source_verified_handoffs=photos,early_photos_supporting_correct_fields=supported,
        early_photos_cited_in_final=cites,early_field_violations=field_errors),
        closed_external_attempts=terminal_external,unresolved_endpoint_limits=[f["code"] for f in unavailable],
        zero_model_navigation=calibration,source_binding_assay=source_assay,
        phase="paired native development and joint acquisition/perception research",
        pending_workers=workers,paired_rgb_assay=dict(sends=assay_sends,results=assay_results),
        accounted_sends=dict(historical_formal=3842,historical_revision_development=93,
          continued_native_development=totals,archived_rgb_assay=assay_sends),
        next="Verify source-bound coverage and the usefulness of measured camera poses on the second development cohort. Retain the paired-RGB timeouts and continue the independent single-decision route.")
    save(Path("RESEARCH_STATE.json"),doc)
    tally="；".join(f"{k} 主对照 {v['primary']}/6、强对照 {v['comparators']}/6" for k,v in cohorts.items())
    stat=f"UTC {timestamp[:19]}。{tally} 已取得完整评分。严格通过 {sum(r['strict'] for r in scored)}/{len(scored)}。新完整任务累计实发 {totals} 次，其中取得评分的回合 {closed} 次。独立核对提前交图 {photos} 次，其中 {supported} 张支持正确字段，最终报告直接引用提前图 {cites} 次。单独的历史照片感知试验实发 {assay_sends} 次。{status}"
    table=["| 场景 | 方法 | 原评分 | 完整通过 | 完整用时（秒） | 按期限计分（秒） | 真实请求 |","|---|---|---|---|---:|---:|---:|"]
    labels={"cracked-row":"设备外壳","warehouse-filter":"货位筛查","native-deferred":"提前采集、延迟判读",
          "native-off":"同优化串行","original-gpt":"原版 GPT","native-prefetch":"强条件预取",
          "native-async":"普通异步","native-planner":"通用规划"}
    labels.update({"cracked-row-02":"新种子设备外壳","warehouse-filter-02":"新种子货位筛查",
        "evidence-deferred":"证据调度","evidence-off":"同优化串行","evidence-async":"普通异步",
        "evidence-prefetch":"强条件预取","evidence-planner":"通用规划"})
    for r in checked:
        table.append("| %s | %s | %s | %s | %.2f | %.2f | %s |"%(labels[r["case"]],labels[r["method"]],
              r["legacy"],"是" if r["strict"]=="True" else "否",float(r["whole_s"]),float(r["restricted_s"]),r["requests"]))
    for f in terminal_external:
        table.append("| %s | %s | 套餐限额 | 未交付 | %.2f | %.2f | %d |"%(labels[f["case"]],
              labels[f["method"]],f["task_seconds"],f["restricted_if_counted_s"],f["actual_sends"]))
    upsert(Path("PROGRESS.md"),"## 持续研发最新实测",stat+"\n\n"+"\n".join(table))
    upsert(Path("NEXT_ACTION.md"),"## 最近一次恢复位置",stat+"\n\n在仓库根目录，用下文相同的 Python 环境运行 `scripts/continuing_recovery.py` 零请求列出断点。账户端确认同一套餐恢复后，运行 `scripts/continuing_recovery.py --run --restoration-evidence '这里填实际恢复凭据或确认时间'`，只启动预登记且尚未发送的主对照。保留中断回合。用 `scripts/continuing_native_report.py --config config/native_evidence_dev02.json` 独立重查，再运行 `scripts/continuing_checkpoint.py` 更新统计。")
    print(stat)


if __name__=="__main__":
    main()
