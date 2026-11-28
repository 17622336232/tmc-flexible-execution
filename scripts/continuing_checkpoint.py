"""Persist counts from receipts, provenance checks, and the live controller heartbeat."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import os

from experiments.flexible_execution.continuing_jobs import snapshot
from experiments.flexible_execution.revision_audit import read,save


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
    config=read(Path("config/native_views_dev01.json"))
    root=Path("outputs/continuing")/config["id"]
    data=snapshot(root,config)
    scored=[r for r in data if r["state"]=="scored"]
    checked=list(csv.DictReader((root/"audit/episodes.csv").open()))
    if len(checked)!=len(scored):
        raise SystemExit("Run continuing_native_report.py first to reconcile all settled results")
    beat=read(root/"heartbeat.json")
    active=[r for r in data if r["state"] in ("running","starting")]
    controller=beat.get("controller_pid")
    live=bool(controller and Path("/proc/%s"%controller).exists())
    doc=read(Path("RESEARCH_STATE.json"))
    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds")
    counts={b:sum(r["block"]==b for r in scored) for b in ("primary","comparators")}
    totals=sum(r["sends"] or 0 for r in data)
    closed=sum(r["sends"] for r in scored)
    photos=sum(int(r["early_handoffs"]) for r in checked)
    cites=sum(int(r["early_final_citations"]) for r in checked)
    field_errors=sum(int(r["field_grant_violation_count"]) for r in checked)
    assay=Path("outputs/continuing/paired-rgb-assay01")
    reservations=assay/"sends.jsonl"
    assay_sends=len(reservations.read_text().splitlines()) if reservations.is_file() else 0
    assay_results=read(assay/"summary.json") if (assay/"summary.json").is_file() else None
    status=("总控 %s，%s / %s，心跳 %s。"%(controller,beat.get("case"),beat.get("method"),beat["utc"])
            if live else "当前完整任务总控已退出，最后心跳 "+beat["utc"]+"。")
    doc.update(as_of_utc=timestamp,method_version="native-views-dev01 / runtime source 2e8e26cb8f0099bb68d02e5aa1c1e9569423706ec413227f9d9aa7714ffcef2b",
        new_dev=dict(scored_episodes=len(scored),strict_successes=sum(r["strict"] for r in scored),
        scored_sends=closed,in_flight_sends_at_snapshot=totals-closed,blocks_completed=counts,
        early_source_verified_handoffs=photos,early_photos_cited_in_final=cites,early_field_violations=field_errors),
        phase="paired native development and joint acquisition/perception research",
        pending_workers=[beat] if active and live else [],paired_rgb_assay=dict(sends=assay_sends,results=assay_results),
        accounted_sends=dict(historical_formal=3842,historical_revision_development=93,
          continued_native_development=totals,archived_rgb_assay=assay_sends),
        next="Finish independent image-chain checks and strong controls, calibrate joint semantic requests on real archived images, check the full-task evidence scheduler against a generic exact planner.")
    save(Path("RESEARCH_STATE.json"),doc)
    stat=f"UTC {timestamp[:19]}。主对照 {counts['primary']}/6，强对照 {counts['comparators']}/6 已结算。严格通过 {sum(r['strict'] for r in scored)}/{len(scored)}。新完整任务累计实发 {totals} 次，其中已结算回合 {closed} 次。独立核对提前交图 {photos} 次，最终报告直接引用提前图 {cites} 次。单独的历史照片感知试验实发 {assay_sends} 次。{status}"
    table=["| 场景 | 方法 | 原评分 | 完整通过 | 完整用时（秒） | 按期限计分（秒） | 真实请求 |","|---|---|---|---|---:|---:|---:|"]
    labels={"cracked-row":"设备外壳","warehouse-filter":"货位筛查","native-deferred":"提前采集、延迟判读",
          "native-off":"同优化串行","original-gpt":"原版 GPT","native-prefetch":"强条件预取",
          "native-async":"普通异步","native-planner":"通用规划"}
    for r in checked:
        table.append("| %s | %s | %s | %s | %.2f | %.2f | %s |"%(labels[r["case"]],labels[r["method"]],
              r["legacy"],"是" if r["strict"]=="True" else "否",float(r["whole_s"]),float(r["restricted_s"]),r["requests"]))
    upsert(Path("PROGRESS.md"),"## 持续研发最新实测",stat+"\n\n"+"\n".join(table))
    upsert(Path("NEXT_ACTION.md"),"## 最近一次恢复位置",stat+"\n\n实时状态和恢复命令在下文。已登记的图片比较用 `scripts/continuing_paired_rgb.py predict` 续跑，全部预测封存后再执行 `score`。")
    print(stat)


if __name__=="__main__":
    main()
