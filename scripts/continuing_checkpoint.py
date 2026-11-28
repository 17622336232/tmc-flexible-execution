"""Persist counts from receipts, provenance checks, and the live controller heartbeat."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from experiments.flexible_execution.continuing_jobs import snapshot
from experiments.flexible_execution.revision_audit import digest,read,save
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
    parser=argparse.ArgumentParser()
    parser.add_argument("--execution-state",choices=("paused","active"),help="Record the user's execution decision")
    requested_state=parser.parse_args().execution_state
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
    regional=optional("same-frame-region-assay01/summary.json")
    decision=optional("registered-blocks.json")
    measured=optional("condition-range-dev01/summary.json")
    direct_travel=optional("inspection-travel-dev01/calibration/summary.json")
    routed_travel=optional("observed-return-dev02/calibration/summary.json")
    first_view=local/"first-view-contingency-dev01"
    fv_reg=optional("first-view-contingency-dev01/registration.json")
    fv_slots=sum(len(c["methods"]) for c in fv_reg.get("cases",[]))
    fv_ledger=first_view/"sends.jsonl"
    fv_sends=len(fv_ledger.read_bytes().splitlines()) if fv_ledger.exists() else 0
    fv_sealed=(first_view/"sealed_replies.json").is_file()
    fv_replay=optional("first-view-contingency-dev01/replay_summary.json")
    event_grid=optional("concurrent-workflow-dev01/summary.json")
    room_base=local/"matched-event-rooms-dev01"
    room_cal=optional("matched-event-rooms-dev01/calibration/summary.json")
    room_live=optional("matched-event-rooms-dev01/live/summary.json")
    room_bounds=optional("matched-event-rooms-dev01/prior_bounds.json")
    deadline_result=optional("deadline-coalescing-dev01/summary.json")
    limit_result=optional("event-compute-dev02/summary.json")
    ordinary_section=optional("native-section-dev01/proposal_summary.json")
    ordinary_binding=optional("native-action-binding-dev01/summary.json")
    acquisition_clocks=optional("native-acquisition-clocks-dev01/summary.json")
    section_ledger=local/"native-section-dev01/sends.jsonl"
    section_sends=len(section_ledger.read_bytes().splitlines()) if section_ledger.is_file() else 0
    ordinary_workers=[]
    ordinary_scripts={
        b"continuing_native_section_study.py":(b"run",),
        b"continuing_native_acquisition_clocks.py":(b"run",b"worker"),
    }
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args=proc.read_bytes().split(b"\0")
            for script, modes in ordinary_scripts.items():
                for index, arg in enumerate(args[:-1]):
                    if arg.rsplit(b"/",1)[-1]==script and args[index+1] in modes:
                        ordinary_workers.append(dict(pid=int(proc.parent.name),script=script.decode(),
                                                     mode=args[index+1].decode()))
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            pass
    room_workers=[]
    room_slots=0;room_settled=0;room_sends=0
    if (room_base/"live/registration.json").is_file():
        room_slots=sum(len(v) for v in read(room_base/"live/registration.json")["orders"].values())
        for rc in room_base.glob("live/*/*.receipt.json"):
            d=read(rc);pid=d.get("worker")
            ep=rc.parent/d["method"];ledger=ep/"sends.jsonl"
            room_sends+=len(ledger.read_bytes().splitlines()) if ledger.exists() else 0
            room_settled+=d["state"]=="settled"
            if d["state"]=="running" and pid and Path(f"/proc/{pid}").exists():
                room_workers.append(dict(case=d["case"],method=d["method"],pid=pid))
    fv_rows=[]
    if fv_sealed:
        sealed=read(first_view/"sealed_replies.json")
        for c in fv_reg["cases"]:
            for m in c["methods"]:
                rc=first_view/c["id"]/(m+".receipt.json")
                log=first_view/c["id"]/m/"model_calls.jsonl"
                if sealed[c["id"]+"/"+m]!={"receipt":digest(rc),"model_log":digest(log)}:
                    raise ValueError("Sealed first decision does not match its original response")
                row=read(rc)
                fv_rows.append({k:row[k] for k in ("case","method","actual_reply","requested_action","supplied_option","service_s","log_sha256","error")})
    fv_pids=[]
    if fv_reg:
        for p in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                args=p.read_bytes().split(b"\0")
                if any(x.endswith((b"/continuing_first_views.py",b"/continuing_first_view_replay.py")) for x in args) and any(a in args for a in (b"predict",b"run",b"worker")):
                    fv_pids.append(int(p.parent.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):
                pass
    if terminal_external and not unavailable:
        status+="原套餐中断保留，后续未发送位置已使用恢复后的同一服务。"
    if decision.get("stage")=="registered-decision-point":
        status+="第三批两种场景的提前交图都未满足登记条件，8 个后续强对照未发送。"
        if not decision.get("scoped_mechanism_admitted"):
            status+="单图实验未通过校园区域归属要求，2 个依赖它的整项也未发送。"
    if measured:
        scored_ranges=measured["scored"]
        status+=(f"两条本机测距路线均回位。满足门框和障碍物同时可核实要求的原位扫描 "
                 f"{sum(r['positive_geometry_support'] for r in scored_ranges if r['view']=='anchor')}/2，近处扫描 "
                 f"{sum(r['positive_geometry_support'] for r in scored_ranges if r['view']=='close')}/2。")
    if routed_travel:
        status+="首次两条双站巡检路线均到达第二站，直返均被原地图拒绝。改用实测经过点回程后，货位回到起点，设备在去第二站时由原导航停止。两版各有两条校准路线，零模型请求。"
    if fv_reg:
        status+=f"独立的首次决策试验实发 {fv_sends}/{fv_slots} 次。"
        if fv_sealed:
            count=sum(r["supplied_option"] for r in fv_rows if r["method"]=="contingent")
            arms=sum(r["method"]=="contingent" for r in fv_rows)
            status+=f"{sum(r['actual_reply'] for r in fv_rows)} 份真实答复已封存，允许给备选动作的 {arms} 份中有 {count} 份给出备选。"
        if not fv_replay:
            status+="首次决策的仿真动作尚未取得评分。"
        else:
            status+=f"8 份实际答复已完成独立仿真验证，新增模型请求为零，备用动作到位 {fv_replay['guarded_movements']} 次，满足登记证据要求 {fv_replay['registered_witnesses']} 次。"
        if fv_pids:
            status+="当前独立试验进程 "+", ".join(map(str,fv_pids))+"。"
    if planned_assay:
        status+=f"另行登记的单图多区域感知比较实发 {regional_sends}/{planned_assay} 次。"
    if event_grid:
        status+=f"多判断调度的 {event_grid['finished']}/{event_grid['registered']} 项离线计算已落盘。"
    if room_cal:
        status+=f"房屋成本标定实发 {room_cal['actual_sends']} 次，单问与合问的四个字段分别答对 {room_cal['singleton_correct']}、{room_cal['grouped_correct']} 个。"
    if room_slots:
        status+=f"新房屋完整开发任务已结算 {room_settled}/{room_slots}，实发 {room_sends} 次。"
        if room_workers:status+="当前进程 "+", ".join(str(w["pid"])+" "+w["case"]+"/"+w["method"] for w in room_workers)+"。"
    if room_live:
        status+=f"独立核验严格通过 {room_live['strict']}/{room_live['registered']}。"
    if limit_result:
        status+=f"强化对照在 {limit_result['matched_cells']} 组相同墙钟预算下与通用规划的中位可行成本相同 {limit_result['cost_tied_with_generic']} 组。"
    if ordinary_section:
        status+=(f"首轮工作单试验实发 {section_sends}/{ordinary_section['registered']} 次。"
                 f"可选委派放行 {len(ordinary_section['eligible_full_cases'])}/2 个任务。")
    if ordinary_binding:
        status+=(f"用封存首帧核对原版动作与来源，{ordinary_binding['bound']}/{ordinary_binding['registered']} 份答复符合。"
                 f"{ordinary_binding['distinct_correct']}/{ordinary_binding['distinct_fields']} 个不同判断正确。")
    if acquisition_clocks:
        cases=acquisition_clocks["cases"]
        completed=sum(len(case["phases"])==2 for case in cases)
        status+=f"两站转向时钟校准完成 {len(cases)} 次尝试，其中往返均到位 {completed}/{len(cases)}，新增模型请求为零。"
    if ordinary_workers:
        status+="当前首轮答复或动作测量进程 "+", ".join(str(w["pid"]) for w in ordinary_workers)+"。"
    if unavailable:
        next_step="等待现有 Diff-VLA 套餐显示可用，再从未发送且已登记的位置恢复。"
    elif workers:
        next_step="跟踪正在进行的 "+workers[0]["cohort"]+" 批次，逐一保留原始收据并复核完整评分。"
    elif acquisition_clocks:
        next_step=("用已封存的模型和动作时钟，比较先发问并同步取证，与先拿齐两张已许可原图再合问。"
                   "逐条检查真实结果回来前可以回答哪些问题，并列出原版 GPT、优化串行和强预取的请求轮数与时间下界。"
                   "这能判断联合安排是否少掉一轮串行推理。地图拒绝的两条动作保留原始失败，回位收益目前缺少完整路径时钟。")
    elif ordinary_binding:
        next_step="沿用原版正常移动和真实首轮判断，先查实际捕获和回位时钟，再量化合法合问的时间上限。"
    elif ordinary_section:
        next_step="用封存原图独立验证普通决策的来源、动作和已答字段，核对无需另增一轮的交接方式。"
    elif room_live:
        next_step="在原版 GPT 的同一次正常决策中验证，能否由用户原话提出合法的多判断工作单，复用已报告字段，等真实条件回来再合问后续字段。先检验结构和来源，再进入匹配整项。"
    elif room_slots:
        next_step="按已登记的两个公开房屋任务收齐六个对照。相同的工作单、相机与验收让我们检验等待中取证和合问能否缩短整项交付，并量出相同输入的服务波动。"
    elif fv_reg:
        if fv_replay:
            next_step="备用动作的辅助诊断已封存。继续验证多判断、共享现场材料与合法合问在相同信息和计算预算下怎样改变完整任务时间。"
        elif fv_sealed:
            next_step="先在原仿真验证八份已封存回复，新增模型请求为零。随后用第三批两种场景的真实时钟分解整项差异，并在公共证据和同等计算预算下比较多判断、共享材料、合法合问如何改变任务交付的关键路径。"
        else:
            next_step="完成同一首帧上原版与附带可选备用动作的正常决策，封存全部回复后才做独立仿真重放。"
    elif measured:
        next_step="已关闭的相机补图批次保留原判。接下来检查能否在判读现场照片期间执行任务本来就要求的下一段行程，先核验场景、照片归属及路线，再决定是否使用真实模型。"
    else:
        next_step="从各批次收据与原有门槛检查下一项尚未发送的作业，继续当前版本的开发分析。"
    stage="paired native development and measured serial inference dependencies" if not measured else "source-bound evidence and mandatory inspection travel"
    if fv_reg:
        stage="frozen first-decision replies and pending simulator verification"
    if event_grid:
        stage="event-driven scheduling of physical evidence and dependent semantic requests"
    if ordinary_binding:
        stage="native controller source binding and legal multi-view request scheduling"
    execution_state=requested_state or doc.get("execution_state","active")
    if execution_state=="paused":
        if workers or fv_pids or room_workers or ordinary_workers:
            raise ValueError("A study worker is still active at the requested pause")
        status="执行已按用户要求暂停，当前研究任务进程为 0。"+status
    doc.update(as_of_utc=timestamp,method_version=versions,
        execution_state=execution_state,
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
        region_assay_result=regional,measured_range_result=measured,
        mandatory_itinerary_direct=direct_travel,mandatory_itinerary_measured_return=routed_travel,
        event_workflow=event_grid,matched_rooms=dict(calibration=room_cal,prior_bounds=room_bounds,
            scheduled=room_slots,settled=room_settled,sends=room_sends,workers=room_workers,
            graded=None if not room_live else {"strict":room_live["strict"],"registered":room_live["registered"]}),
        deadline_policy_sensitivity=deadline_result,hard_compute_budget=limit_result,
        native_ordinary_frontier=dict(proposals=ordinary_section,binding=ordinary_binding,
                                      acquisition_clocks=acquisition_clocks,workers=ordinary_workers),
        first_decision_study=dict(registered=fv_slots,sends=fv_sends,sealed=fv_sealed,pids=fv_pids,
          sealed_file_sha256=digest(first_view/"sealed_replies.json") if fv_sealed else None,receipts=fv_rows,
          replay_entry_present=Path("scripts/continuing_first_view_replay.py").is_file(),replay=fv_replay),
        registered_block_decision=decision if decision.get("stage")=="registered-decision-point" else {},
        optional_development=dict(region_assay_registered_sends=planned_assay,region_assay_actual_sends=regional_sends,
            closed_mechanism_registered_arms=mechanism.get("bounded_online_order",[]),closed_mechanism_actual_sends=scoped_sends),
        current_hypothesis=next_step,
        registered_slots=planned_counts,stage=stage,phase=stage,
        pending_workers=workers,paired_rgb_assay=dict(sends=assay_sends,results=assay_results),
        accounted_sends=dict(historical_formal=3842,historical_revision_development=93,
          continued_native_development=totals,archived_rgb_assay=assay_sends,one_image_regions=regional_sends,
          explicit_roster_mechanism=scoped_sends,standalone_first_decisions=fv_sends,
          matched_room_cost_calibration=room_cal.get("actual_sends",0),matched_room_tasks=room_sends,
          ordinary_section_first_decisions=section_sends),
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
    focus="本轮研究问题是同一张已取得的照片能否服务多个后续判断，以及机器人在真实回复返回后能否重新安排尚未完成的工作。先比较整项关键路径的上限，再用匹配的开发任务检验。"
    upsert(Path("PROGRESS.md"),"## 持续研发最新实测",stat+"\n\n"+focus+"\n\n"+"\n".join(table))
    recovery="在仓库项目环境中运行 `scripts/continuing_registered_blocks.py` 可核对全部已登记批次与额度状态。运行 `scripts/continuing_region_assay.py grade` 和 `scripts/continuing_condition_range.py grade` 可用已有原始数据独立复核两项诊断。"
    if fv_reg:
        recovery+="用 `scripts/continuing_first_views.py status` 核对首次决策位置，用 `scripts/continuing_first_views.py seal` 核对封存回复，两条命令均为零新增模型请求。"
    if event_grid:
        recovery+="主线的零请求复核命令是 `scripts/continuing_v3_attribution.py`、`scripts/continuing_concurrent_workflow.py summary`。运行 `scripts/continuing_matched_event_rooms.py status` 可核对正式登记的六个开发位置。"
    if ordinary_binding:
        recovery+="可用 `python scripts/continuing_native_action_binding.py grade` 核验四份封存答复，用 `python scripts/continuing_native_acquisition_clocks.py summary` 核对两条实测记录。两条命令均为零新增模型请求。"
    upsert(Path("NEXT_ACTION.md"),"## 最近一次恢复位置",stat+"\n\n"+next_step+"\n\n"+recovery)
    manuscript=Path("docs/DEVELOPMENT_EVIDENCE_CN.md")
    if manuscript.exists():
        upsert(manuscript,"## 当前完整任务样本",f"截至 {timestamp[:19]} UTC，新的自然在线开发共有 {sum(bool(r['sends']) for r in data)} 个已启动回合，{len(scored)} 个已有完整评分，其中 {sum(r['strict'] for r in scored)} 个严格通过。全部已启动回合累计发送 {totals} 次。以下时间下限与决策重算使用恢复前封存的 16 个有评分回合。逐回合新结果和中断记录见[完整进度表](../PROGRESS.md)。")
    print(stat)


if __name__=="__main__":
    main()
