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
    def native_development(name):
        roster=[]
        for case in optional(name+"/registration.json").get("cases",[]):
            for method in case["order"]:
                key=case["id"]+"/"+method
                episode=local/name/key
                receipt=optional(f"{name}/{key}.receipt.json")
                worker=optional(f"{name}/{key}.worker.json")
                ledger=episode/"private/robot/sends.jsonl"
                roster.append(dict(case=case["id"],method=method,settled=bool(receipt),
                    started=episode.with_suffix(".start.json").is_file(),
                    sends=len(ledger.read_bytes().splitlines()) if ledger.is_file() else 0,
                    endpoint_block=receipt.get("endpoint_block"),pid=worker.get("pid")))
        return dict(slots=roster,sends=sum(r["sends"] for r in roster),settled=sum(r["settled"] for r in roster),
                    result=optional(name+"/summary.json"),audit=optional(name+"/capture_audit.json"))
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
    clock_audit=optional("online-cost-dev01/clock_audit.json")
    cost_replay=optional("online-cost-dev01/summary.json")
    cross_source=optional("cross-source-assay-dev02/summary.json")
    natural_probe=optional("native-natural-frontier-dev01/summary.json")
    permission_cache=optional("online-cost-dev01/permission-cache/summary.json")
    permission_wall=optional("online-cost-dev01/permission-cache/wall/summary.json")
    corrected_budget=optional("online-cost-dev01/permission-cache/budget-v2-guarded/summary.json")
    switch_witness=optional("online-cost-dev01/switch-witness/summary.json")
    native_followup=optional("native-natural-followup-dev01/summary.json")
    followup_registered=(local/"native-natural-followup-dev01/registration.json").is_file()
    joint=native_development("native-joint-execution-dev01")
    pending=native_development("native-pending-photo-dev02")
    continuation=native_development("native-reply-continuation-dev03")
    reply_permission=native_development("native-reply-permission-dev04")
    resolved_scope=native_development("native-resolved-scope-dev05")
    shared_source=native_development("native-shared-source-dev06")
    clock_prior=optional("native-service-history-dev01.json")
    native_public=optional("native-public-preflight-dev01/summary.json")
    joint_slots,joint_sends,joint_settled,joint_result=(joint[k] for k in ("slots","sends","settled","result"))
    pending_slots,pending_sends,pending_settled,pending_audit=(pending[k] for k in ("slots","sends","settled","audit"))
    continuation_slots=continuation["slots"]
    continuation_result=continuation["result"]
    section_ledger=local/"native-section-dev01/sends.jsonl"
    section_sends=len(section_ledger.read_bytes().splitlines()) if section_ledger.is_file() else 0
    ordinary_workers=[]
    ordinary_scripts={
        b"continuing_native_section_study.py":(b"run",),
        b"continuing_native_acquisition_clocks.py":(b"run",b"worker"),
        b"continuing_cross_source_assay.py":(b"run",),
        b"continuing_natural_frontier_probe.py":(b"run",),
        b"continuing_native_bound_followup.py":(b"run",),
        b"continuing_native_joint_execution.py":(b"run",b"worker",b"grade"),
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
    if clock_audit:
        parts=clock_audit["components_s"]
        status+=(f"房屋56联合版整项慢 {clock_audit['joint_minus_serial_s']:.2f} 秒，"
                 f"增加 {parts['model_on_path_s']:.2f} 秒串行服务时间，少占 {-parts['physical_exposed_s']:.2f} 秒动作时间。")
    if cross_source:
        status+=(f"固定原图的跨图合问标定实发 {cross_source['actual_sends']}/6 次，"
                 f"整对答对合问 {cross_source['both_correct_joint']}/2、逐问 {cross_source['both_correct_separate']}/2。")
    if natural_probe:
        accepted=sum(r["ready_for_new_closed_loop"] for r in natural_probe["rows"])
        status+=f"无编号任务用原版提示实发 {natural_probe['closed']}/2 次，{accepted}/2 达到预先登记的接入条件。"
    if permission_wall:
        status+=(f"初版跨图事件模型完成 {permission_wall['trials']} 次名义等时预算规划，"
                 f"合并版对通用规划更低的可行成本出现在 {permission_wall['residual_better']}/{permission_wall['cells']} 组。")
    if corrected_budget:
        n=corrected_budget["comparison_to_generic"]["residual"]
        status+=(f"按实际合问代价重测 {corrected_budget['trials']} 次，"
                 f"合并版 {n['better']} 组低于通用规划，{n['tied']} 组持平，{n['worse']} 组更高。"
                 f"搜索晚于预算 {corrected_budget['total_timing_miss']} 次，按预选动作计分。")
    if switch_witness:
        status+=f"真实时钟驱动的三问换序记录 {switch_witness['count']} 个离线条件。"
    if followup_registered and not native_followup:
        status+="一次仓库侧面后续判断已登记，当前无发送收据。"
    if native_followup:
        status+=(f"仓库已授权的侧面两问合问实发 {native_followup['actual_sends']} 次，"
                 f"独立通过 {int(native_followup['independent_correct_combination'])}/1。")
    if joint_slots:
        status+=f"仓库原版闭环匹配开发已结算 {joint_settled}/{len(joint_slots)} 个位置，实发 {joint_sends} 次。"
    if joint_result:
        status+=f"其中严格整项通过 {joint_result['strict']}/{len(joint_result['rows'])}。"
    if pending_slots:
        status+=f"仓库快门时序消融已结算 {pending_settled}/{len(pending_slots)}，实发 {pending_sends} 次。"
    if pending_audit:
        count=sum(r["verified"] for r in pending_audit["rows"])
        status+=f"在等待中取得原图、随后使用且严格交付的消融位置有 {count} 个。"
    if continuation_slots:
        status+=(f"原版常规回复待办提议已结算 {continuation['settled']}/{len(continuation_slots)}，"
                 f"实发 {continuation['sends']} 次。")
    if continuation_result:
        admitted=sum(r.get("continuation_proposal_admitted",False) for r in continuation_result["rows"])
        status+=f"通过待办校验 {admitted} 个，严格整项通过 {continuation_result['strict']} 个。"
    if reply_permission["slots"]:
        status+=(f"待办回复格式与提前许可对照已结算 {reply_permission['settled']}/{len(reply_permission['slots'])}，"
                 f"实发 {reply_permission['sends']} 次。")
    if reply_permission["result"]:
        proof=reply_permission["result"]
        status+=f"新版待办获准 {sum(r['continuation_proposal_admitted'] for r in proof['rows'])} 个，整项通过 {proof['strict']} 个。"
    if resolved_scope["slots"]:
        status+=(f"条件等价与历史成本的整项验证已结算 {resolved_scope['settled']}/{len(resolved_scope['slots'])}，"
                 f"实发 {resolved_scope['sends']} 次。")
    if resolved_scope["result"]:
        proof=resolved_scope["result"]
        complete_chain=sum(any(p["verified_complete_side_chain"] for p in r.get("pending_captures",()))
                           for r in proof["rows"])
        status+=(f"提议获准 {sum(r['continuation_proposal_admitted'] for r in proof['rows'])} 个，"
                 f"严格整项通过 {proof['strict']} 个，等待中原图进入完整报告 {complete_chain} 个。")
    if native_public:
        verified=sum(all(x["data"]["checks"].values()) for x in native_public["rooms"].values())
        status+=f"公开房屋复用原版执行器的实际取图、转向和回位预检 {verified}/{len(native_public['rooms'])}，新增模型请求 0 次。"
    if shared_source["slots"]:
        status+=(f"原图复用与末次判读直接交付的跨场景对照已结算 {shared_source['settled']}/{len(shared_source['slots'])}，"
                 f"实发 {shared_source['sends']} 次。")
    if shared_source["result"]:
        proof=shared_source["result"]
        complete_chain=sum(any(p["verified_complete_side_chain"] for p in r.get("pending_captures",()))
                           for r in proof["rows"])
        status+=f"严格整项通过 {proof['strict']} 个，整项引用等待中同一原图的轨迹 {complete_chain} 个。"
    if ordinary_workers:
        status+="当前首轮答复或动作测量进程 "+", ".join(str(w["pid"]) for w in ordinary_workers)+"。"
    if unavailable:
        next_step="等待现有 Diff-VLA 套餐显示可用，再从未发送且已登记的位置恢复。"
    elif workers:
        next_step="跟踪正在进行的 "+workers[0]["cohort"]+" 批次，逐一保留原始收据并复核完整评分。"
    elif pending_slots and not pending_audit:
        next_step=("收齐快门时序消融并封存。核验派发、取图、回包解锁、同图合法合问与最终报告是否使用同一照片。")
    elif shared_source["slots"] and not shared_source["result"]:
        next_step=("核对仓库证据复用和直接交付是否减少额外调用。"
                   "继续收齐同一公开房屋中的原版、串行、预取、通用规划和联合版，封存后独立验收。")
    elif shared_source["result"]:
        next_step=("复核仓库实际原图是否贯穿两项判读和完整报告，分析公开房屋五种执行的关键路径。"
                   "先量出公共优化、共享证据与服务长尾各自的贡献，依据开发证据设计差异明显的小规模跨任务验证。")
    elif resolved_scope["slots"] and not resolved_scope["result"]:
        next_step=("核对真实首轮的原文范围与来源绑定，等待两条相反权限的完整任务结算。"
                   "核验新旧照片、实际发问、严格报告以及历史时间预算所改变的执行选择。")
    elif resolved_scope["result"]:
        next_step=("保留本轮完整任务的全部通过与失败。继续把同一个自主执行器接到已暴露的公开房屋，"
                   "以真实动作时钟和独立视觉验收验证不同场景下的共同方法。")
    elif reply_permission["slots"] and not reply_permission["result"]:
        next_step=("收齐允许与禁止提前的两条指令，核对普通回复提议是否保留了所有条件和明确的物理权限。"
                   "等待中相机产生的实际图像须由完整报告引用才计为机制整项通过。")
    elif reply_permission["result"]:
        next_step=("按原文或封存回复核对两条许可对照。"
                   "改用边界明确的自然任务检验是否能接受合法分解并实际运行，然后用已封存服务时间和错误记录检验成本选择。")
    elif continuation_result and not continuation_result["strict"]:
        next_step=("根据封存回复诊断为何常规回复未带可核对的待办。"
                   "保留原长指令失误，把提议合同直接写进一次正常调用的响应格式，再用无技术记录要求的普通文字检验。")
    elif joint_result:
        next_step=("让原版正常回复顺带提议剩余任务及依赖，独立校验完整用户原文和任务许可。"
                   "用改写指令、明确禁止提前和没有收益的匹配开发条件，检验执行器能否据此选择不同顺序。")
    elif joint_slots:
        next_step=("核对已登记的五种仓库对照，保留每轮原始请求和仿真动作。"
                   "待封存后核验发问与取证的因果顺序、证据和严格整项验收。")
    elif native_followup:
        next_step=("将已预登记并通过侧面复核的材料许可接进原版完整执行。"
                   "用同一任务比较原版、串行、强预取、通用规划与联合调度。")
    elif followup_registered and not native_followup:
        next_step=("用仓库封存的真实首轮判断，放行实际侧面照片里的两项已开放问题。"
                   "先复核登记，只使用一轮原版模型回复并独立验收来源和许可。"
                   "该检查用于验证从用户自然语言到真实后续请求的连接。"
                   "三问顺序切换的整项匹配仍须保留通用和强预取，不使用正式数据。")
    elif natural_probe and cost_replay:
        next_step=("为实际闭环准备一项新的无编号自然任务。以独立来源、实际返回的字段和测得的动作代价为输入，"
                   "先检查现有证据能否让跨图合问在合理的服务波动下少一轮串行请求。"
                   "原版、共用优化串行和强预取同时获得相同的材料许可、时钟和合问能力。")
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
    if cost_replay:
        stage="causal timing, cross-camera batch calibration and native task discovery"
    if joint_slots:
        stage="native controller closed-loop matched development"
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
        native_joint_execution=joint,native_pending_photo=pending,native_reply_continuation=continuation,
        native_reply_permission=reply_permission,native_resolved_scope=resolved_scope,
        native_service_history=clock_prior.get("profile",{}),native_public_preflight=native_public,
        native_shared_source=shared_source,
        adaptive_cost_development=dict(clock=clock_audit,cross_source_calibration=cross_source,
            native_natural=natural_probe,cost_scenarios=cost_replay.get("comparisons",0),
            node_budget={k:v for k,v in permission_cache.items() if k!="rows"},
            wall_budget={k:v for k,v in permission_wall.items() if k!="comparisons"},
            corrected_budget={k:v for k,v in corrected_budget.items() if k!="details"},
            three_question_switch={k:v for k,v in switch_witness.items() if k!="rows"},
            native_fixed_followup=dict(registered=followup_registered,result=native_followup)),
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
          ordinary_section_first_decisions=section_sends,
          cross_source_calibration=cross_source.get("actual_sends",0),
          natural_ordinary_gate=natural_probe.get("closed",0),
          natural_bound_followup=native_followup.get("actual_sends",0),native_joint_closed_loop=joint_sends,
          native_pending_photo=pending_sends,native_reply_continuation=continuation["sends"],
          native_reply_permission=reply_permission["sends"],native_resolved_scope=resolved_scope["sends"],
          native_shared_source=shared_source["sends"]),
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
    if cost_replay:
        recovery+="签出当前版本后运行 `python scripts/continuing_room_clock_replay.py`、`python scripts/continuing_online_cost_replay.py`。使用 `python scripts/continuing_cross_source_assay.py grade` 和 `python scripts/continuing_natural_frontier_probe.py grade` 复核真实回复。这四个命令均不发送请求。"
    if switch_witness:
        recovery+="用 `python scripts/continuing_switch_witness.py` 核对三问换序。用 `python scripts/continuing_native_bound_followup.py register` 核对仓库真实来源与两项许可，这两条也不发送请求。"
    if joint_slots:
        recovery+="第一批仓库匹配开发按原始代码版本 `a394efa` 的 `python scripts/continuing_native_joint_execution.py status` 或 `grade` 零请求核对。"
    if pending_slots:
        recovery+="快门消融用原始代码版本 `ea4ea42` 的 `python scripts/continuing_native_joint_execution.py status --study native-pending-photo-dev02` 核对登记。当前可运行 `python scripts/continuing_pending_capture_audit.py` 零请求复核。"
    if continuation_slots:
        recovery+="最早的待办提议检查使用版本 `9d0903e` 的 `python scripts/continuing_native_joint_execution.py status --study native-reply-continuation-dev03`。"
    if reply_permission["slots"]:
        recovery+="用 `python scripts/continuing_native_joint_execution.py status --study native-reply-permission-dev04` 核对许可对照。"
    if resolved_scope["slots"]:
        recovery+="用 `python scripts/continuing_native_joint_execution.py status --study native-resolved-scope-dev05` 核对本版整项登记。收齐后的 `grade --study native-resolved-scope-dev05` 不发请求。"
    if clock_prior:
        recovery+="用 `python scripts/continuing_native_request_history.py` 零请求核对长度、超时和准确性的历史估计。"
    if native_public:
        recovery+="用 `python scripts/continuing_native_room_preflight.py` 核对两个原已暴露的公开房屋和实际相机动作。"
    if shared_source["slots"]:
        recovery+="用 `python scripts/continuing_native_joint_execution.py status --study native-shared-source-dev06` 核对登记。若进程已退出且没有外部限制，按同版本的 `run --study native-shared-source-dev06` 从未启动位置续跑。收齐后的 `grade --study native-shared-source-dev06` 无新增请求。"
    upsert(Path("NEXT_ACTION.md"),"## 最近一次恢复位置",stat+"\n\n"+next_step+"\n\n"+recovery)
    manuscript=Path("docs/DEVELOPMENT_EVIDENCE_CN.md")
    if manuscript.exists():
        upsert(manuscript,"## 当前完整任务样本",f"截至 {timestamp[:19]} UTC，新的自然在线开发共有 {sum(bool(r['sends']) for r in data)} 个已启动回合，{len(scored)} 个已有完整评分，其中 {sum(r['strict'] for r in scored)} 个严格通过。全部已启动回合累计发送 {totals} 次。以下时间下限与决策重算使用恢复前封存的 16 个有评分回合。逐回合新结果和中断记录见[完整进度表](../PROGRESS.md)。")
    print(stat)


if __name__=="__main__":
    main()
