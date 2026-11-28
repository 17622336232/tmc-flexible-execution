"""Generate the bounded research decision from completed, independent-scored receipts."""
import csv
from pathlib import Path

from experiments.flexible_execution.revision_audit import BASE, OUT, digest, read, save


NAMES={"original-gpt":"原版 GPT","conditional-async":"共同执行器＋普通异步",
       "conditional-a":"共同执行器＋A","conditional-planner":"共同执行器＋通用规划"}
CASES={"dev-cracks-prim":"裂纹巡检","dev-locate-control":"指定货位盘点"}


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(r,k):
    return float(r[k])


def main():
    first=table(OUT/"development_results.csv")
    last=table(OUT/"round3/development_results.csv")
    assert len(first)==8 and len(last)==3, "Development block still incomplete"
    assert all(r["shared_fragment_regrade"]==r["strict_success"] for r in first+last), "Review scoring difference"
    rows=[]
    for block,values in (("首组",first),("第三轮",last)):
        for r in values:
            passed=r["strict_success"]=="True"
            state="通过" if passed else "到期" if r["outcome"]=="deadline" else "未通过"
            rows.append(f"| {block} | {CASES[r['case']]} | {NAMES[r['method']]} | {state} | {number(r,'whole_s'):.1f} | "
                        f"{number(r,'restricted_s'):.1f} | {r['sends']} | {number(r,'compile_s'):.1f} | {number(r,'overlap_s'):.1f} |")
    ap=[r for r in last if r["method"]=="conditional-a"][0]
    orig=next(r for r in first if r["method"]=="original-gpt" and r["case"]==ap["case"])
    off=next(r for r in last if r["method"]=="conditional-async")
    planner=next(r for r in last if r["method"]=="conditional-planner")
    draft=read(OUT/"round3/independent_draft_check.json")
    def comparison(r):
        change=number(ap,"restricted_s")-number(r,"restricted_s")
        return f"{change:+.1f} 秒（{100*change/number(r,'restricted_s'):+.1f}%）"
    n_sends=sum(int(r["sends"]) for r in first+last)
    duration=sum(number(r,"whole_s") for r in first+last)/60
    symbolic=read(OUT/"symbolic_summary.json")
    src=read(OUT/"shared_fixes.json")
    refs=read(OUT/"finite_decision_reference.json")
    res=table(OUT/"sampled_resources.csv") if (OUT/"sampled_resources.csv").exists() else []
    energies=(f"本机子进程的 {len(res)} 个采样窗口内，最高同时驻留内存合计 "
              f"{max(number(r,'peak_sampled_rss_mib') for r in res):.0f} MiB；共享内存可能重复计算。") if res else ""
    am=[r for r in table(OUT/"fresh_mechanism.csv") if r["method"]=="conditional-a"]
    early=sum(int(r["conditional_moves_started_before_bound_reply"]) for r in am)
    uses=sum(int(r["conditional_early_materials_used_in_report"]) for r in am)
    text=f"""# 修订研究决策

## 可复核结论

当前证据支持一台机器人在许可范围内，边等判断边移动，并在部分任务中把提前拍的图用于最终报告。现有数据不足以支持修订 A 在整项任务上普遍比原版 GPT 快 20%，也不足以支持 A 优于强预取或通用规划。第三轮同一裂纹场景中，A 相对早先的原版 GPT 的受限用时差为 {comparison(orig)}，相对同轮普通异步为 {comparison(off)}，相对同轮通用规划为 {comparison(planner)}。差值取 A 减对照。原版参照来自另一时间段。

## 新自然在线开发

使用相同 GPT-6.1 Terra、max、仿真传感器、视觉资产、导航和 360 秒期限。各臂重新看图，均无训练。11 个新回合来自两个独立登记的场景，第三轮只重复其中一个。总共 {n_sends} 次发送，任务运行时间合计 {duration:.1f} 分钟。自然服务时延有波动，两场不适合给出总体效应区间。失败按期限计入受限用时，人工或远程验收等附加工序在各方法实际执行时计时。

| 版本 | 场景 | 方法 | 严格验收 | 整项实测秒 | 受限秒 | 请求 | 远程任务编译秒 | 运动与请求重叠秒 |
|---|---|---|---|---:|---:|---:|---:|---:|
"""+"\n".join(rows)+f"""

原版循环在裂纹巡检中直接决定观察、移动和报告，8 次请求完成任务。首组 A 的任务编译用了 46.9 秒，材料判读用了 105.7 秒，整项视觉验收用了 105.5 秒。前段材料判读未核实物体和所需字段，进入后续控制器的合格字段为 0。同场普通异步的任务编译用了 117.5 秒。第三轮把能从完整指令精确读出的要求交给共同文字解析器，实时图像判读、导航和独立评分照常运行。由这个改动产生的收益属于共同执行器。

## 三轮决定

1. 修复共同执行器，分清可选的条件判断与必须完整表达的任务。18 份冻结回复均保住了完整任务程序。原编号与照片双重验证成功时，场景 172 恢复了一条裂纹事实。历史终态未改。归档的带前段串行对照有 43 次尝试、24 个任务族，其中 32 次旧评分正确，39 次有可重评报告，19 次严格通过。原版两次同源历史复现另计。
2. 允许部分推进后按真实结果继续。十类任务交叉许可、延迟和移动成本，得到 120 个符号开发配置，每种策略都独立验收 120/120。普通异步平均 {symbolic['conditional-async']['restricted_model_seconds']:.2f} 个模拟秒，A、强预取、通用规划都约 11.99 个模拟秒。独立穷举对初始决策打分，A、预取、通用规划都是 120/120，普通异步是 {refs['conditional-async']['optimal']}/120。有限模型允许同样好的多个动作，假设材料可读，枚举器没有接收真实分支。
3. 共同文字解析器冷启动，在 43 种归档指令与评分规则组合中识别 20 种、涉及 10 个任务族。识别成功者通过 1,278 次局部语义核验，附加限制的 129 种变体均交回通用解析。三个自然在线复测均重新采集图像。

单独的消融在同一任务和期限下关闭某项能力。六个共享取证配置中，禁止共享观察后拍照从 6 次增至 12 次；六个复用配置中，关闭初始证据复用后从 0 次增至 6 次。原高成本格子还留有便宜的替代视点，因此另存六个两条路都绕行的诊断配置，其中五个选择等待。这些都是离散事件结果。生成的图中延迟与移动成本均明确标为模型量。

三个新 A 回合合计有 {early} 次经原始请求时钟核对的回包前条件动作，严格通过的最终报告采用提前材料 {uses} 次。逐个目标的冻结视觉判断可以触发调度。目前一旦移交给整项 GPT 控制器，其普通动作决策与最终验收仍主要同步进行。这一实现边界限制了能被重排的时间。离线的多判断规划及公开房屋的前图转侧图链，不能替代一套贯穿整项真实控制的多判断执行器。

第三轮 A 最能解释两种失败怎样叠加。一次材料判读花了 119.2 秒，把本体编号读成 {draft['accepted_material_values'][0]}，前段检查接收了这个值，整项控制继承了一条事实。草稿继续使用了这个编号。事后以原评分器独立核对，正确的裂纹设备编号是 {draft['expected'][0]}，草稿漏掉了它。末尾同一验收请求先等待 120.0 秒并超时，重试又等了 58.3 秒后超时。即便保留原草稿直接评分也不正确。对象、照片和时间能够追溯，读数正确性仍需单独验证。[草稿复核与原图指纹](round3/independent_draft_check.json)

## 旧正式实验的证据边界

自建 CPU 仿真包括 72 个场景、五臂共 360 回合。旧 A 通过 26/72，规划器 20/72。按 70 个独立来源组配对，受限用时差 −26.65 秒，95% 区间 [−52.10, −4.58]。成功率差 9.29 个百分点，95% 区间 [−0.71, 20.00]。公开房屋 24 个、四臂 96 回合。A 与规划器各通过 20/24，受限用时差 +2.74 秒，95% 区间 [−5.93, 13.14]。这些都是原冻结版本的估计。

自建 A 的 72 场有 49 场提交视觉判断，36 场留下依结果而异的材料要求，3 场明确允许提前采集，2 场在回包前拍完。回包前拍完且后来核验为所需材料的场景为 0。两张阳性设备编号照是在提前开始移动后、回包之后拍完。逐场明细保留无法观察的反事实等待节省。对这份旧名单，如果只改变允许提前取编号的三场，就算把三场规划器的全部受限时长归零，合计也只省 5.84%。这个上界的分母是 72 场规划器的受限用时总和，不适用于更广泛的变通策略。

另按原登记指令逐项判读，50 场要求按结果筛选报告，16 场仅查指定编号，6 场必须逐个盘点。50 场中有 37 场实际提交判断，36 场的运行程序留下条件材料要求。仅凭“只报告符合条件的设备”无法证明另一次拍摄既合法又必要。三个旧评分中存在额外阳性报告字段的通过回合，补做来源绑定复核后均找到了相关字段。

公开房屋原始记录中有 23 条“前图阳性返回后，另向模型发送一张新侧图”的双阶段链，分布在 6 间房。A 的一张侧图在回包前 0.055 秒完成，后来用于通过验收的报告。照片级复核中，93 个可评正面判断有 81 个正确，23 个可评侧面判断均正确。四个方法共享部分照片，这些判断不能当作 116 个独立场景。所有 457 次历史尝试的 3,842 条模型记录里，没有发现成功后重复发送完全相同输入的请求。视觉上相近的补拍以及材料与最终报告的两种核验仍可产生大量耗时。

## 后续验证边界

保持独立测试封存。需要在多个独立来源中观察到“提前取得了合格材料、真实后续采用、并节省整项时间”，再检验能否超过强规划器。可以把首要研究问题缩小为哪些许可、现场成本和判断时长组合值得提前取证。当前没有测得整条因果链的总体净节省，也没有具备统计把握的成功率非劣性结论。

新公开储备包含两套互不重叠的 32 间房。一套检查任务涉及的资产与已用资产互斥，另一套按房形划组并独立抽取，背景资产可能重叠。房屋未渲染、未验收，数量只表示储备容量。公开房屋与自建程序生成任务均不能代表真实巡检任务在部署中的发生频率。

{energies} 请求日志未给出 token 用量。历史逐方法时间、CPU、首次记录到的动作和请求重叠见成本表。真实阻塞节省、全程峰值 GPU、Orin 实测能量、飞行时间和接管次数尚无实测。没有启动实体无人机、无人车或机器狗，实体动作需现场操作者确认。

## 图表与复现

- [开发逐回合表](development_results.csv) · [第三轮逐回合表](round3/development_results.csv) · [输入一致性与原版提示指纹](run_provenance.json)
- [历史逐方法成本](historical_costs.csv) · [按任务族与变化组分组](historical_subgroups.csv) · [457 次原始尝试](all_raw_attempts.csv)
- [72 场机会漏斗](opportunities.png) · [逐场证据](opportunities.csv) · [已观察到的提前动作](observed_early_motion.csv)
- [独立决策核验](finite_decision_reference.csv) · [机制消融](ablation-verified/ablation_summary.csv) · [延迟与移动成本曲线](modeled_latency_cost.png)
- [任务文字与真实链条逐场对照](opportunities_with_instruction_reference.csv) · [新回合的动作和材料采用](fresh_mechanism.csv)
- [原图案例](recorded_examples.png) · [带哈希的图片来源](recorded_examples.csv) · [照片级重评](archived_vision_decisions.csv) · [资源采样](sampled_resources.csv)
- [原版串行重评](original_regraded.csv) · [带额外前段的串行重评](augmented_serial_regraded.csv) · [旧验收的附加字段敏感性检查](positive_field_sensitivity.csv)
"""
    (OUT/"RESEARCH_DECISION_CN.md").write_text(text,encoding="utf8")
    def public(rows):
        return [dict(case=r["case"], method=r["method"], strict_success=r["strict_success"]=="True",
                     sends=int(r["sends"]), raw_result_sha256=r["raw_result_sha256"],
                     **{k:number(r,k) for k in ("whole_s","restricted_s","compile_s","overlap_s")}) for r in rows]
    save(Path("docs/revision_results.json"),dict(role="exposed-development; never independent formal evidence",
         units="seconds", first_block=public(first), third_round=public(last), sends=n_sends,
         integrity=read(OUT/"integrity.json"), historic_funnel=read(OUT/"history.json")["funnel"],
         symbolic_initial_decisions=refs, early_a_actions=early, used_early_materials_in_passing_a_reports=uses,
         final_draft=dict(claimed=draft["claimed"],expected=draft["expected"],correct=draft["draft_correct"],
                          acceptance_timeout_seconds=draft["verifier_durations"]),
         decision_report_sha256=digest(OUT/"RESEARCH_DECISION_CN.md")))


if __name__=="__main__":
    main()
