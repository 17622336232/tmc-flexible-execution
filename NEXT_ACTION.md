# 当前断点

<!-- continuing-snapshot -->
## 最近一次恢复位置

UTC 2026-11-28T10:44:52。主对照 6/6，强对照 2/6 已结算。严格通过 5/8。新完整任务累计实发 54 次，其中已结算回合 45 次。独立核对提前交图 2 次，最终报告直接引用提前图 0 次。单独的历史照片感知试验实发 0 次。总控 3661181，cracked-row / native-async，心跳 2026-11-28T10:44:49.417961+00:00。

实时状态和恢复命令在下文。已登记的图片比较用 `scripts/continuing_paired_rgb.py predict` 续跑，全部预测封存后再执行 `score`。
<!-- /continuing-snapshot -->

状态为 RESEARCH_IN_PROGRESS。旧 456 回合和 11 个开发回合均封存。先读取 RESEARCH_STATE.json、最新迭代记录和新研究目录的账本；后续开发不沿用旧的三轮上限。

11 个旧开发回合的 93 次请求已拆解；64 份历史读号样本已做拒答式 OCR 诊断。新批次使用相同场景、360 秒任务期限及每回合至多 24 次真实请求。首个裂纹回合已通过。两张提前拍摄的照片进入后续普通判断，最终报告引用另两张常规照片。应逐张核对原图哈希、实发时间、许可、是否进入请求和最终报告分别引用了什么。

运行位置为本仓库根目录。以下第一行只查看状态。第二行在总控已退出时恢复，已有结果由收据校验。强对照使用同一入口追加 `--block comparators`，按登记条件另启。已有账户可用次数没有读数。

```bash
env PYTHONPATH=bridge:research/attack_learning:. /home/tdszy/uav/Embodied-intelligent-behavior/.venv/bin/python -m experiments.flexible_execution.continuing_jobs
env PYTHONPATH=bridge:research/attack_learning:. /home/tdszy/uav/Embodied-intelligent-behavior/.venv/bin/python -m experiments.flexible_execution.continuing_jobs --run
```

下一项本地工作是从实际轨迹计算有来源的取证交接、真实传输与动作交叠、额外返程、错误/无用拍摄，把整项结果与同优化串行基线和原版配对。随后检验多判断场景下联合安排现场采集和语义请求能否改变关键路径。同信息规划基线也可选择所有合并与取证动作。
