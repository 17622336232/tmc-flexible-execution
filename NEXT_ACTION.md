# 当前断点

<!-- continuing-snapshot -->
## 最近一次恢复位置

UTC 2026-11-28T12:55:54。native-views-dev01 主对照 6/6、强对照 6/6；native-evidence-dev02 主对照 4/6、强对照 0/6 已取得完整评分。严格通过 10/16。新完整任务累计实发 88 次，其中取得评分的回合 84 次。独立核对提前交图 3 次，其中 1 张支持正确字段，最终报告直接引用提前图 0 次。单独的历史照片感知试验实发 18 次。当前完整任务总控已退出，最后心跳 2026-11-28T11:10:57.188369+00:00。有 1 回合因套餐限额中断，已发 4 次。服务未返回重置时间。已另做 8 次零模型仿真路线检查，完成读号并回位为正面 2/4、同距离沿原方向 0/4。

在仓库根目录，用下文相同的 Python 环境运行 `scripts/continuing_recovery.py` 零请求列出断点。账户端确认同一套餐恢复后，运行 `scripts/continuing_recovery.py --run --restoration-evidence '这里填实际恢复凭据或确认时间'`，只启动预登记且尚未发送的主对照。保留中断回合。用 `scripts/continuing_native_report.py --config config/native_evidence_dev02.json` 独立重查，再运行 `scripts/continuing_checkpoint.py` 更新统计。
<!-- /continuing-snapshot -->

状态为 RESEARCH_IN_PROGRESS。旧 456 回合和 11 个开发回合均封存。先读取 RESEARCH_STATE.json、最新迭代记录和新研究目录的账本；后续开发不沿用旧的三轮上限。

11 个旧开发回合的 93 次请求已拆解；64 份历史读号样本已做拒答式 OCR 诊断。新批次使用相同场景、360 秒任务期限及每回合至多 24 次真实请求。首个裂纹回合已通过。两张提前拍摄的照片进入后续普通判断，最终报告引用另两张常规照片。应逐张核对原图哈希、实发时间、许可、是否进入请求和最终报告分别引用了什么。

运行位置为本仓库根目录。以下第一行只查看状态。第二行在总控已退出时恢复，已有结果由收据校验。强对照使用同一入口追加 `--block comparators`，按登记条件另启。已有账户可用次数没有读数。

```bash
env PYTHONPATH=bridge:research/attack_learning:. /home/tdszy/uav/Embodied-intelligent-behavior/.venv/bin/python -m experiments.flexible_execution.continuing_jobs
env PYTHONPATH=bridge:research/attack_learning:. /home/tdszy/uav/Embodied-intelligent-behavior/.venv/bin/python -m experiments.flexible_execution.continuing_jobs --run
```

下一项本地工作是从实际轨迹计算有来源的取证交接、真实传输与动作交叠、额外返程、错误/无用拍摄，把整项结果与同优化串行基线和原版配对。随后检验多判断场景下联合安排现场采集和语义请求能否改变关键路径。同信息规划基线也可选择所有合并与取证动作。
