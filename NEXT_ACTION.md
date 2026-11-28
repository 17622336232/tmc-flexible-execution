# 当前断点

<!-- continuing-snapshot -->
## 最近一次恢复位置

UTC 2026-11-28T14:26:39。native-views-dev01 主对照 6/6、强对照 6/6；native-evidence-dev02 主对照 5/6、强对照 5/6；coordinated-handoff-dev03 主对照 1/6、强对照 0/8 已取得完整评分。严格通过 17/23。新完整任务累计实发 133 次，其中取得评分的回合 121 次。独立核对提前交图 5 次，其中 2 张支持正确字段，最终报告直接引用提前图 0 次。单独的历史照片感知试验实发 18 次。总控 3793943，equipment-inspection-03 / coordinated-deferred，心跳 2026-11-28T14:26:37.678723+00:00。另有 1 个运行错误未交付，保留 5 次发送。已另做 8 次零模型仿真路线检查，完成读号并回位为正面 2/4、同距离沿原方向 0/4。已有 9 次普通回复选了补图，其中 8 次下一轮选移动。另行登记的单图多区域感知比较实发 0/6 次。

跟踪正在进行的 coordinated-handoff-dev03 批次，逐一保留原始收据并复核完整评分。在仓库根目录使用项目 Python 环境运行 `scripts/continuing_recovery.py` 可零请求列出第二批主对照断点，追加 `--block comparators` 查看强对照。确认同一套餐恢复后可用 `--run --restoration-evidence '实际恢复凭据或确认时间'` 恢复首个未发送位置。用 `scripts/continuing_native_report.py --config config/native_evidence_dev02.json` 独立重查，再运行 `scripts/continuing_checkpoint.py` 更新统计。
<!-- /continuing-snapshot -->

状态为 RESEARCH_IN_PROGRESS。旧 456 回合和 11 个开发回合均封存。先读取 RESEARCH_STATE.json、最新迭代记录和新研究目录的账本；后续开发不沿用旧的三轮上限。

整个已登记开发序列的只读入口为 `python scripts/continuing_registered_blocks.py`。运行模式添加 `--run`，会核对已消耗的收据、接管仍在跑的同一批次，并按原证据门槛继续尚未发送的位置。总控状态保存在 `outputs/continuing/registered-blocks.json`。若出现新的套餐限制会保留所有已发送请求；恢复证据仍由原断点程序登记。

11:10 UTC 的套餐限制保留原始收据。14:00 UTC 经官方用量页核对，Diff-VLA 已启用现有套餐，新周期在该中断后开始。已从原工作副本 `b15c49c` 启动第二批此前尚未发送的货位主对照 `evidence-deferred`，首条真实回复成功返回。后续以本页顶端的回合计数和心跳为准。第二批的强对照按原门槛核验。第三批 `config/coordinated_handoff_dev03.json` 单独保存。在项目 Python 环境中运行下面的命令可核验登记、回算调度机会及生成四个布局的实拍对照图，均不发模型请求。

```bash
python -m experiments.flexible_execution.continuing_jobs --config config/coordinated_handoff_dev03.json
python scripts/continuing_wait_bounds.py
python scripts/continuing_scope_replay.py
python scripts/continuing_camera_figure.py
```

对于这次新发现的串行等待问题，可用 `python scripts/continuing_critical_path.py` 重建每个回合的时间下界，用 `python scripts/continuing_actual_offer_planners.py` 在 19 个真实选项上重查规划器。后者会重新计时，本机毫秒读数会波动。运行 `python scripts/continuing_region_assay.py register` 核对两张已冻结的图片，`python scripts/continuing_scoped_mission.py preflight` 从八次已记录的起点相机原图重算清单配准。这些操作均不发送模型请求。

套餐恢复且第二批预留的货位主对照结算后，第三批正常整项试验入口为 `python -m experiments.flexible_execution.continuing_jobs --config config/coordinated_handoff_dev03.json --run`。另行的低成本感知比较入口为 `python scripts/continuing_region_assay.py predict`，随后用 `grade` 独立核验每个区域。清单机制的两种动作策略只在预登记的感知正确性门槛满足后，按 `config/scoped_mission_dev01.json` 分别运行 `python scripts/continuing_scoped_mission.py run-next` 并最终 `grade`。仿真入口沿用项目 ROS 和既有 Python 环境。所有失败保留，登记位置用过后不会重发。

11 个旧开发回合的 93 次请求已拆解；64 份历史读号样本已做拒答式 OCR 诊断。新批次使用相同场景、360 秒任务期限及每回合至多 24 次真实请求。首个裂纹回合已通过。两张提前拍摄的照片进入后续普通判断，最终报告引用另两张常规照片。应逐张核对原图哈希、实发时间、许可、是否进入请求和最终报告分别引用了什么。

运行位置为本仓库根目录。以下第一行只查看状态。第二行在总控已退出时恢复，已有结果由收据校验。强对照使用同一入口追加 `--block comparators`，按登记条件另启。已有账户可用次数没有读数。

```bash
env PYTHONPATH=bridge:research/attack_learning:. /home/tdszy/uav/Embodied-intelligent-behavior/.venv/bin/python -m experiments.flexible_execution.continuing_jobs
env PYTHONPATH=bridge:research/attack_learning:. /home/tdszy/uav/Embodied-intelligent-behavior/.venv/bin/python -m experiments.flexible_execution.continuing_jobs --run
```

下一项本地工作是从实际轨迹计算有来源的取证交接、真实传输与动作交叠、额外返程、错误/无用拍摄，把整项结果与同优化串行基线和原版配对。随后检验多判断场景下联合安排现场采集和语义请求能否改变关键路径。同信息规划基线也可选择所有合并与取证动作。
