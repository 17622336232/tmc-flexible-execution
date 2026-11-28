# 当前断点

<!-- continuing-snapshot -->
## 最近一次恢复位置

UTC 2026-11-28T17:34:53。native-views-dev01 主对照 6/6、强对照 6/6；native-evidence-dev02 主对照 5/6、强对照 5/6；coordinated-handoff-dev03 主对照 6/6、强对照 0/8 已取得完整评分。严格通过 20/28。新完整任务累计实发 168 次，其中取得评分的回合 159 次。独立核对提前交图 5 次，其中 2 张支持正确字段，最终报告直接引用提前图 0 次。单独的历史照片感知试验实发 18 次。当前完整任务总控已退出，最后心跳 2026-11-28T14:45:52.946380+00:00。另有 1 个运行错误未交付，保留 5 次发送。已另做 8 次零模型仿真路线检查，完成读号并回位为正面 2/4、同距离沿原方向 0/4。已有 9 次普通回复选了补图，其中 8 次下一轮选移动。原套餐中断保留，后续未发送位置已使用恢复后的同一服务。第三批两种场景的提前交图都未满足登记条件，8 个后续强对照未发送。单图实验未通过校园区域归属要求，2 个依赖它的整项也未发送。两条本机测距路线均回位。满足门框和障碍物同时可核实要求的原位扫描 1/2，近处扫描 0/2。首次两条双站巡检路线均到达第二站，直返均被原地图拒绝。改用实测经过点回程后，货位回到起点，设备在去第二站时由原导航停止。两版各有两条校准路线，零模型请求。独立的首次决策试验实发 8/8 次。8 份真实答复已封存，允许给备选动作的 4 份中有 1 份给出备选。8 份实际答复已完成独立仿真验证，新增模型请求为零，备用动作到位 1 次，满足登记证据要求 0 次。另行登记的单图多区域感知比较实发 6/6 次。多判断调度的 296/296 项离线计算已落盘。房屋成本标定实发 6 次，单问与合问的四个字段分别答对 4、4 个。新房屋完整开发任务已结算 6/6，实发 9 次。独立核验严格通过 4/6。强化对照在 108 组相同墙钟预算下与通用规划的中位可行成本相同 108 组。首轮工作单试验实发 4/4 次。可选委派放行 0/2 个任务。用封存首帧核对原版动作与来源，4/4 份答复符合。3/3 个不同判断正确。两站转向时钟校准完成 2 次尝试，其中往返均到位 0/2，新增模型请求为零。房屋56联合版整项慢 3.14 秒，增加 3.34 秒串行服务时间，少占 0.20 秒动作时间。固定原图的跨图合问标定实发 6/6 次，整对答对合问 2/2、逐问 1/2。无编号任务用原版提示实发 2/2 次，1/2 达到预先登记的接入条件。初版跨图事件模型完成 432 次名义等时预算规划，合并版对通用规划更低的可行成本出现在 5/48 组。按实际合问代价重测 960 次，合并版 5 组低于通用规划，43 组持平，0 组更高。搜索晚于预算 52 次，按预选动作计分。真实时钟驱动的三问换序记录 18 个离线条件。一次仓库侧面后续判断已登记，当前无发送收据。

用仓库封存的真实首轮判断，放行实际侧面照片里的两项已开放问题。先复核登记，只使用一轮原版模型回复并独立验收来源和许可。该检查用于验证从用户自然语言到真实后续请求的连接。三问顺序切换的整项匹配仍须保留通用和强预取，不使用正式数据。

在仓库项目环境中运行 `scripts/continuing_registered_blocks.py` 可核对全部已登记批次与额度状态。运行 `scripts/continuing_region_assay.py grade` 和 `scripts/continuing_condition_range.py grade` 可用已有原始数据独立复核两项诊断。用 `scripts/continuing_first_views.py status` 核对首次决策位置，用 `scripts/continuing_first_views.py seal` 核对封存回复，两条命令均为零新增模型请求。主线的零请求复核命令是 `scripts/continuing_v3_attribution.py`、`scripts/continuing_concurrent_workflow.py summary`。运行 `scripts/continuing_matched_event_rooms.py status` 可核对正式登记的六个开发位置。可用 `python scripts/continuing_native_action_binding.py grade` 核验四份封存答复，用 `python scripts/continuing_native_acquisition_clocks.py summary` 核对两条实测记录。两条命令均为零新增模型请求。签出当前版本后运行 `python scripts/continuing_room_clock_replay.py`、`python scripts/continuing_online_cost_replay.py`。使用 `python scripts/continuing_cross_source_assay.py grade` 和 `python scripts/continuing_natural_frontier_probe.py grade` 复核真实回复。这四个命令均不发送请求。用 `python scripts/continuing_switch_witness.py` 核对三问换序。用 `python scripts/continuing_native_bound_followup.py register` 核对仓库真实来源与两项许可，这两条也不发送请求。
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
