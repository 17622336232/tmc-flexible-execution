# 巡检执行调整实验

当前分支实施 TMC 1.1 协议的本地启动。运行目录包含源代码校验、任务盘点、资源记录、每回合事件、独立重评分和断点状态。

## 本地机制诊断

从本仓库根目录运行。依赖为项目已有的 Python 和 PyYAML。

```bash
python3 -m experiments.flexible_execution.cli audit --profile local5060
python3 -m experiments.flexible_execution.cli run --profile local5060 --resume
python3 -m experiments.flexible_execution.cli status --profile local5060
python3 -m experiments.flexible_execution.cli reproduce --run-id local5060-start
python3 -m unittest tests.test_flexible_execution
```

默认输出在 `outputs/local5060-start`。用新的 `--run-id` 隔离另一次运行。恢复时原始代码和配置需与运行时匹配，已落盘的回合保留。

三个方法共享同一任务和验收。ASYNC 做已获准的独立必做工作，使用已有材料，并在需要时返访。PREFETCH 对不同结果所需的材料做共享采集和成本选择。A 对材料与技能建立索引，检查提前取证后是否有共同可执行的终态动作。

这 12 个案例用于验证机制，动作和查询耗时按案例设定，计算开销使用本机实测。自然服务延迟、真实视觉质量和任务泛化需要后续配对实验。

## 原导航仿真接通

使用原精简仓库已经构建的 ROS Noetic 仿真与 Python 环境。入口将首张仿真机载图提交后台任务，用正常导航接口移动到附近观测点并采新图，再验收原查询的回调。

```bash
bash scripts/run_flexible_cpu_smoke.sh --output outputs/native-async-recheck
```

已有同名输出时选择新目录。可用 `--ros-port` 和 `--gazebo-port` 指定空闲端口，原精简仓库地址可通过 `VLA_LEGACY_ROOT` 指定。输出包括两张实际仿真相机图、导航日志和 `async_navigation_probe.json`。后台任务用受控返回时刻验证并行语义，图像解码耗时单独记录。

## 复用现有 GPT 双阶段运行

新入口沿用巡检项目的 ChatGPT 授权、配置的模型和推理强度。`--chatgpt-project` 指向当前电脑上已经完成项目登录的目录。仿真使用现有 ROS 导航和 RGB，相机画面由该回合实际采集。服务由账户的 ChatGPT 计划授权。

```bash
bash scripts/run_flexible_clean.sh --method serial --backend cpu \
  --chatgpt-project /home/tdszy/uav/Diff-VLA \
  --family cracks --seed 1 --equipment-count 2 --episode-timeout 300 \
  --output outputs/existing-gpt-pilot/serial-reproduction

bash scripts/run_flexible_clean.sh --method async-probe --backend cpu \
  --chatgpt-project /home/tdszy/uav/Diff-VLA \
  --family cracks --seed 1 --equipment-count 2 --episode-timeout 180 \
  --output outputs/existing-gpt-pilot/frozen-query-reproduction
```

`serial` 沿用原任务提示词与串行决策循环。在 `private/robot/timeline.jsonl` 记录同一时钟下的请求、动作和相机时间。`private/robot/model_calls.jsonl` 保留原模型名、实际图像哈希和每次调用耗时。`clean_timing.json` 可从时间轴逐段重算。

`async-probe` 先从指令整理一个逐对象判断任务，再用实拍图定位可见目标。首图足够时会复用首图。需要补拍的目标通过原视觉定位把图像框换成地图核验过的观测位置。对一张冻结的照片开始单独判读后，执行器可选择仍需补拍的独立目标。判读回复只允许提供判断结果，并与提交时冻结的图片绑定。`private/robot/gpt_probe.json` 记录这一接口的结果及是否取得了另一张照片。整项巡检报告标为未完成，因为探针只验收一条判读。

当前任务整理入口支持一组对象上的单项可见判断，以及由结果触发的材料要求。材料的提前许可由原指令中的引用保留在草稿中。探针据此展示条件要求，实际移动只做独立必做的检查。条件分支取证、完整新评分和公平配对需要在同一任务前端下接通后再进入 48 个开发实例。

目标框深度无法可靠确定时，视点模块可以从同一画面上的实际雷达表面回波提出补充观察位。候选通过原地图准入，移动时原导航再次检查。表面回波只用于选择下一处观察，相机材料和对象身份以后续实际拍摄为准。

`paired-serial` 和 `paired-async` 使用同一任务解析和观测选择。串行组在收到原照片的判读后补拍，异步组允许在等待期间补拍。两组都把哈希匹配的相机图交给原完整任务报告入口，保留各自实际采到的视野。原报告验收和新增逐对象覆盖验收分别保留。固定延迟的本地回放归入接线诊断。

可对已有 `clean` 完整运行离线核对报告和逐对象记录。覆盖检查使用私有评分真值、实际发送的图片哈希和对应的模型输出，不启动模型请求。

```bash
python3 -m experiments.flexible_execution.cli audit-clean \
  --episode-dir outputs/existing-gpt-pilot/cracks-serial-seed1-recheck
```

开发实例排程从现有 24 项任务各安排两组种子，并固定每组的串行和异步运行顺序。准备和状态命令只读写本地文件。执行命令核对冻结协议，跳过已经验收的回合，并在有记录的剩余预算内接续。

```bash
python3 -m experiments.flexible_execution.cli pilot-prepare --pilot-dir outputs/natural-pilot
python3 -m experiments.flexible_execution.cli pilot-status --pilot-dir outputs/natural-pilot
python3 -m experiments.flexible_execution.cli continue --profile local5060 --resume --max-jobs 2
python3 -m experiments.flexible_execution.cli pilot-report \
  --pilot-dir outputs/natural-pilot --output outputs/natural-pilot/reports/base
```

配对目录用 `pair-001/paired-serial` 和 `pair-001/paired-async` 这样的相对路径。状态核对同一场景种子、任务、设备数量、后端、时限、模型和推理强度，两边都产生运行结果才记为已记录，整项覆盖另外统计。

总控优先恢复记录中断的排程。已识别的临时服务错误在同批还有启动名额时，分别等待 15 秒和 30 秒后恢复原作业，沿用累计时限和请求预算。达到三次基础设施尝试上限、额度或权限受阻，以及遇到未知错误时，保留现场供诊断。

## 候选排序和公开视觉

逐臂报表从尝试日志累计本机工作进程及其已回收子进程的用户态与内核态 CPU 秒，同时记录模型请求、搜索、运动和取证时间。本机 CPU 缺测的尝试标为缺测。模型服务侧的计算量无法从这些日志取得。能量字段当前为未测。

受限时间沿用协议登记的累计尝试时钟。尝试时钟齐全时，逐臂表也报告从第一次接收任务到末次退出的总历时，并列出尝试间隔。

从已重算的汇总绘制整项成功率差和受限时长差。图中每行标注独立来源组数，线段为按来源组重采样的 95% 区间。受限时长按登记时限计入失败回合。

```bash
python3 -m experiments.flexible_execution.evidence_figure \
  --paired-summary outputs/natural-pilot/reports/base/summary.json \
  --output outputs/figures/base-paired.png
```

在 `pilot-report` 加 `--coverage-object CP-02 --coverage-domain campus` 可从原始覆盖验收和模型决策逐回合复算缺证。可见性还需要核对机载原图。

在线候选先通过廉价的许可和路线准入。普通成功标签模型和完整代价标签模型使用同一候选及特征，排序以后才逐项检查共同后续。C 在同一个检查上限下按预测净收益提前停止。动作派发前仍用当前状态复查许可、可达性和预留时间。符号训练、实际可见状态回放和在线视觉对照分别记账。

检索组从与 B1 完全相同的训练子集取七个最近的候选，以标准化结构特征为距离，用这些经程序验证的符号标签的平均收益排序。取样、归一化和索引文件都在运行前锁定。检索、启发式、普通监督、B1 和 C 共用候选池、检查器和时钟预算。

符号选参先对比 1,000、3,000、10,000 个训练状态在三个种子下的受限时间，时间并列选数据较少的一组。视觉选参固定同一训练规模，在同一批 48 个独立场景中匹配比较最多检查 4 或 8 个候选。五种排序方法和两种上限均进入每个场景。优先选择通过整项验收更多的上限，其次比较失败按截止时间计入的平均时长，最后选择较小上限。正式测试为普通监督、B1 和 C 各保留三个训练种子。统计先把同一基础场景中同种子的结果配对，再按场景来源分组计算区间。

真实闭环中，同一张传感器图仍可能得到不同的模型目标框和要求。固定候选池回放从真实训练场景的已绑定照片与判读输入恢复同一候选列表，所有排序器在该列表上用同一个许可检查器。回放统计检查量和预计完成时间，实体照片与最终报告仍由各自的真实轨迹独立验收。 新增选参和自建测试的组内参照由最先完成任务解释和目标定位的作业登记。每臂仍各自发出真实解释和定位请求，计入发送和耗时。当场景、授权任务、模型、推理设置和首图字节一致时，控制器使用同一份解释和目标框。实际机载照片与导航许可在本臂重新验证。首图不同就沿用本臂的模型结果，并在报告记录差异。

```bash
python3 -m experiments.flexible_execution.cli ranker-diagnostic \
  --data-dir outputs/ranker-diagnostic-v1 --sizes 1000 3000 10000
python3 -m experiments.flexible_execution.cli ranker-audit \
  --data-dir outputs/ranker-diagnostic-v1 --output outputs/ranker-diagnostic-v1/training_audit.json
python3 -m experiments.flexible_execution.cli ranker-configurations \
  --data-dir outputs/ranker-diagnostic-v1 --output outputs/natural-pilot/ranking-configurations-v1
python3 -m experiments.flexible_execution.cli native-training-prepare \
  --development-manifest outputs/development-visual-v2/split_manifest_v2.json \
  --cost-config outputs/natural-pilot/conditional-comparison-v4/calibrated_costs.json \
  --ranker-config outputs/natural-pilot/ranking-configurations-v1/cap-4.json \
  --output outputs/natural-pilot/native-training-check-v2
python3 -m experiments.flexible_execution.cli pilot-run \
  --pilot-dir outputs/natural-pilot/native-training-check-v2 --resume --max-jobs 7
python3 -m experiments.flexible_execution.cli ranker-native-replay \
  --pilot-dir outputs/natural-pilot/native-training-check-v2 \
  --output outputs/natural-pilot/reports/native-training-fixed-pools-completed.json
python3 -m experiments.flexible_execution.evidence_figure \
  --fixed-pools outputs/natural-pilot/reports/native-training-fixed-pools-completed.json \
  --output outputs/figures/native-ranking-replay.png
```

公开验证入口在 `experiments.flexible_execution.public_task`。AI2-THOR 固定在 5.0.0，从官方 ProcTHOR 验证集加载房屋。相机通过软件 OpenGL 取得初始和侧向图像。模型收到任务文字和执行过程中拍到的 RGB。独立评分用姿态和实例分割核对完整报告。公开房间是本研究的派生任务，正式测试取官方测试房屋，验证房用于调试。 [ProcTHOR 官方数据](https://github.com/allenai/procthor-10k/tree/439193522244720b86d8c81cde2e51e3a4d150cf)

正式公开比较在登记前读取四间验证房的实际动作、取景和模型请求用时。先在每间房内计算中位数，再跨房计算初始判读中位数和后续开销的上四分位数。正式登记锁定这些估计、样本来源和四种方法的共同预算。逐臂报表用已写完的动作记录复核原指令许可，基础设施重试中的转向也计入。

```bash
python3 -m experiments.flexible_execution.cli audit-public \
  --episode-dir outputs/public-visual/closed-loop-attempt-002
python3 -m experiments.flexible_execution.cli public-plan \
  --test-archive outputs/public-visual/procthor/test.jsonl.gz --output outputs/public-visual/test-plan
```

`public-plan` 校验压缩包并按固定哈希抽取正式房号。每房四项任务，准备时无需读取房屋内容。最多四种方法，共 800 个待冻结协议的作业位。

校准和选参布局可由同一仿真场景生成器实物化。布局种子控制站位，场景种子控制摆在站位里的材料。两组各 48 个布局，覆盖六种预定条件。正式运行前核对几何、材质和原始任务文字的哈希。选参使用校准阶段的实际判读、相机、动作和报告计时。移动成本从通过校验的动作前后照片位姿拟合。没有回包的等待只作为下限证据，样本量不足时保留早期实测配置。选参和自建测试保留机载证据与完整任务日志。跑分回合可通过登记的存储选项跳过外部旁观录像，并把低层传感器与 ROS 日志按可校验的 gzip 文件保存。

```bash
python3 -m experiments.flexible_execution.cli visual-development-prepare \
  --output outputs/development-visual-v2 \
  --pilot-dir outputs/natural-pilot/visual-calibration-v3 \
  --cost-config outputs/natural-pilot/conditional-comparison-v4/calibrated_costs.json
```

TimelyLLM 的作者代码在 CPU 上复算了随附无人机样本中 532 个任务的 1313 段执行时长，逐项与作者总时长一致。原服务调度还需 CUDA 与其固定运行时，默认触发轨迹未随这个公开快照提供。该复算只用于核对分段和计时口径。 [作者代码](https://github.com/Neawhen/TimelyLLM/tree/669f2490a112a7415fe06e902e1ea82da02a3342)

本地 STREAM 是客户端分段执行。每次导航先生成一段完整动作，再用同一张冻结照片生成已有事实和下一视角的检查计划。计划后半段只使用前一动作的意图，之后仍需用实际工具反馈及新照片验证。STREAM 在第二段请求发出后按原导航规则执行第一段。匹配的等待组收齐两段再执行。同一个模型、两段输入和发送计时适用于两组。完整任务仍使用共同的报告验收。该设计保留分段产生后即可执行的对照，未复现作者服务端的抢占、调度及算力使用。

ALFWorld 0.4.2 的官方文字发布包有 3,553 个训练、140 个已见验证和 134 个未见验证游戏文件。用 TextWorld 1.6.2 实际回放了一个已见验证的蜡烛放置任务，官方轨迹四步结束，环境确认完成。使用原始文字状态和发布轨迹，无模型请求。证据只用于核对文字任务的执行语义。 [ALFWorld 官方发布](https://github.com/alfworld/alfworld/releases/tag/0.4.2)

OpenEQA 保持为可选的公开视觉问答参考。作者列出的 HM3D RGB 历史包约 12 GB，超过本机保留 30 GiB 后可新增空间。本轮优先用已经可运行的 ProcTHOR 进行主动取证。 [OpenEQA 数据说明](https://github.com/facebookresearch/open-eqa/blob/main/data/README.md)

## 参考工作的数据和训练方式

| 工作 | 数据与实验 | 本项目采用的做法 |
| --- | --- | --- |
| SELP | 自建无人机导航及桌面操作任务，有领域内规划微调、约束核验和仿真实验。 [论文](https://arxiv.org/html/2409.19471) | 由任务约束和环境核验候选，用独立规则核对成功。 |
| TypeFly | 作者编排难度递增的飞行任务，用提示词让云端模型生成 MiniSpec，视觉在边缘处理，执行中可继续查询。规划模型采用提示工程。 [项目](https://typefly.github.io/) | 使用现有模型，保留任务总时延、查询和真实动作的账；分段执行需另行比较。 |
| Hi Robot | 用遥操作示范和基于观测构造的对话训练高层策略，同时训练底层控制，在实体机器人上验证。 [论文](https://arxiv.org/html/2502.19417) | 用实拍图和已执行动作生成监督依据，分别记录上层规划与底层执行。 |
| InteLiPlan | 300 条自建文本样本覆盖正常及失败恢复场景，视觉与可行性转为文本输入，LoRA 微调 Mistral 7B 和 DeepSeek 8B。 [论文](https://arxiv.org/html/2409.14506v4) | 保留失败与恢复的训练标签，冻结在线视觉及证据检查。 |

当前闭环模型为 GPT-6.1 Terra（max），A 无需微调。B1 在本机训练小排序器。Qwen3-1.7B 留作 B2 的压缩对照。
