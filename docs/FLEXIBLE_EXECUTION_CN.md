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

## 候选排序和公开视觉

在线候选先通过廉价的许可和路线准入。普通成功标签模型和完整代价标签模型使用同一候选及特征，排序以后才逐项检查共同后续。C 在同一个检查上限下按预测净收益提前停止。动作派发前仍用当前状态复查许可、可达性和预留时间。符号训练、实际可见状态回放和在线视觉对照分别记账。

```bash
python3 -m experiments.flexible_execution.cli ranker-diagnostic \
  --data-dir outputs/ranker-diagnostic-v1 --sizes 1000 3000 10000
python3 -m experiments.flexible_execution.cli ranker-audit \
  --data-dir outputs/ranker-diagnostic-v1 --output outputs/ranker-diagnostic-v1/training_audit.json
python3 -m experiments.flexible_execution.cli pilot-run \
  --pilot-dir outputs/natural-pilot/learned-order-v1 --resume --max-jobs 4
```

公开验证入口在 `experiments.flexible_execution.public_task`。AI2-THOR 固定在 5.0.0，从官方 ProcTHOR 验证集加载房屋。相机通过软件 OpenGL 取得初始和侧向图像。模型收到任务文字和执行过程中拍到的 RGB。独立评分用姿态和实例分割核对完整报告。公开房间是本研究的派生任务，正式测试取官方测试房屋，验证房用于调试。 [ProcTHOR 官方数据](https://github.com/allenai/procthor-10k/tree/439193522244720b86d8c81cde2e51e3a4d150cf)

```bash
python3 -m experiments.flexible_execution.cli audit-public \
  --episode-dir outputs/public-visual/closed-loop-attempt-002
python3 -m experiments.flexible_execution.cli public-plan \
  --test-archive outputs/public-visual/procthor/test.jsonl.gz --output outputs/public-visual/test-plan
```

`public-plan` 校验压缩包并按固定哈希抽取正式房号。每房四项任务，准备时无需读取房屋内容。最多四种方法，共 800 个待冻结协议的作业位。

校准和选参布局可由同一仿真场景生成器实物化。布局种子控制站位，场景种子控制摆在站位里的材料。两组各 48 个布局，覆盖六种预定条件。正式运行前核对几何、材质和原始任务文字的哈希。选参使用校准阶段量得的时延和报告成本估计。

```bash
python3 -m experiments.flexible_execution.cli visual-development-prepare \
  --output outputs/development-visual-v1 \
  --pilot-dir outputs/natural-pilot/visual-calibration-v1 \
  --cost-config outputs/natural-pilot/conditional-comparison-v4/calibrated_costs.json
```

TimelyLLM 的作者代码在 CPU 上复算了随附无人机样本中 532 个任务的 1313 段执行时长，逐项与作者总时长一致。原服务调度还需 CUDA 与其固定运行时，默认触发轨迹未随这个公开快照提供。该复算只用于核对分段和计时口径。 [作者代码](https://github.com/Neawhen/TimelyLLM/tree/669f2490a112a7415fe06e902e1ea82da02a3342)

本地 STREAM 是客户端分段执行。每次导航先生成一段完整动作，再用同一张冻结照片生成已有事实和下一视角的检查计划。计划后半段只使用前一动作的意图，之后仍需用实际工具反馈及新照片验证。STREAM 在第二段请求发出后按原导航规则执行第一段。匹配的等待组收齐两段再执行。同一个模型、两段输入和发送计时适用于两组。完整任务仍使用共同的报告验收。该设计保留分段产生后即可执行的对照，未复现作者服务端的抢占、调度及算力使用。

## 参考工作的数据和训练方式

| 工作 | 数据与实验 | 本项目采用的做法 |
| --- | --- | --- |
| SELP | 自建无人机导航及桌面操作任务，有领域内规划微调、约束核验和仿真实验。 [论文](https://arxiv.org/html/2409.19471) | 由任务约束和环境核验候选，用独立规则核对成功。 |
| TypeFly | 作者编排难度递增的飞行任务，用提示词让云端模型生成 MiniSpec，视觉在边缘处理，执行中可继续查询。规划模型采用提示工程。 [项目](https://typefly.github.io/) | 使用现有模型，保留任务总时延、查询和真实动作的账；分段执行需另行比较。 |
| Hi Robot | 用遥操作示范和基于观测构造的对话训练高层策略，同时训练底层控制，在实体机器人上验证。 [论文](https://arxiv.org/html/2502.19417) | 用实拍图和已执行动作生成监督依据，分别记录上层规划与底层执行。 |
| InteLiPlan | 300 条自建文本样本覆盖正常及失败恢复场景，视觉与可行性转为文本输入，LoRA 微调 Mistral 7B 和 DeepSeek 8B。 [论文](https://arxiv.org/html/2409.14506v4) | 保留失败与恢复的训练标签，冻结在线视觉及证据检查。 |

当前闭环模型为 GPT-6.1 Terra（max），A 无需微调。B1 在本机训练小排序器。Qwen3-1.7B 留作 B2 的压缩对照。
