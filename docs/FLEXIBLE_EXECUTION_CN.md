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
