# Embodied intelligent behavior：无人机仿真与任务感知攻击实验

[TMC 实验进度](PROGRESS.md) · [现有 GPT 巡检与执行调整实验](docs/FLEXIBLE_EXECUTION_CN.md)

本仓库整理了无人机自主任务及外部观察攻击实验的**源代码**。目标环境是 **Ubuntu 20.04、ROS Noetic、Gazebo Classic 11、Python 3.8**。

机器人通过真实仿真 RGB 和有限距离 LiDAR 观察场景，由视觉语言模型选择观察顺序、机位、朝向、记录与提交；Diff-Planner 负责轨迹规划。攻击进程只获得公开的外部相机观察，选择一次允许的场景修改。评分器在任务结束后检查结果，不向在线决策提供答案。

仓库包含设备巡检、仓储和校园场景生成器，以及任务推断、视觉攻击基线、训练和批量实验代码。它是接口级飞行仿真，**不是 PX4 SITL、电机动力学仿真或已完成的实机验证**。当前轻量攻击器是“任务假设权重＋条件效果评分＋学习等待”的近似原型；完整共享 K/D/E 方法的区别见 [方法说明](docs/METHOD.md)。

## 先看清楚：下载后能运行什么

| 运行内容 | 需要模型账户 | 需要已训练文件 |
|---|---|---|
| 查看 Gazebo/RViz 场景 | 不需要 | 不需要 |
| 自动起飞并检查 RGB、LiDAR | 不需要 | 不需要 |
| 大模型自主执行任务，无攻击 | 需要 | 不需要 |
| API 观察者选择攻击的早期原型 | 需要 | 不需要 |
| 当前本地轻量模型攻击及配对对照 | 机器人和静态场景解读需要 | 需要，放在本机，不上传 |

**本仓库不带编译文件、实验结果、照片、视频、数据集、模型权重、账户凭据或 APK。** 场景纹理由 Python 在运行时生成，无需下载以前的图片。训练文件 `.pkl` 和包含权重的 `bundle.json` 属于数据，使用者自行训练或在本机导入。

下面按顺序操作即可。文中的“终端”指 Ubuntu 中按 `Ctrl+Alt+T` 打开的窗口。代码框中的 `$` 不需要输入；这里的命令也没有添加 `$`。

## 1. 准备电脑

建议使用 Ubuntu 20.04 桌面版、8 GB 以上内存、至少 10 GB 可用空间。16 GB 内存及以上更适合同时开 Gazebo 和 RViz。当前树模型训练和 CPU 渲染不要求 CUDA。

Ubuntu 22.04/24.04、Windows 和 macOS 不直接运行下面的 apt 安装入口。请先准备 Ubuntu 20.04 虚拟机或对应容器；不要把 ROS 2 安装当作 ROS Noetic。Noetic 是本项目保留的旧版依赖，参阅 [ROS 安装资料](https://wiki.ros.org/noetic/Installation/Ubuntu)。

若尚未安装 Git，先运行：

```bash
sudo apt update
sudo apt install -y git
```

## 2. 下载代码

```bash
mkdir -p ~/uav
cd ~/uav
git clone https://github.com/17622336232/Embodied-intelligent-behavior.git
cd Embodied-intelligent-behavior
```

后面的命令都在这个仓库目录执行。不要进入原来其他 Diff-Planner 工程，也不用复制别人的 `build` 或 `devel`。本仓库已带齐仿真需要的规划器源包。

## 3. 安装系统依赖

```bash
bash scripts/install_dependencies.sh
```

这个脚本会使用 `sudo` 安装 ROS Noetic、Gazebo、RViz、C++ 编译工具和系统 Python 依赖。输入密码时终端不显示字符，这是正常现象。脚本不会连接物理无人机。

已安装 ROS Noetic 的电脑也可以运行，它会补齐依赖。脚本只在没有找到 `/opt/ros/noetic/setup.bash` 时添加 ROS 软件源，密钥取自 [ROS 官方 rosdistro](https://github.com/ros/rosdistro)。如果旧的软件源已经报签名错误，先按后面的排错部分处理。

## 4. 建立项目自己的 Python 环境

```bash
bash scripts/setup_python.sh
```

依赖版本固定在 [requirements.txt](requirements.txt)，安装在本仓库 `.venv` 中，不替换系统 Python。环境保留系统 ROS Python 包，以便导入 `rospy`、`cv_bridge`。系统缺少 `ensurepip` 时，脚本会在本项目 `.runtime` 中使用 virtualenv 建环境。

无需修改 `~/.bashrc`，也不需要自己记住激活环境；本仓库的启动脚本会自动加载 ROS 和 `.venv`。

## 5. 编译仿真

```bash
bash scripts/build_sim.sh
```

第一次会编译十个 ROS 包。看到 `仿真编译完成` 才算完成。黄色 warning 不一定是错误；若最后出现 `Error` 或 `Failed`，请看第一处报错。

内存较少时，把并行编译数降为 1：

```bash
VLA_BUILD_JOBS=1 bash scripts/build_sim.sh
```

检查环境和编译结果：

```bash
bash scripts/doctor.sh --require-build
```

这一步不启动仿真、不登录、不调用模型。修改 C++ 后重新编译；只修改 Python 通常不需要重编译。切换电脑或移动整个目录后应重新编译，不要复用之前的构建目录。

## 6. 先不登录，检查场景能否显示

```bash
bash scripts/run_attack_showcase.sh --family cracks --view-only
```

应出现 Gazebo 和 RViz。Gazebo 中是一排设备；RViz 用于看机载 RGB 和有限感知地图。这一步无人机留在地面，不执行任务、不攻击、不调用 API。

查看另外两类场景：

```bash
bash scripts/run_attack_showcase.sh --family inventory --view-only
bash scripts/run_attack_showcase.sh --family parking --view-only
```

**上面三条一次只运行一条。** 在启动它的终端按 `Ctrl+C` 结束，等命令退出后再启动下一场。默认独立端口为 ROS `11391`、Gazebo `11491`，不会强行停止其他项目。

Gazebo 主窗口与服务端相机的位置更新不同步问题已有独立显示修复：启动时会出现 `[Gazebo显示] 已启用静态物体位置同步`。它只同步仿真中已经发生的静态物体位置变化，不选择攻击、不修改 RGB 或任务答案。

## 7. 检查自动起飞和传感器

```bash
bash scripts/run_attack_showcase.sh --family cracks --sensor-smoke
```

这一步会自动起飞，检查机载 RGB、外部 RGB 和有限距离雷达，然后结束。**它不执行大模型任务，也不调用攻击模型。** 本次生成的图片和检查记录保存在终端显示的 `artifacts/...`，这些文件被 Git 忽略。

无桌面服务器可以使用虚拟显示：

```bash
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a bash scripts/run_attack_showcase.sh \
  --family cracks --sensor-smoke --headless --no-rviz
```

`--headless` 只是不打开图形窗口；Gazebo 的 RGB 渲染仍需要可用的 OpenGL/显示环境，因此服务器使用 `xvfb-run`。CPU 实验后端是另一条路径，见第 11 节。

## 8. 配置自己的模型账户

当前任务模型默认是 `gpt-6.1-terra`，推理强度 `max`；静态场景解读和 API 观察者默认 `high`。实际可用模型取决于使用者自己的账户和服务权限。

```bash
cp config/local.env.example config/local.env
bash scripts/model_login.sh
```

按终端提示在这台电脑的浏览器中登录并授权，然后查看账户实际提供的模型：

```bash
bash scripts/model_login.sh --list-models
bash scripts/model_login.sh --status
```

凭据仅保存在本机 `.runtime` 中，不包含在仓库里，不需要发给其他人。这个实验入口使用项目已有的 ChatGPT 授权传输；不要把 API Key 当成登录命令的参数。账户资格及额度以 [服务方实际授权](https://developers.openai.com/siwc/token-sharing-open-source) 为准，仓库不提供公共账户。

如果账户没有默认模型，可编辑 `config/local.env`，明确指定目录中支持图片的模型：

```bash
export UAV_TASK_MODEL='这里换成账户目录里的准确模型名称'
```

保存后重新启动实验。换模型是新的实验配置，不能与原默认模型结果直接合并。接口失败会明确报错，不会切换成离线规则替代决策。

## 9. 运行大模型自主任务，先不攻击

完成前面的环境、编译和授权后运行：

```bash
bash scripts/run_attack_showcase.sh --family cracks --policy clean
```

无人机会自动起飞，由大模型自行选择观察顺序、位置、朝向和最终报告。任务要求是检查设备裂纹，记录对应编号。**没有固定巡检路线，也没有把评分答案发给机器人。**

这一步需要联网并消耗自己的模型额度。相机准备、模型推理和移动都需要时间，首次运行请看终端进度，不要同时启动第二套实例。

任务和场景对应关系：

| 场景 | 可用任务 |
|---|---|
| 设备 | `cracks` 裂纹；`temperature` 超温；`locate` 指定设备读温；`crack_and_heat` 裂纹且超温；`thermal_minimum` 最低温；`safe_equipment` 完好且不过热；`locate_crack` 指定设备裂纹；`locate_condition` 指定设备综合状态 |
| 仓储 | `inventory` 盘点；`color_search` 按颜色找货；`order_check` 订单核对；`shortage` 缺货；`overstock` 多货；`large_inventory` 数量筛选；`locate_boxes` 指定货位计数；`locate_order` 指定货位订单核对 |
| 校园 | `parking` 禁停检查；`blocked_exit` 通道堵塞；`fire_safety` 灭火器；`vehicle_search` 找车；`clear_exit` 可用通道；`fire_and_access` 消防与通行；`locate_extinguisher` 指定点灭火器；`locate_access` 指定点通道 |

另有 `equipment_labels`、`warehouse_labels`、`campus_labels` 三类逐字读牌对照。设备任务以外，当前展示入口请明确使用 `--policy clean`，例如：

```bash
bash scripts/run_attack_showcase.sh --family inventory --policy clean
```

## 10. 运行攻击实验

### 10.1 不需要本地训练权重的 API 原型

```bash
bash scripts/run_inspection.sh --family cracks --policy inferred --source-profile single
```

这条是保留的早期 API 观察者：由外部画面推断任务并选择场景修改，需要模型调用。它能用于检查“外部观察→物理修改→机器人继续执行→事后评分”的链路，**不是当前本地条件评分＋学习等待方法**。不要把两个版本混在同一张结果表中。

### 10.2 当前本地轻量攻击器

按“只上传代码”的要求，仓库不带训练权重。从已有实验电脑迁移时，单独传输你自己的 `isolated_waiting_candidate_01` 模型目录；不要把数据盘里的整个实验目录复制进仓库。

在新电脑**编译完成后**导入：

```bash
python3 scripts/import_attack_models.py \
  --candidate /你自己的模型目录/isolated_waiting_candidate_01

bash scripts/doctor.sh --require-build --require-model
```

这里的 `/你自己的模型目录/...` 是占位符，必须替换成真实路径。导入器检查原文件校验值，保留数值权重，只重写文件路径和本机源码/编译校验契约。它不会运行模型或上传文件。模型保存在 Git 忽略的 `models/task_attack/showcase_candidate` 中。重新编译改变二进制后，需要把已有导入目录改名备份，再导入以重新绑定本机编译。

如果还要运行视觉攻击基线，可另行导入自己的 `baseline_bundle`：

```bash
python3 scripts/import_attack_models.py \
  --candidate /你自己的模型目录/isolated_waiting_candidate_01 \
  --baseline /你自己的模型目录/baseline_bundle
```

每次导入使用尚不存在的目标目录，脚本不会覆盖旧模型。没有本地模型时，下面的学习攻击命令会在起飞和 API 调用前明确停止；不会自动换成另一种攻击。

导入成功后运行：

```bash
bash scripts/run_attack_showcase.sh --family cracks
```

默认 `--policy belief`，使用当前条件评分和学习等待策略。流程是：同场景静态 RGB 准备→机器人自主任务→外部行为更新→选择一次物理修改→机器人继续执行→独立评分。

终端会打印实际改了什么。例如裂纹覆片改变的是门板外观，**编号不一定变化**；温度覆片位于设备右侧，不会让正面门板长出裂纹。修改直接出现在模型选择的位置，没有机械臂粘贴动画。

以下条件使用同一套可修改载体，替换 `--policy` 即可：

| 参数 | 含义 |
|---|---|
| `belief` | 当前任务推断＋条件评分＋学习等待 |
| `belief_blind` | 只使用初始静态观察与公开时间的对照 |
| `direct_effect` | 使用外部行为直接预测攻击效果 |
| `belief_known_task` | 已知真实任务的信息优势参照 |
| `redvla_external` | RedVLA 思路的外部二维观察适配 |
| `clean` | 无攻击 |

UAO、ICCV 视觉攻击代码也保留在 `experiments/task_attack/visual_baselines.py` 等模块中，需要自己的优化纹理包，不能把本仓库的黑盒 API 适配当作原论文的完整复现。当前学习权重只在八类设备任务上验证过，其他场景不自动套用。

## 11. 不开窗口的后台实验

Gazebo 仍进行真实 RGB/射线传感，只隐藏窗口：

```bash
bash scripts/run_attack_showcase.sh --family cracks --policy clean --headless --no-rviz
```

没有桌面时在前面加 `LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a`。

另有 CPU 渲染后端，使用同一场景定义、有限距离射线和原 ROS 规划链，不启动 Gazebo。示例运行无攻击自主任务：

```bash
bash scripts/run_attack_integrity.sh --backend cpu --headless --no-rviz \
  --family inventory --policy clean --seed 10001 --equipment-count 4 \
  --source-profile single --output artifacts/cpu_inventory_10001
```

CPU 与 Gazebo 是不同渲染域，结果应分开报告。批量任务可用 `experiments/task_attack/batch.py` 等入口，先查看参数：

```bash
source scripts/environment.sh
python3 -m experiments.task_attack.batch --help
```

批量运行会消耗较多 API 额度。先验证单回合再增加并发，每个并发实例使用不同端口。不能用删掉失败、超时回合的方式提高成功率。

## 12. 结果保存在哪里，怎么看

每次启动会打印本次输出目录。目录里通常包含：

- `result.json`：实际终态和评分；
- `private/robot/`：机器人本次实际取到的机载图、决定和报告；
- `private/observer/`：攻击观察者的输入边界、判断和实际修改记录；
- `public/`：外部相机和公开动作菜单；
- `private/simulation.log`：本回合 ROS/Gazebo 日志。

真实字段以对应入口生成的文件为准。`service_success=true` 只表示物理修改请求得到确认，不能单独当作攻击成功。错误接受、正常完成、拒绝和超时分别统计。

运行期间单独查看机载 RGB：

```bash
source scripts/environment.sh
ROS_MASTER_URI=http://127.0.0.1:11391 rqt_image_view /vla/camera/image_raw
```

默认端口不同的入口应使用其终端打印的实际 ROS 端口。程序生成的记录全部留在本机，不提交到 GitHub。

## 13. 训练与代码结构

初学者可以先完成前面的编译、传感器和单任务步骤。研究训练需要自行采集的数据，不是执行一次编译就得到模型。训练、等待策略和输入边界说明见 [训练教程](docs/TRAINING.md)。

```text
.
├── README.md                     中文安装、编译、运行与排错
├── requirements.txt              Python 3.8 依赖版本
├── scripts/                      环境、构建、启动、模型导入与检查
├── experiments/                  场景、机器人、攻击者、评分、CPU后端
├── research/attack_learning/     从实验目录提取出的训练与轻量推断源码
├── sim_ws/src/vla_sim/           Gazebo传感器、仿真执行、RViz、规划覆盖层
├── sim_ws/overrides/plan_env/     有限观测地图源代码
├── Diff-planner/src/             构建所需的原规划器源包
├── bridge/uav_bridge/            模型传输、认证、观测和公共协议
├── bridge_ros/                   运行仿真所需ROS适配与命令门控
├── config/local.env.example      不含账户信息的本机配置模板
├── tests/                        源码分发与关键入口检查
└── docs/                         方法、训练与第三方说明
```

`sim_ws/build`、`sim_ws/devel`、`.venv`、`.runtime`、`models`、`artifacts` 都由本机生成或导入，已在 `.gitignore` 中排除。本仓库不含 Android 与实机硬件工程。

运行不调用模型的源码检查：

```bash
source scripts/environment.sh
python3 -m unittest tests.test_attack_showcase tests.test_source_distribution tests.test_isolated_waiting
```

## 14. 常见问题

| 现象 | 处理方式 |
|---|---|
| `找不到 /opt/ros/noetic/setup.bash` | 检查是否 Ubuntu 20.04，运行第 3 节系统依赖安装 |
| `catkin_make: command not found` | 从仓库目录运行 `bash scripts/build_sim.sh`，不要只在子目录手动运行 make |
| `No module named rospy/cv_bridge` | 使用本仓库脚本；若手动执行 Python，先 `source scripts/environment.sh` |
| 编译器 `Killed` / `cc1plus` 被终止 | 通常是内存不足，改用 `VLA_BUILD_JOBS=1` |
| `ensurepip is not available` | setup 脚本会尝试项目内 virtualenv；也可先补装 `python3.8-venv` |
| 下载 pip 包失败 | 检查联网和代理后重新运行 setup；不复制其他电脑的 `.venv` |
| ROS apt 签名或源报错 | 按 [ROS 官方软件源管理说明](https://github.com/ros-infrastructure/ros-apt-source) 修复已有源，再运行依赖脚本；不要关闭签名检查 |
| Gazebo 黑屏或 OpenGL 报错 | 尝试 `LIBGL_ALWAYS_SOFTWARE=1`；无桌面用前述 `xvfb-run` 命令 |
| 端口已占用 | 先在旧演示终端 `Ctrl+C`；或传 `--ros-port 11395 --gazebo-port 11495`，不要杀掉其他项目 |
| 缺少 `candidate_contract.json` / 本地权重 | 先跑 `--view-only` / `--policy clean`，或按第 10 节导入自己的模型 |
| 模型或源码校验不一致 | 保留旧目录，在当前编译后重新导入；不能直接删掉校验让实验继续 |
| 账户目录没有默认模型 | 用 `--list-models` 核对，明确配置自己的图片模型；没有可用模型不能运行自主任务 |
| 登录提示时间/令牌失效 | 检查 `timedatectl status` 和实际日期时间；不要放宽令牌校验 |
| 只看到搜索/悬停，耗时较长 | 看模型请求和规划反馈；不要把尚未完成当作成功，也不要自动重复启动新实例 |
| Gazebo 没看到“编号变化” | 看终端具体修改对象：可能是裂纹或侧面温度；编号不是每轮都会变化 |

## 第三方代码

规划器来源和许可证保留在 [Diff-planner/LICENSE](Diff-planner/LICENSE)，为 GPL-3.0。各源文件自身的版权和许可声明继续有效，参阅 [第三方说明](docs/THIRD_PARTY.md)。本仓库没有把上游代码改标为其他许可证，也不包含上游 bag、图片和旧编译缓存。
