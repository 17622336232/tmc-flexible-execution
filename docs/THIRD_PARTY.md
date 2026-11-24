# 第三方与源码来源

本仓库是原 Diff-VLA 项目的仿真实验源码整理，不包含原实机工作区、Android构建、历史实验数据及账户信息。

## Diff-Planner

仅保留仿真构建所需源包：quadrotor_msgs、pose_utils、uav_utils、path_searching、traj_utils、traj_opt、plan_manage。原许可证完整保存在 `Diff-planner/LICENSE`（GNU GPL version 3）。源文件中其他版权和许可声明保持有效；没有改标为MIT。

`sim_ws/overrides/plan_env` 和 `sim_ws/src/vla_sim` 是项目已有的有限感知地图、仿真和规划适配层。源码整理没有重写规划器或新增飞行任务约束。

两个ROS包清单中移除了已经在对应CMake中注释、不实际使用的 `cmake_utils` 依赖，方便干净机器安装。构建时只建立仓库内部相对链接，不依赖用户电脑上的其他Diff-Planner目录。

## 攻击训练源码

`research/attack_learning` 提取自原轻量攻击候选的源码快照，包含在线策略、代理数据处理、条件评分和等待拟合所需模块及其导入依赖。原先藏在实验输出目录中的源码现已独立，不再把该目录当作依赖安装位置。

整理只调整了模块位置、导入路径和路径根目录。模型导入器保留数值参数，将本机路径与代码/编译契约重新绑定，并保存原包的契约摘要。这种可移植构建不能冒充历史二进制或历史结果。

场景几何和纹理通过仓库的Python源代码生成，不打包此前的渲染图、照片、视频、PCD和训练权重。ROS、Gazebo、Python库通过系统/Python包管理器安装，各自使用其原许可证。
