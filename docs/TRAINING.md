# 本地轻量模型的训练与迁移

这部分面向后续研究开发。首次使用请先按根目录README完成编译、传感器检查和单回合自主任务。本仓库提供代码，不带历史训练数据、结果或参数。

## 1. 训练的是什么

- 任务推断：外部行为前缀与候选任务描述组合，预测各假设的权重。现用ExtraTrees为96棵树。
- 条件效果评分：给定同样外部信息、候选任务与具体修改，学习清洁执行轨迹上的反事实代理效果。
- 等待价值：比较立即修改、等待不同时间尺度和放弃；使用64棵树及迭代回传。
- 辅助观察模型：从允许的外部特征估计可能的观察对象/部位等，不直接读取在线真实机位或机载图。

训练GPT不是这些脚本的功能。仿真任务执行与静态场景标注需要模型API；树模型拟合主要使用CPU。

## 2. 原始数据与训练标签

在自有机器人/仿真上运行任务，保留外部RGB与像素事件、实际机载取图时间、感知与记录过程、真实任务及事后报告。内部日志用于离线制作监督标签，不能进入未知任务攻击者的在线输入。

数据按完整场景分组，而不是把同一回合的不同帧随机分开。任务推断模型以真实授权任务作离线标签；效果代理结合任务语义、正常轨迹的实际后续读取机会与经验采用系数；等待目标由后续已学习策略的价值迭代产生。

这些代理标签不等于每个动作都已实际攻击。受害者受到修改后可能改路线或恢复，仍需完整配对执行。输出中的 `proxy`、`not ASR` 不应改写成真实攻击成功率。

## 3. 当前源码入口

`research/attack_learning` 已从原实验结果目录提取出来，按以下命令运行；它们不会默认找数据盘路径。

```bash
bash scripts/research.sh crossfit_dependency_stopping_data --help
bash scripts/research.sh fit_isolated_waiting --help
bash scripts/research.sh publish_conditional_critic_candidate --help
```

| 模块 | 用途 |
|---|---|
| `collect_target_crossed_training` | 规划/采集指定对象参数交叉的自有正常执行数据 |
| `build_local_operator_dataset` | 将原始执行和操作标注整理成局部训练集 |
| `augment_named_local_dataset` | 补充指定对象配置与对应标注 |
| `prepare_common_operator_baselines` | 用同一自有数据构造各方法可共享的效果代理标签 |
| `train_common_operator_baselines` | 拟合相同资源条件下的评分器 |
| `crossfit_dependency_stopping_data` | 五折场景交叉拟合，生成任务推断与等待训练轨迹 |
| `fit_isolated_waiting` | 分离各折的整个等待迭代链，选择训练迭代后再诊断校准集 |
| `publish_conditional_critic_candidate` | 将权重、运行契约与源码快照冻结为新的本地候选 |

保留这些模块的导入依赖，避免脚本挪出实验目录后缺少内部模块。`programs/equipment.json` 是有限任务程序配置，不含某个测试场景的设备答案。

## 4. 已有准备数据时的实际命令

这不是从空目录就能完成的三条命令。先准备下列**本地输入**，它们不在Git仓库中：

1. 局部操作数据目录，包含 `dataset.json.gz`；
2. 共同效果标签目录，包含 `training_values.json.gz`；
3. 共同评分器目录，包含 `ranker_candidates.json`；
4. 这些数据所引用的原始执行目录及静态场景解释；
5. 兼容的基底参数目录 `models/task_attack/integrity_v19`，及数据构造器需要的其他基底参数。

迁移自己的已有基底参数时，可在根目录README的导入命令上增加：

```bash
python3 scripts/import_attack_models.py \
  --candidate /自己的模型/isolated_waiting_candidate_01 \
  --supporting-models /自己的旧工程/models/task_attack
```

已有导入目标先改名备份，避免覆盖。原始数据中如果仍记录旧电脑的绝对目录，先在**新数据副本**中迁移路径，保留原始记录，不覆盖历史结果。路径重写不能改变训练/校准划分、标签或评分。

下面 `/你的数据/...` 都要替换成自己的真实目录。输出路径必须是新目录。

```bash
bash scripts/research.sh crossfit_dependency_stopping_data \
  --dataset /你的数据/local_operator_dataset \
  --values /你的数据/common_training_values \
  --rankers /你的数据/common_rankers \
  --output artifacts/new_stopping_data

bash scripts/research.sh fit_isolated_waiting \
  --dataset artifacts/new_stopping_data \
  --output artifacts/new_waiting_fit

bash scripts/research.sh publish_conditional_critic_candidate \
  --data artifacts/new_stopping_data \
  --stopping artifacts/new_waiting_fit \
  --output models/task_attack/new_candidate
```

这会生成新的候选；不会自动替换现有展示模型。检查候选后，按根目录README的导入流程安装到本机展示目录。所有方法应使用同样的数据与资源预算，不能为某个方法单独选择有利数据。

## 5. 从头采集时要注意什么

可先用普通自主任务入口，在新的 `--output` 目录采集正常回合；运行前规划固定任务、场景种子、来源条件和训练/校准分组。`collect_target_crossed_training` 保留了18种任务参数配置的采集代码，依赖已有基底参数，并不是全新项目的一键自举训练器。

历史构造脚本仍包含其声明的输入格式和阶段依赖。仓库提供了这些源代码，但没有把此前多阶段训练材料包装成不存在的“开箱即得模型”。新的数据采集、标签构造和完整终点验证仍需实际执行，不应生成假数据来填满目录。

## 6. 当前仍未完成的部分

当前轻量模型是有限候选空间上的判别式任务权重、条件代理评分和独立等待学习。五折等待递推隔离不意味着整条历史特征管线已经完成严格嵌套的端到端交叉验证。

完整方案需要共享K/D/E解释外部行为，并从同一执行状态推演修改和不修改到提交终点，等待观察也由同一模型生成。现有训练文件不能代替该完整实现，更不能把代理训练分数当作方法优越性的证明。
