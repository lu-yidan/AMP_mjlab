# AMP_mjlab

[English README](README.md)

部署集成代码位于 [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm) 项目中的 `MJAmp State`。

基于 mjlab + rsl_rl 的 G1 AMP 运动控制项目。

本项目的核心特点是：

- 使用同一个 policy 同时学习 locomotion（走/跑）与 recovery（跌倒恢复）
- 通过 AMP 判别器约束动作风格与运动先验
- 在训练与导出链路中保持一致，支持直接导出 ONNX policy

## 核心思路

传统做法常把“走跑策略”和“恢复策略”分开训练并做切换；本项目将两类能力放入一个策略中统一学习。

实现要点：

- 运动数据分组：
	- Walk/Run 数据目录：`src/assets/motions/g1/amp/WalkandRun`
	- Recovery 数据目录：`src/assets/motions/g1/amp/Recovery`
- 延迟重置机制（Delayed Termination）：
	- 一部分环境在触发终止后不立即 reset，而是给定恢复窗口
	- 该子集环境优先从 Recovery 片段采样 reset 状态
- 统一 AMP 训练：
	- 单一 actor-critic + 单一 AMP discriminator
	- 在同一训练过程中学习速度跟踪、抗扰动与恢复能力

这样可以减少策略切换带来的状态不连续问题，得到更一致的行为。

## 环境要求

- Linux
- Python 3.11（建议）
- 已可用的 MuJoCo / GPU 驱动环境

## 快速开始

### 1. 安装仓库

```bash
conda activate mjlab
cd AMP_mjlab
python -m pip install -e .
cd rsl_rl
python -m pip install -e .
```

### 2. 应用 mjlab 补丁（可选）

如果不打这个补丁，则需要在代码中去掉 `history_ordering` 配置。

补丁作用说明：

- 增加了历史观测的展开方式选项，可选择按时间维(`time`)或按观测项(`term`)展开。
- mjlab 默认仅支持按 `term` 展开。

补丁文件：

- `mjlab_patch/mjlab/managers/observation_manager.py`

示例覆盖命令：

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
	/home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

### 3. 查看可用任务

```bash
python scripts/list_envs.py --keyword AMP
```

主要任务：

- `Unitree-G1-AMP-Rough`
- `Unitree-G1-AMP-Flat`

## 训练


```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```


日志默认在：

- `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`

## 训练曲线说明（重要）

- 在约 `2w` 轮（约 20k iterations）附近，策略通常会突然学会“跌倒后恢复”行为。
- 对应地，`logs` 中多个指标会出现明显突变（阶跃式变化），这是正常现象，不一定是训练异常。

![训练日志突变示例](logs.png)

## 评估与可视化

使用已训练权重回放：

```bash
python scripts/play.py Unitree-G1-AMP-Rough \
	--checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt 
```

说明：训练与回放阶段都支持 ONNX 导出（默认开启）。

## 运动数据准备

仓库提供 CSV 到 NPZ 的转换脚本：

```bash
python scripts/csv_to_npz.py --help
```

推荐目录组织：

- 原始 CSV：`motion_data_csv/amp`
- 转换后 NPZ：`src/assets/motions/g1/amp/WalkandRun` 与 `src/assets/motions/g1/amp/Recovery`

只要上述目录中存在可用 NPZ，训练配置会自动加载。

## 目录说明

- `src/tasks/amp_loco`：AMP locomotion/recovery 任务实现
- `src/tasks/amp_loco/config/g1`：G1 任务注册、环境与 RL 配置
- `src/tasks/amp_loco/mdp`：奖励、观测、事件、终止逻辑
- `scripts/train.py`：训练入口
- `scripts/play.py`：回放入口
- `scripts/csv_to_npz.py`：动作数据转换工具
- `mjlab_patch`：依赖的 mjlab 本地补丁

## 项目亮点总结

- 单一策略统一覆盖走跑与跌倒恢复
- AMP + 速度任务联合优化，兼顾风格与任务性能
- 延迟重置与 recovery 采样机制，显式强化恢复能力
- 训练到部署链路完整，支持 ONNX 导出

## 致谢

- 感谢 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 项目的开源工作与启发。
- 感谢 [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)，本项目在 rsl_rl 的 AMP 部分参考了该实现。

## 状态门控双判别器实验

分支 `lyd-multi-discriminator-amp` 基于上游 `6c7a294`，新增任务
`Unitree-G1-MultiAMP-Flat`。原 `Unitree-G1-AMP-Flat` 保留作单判别器对照。
两者使用同一套环境重置和终止规则（40% 环境允许连续 250 步失败，20 秒超时）。

新任务使用独立的恢复判别器与速度条件化走跑判别器，各有优化器、归一化统计
和回放缓冲区。恢复数据匹配 `fallAndGetUp*`，跑步匹配 `run*`、`jog*`、
`arc_jog*`，其余支持的走路/静止片段归入走路组；未知名称报错。
参考动作先等概率选片段，再选相邻帧，不跨片段末尾。
走跑判别器的条件为平面指令速度模长除以 `speed_max`（截断到 [0,1]）；
专家动作按该条件作为概率选择跑步，否则选择走路。它是初版条件定义，
尚未按实际参考速度校准，也未单独条件化运动方向。

门控在动作执行前采集，条件随样本保存：`abs(gravity_z + 1) > 0.6`
或头高低于 0.9m 时进入恢复分支。头高使用 `torso_link` 局部
`(0,0,0.43)` 的头部碰撞球中心相对环境地面原点的高度，仅适用于平地。
`use_head_height=False` 可做仅姿态门控对照。0.6、0.9m 和 3m/s
都是待标定参数，不代表已验证最优值。配置位于 `g1_multi_amp_ppo_runner_cfg`。

继续采用基线 LSGAN 损失与奖励混合方式。自动 reset 的终止步暂不参与
AMP 奖励和回放训练，保留该步完整任务奖励，避免跨回合伪样本；后续若环境
提供准确的 reset 前观测，可加入真正的终止转移。检查点包含两个网络、
归一化统计、优化器和配置；恢复时回放缓冲区重新采集。
单/双判别器检查点不能直接混用。当前只支持每次训练使用单 GPU。

```bash
python scripts/train.py Unitree-G1-MultiAMP-Flat --env.scene.num-envs 16 --agent.max-iterations 2 --gpu-ids '[2]'
python -m unittest discover -s tests -p test_multi_amp.py -v
```

仍需按前文应用仓库的 mjlab 观测历史补丁。测试与短训练通过只证明实现能运行，
不代表已复现论文表现；正式评估应固定数据、种子、训练预算及跌倒初始状态。

2026-10-16 在指定服务器 RTX 4090 上验证：3 项单元测试通过；新任务以
16 环境完成 2 次迭代（768 时间步），检查点和 ONNX 均成功保存。
最终回放样本数 recovery=201、locomotion=559，两个判别器损失均有限。
运行记录：`logs/rsl_rl/g1_multi_amp/2026-10-16_22-35-06`（服务器）。

### 分层 reset 对照（2026-10-17）

新增 `Unitree-G1-AMP-Stratified-Flat` 和 `Unitree-G1-MultiAMP-Stratified-Flat`。
4096环境固定配额为走跑2462、自然后段409、自然中段409、四个低位方向
各204（每方向自然153＋程序51）。自然姿态先均匀选原始clip，再选帧；
程序姿态均匀抽取。恢复初态速度清零，走跑沿用旧初始化。
恢复组沿用250步连续失败窗口与20秒超时。判别器专家数据和门控均未改变。

初态使用L4 natural_curriculum_v1/train与procedural_low_v1/train，经
`prepare_stratified_reset.py` 按当前AMP模型重新映射关节、修正地面高度，
拒绝限位违规和超过2mm的接触穿透。接受自然2895、程序4096，来源hash及
分组统计在 `src/assets/motions/g1/reset_banks/l4_amp_train.json`。
阶段与方向标签继承L4训练池，不使用其validation/test姿态训练。

两组4096环境、2迭代预检均通过，使用相同seed=42与分组计数。
正式各100000迭代、24步/环境、每1000保存，从零开始；单判别器GPU1，
双判别器GPU3，W&B项目lyd-multi-amp。旧reset对照GPU0/2继续运行。

### 恢复引导消融

`Unitree-G1-MultiAMP-Stratified-Guidance-Flat` 在分层双判别器上只添加
任务奖励：0.7×头部上升速度分数＋0.3×归一化头高。头部点采用torso局部
(0,0,.43)，点速度包含角速度叉乘偏移。上升0.1–0.3m/s满分，区间外指数衰减。
两项在头高≤.9m时全额激活，在.9–1.15m间线性退出，高于1.15m为零。
这是借鉴L4的有界任务引导，不包含SMP prior，未移除旧根部高度奖励。
奖励项经环境dt缩放后参与原任务/AMP混合；普通非终止步新增上限0.015，
终止步按现有task-only处理上限0.02。需评估低位停留或反复起落的奖励投机。
两项边界测试与4096环境2迭代短训练通过，正式seed42、100k、GPU4、W&B，
从零开始，不改reset、门控、专家数据或终止规则。

### 2026-10-18：统一60k检查点评测与曲线诊断

脚本 `scripts/evaluate_amp_common.py`：对当前机器人重新验证L4 validation池，
保留756自然姿态（其中376低位）和512程序低位；所有策略初始状态hash相同。
20秒、确定性actor、标称物理、恢复命令为0、关闭自动reset；根部>.65m且
投影重力gz<-.9连续1秒判成功，另记录最后5秒直立比例。结果是独立于reset
训练池的验证诊断，非真机、非随机推倒测试，单训练种子，非最终test集终评。
自然低位314/376为仰卧，须同时看均衡程序四方向。

60k自然低位/程序低位成功率：原单91.5/74.2%，原双98.4/85.2%，
分层单99.7/100%，分层双12.0/0%，分层双+引导100/100%。
程序俯卧分别0/44.5/100/0/100%。本轮另有128个相同标称站立状态的走跑指令
副本，用于基础速度跟踪检查，不应作为128个独立泛化试验。

台阶前后验证：分层单16k自然/程序低位0/0%，18k为99.7/63.3%；
引导组18k为0/0%，20k为100/75%。两组台阶后的程序俯卧仍为0%，
故平均episode length接近1000并不保证覆盖所有恢复姿态。

日志mean_episode_length只平均最近100个已结束回合。分层组早期(1k–5k)
原始序列125迭代滞后相关约.91–.93，125×24=3000控制步，是250失败窗口和
1000超时的公倍数；与失败环境同步重置及小统计窗口产生的周期尖峰一致。
20%环境250步失败、80%1000步超时，结束回合加权均值约625步，接近分层双
40k–60k的618步平台。双判别器不会直接平滑这种统计量。

需注意当前单双实现除网络数量外，还不同于判别器学习率/优化器、梯度惩罚
归一化路径与终止步AMP处理。因此不应把结果仅归因于判别器数量。
现有训练未因本次诊断被修改。原始评测JSON、历史导出与曲线保存在
`logs/eval60k`（本地）及服务器`records/eval60k`。

### 60k 速度跟随补测（2026-10-18）

使用 `scripts/evaluate_amp_commands.py` 对五组策略做相同初态命令测试，每种命令 8 个初始关节扰动，
先站立 2 s，再保持指令 10 s，统计最后 5 s；不重置，无部署平滑。结果保存在本地 `logs/commands60k/`。
纯侧移与原地旋转在原始单判别器中也很差；原始双判别器前后跟随较好。分层双判别器加引导组
对 -0.5 m/s 后退几乎不响应，对静止直接切换 2 m/s 前进也几乎不响应，而此前渐进变速可前进。
不要将恢复成功率或 mean_episode_length 当作指令跟随指标。部署配置还把侧向限制为 ±0.01 m/s。
需优先验证命令分模式采样、跟随奖励尺度及转换过程，再用受控消融分析判别器和恢复奖励。

### 从零训练：分层单判别器＋命令跟随修复

新任务 `Unitree-G1-AMP-Stratified-Tracking-Flat`；沿用分层 reset 和原 AMP 专家数据，不加 recovery guidance。
4096 环境、24 steps、100k iterations、seed 42、每 1000 保存、resume=False。现有旧任务配置不变。

- 命令模式：前进25%、后退15%、侧移15%、原地转向15%、混合20%、站立10%；前进样本额外保留35%的小速度采样。
- 每3～8秒重采样，50%阶跃、50%以1.5 SI单位/s限速变化；恢复阶段命令为零。
- 恢复进入：头高<0.9m或|gz+1|>0.6；退出：根高>0.65m、gz<-0.9且头高>1.05m持续0.5秒。
- xy跟随改为torso yaw坐标系平面误差、std=0.5；yaw跟随仅使用世界z角速度误差、std=0.5；另加权重-0.2的垂直速度平方成本。三项仅loco启用。
- 课程分三级：vx [-0.5,1]/[-1,2]/[-1,3]，vy ±0.3/0.6/1，wz ±0.5/1/1.5。混合模式vx上限保持1m/s。
- 每5000控制步统计窗口，进入下一阶段要求各模式至少5000有效样本、xy误差<0.3、yaw误差<0.35、站立比例>95%，恢复占比<15%；最早在5k和15k iterations升级。不达标则停留，不强行扩速。
- 模式指标使用最近一个完整窗口，首个窗口完成前为零；不会被episode reset永久清零。

同时修复 `src/tasks/amp_loco/rl/runner.py` 的保存副作用：旧ONNX导出调用eval会使在线actor观测归一化器停止更新。
已检查旧分层单判别器60k checkpoint的统计计数只有98,304（第一轮4096×24），而非持续积累。
新导出只复制actor网络和归一化器，避免改变在线训练状态或复制临时分布计算图；这是一项两种判别器方案都需要的共同修复，不能据此单独证明旧模型跟随差的全部原因。

源码快照与启动清单位于本地 `logs/tracking_scratch/`，服务器 `records/tracking-scratch-source/`，
配置也随训练run保存。训练工作分支仍为 `lyd-multi-discriminator-amp`，当前源码改动尚未提交。

正式run：[4uvuh77l](https://wandb.ai/tabletennis/lyd-multi-amp/runs/4uvuh77l)，GPU 2，目录 `2026-10-19_00-03-26_stratified-single-tracking-scratch-4096x24-100k`。前两次启动检查run因指标记录和导出副作用修复而停止，正式run没有加载它们的checkpoint。修复后4096环境两轮短训练的actor归一化计数为196,608，确认保存后仍继续更新。


### 双判别器速度跟随修正版（2026-10-19）

分支 `lyd-multi-discriminator-amp`，当前工作区修改；尚未提交。新增任务：

- `Unitree-G1-MultiAMP-Stratified-Tracking-Flat`：无额外恢复引导。
- `Unitree-G1-MultiAMP-Stratified-Tracking-Guidance-Flat`：仅恢复状态添加引导。

两者复用单判别器 Tracking 的分层 reset、命令采样、跟随奖励和达标后升至
3 m/s 的课程，以及已修复的 ONNX 导出/归一化流程。旧任务定义保持原行为。
默认 100000 iterations，每 1000 保存；本次只进行短训练验证，不自动启动长期实验。

运动判别器条件由速度大小改为有符号 `[vx, vy, wz]`，归一化尺度 `[3, 1, 1.5]`。
专家标签由每帧 torso 的 yaw 坐标系平面速度和世界 z 轴角速度提取。
匹配每维容差 `[0.3 m/s, 0.2 m/s, 0.3 rad/s]`，在最近 16 个合格转移中随机选取，
策略和专家使用同一个真实专家条件（离散近似），避免类别标签分布泄漏。
回放更新使用该条件的最近专家，防止重复匹配造成条件漂移。
没有合格专家的运动转移使用原任务奖励，不进入运动判别器回放。
`loco_expert_coverage` 是最近一次有运动样本的批次匹配率，不是全局平均成功率。
这会改变不同指令下的风格奖励覆盖，尚需长训练验证；不能视为补齐了侧移/转向数据。

路由直接读取 TrackingCommand 的恢复状态，进入阈值为倾斜条件或头低于 0.9m，
退出要求根高度 >0.65m、gz<-0.9、头高>1.05m 持续 0.5s。
恢复时速度指令为零；可选引导在 locomotion 状态严格为零，恢复中保留原高度渐隐。
新运动判别器输入维度改变，不能加载旧双判别器 checkpoint 续训，需从零训练。
验证包含旧版回归、方向匹配/不支持指令、checkpoint 往返、loco 引导归零，
以及服务器 64 环境 ×24 steps ×2 iterations 的 guided 任务短训练。


### 训练版本保存与最终单判别器存档（2026-10-26）

本分支保存分层 reset、单/双判别器 Tracking 任务、恢复引导对照、
观测归一化导出修复，以及统一恢复/命令评测和视频脚本。
训练时的原始代码快照仍是运行来源；本提交是训练后整理保存，不宣称当时由此提交启动。
单判别器 W&B： https://wandb.ai/tabletennis/lyd-multi-amp/runs/4uvuh77l 。
服务器最终模型：
`/root/workplace/lyd-multi-discriminator-amp/logs/rsl_rl/g1_amp_stratified_tracking/2026-10-19_00-03-26_stratified-single-tracking-scratch-4096x24-100k/model_99999.pt`。
迭代从 0 编号，因此 99999 对应完成 100000 轮，不是 PPO 内部 epoch 数。
本地模型备份与最终评测目录：`logs/eval100k_single/`（忽略目录，不随 Git 提交大权重）。
服务器在本次检查时没有可用 CUDA 设备；最终评测尝试使用本机 GPU 与复制的原训练 mjlab runtime，
环境差异与实际完成状态见该目录的评测记录。


最终存档评测已完成，结果与模型 SHA-256 见 `reports/single_tracking_100k_evaluation.json`。
本机 RTX 4070 Ti，使用服务器复制的 mjlab runtime，MuJoCo 3.5.0、mujoco-warp 3.5.0.2、
Warp 1.13.0.dev20260225；本机 Python 3.13/PyTorch 2.8 与训练端 Python 3.12/PyTorch 2.10 不同。
376 个自然低位姿态和 512 个程序低位姿态均 100% 成功，平均成功时间分别为 1.846s、1.881s。
成功判据为 20 秒内根高度 >0.65m、gz<-0.9 连续 1 秒，统计时间含该 1 秒保持期。
命令评测每指令 8 个配对初态，2s 站立后执行指令 10s，取末 5s；没有自动 reset、推扰或部署平滑。
目标前进 3m/s 实际 2.889m/s，后退 -1m/s 实际 -1.013m/s，左右侧移目标 ±0.5m/s
实际 0.439/-0.412m/s，左右转向目标 ±1rad/s 实际 0.827/-0.916rad/s。
所有命令末段站立率 100%。这不是复杂地形、非零指令倒地恢复或实机验证。
原服务器与本机隔离环境均通过 10 项单元测试；本地工作区保存为 Git 提交，大模型仍在忽略的 logs 目录。

### SuperMimic 站立／行走／跑步速度指令版

分支：`lyd-supermimic-stand-walk-run`，基于 `lyd-multi-discriminator-amp` 的
`40a8a0c`。代码位于 `src/supermimic/`，独立入口，不改变原 AMP/PPO 任务。
来源为本地 `supermimic_vibed` 的 `a1508d42c9d1699a52e8233ab16d08c488441ab9`，
作者、MIT 许可和改动边界保存在该目录的 `ATTRIBUTION.txt` 和 `LICENSE`。

这是 **SuperMimic-inspired 速度控制改编版**，不是原论文或原实现的动作跟踪基准复现。
原实现输入未来参考动作，使用 Isaac Lab/Newton；本版输入 `[vx, vy, wz]`，
使用本仓库 G1/mjlab/MuJoCo-Warp、电机顺序和逐关节动作缩放。
保留可微旋转/状态表示、预测局部状态增量的世界模型、多步自回归模型拟合、
想象轨迹 BPTT 策略优化、soft survival 和 lambda-return critic。
不运行 PPO 或 AMP 判别器，不使用动作示范或旧 AMP 策略权重。
因此本版本只证明训练链路集成，不保证动作自然度，也不声称复现原项目的训练效率。

第一版范围：平地、标称动力学、默认站姿 reset、站立/前进走跑命令等概率采样，
走路速度 0.1–1m/s，跑步 1–3m/s（可用 `--max-speed` 限制）；尚未训练侧移/转向，
也没有加入倒地恢复、复杂地形、域随机化、示范跟踪、数据驱动 reset 课程。
“站”在本版指零速度站立，**不等于跌倒后站起**。
真实和想象环境共用解析跟随/站立/动作平滑奖励与终止头。
仿真先记录终止前的 `s1`，再手动 reset；世界模型窗口不能跨 episode，
历史观测同样不能跨 reset。非有限损失/梯度会报错，不静默继续训练。

默认规模按普通 GPU 下调：1024 环境，256 步回放，256 模型 batch、256 想象 batch；
网络仍默认使用原实现的大网络。使用与本仓库兼容的 mjlab runtime 和 Python 环境运行，
本机验证环境为 `/tmp/amp-eval100k-venv/bin/python`，
`PYTHONPATH=/tmp/amp-eval100k-runtime:.`；这些临时路径不是安装包的一部分。

```bash
# 在仓库根目录，先激活兼容 mjlab 的 Python 环境并配置 PYTHONPATH。
python -m src.supermimic.train --output logs/supermimic/stand_walk_run \
  --num-envs 1024 --iterations 100000 --max-speed 3 \
  --wandb-project lyd-supermimic

# 资源有限时先跑小规模端到端检查，不能用其模型判断最终能力。
python -m src.supermimic.train --num-envs 8 --iterations 2 \
  --buffer-steps 16 --history 2 --horizon 2 --wm-window 2 \
  --wm-batches 1 --wm-batch 8 --policy-batch 8 --warmup-steps 4 \
  --wm-only-iters 0 --small --output logs/supermimic/smoke

python -m src.supermimic.evaluate \
  --checkpoint logs/supermimic/stand_walk_run/model_100000.pt \
  --output logs/supermimic/stand_walk_run/evaluation.json

python -m src.supermimic.export \
  --checkpoint logs/supermimic/stand_walk_run/model_100000.pt \
  --output logs/supermimic/stand_walk_run/policy.onnx
```

恢复训练使用 `--resume PATH`，其余网络/历史参数必须匹配；`--iterations` 是目标总轮数。
保存网络、优化器、已完成轮数、关节顺序和默认关节姿态。恢复时重新填充 replay、
重置仿真，故不是逐位一致续训。默认每 1000 轮和结束时保存，文件编号为已完成轮数。
可选 W&B；默认只写本地 `metrics.jsonl`，没有自动启动后台长期训练或自动上传。

策略输入：`[batch, history, 93]`，历史从旧到新，内容依次为
`q-default_q(29), qd*0.1(29), pelvis body-frame angular velocity*0.25(3),
projected gravity(3), previous normalized action(29)`；另输入 `[batch,3]` 速度指令。
输出 29 个电机顺序的归一化位置偏移，限幅 ±5；使用本仓库 G1_ACTION_SCALE 与默认姿态。
actor 自带运行统计归一化，世界模型只用于训练，部署只需要 actor 和历史缓冲。
这**不兼容旧 AMP 的 384 维单输入／0.25 标量动作接口**，不能直接只替换部署模型路径。
导出同时写接口说明 JSON；未实现旧部署接口适配。

评测从同一默认站姿发出 0/0.5/1/2/3m/s 指令，无自动 reset；默认每命令重复 8 次，
在无噪声标称仿真下这些重复不构成独立统计样本，报告不可当作随机初态泛化成功率。
当前只完成训练、checkpoint 恢复、评测、ONNX 导出的短流程验证，尚无充分训练的模型。

本次验证：14 项单元测试通过；8 环境、2 轮小网络训练包含世界模型和 BPTT/critic 更新，
从第 2 轮存档恢复到第 3 轮成功；5 条速度指令的短评测入口跑通（短训模型不具备任务能力）。
强制 timeout 的仿真检查确认终止前状态进入 replay、奖励与解析头一致、reset 后 previous action 清零。
导出 ONNX 在随机 batch=3 输入下与 PyTorch 最大误差为 1.19e-7。
