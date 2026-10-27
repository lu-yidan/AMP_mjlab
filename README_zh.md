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

### 2026-10-27：双判别器稳定性与恢复专家消融

分支 `lyd-amp-recovery-stability`，基于 `lyd-multi-discriminator-amp`。
四组从零开始：A 原分层单判别器 Tracking；B `MultiAMP-Stable-Tracking`；
C `MultiAMP-Phase-Tracking`；D `MultiAMP-Smooth-Tracking`（完整任务名前缀
`Unitree-G1-`、后缀 `-Flat`）。统一 4096 环境、24 steps、100000 iterations、
seed42、1000轮存档；保持3m/s课程，不新增恢复引导。

B 固定任务权重0.75、风格权重0.25；无专家匹配或终止时保留0.75任务奖励。
每个判别器每轮最多更新20次，并限制更新样本数不超过新入回放样本数的2倍；
无新样本不更新。回放50轮过期，批次一半最近两轮、一半有效历史，缺少近期时回退。
归一化每轮首次更新一次，梯度范数上限1。记录新样本数、回放年龄、更新次数、
重复使用率、专家/策略分数、风格奖励及零奖励比例、专家覆盖率。
这些配置为待验证实验值，不宣称已经解决退化。

C 在B上用SMP训练专用恢复池替换原整段fallAndGetUp恢复专家；走跑专家不变。
`scripts/prepare_smp_amp_experts.py` 按现有训练reset bank的96个自然clip清单（含镜像）
读取SMP CSV，检查与validation原片段及镜像零交叉，按关节名称映射到当前AMP模型，
以50Hz插值并用当前模型前向运动学、关节速度和body-origin Jacobian计算AMP特征。
从低位最低点保留到连续直立0.3秒，排除之前跌落和长站立尾段。
94个片段贡献16336个相邻转移；没有超过0.1rad限位偏差而被拒绝的片段。
该检查不代表动态可行性或无自碰撞。专家池和来源hash见
`reports/amp_stability_smp_expert_audit.json`，大数据放在忽略目录
`logs/amp_stability/smp_recovery_train.npz`。
按头高0.6/0.9/1.15m分四阶段，按躯干局部重力xy主方向及直立分五类；
策略回放按现有阶段/方向格均衡，专家匹配同格；缺方向只回退同阶段，缺阶段不给风格奖励。
C相对B同时改变数据与采样，不能把效果仅归因于阶段采样。
原baseline整段动作与自然验证池可能共享源动作，不能将整个A/B/C/D比较称为完全源隔离终评；
程序低位验证与后续动态推倒测试须单独报告。

D在C上仅增加奖励软混合：恢复状态下，头高0.9→1.15m与躯干直立度0.4→0.9
共同形成smoothstep就绪度，恢复/走跑风格权重相加为1。命令恢复滞回不变。
重叠区转移进入两个判别器回放；无对应走跑专家时不补造专家，仍记录覆盖率。
任务奖励尺度保持固定。A继续采用原单判别器流程（包括原终止步近似处理），
因此本实验比较完整实现，不是只改变判别器数量的严格消融。

测试覆盖终止/缺专家时任务权重、无新样本停止更新、回放过期与环绕、恢复切片排除跌落、
平滑混合端点、检查点往返，以及旧单双任务回归。正式启动清单另存于reports。

D额外对混合权重施加0.2秒速率限制（控制dt=0.02，每步权重最多改变0.1），
防止恢复状态退出阈值与高度渐变区间不同导致瞬时跳变。episode结束后的第一步
按新初态初始化，不继承上一回合权重；该短时状态不用于跨episode续训。
最初D启动后约10轮在代码复核中被停止；修正后的D从零启动，独立run，不续训旧权重。
A/B/C仍使用e48bd61快照；新增时间渐变只影响D。各run来源以启动清单为准。

正式启动清单：`reports/amp_stability_launch_20261027.json`。
全部从零开始、W&B项目`tabletennis/lyd-multi-amp`，group为`amp-recovery-stability-20261027`。

| GPU | 实验 | W&B run |
|---|---|---|
| 0 | A 分层单判别器 Tracking | https://wandb.ai/tabletennis/lyd-multi-amp/runs/42b53ed8 |
| 1 | B 双判别器稳定更新 | https://wandb.ai/tabletennis/lyd-multi-amp/runs/372370e0 |
| 2 | C 稳定更新＋SMP恢复专家＋阶段采样 | https://wandb.ai/tabletennis/lyd-multi-amp/runs/1ed80983 |
| 3 | D 在C上增加空间及0.2s时间平滑 | https://wandb.ai/tabletennis/lyd-multi-amp/runs/f011bd5c |

旧D `a3a09677` 已停止并被替换，不计入正式对照。16项单元测试通过；四组均通过4096×24×2轮
GPU短训练、checkpoint及ONNX导出；短训练专家预载2048，正式恢复默认200000。
SuperMimic三个活跃训练已按要求停止，最近周期存档分别为原版24步84k、分组24步73k、分组6步96k；
两组3步已完成100k。未删除模型或停止其他项目进程。

实验固定代号：AMP-A=分层单判别器基线；AMP-B=双判别器稳定更新；
AMP-C=稳定更新＋SMP恢复专家＋阶段采样；AMP-D=C＋空间/时间平滑。
W&B名称已加对应代号前缀；启动清单含`code`字段。后续新配置继续分配新字母，
不复用已有代号；checkpoint或评测名称应包含代号和训练轮数，避免同名混淆。

2026-10-27首次视频对照：AMP-A 14k、AMP-B 17k、AMP-C 16k、AMP-D 17k。
`scripts/record_amp_stability.py` 用相同6个标称初态，确定性actor，不自动reset或施加推扰。
4个恢复初态取验证池每方向第一个程序低位姿态，零命令录10秒；走跑与原地转向各25秒，
每5秒切换命令（走跑0/.5/1/2/3m/s，转向0/+1/-1/+1/-1rad/s）。
与既有评测一致，命令切换经过一个0.02s控制步进入观测历史。
`scripts/compose_amp_stability.py` 校验初态hash一致后合成四宫格，左上A、右上B、左下C、右下D。
6个场景原片在服务器`/root/workplace/amp-stability-videos-20261027/latest-a14-b17-c16-d17/`；
本地合集在`logs/amp_stability/videos/latest-a14-b17-c16-d17/`。
权重hash、验证池hash、初态hash及帧数在`reports/amp_stability_videos_20261027.json`。
这是最新存档的固定场景展示，不是相同训练预算或完整恢复成功率评测。

### AMP-E：C 的 MBRL SuperMimic 对照

分支 `lyd-amp-c-supermimic`。使用 `MBRL_supermimic` 的独立库版本
`eb7a82bedad9e9e25a93cff3318d24e4a9f88cb2`，不恢复之前已停止的 SuperMimic 任务。
保持 AMP-C 的真实环境、分层 reset、SMP 恢复数据、双判别器与奖励比例，
以世界模型＋BPTT 替换 PPO。计划 4096×24、100k 轮、每1000轮保存，
32步想象展开。判别器只用真实采样训练，想象中冻结参数、保留状态梯度。

接入包含恢复滞回、250步延迟终止、命令平滑与4帧观测历史。
接触代价及子步加速度由世界模型辅助头预测；命令专家条件在短想象展开内固定，
这些近似与 mjlab 子步位姿时序偏差均在 [详细说明](docs/AMP_E_SUPERMIMIC.md) 记录。
数值验收区分“实际奖励时刻的公式一致性”和“步末状态的时序误差”。
比较时需同时看真实恢复成功率、速度跟随、真实采样量和墙钟时间，不能仅看想象奖励。
入口：`scripts/train_amp_supermimic.py`；带验收门禁的启动器：`scripts/launch_amp_supermimic.py`。

AMP-E 已于2026-10-27在GPU4启动：W&B run `079f010b`，
https://wandb.ai/tabletennis/lyd-multi-amp/runs/079f010b 。
训练源码提交 `a3219c8`，启动与参数记录在 `reports/amp_e_launch_20261027.json`。
19项测试通过；256环境数值验收通过；4096×24、32步想象的3轮完整尺寸短训练通过，
检查点206个张量均有限。首次编译后短测约2.1秒/轮，不能视为长期吞吐保证。
正式训练前300轮只预热世界模型，然后开始更新actor/critic。
