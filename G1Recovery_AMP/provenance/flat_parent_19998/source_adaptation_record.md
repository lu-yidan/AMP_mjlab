# AMP 到历史 A6 受困场景：迁移记录

本文件记录实测事实、待验证假设和下一步验收门槛；它不是已经完成 A6 复现的声明。
协议来源：`/mnt/d/learn in HKU/dissertation/learn/help.md`（2026-09-28）。
师兄最终交付提交：`smp@5f4a6a9787...`（reset bank、校验和、交接脚本与文档）。
历史 A6 训练实现参考提交：`smp@1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2`（环境、动力学随机化和训练逻辑）。两者用途不同，并不是新提交覆盖或否定了旧训练实现。

## 阶段状态

| 阶段 | 状态 | 验收标准 |
| --- | --- | --- |
| 平地 AMP 起点体检 | 初步完成 | 同一批固定起点比较候选 checkpoint，记录配置与哈希 |
| 平地四方向正式评测 | 待做 | 独立初态、四方向、稳定站立与物理安全指标 |
| 物理步长对齐 | 独立任务已注册，三个固定起点初测及平地 bank 9.98 s 数值检查通过；完整能力验收待做 | 0.002 s × 10 子步仍为 0.02 s 控制周期，四方向平地能力不异常退化 |
| A6 场景与 reset | 三场景实体、50/25/25 配额、L/M/H、安全接管和板状态机已接线；reset 与 100 步策略 rollout 通过 | 平地/导向板/自由板配额、四方向、bank SHA、板位置、部分 reset、完成/无效判据均通过 |
| 奖励与状态历史映射 | 状态历史、五项障碍奖励、四阶段状态、九项共同正向任务、九项公共成本、Q/L、alpha 及 G−/G+ 开关已完成 | 单项数值、dt、G−/G+ 开关及 reset 清空逐项通过 |
| A6 动力学随机化 | 成组质量/惯量、六组电机增益、脚底摩擦、编码器偏差、躯干质心、速度推力和 0–10 ms 动作延迟已接入 | 纯函数测试、reset 审计及 G−/G+ 短 rollout 通过；GPU 长训练待做 |
| 平地 checkpoint 迁移初始化 | 按协议完整恢复 AMP 原生训练状态，并单独重置 A6 课程时钟 | 完整恢复测试与 A6 课程时钟测试通过 |
| G−/G+ 配对短训练 | 已完成 seed 42、512 环境、各 50 次更新预检 | 同一父 checkpoint 和 seed，跨越完整 10 s episode，无数值异常 |
| 正式训练与独立评测 | 待做 | 按协议预算及 20 s 测试报告逐场景结果 |

## 当前候选 checkpoint

| 候选 | SHA256 | 来源 |
| --- | --- | --- |
| `logs/downloaded/2026-09-29_15-01-41_fresh_old8_4096_15000/model_14999.pt` | `b1ea03536cb82624efbcf8a3be996c1acd9dee3bd1d40ec2812cd6626a49aac8` | 8 动作、4096 环境、从头训练 15000 次更新 |
| `logs/downloaded/2026-09-30_01-04-15_low_height_progress_from_14999_gpu3_5000/model_19998.pt` | `4280efe9de1d6772e33f93847208896c4f7126e62630fc95b1cfeafe72e9f47c` | 从前者续训 5000 次更新；环境仅新增权重 0.5 的 `low_height_progress` |

两份展开配置的环境差异仅为上述奖励；agent 配置差异为续训元数据、运行名和更新次数。当前 AMP 混合比例仍为任务 50% / 风格 50%。

## 平地固定起点体检（2026-09-30）

使用 `scripts/evaluate_recovery_checkpoint.py`，CPU、seed 42、play 配置、每个动作从首帧开始、最多 500 个控制步。表中数字是“高度 > 0.65 m 且躯干上方向竖直分量 > 0.7”的最长连续时间（秒）；**不是**正式 SR1/SR10，也没有检验双脚支撑和全身速度。

| 动作文件（省略 `.npz`） | `14999` | `19998` |
| --- | ---: | ---: |
| `fallAndGetUp1_subject1_1060_1150` | 7.08 | 7.02 |
| `fallAndGetUp1_subject1_1400_1480` | 0.00 | 0.00 |
| `fallAndGetUp1_subject1_2100_2200` | 7.72 | 7.64 |
| `fallAndGetUp1_subject5_2500_2600` | 6.94 | 7.00 |
| `fallAndGetUp2_subject2_850_1050` | 0.00 | 2.64 |
| `fallAndGetUp6_subject1_530_600` | 8.48 | 8.52 |
| `fallAndGetUp6_subject1_650_700` | 9.98 | 9.98 |
| `fallAndGetUp6_subject1_1630_1690` | 8.34 | 8.58 |

初步选择 `model_19998.pt` 作为下一关的**候选**父模型，因为它未在这 8 个起点上明显丢失原有能力，并使其中一个此前失败的起点短暂站起。不可将 7/8 当作真实成功率：这 8 个样本不独立，也未覆盖协议要求的四方向和板下初态。两次模型相差奖励**及** 5000 次训练更新，不能将改善单独归因于奖励。当前 play 配置仍有启动时摩擦/质量随机化；固定 seed 体检不等于名义物理评测。

复现命令（把 checkpoint 参数换成上表任一路径）：

```bash
cd /home/ray1/code/G1Recovery_AMP
PYTHONPATH="$PWD/src:$PWD" ../mjlab/.venv/bin/python \
  scripts/evaluate_recovery_checkpoint.py \
  --checkpoint-file logs/downloaded/2026-09-30_01-04-15_low_height_progress_from_14999_gpu3_5000/model_19998.pt \
  --device cpu
```

## 联合训练前必须解决的差异

- 当前控制周期为 `4 × 0.005 s = 0.02 s`；历史 A6 为 `10 × 0.002 s = 0.02 s`。不能只改 `decimation` 或只改 `timestep`。
- 当前 actor 为 480 维历史观测，动作为 29 维关节位置目标，AMP 判别器输入是 10 × 85；保持它们与 checkpoint 兼容，不强制改成 SMP 的 93 维 actor。A6 task 只在 reset 后前 10 个控制周期平滑接管目标，并按实时 PD 参数限制目标角，不改变网络输入输出形状。
- 默认 AMP 环境仍只在平地从 8 个动作文件 reset；独立 A6 任务使用冻结低位 bank 和自然 L/M/H curriculum bank，并已加入平地 50%、竖直导向板 25%、自由板 25% 的固定场景分配。
- `src/g1recovery_amp/data/reset_banks/a6/multiterrain_train.npz` 与 `natural_curriculum_train.npz` 已按交接包原样复制并分别通过 SHA256 `287eae8e...1027a`、`e9f94540...44b9` 校验。读取器拒绝缺字段、错误形状、非有限 qpos、非单位四元数或哈希不符的数据。
- 独立 A6 task 已用九项历史共同任务替换六项功能重叠的原生 AMP recovery 奖励，并记录 AMP 的 50:50 混合前后有效权重与 `dt`；默认平地 AMP task 不变。G−/G+ 必须从同一父 checkpoint 启动。

## 物理步长过渡体检（2026-09-30）

独立任务 `Mjlab-Recovery-AMP-A6-Unitree-G1-Dev` 和评测脚本的 `--a6-timing` 都同时设置物理步长 `0.002 s` 与 `decimation=10`；默认平地任务仍保持 `0.005 s × 4`。二者在切换前检查控制周期没有改变。对候选 `model_19998.pt` 的三个固定起点，均运行 500 个控制步、未出现 NaN/Inf 或接触容量错误：

| 动作文件（省略 `.npz`） | 默认物理步长直立时长 | A6 物理步长直立时长 |
| --- | ---: | ---: |
| `fallAndGetUp1_subject1_1060_1150` | 7.02 s | 7.04 s |
| `fallAndGetUp1_subject1_1400_1480` | 0.00 s | 0.00 s |
| `fallAndGetUp2_subject2_850_1050` | 2.64 s | 3.14 s |

环境运行输出确认物理步长 `0.002 s`、控制周期 `0.02 s`。这只是三个起点的初步对比：仍需四方向、多初态、稳定站立判据和物理负载统计，不能视为受困场景验证通过。

## 平地低位 reset bank 接线（2026-09-30）

独立 A6 任务关闭 motion command 对物理机器人的二次传送，但继续保留动作文件供 AMP demonstration 和参考数据使用。32 环境 CPU 行为检查确认：四方向各 8 个环境，自然/程序化来源为 24/8，根与关节 qpos 逐行等于所抽 bank 状态，根与关节速度为零，actor 每个观测项的五个历史槽都等于 reset 后首帧。候选 `model_19998.pt` 在该任务上完成 actor、critic、AMP 判别器与优化器的完整恢复，并推进一个控制步而无 NaN/Inf。

## 平地 bank 多步稳定性与控制接管（2026-09-30）

师兄 A6 的 `DeploymentPositionAction` 并非普通位置动作：它在 reset 后用 10 个控制周期从 bank 关节角平滑过渡到策略目标，并把目标投影到当前 `kp`、`kd` 和扭矩上限允许的范围。AMP-A6 已迁移同一类保护，但保留 AMP 原有默认姿态、`0.25` action scale、29 维 action 和原执行器参数。训练模式的 PD 增益随机化后，保护会读取当次环境的实际增益，而不是使用写死的常数。

候选 `model_19998.pt` 在 8 个环境中覆盖四方向 × 自然/程序化两个来源，运行 499 个控制步（9.98 s）。结果：无 NaN/Inf、无提前 reset、无物理告警越界；实际 rollout 峰值为根角速度 `8.35323 rad/s`、关节速度 `19.9788 rad/s`、关节加速度 `1309.68 rad/s²`、接触穿透 `0.0211689 m`。reset 后但第一帧 action 尚未施加时的 MuJoCo 瞬时 `qacc` 为 `3005.65 rad/s²`，它被单独报告，不再混入实际 rollout 峰值。迁移安全接管前，相同类型检查曾把 reset 瞬时值计入峰值并得到 `5231.46 rad/s²`；拆分指标并加入接管保护后，实际动态峰值明显降低。

## 平地 L/M/H curriculum（2026-09-30）

平地 reset 已从“全部低位”升级为历史 A6 的 L/M/H 固定环境配额：低位 40%、中间 40%、近站立 20%。L 继续从 multi-terrain bank 的平地池抽样，保持四方向均衡，并在每个方向内部使用约 75% 自然、25% 程序化状态；M/H 全部从 `natural_curriculum_train.npz` 抽取。自然 bank 的镜像片段与原片段合并为同一 clip 分配概率，避免帧数多或带镜像的片段被过度采样。

在历史 A6 的 2048 个平地环境规模下，固定配额测试得到 L/M/H=`820/820/408`，其中低位每方向 205 个，每方向自然/程序化=`154/51`，合计程序化 204 个。32 环境实际 reset 检查因整数取整得到 `12/12/8`，低位四方向=`3/3/3/3`、自然/程序化=`8/4`；qpos 逐行匹配相应 bank、速度为零、五帧历史一致，且对偶数环境执行 partial reset 后，奇数环境的物理状态与 bank 行记录保持不变。

候选 `model_19998.pt` 在 L/M/H 各 4 个环境中运行 499 个控制步（9.98 s）：无 NaN/Inf、无提前 reset、无物理告警越界。实际 rollout 最大关节加速度分别为 L `1162.32`、M `1601.49`、H `1877.45 rad/s²`。旧策略对自然 M/H 样本的瞬态反应比低位更分散，后续短训练与多 seed 评测仍需重点观察，但当前没有 reset 数值失稳证据。

## 三场景物理 reset（2026-09-30）

独立 A6 任务现已加入尺寸固定为 `0.90 × 0.64 × 0.07 m` 的两块板实体。竖直导向板由 mocap 锚点锁定水平位置和姿态，只保留一个竖直滑动关节；自由板使用 free joint，可由接触推动并发生平移、旋转。每个世界虽然都编译两块板，但 reset 会把非本场景的板停放在远处，因此一个环境只激活平地、导向板或自由板之一。

4096 环境的纯配额测试精确得到场景 `2048/1024/1024`、L/M/H=`2868/820/408`、自然/程序化=`3380/716`。板场景全部使用低位状态，并分别从 multi-terrain bank 的 stratum 1 与 2 抽样，而不是把平地 stratum 0 静默复用给所有场景。板底面按当前机器人碰撞几何最高点再加 `2 mm` 放置；质量在课程开始为 `4–6 kg`，随 `common_step_counter` 在 100000 步内扩展到 `4–12 kg`。

32 环境 CPU reset 验收得到平地/导向板/自由板=`16/8/8`，actor 仍为 480 维；已验证板底间隙、bank stratum、零初速度、五帧静态历史和 partial reset 隔离。32 环境的程序化计数为零是冻结实现按每个小场景方向单独四舍五入的极小样本现象，正式 4096 配额为 716。后续已经迁入共同正向恢复任务，但 G−/G+ 开关仍未完成，也尚未证明候选策略能在板下稳定脱困。

状态机接入前，候选 `model_19998.pt` 在 32 环境三场景中完成 100 个控制步的无训练 rollout：没有 NaN、提前 reset 或现有物理告警越界。引导板峰值接触力约 `1029 N`，水平位移为 `0`、竖直位移约 `0.148 m`，符合它只能竖直运动的结构；自由板水平位移约 `1.09 m`、竖直位移约 `0.369 m`。自由板曾出现约 `0.039 m` 接触穿透，超过历史 A6 的 `0.020 m` 无效阈值，这成为迁移状态机的直接依据。

## 板接触、完成与无效状态机（2026-09-30）

板状态机现已独立记录每个环境是否曾接触板、连续净空计数、是否完成逃脱、几何进度和无效原因。只有先接触过板，随后机器人碰撞几何在板平面外至少留下 `0.025 m` 净空，并连续保持 15 个控制步，才标记为逃脱；短暂离开后再次接触会把计数清零。平地环境不参与这些判据。

板场景出现以下任一情况会立即终止并重新抽取 reset 状态：接触力大于 `1500 N`、穿透深度大于 `0.020 m`，或第 26 个控制步仍从未接触板。未接触超时同时应用于导向板和自由板，与冻结的历史 A6 实现一致。7 个定向状态机测试覆盖边界值、连续保持、重新接触、平地旁路和三种无效原因；A6 配置测试确认终止项只接入独立 A6 task。

接入状态机后再次用 `model_19998.pt`、32 环境、100 控制步做 CPU rollout：无 NaN/Inf，且既有四项物理警戒均未越界。导向板 8 个环境全部接触过板，逃脱 0 个、无效终止 0 次；自由板 8 个环境全部接触过板，1 个满足连续净空完成条件，共发生 4 次无效终止，诊断均为穿透超过 `0.020 m`，没有过力或未接触超时。自动 reset 后读取的普通瞬时峰值会遗漏终止前一帧，因此判断无效原因应以状态机保存的终止原因计数为准。

此后的奖励迁移已补齐历史 A6 的九项共同正向恢复任务，并保持 AMP 任务/风格混合关系可解释；`alpha` 与 G−/G+ 配对开关也已完成。当前仍不应直接启动大规模联合训练，应先完成同父 checkpoint、同 seed 的配对短训练。

## 五项障碍交互奖励（2026-09-30）

已按冻结 YAML 接入 G/几何进展 `+0.45`、dense clearance `+0.08`、completion `+0.60`、excess force `−0.03` 和 separation `+0.01`。G 使用单独的无“高于板”豁免水平投影，并要求实际手掌接触地面、头高不超过 `0.90 m`、曾接触板、尚未脱困且状态有效；因此竖直抬高身体不能冒充水平脱困。完成判据仍使用上一节的保守几何与 15 步确认，两套几何没有合并。

4 个奖励纯函数测试与 1 个水平投影反例测试通过。`model_19998.pt` 的 32 环境、100 步 CPU rollout 中，平地五项积分全部为零；引导板的平均加权积分为 G `0.02173`、clearance `0.02077`、completion `0`、force `−0.00348`、separation `0.00082`；自由板依次为 `0.02567/0.03911/0.06750/−0.00015/0.00443`。这些值已经包含配置权重与 `dt=0.02`，尚未包含 AMP 算法随后执行的 50:50 task/style 混合。力软代价在这段旧策略轨迹上没有压倒正向项，但样本很小，不能据此固定最终训练结论。

详细的原生项、共享项、重复风险和 AMP 混合后有效系数记录在 `A6_REWARD_MAPPING.md`。正式 G−/G+ 配置已经可构造，但尚未用配对短训练证明训练数值稳定或几何引导有效。

## 四阶段状态与控制步公共成本（2026-09-30）

独立 A6 task 已加入冻结版四阶段状态。reset 后不使用 bank 文件标签直接指定阶段，而是读取当前机器人实际状态：头部高度和躯干直立程度决定初始阶段。运行时再结合双膝角度、头部竖直速度和双脚最小地面载荷，连续满足 10/10/25 个控制步后才进入下一阶段；机器人重新跌倒时退回阶段 0。新增 `quality_feet` 传感器只供环境内部奖励和阶段判断使用，没有加入 observation，因此 actor/critic/discriminator 仍保持 480/590/10×85，29 维动作也未改变。

旧 AMP 的 `action_rate_l2(-0.005)` 已被冻结 A6 的动作变化成本 `−0.0015` 替换，并新增动作二阶变化成本 `−0.0012`；两者分别按阶段乘 0.5/0.75/1/1 与 0.3/0.6/1/1。旧软关节限位使用同一原始公式，但权重为 `−10`，现已替换为 A6 的 `−0.1`。旧 `joint_acc_l2` 与 `joint_torques_l2` 也已删除，由下述十物理子步平均版本取代，不再存在重复惩罚。

5 个纯函数测试覆盖 reset 阶段、连续保持、双脚载荷严格边界、跌倒回退和阶段成本系数。32 环境、30 步 CPU 零动作 rollout 中两个动作成本均为零，软限位加权积分均值为 `−0.00416`；同一规模的 `model_19998.pt` rollout 严格加载旧 checkpoint，三个成本积分均值分别为 `−0.00110/−0.00200/−0.00437`，证明新成本经过 reward manager 实际执行。两次 rollout 都保持有限 observation/reward 且没有超过已有物理告警阈值。策略 rollout 的自由板环境出现 4 次穿透无效终止，属于旧策略进入新场景后的失败样本，不是 checkpoint 形状或成本计算故障。

## 六项物理子步公共成本（2026-09-30）

`metrics_manager.compute_substep()` 现在在每个 0.002 s 物理子步后调用 A6 成本采样器；每个 0.02 s 控制周期积累十次，再由 reward 读取平均值。接入项为 joint acceleration `−5e−8`、torque `−1e−6`、joint overspeed `−0.02`、joint overpower `−2e−6`、head overspeed `−1.0` 和 sustained effort `−0.05`。持续用力使用 0.5 s 指数移动平均，并取负载最高的三个关节；其归一化上限按当前 AMP 模型实际 actuator limit 读取，而不是硬编码 SMP 控制器的不同上限。

手算测试覆盖阶段系数、速度/功率阈值、头部上下行阈值以及 0.5 s 记忆公式。32 环境、30 步 CPU 策略 rollout 中六项加权积分依次为 `−0.00407/−0.00157/−0.11158/−0.00045/−0.26766/−0.00081`；零动作对照为 `−0.00454/−0.00098/−0.23775/−0.00113/−0.31554/−0.00103`。旧策略减少了超速和头部超速，但使用了更多扭矩，符合主动起身的控制权衡。两组均无 NaN/Inf 或既定物理告警越界。

## Q 脚部运动与 L 单关节卡死（2026-09-30）

Q 使用左右踝刚体的世界线速度，但只在机器人接近直立时逐渐开启：头高从 `0.85 m` 到 `1.15 m`、躯干直立度从 `0.70` 到 `0.93` 分别做线性门控。两脚速度平方的平均值以 `0.1 m/s` 为尺度并封顶为 10，配置权重为 `−0.03`。因此机器人倒地翻身时不会因为脚移动而被 Q 阻碍，接近站稳后才会被要求减少滑脚。

L 对 29 个关节分别计时。只有扭矩严格超过当前关节实际 actuator effort limit 的 70% 才累计，高负载中断就把该关节计时清零；负载门控到 95% 拉满，持续时间从 `0.30 s` 到 `1.00 s` 拉满，同时要求关节速度低于 `0.30 rad/s`。每控制步再用最近 25 帧（`0.50 s`）关节位置跨度检查它是否真的几乎没动，最终取最坏的一个关节，配置权重为 `−0.20`。这种逐关节最大值不会让其余正常运动的关节掩盖某个卡住的关节。

3 个新增手算测试覆盖 Q 的门槛/封顶、L 的严格 70% 边界和最坏关节选择。4 环境、30 步 `model_19998.pt` CPU rollout 严格加载旧 checkpoint，累计 300 次子步采样，无 NaN/Inf、提前 reset 或已有物理告警越界。该小样本始终处于阶段 0，且没有关节满足完整卡死条件，所以 Q/L 积分均为 0；这符合门控定义，不能单凭此轨迹证明它们在训练分布中的触发频率，后续应在完整 L/M/H 批次记录非零率和分位数。

## 九项共同正向恢复任务与辅助力清理（2026-09-30）

独立 A6 task 已接入冻结 A6 的九项正向任务：阶段姿态、头部竖直速度、头高、躯干直立、双脚安静、身体根部安静、角速度安静、关节速度安静和 action 平滑。权重依次为 `0.22/0.18/0.10/0.15/0.08/0.07/0.07/0.06/0.07`。其中阶段姿态与头部速度读取实时四阶段状态；脚部和根部安静项只在接近直立时开启；action 平滑项在 reset 后首步把差值置零，避免把新 episode 与上一个 episode 的 action 相减。

为避免同一目标重复计分，独立 A6 task 删除了原生 AMP 中功能重叠的 `ang_vel_xy_l2`、`lin_vel_xy_l2`、`target_orientation`、`target_base_height`、`target_joint_deviation_l2` 和 `low_height_progress`。这些删除只作用于独立 A6 task，默认平地 AMP task 保持不变。独立 A6 task 还移除了继承的 200 N 起身辅助 command、物理施力 event 及其 curriculum；`motion` command 仍保留，因为 AMP 判别器需要参考动作数据，但它不会再把机器人传送到动作帧。

5 个纯函数测试覆盖平滑门控、阶段目标、上升速度目标、近站立门控和 reset 首步 action 掩码；配置测试检查九项名称、权重、函数索引、旧重叠奖励移除以及辅助力链路清理。4 环境、5 步 CPU 零动作 rollout 和 4 环境、30 步 `model_19998.pt` 策略 rollout 均正常执行；旧 checkpoint 的 actor/critic 严格加载，九项积分均为有限值，未出现 NaN/Inf、意外 reset 或已有物理警告越界。该验证证明接线和数值基本健康，但不能证明奖励权重已经训练有效。

下一关进行同父 checkpoint、同 seed 的 G−/G+ 短配对训练，先比较各奖励积分、无效终止原因、净空完成率和物理安全指标，再决定是否扩大预算。

## G−/G+ 与受困倍率（2026-09-30）

新增两个明确的训练入口：`Mjlab-Recovery-AMP-A6-GMinus-Unitree-G1-Dev` 与 `Mjlab-Recovery-AMP-A6-GPlus-Unitree-G1-Dev`。旧的 `Mjlab-Recovery-AMP-A6-Unitree-G1-Dev` 保留为 G+ 兼容入口，已有检查命令不会失效；正式配对实验必须使用名字明确的两个入口，避免日志身份含糊。

G− 将九项正向恢复任务倍率固定为 `1.0`，并关闭 geometry progress、dense clearance 和 separation。G+ 在板场景尚未满足连续净空逃脱条件时把九项正向任务乘 `0.05`，在平地或逃脱后恢复 `1.0`，同时开启三项几何引导。两组都保留 completion、force、Q/L、其他公共成本和 AMP style reward；网络接口仍为 actor 480、critic 590、discriminator 10×85、action 29。

纯函数测试确认倍率在“平地/板下/已脱困”三种状态依次为 `1/0.05/1`，配置测试确认两组 observation/action 完全相同且奖励差异严格限定在上述位置。4 环境 × 5 控制步 CPU 实际运行中，G− RewardManager 为 22 项，G+ 为 25 项；两组均完成 50 次物理子步采样，无 NaN/Inf、意外 reset 或既定物理警告越界。G+ 的板下正向任务积分被缩小，安全成本与 G− 保持一致。该结果验证接线，不代表 G+ 已优于 G−。

## A6 适配课程时钟（2026-09-30）

父平地 checkpoint 保存的 `common_step_counter=480000`，但该数字是过去平地训练的年龄，不应让新加入的板质量课程直接跳到终点。A6 现在单独保存 `a6_curriculum_start_counter`，板质量进度使用 `(common_step_counter - start_counter) / 100000`：首次从平地 checkpoint 进入 A6 时把起点重置为当前总计数；以后从 A6 checkpoint 续训时恢复原起点。因此模型编号、PPO/AMP 优化器和全局计数能够连续，而板质量仍从 `4–6 kg` 开始逐渐扩展到 `4–12 kg`。

## A6 动力学随机化与动作延迟（2026-09-30）

独立 A6 训练任务已按历史训练实现补齐动力学随机化，而默认平地 AMP 任务保持不变。每次环境 reset 时，以 25% 概率使用名义参数；其余环境分别对双腿、上身、骨盆/腰三组刚体统一缩放质量与惯量，并对腰、髋、膝、踝、手臂、手腕六组电机统一缩放 `Kp/Kd`。同组使用同一个倍率，避免把一条腿或一个电机组内部随机成互相矛盾的机械系统。倍率宽度在 A6 的前 2000 次 PPO 更新中从 `±10%` 线性增加到 `±20%`。

训练模式另外启用：双脚摩擦 `0.3–1.2`、关节编码器偏差 `±0.015 rad`、躯干质心偏移（x/y `±0.025 m`、z `±0.030 m`）、每 `1–3 s` 一次的随机速度推力，以及 `0–5` 个 2 ms 物理子步的控制延迟，即 `0/2/4/6/8/10 ms`。play 模式关闭这些随机项并使用名义动力学和固定 `6 kg` 板，便于复现实验结果。

64 环境训练配置 reset 审计得到质量/惯量倍率 `0.9002–1.0992`、电机增益倍率 `0.9002–1.0997`、六档延迟全部出现，18/64 个环境使用名义参数；质量与惯量配对、`Kp/Kd` 配对及 effort limit 不变均通过检查。随后 G+ 32 环境零动作 30 步、G− 16 环境零动作 15 步，以及 G+ 4 环境加载 `model_19998.pt` 的策略 20 步均无 NaN/Inf 和既定物理告警越界。reset 瞬间仍可出现约 `3000–5000 rad/s²` 的关节加速度尖峰，已与 rollout 动态峰值分开记录；它是后续 GPU 短训练必须继续观察的风险，当前短验证不能证明长期训练稳定或策略已经适应这些随机量。

## 平地 AMP checkpoint 到 A6 的加载边界（2026-09-30）

`help.md` 要求外部 AMP 方法保留自身的 actor、normalizer、discriminator、动作历史及算法所需训练状态。因此首次从平地 `model_19998.pt` 进入 A6 时使用完整恢复：actor、critic、PPO optimizer、AMP discriminator、AMP optimizer、迭代编号和总步数都接续。虽然 critic 最初仍带有平地任务的价值估计偏差，但清空它或优化器会改变原生 AMP 的继续适配条件，不符合当前配对协议；G−/G+ 必须从同一份完整状态开始，让两组之间唯一约定差异保持为稠密脱困引导。

A6 新增课程使用独立的 `a6_curriculum_start_counter`。平地 checkpoint 没有该字段时，以载入时的总步数作为 A6 零点；A6 checkpoint 后续保存并恢复该字段。因此“完整恢复 AMP 训练状态”和“板质量/动力学宽度从 A6 起点开始”可以同时成立。play/评测仍显式只载入 actor，不会恢复训练优化器。

4 环境 CPU 真实加载检查得到：模型迭代号 `19998`、总控制步 `480000`、A6 课程起点 `480000`，PPO optimizer 恢复 17 个参数状态，AMP optimizer 恢复 6 个参数状态，验证结果为 `full_amp_resume_valid=True`。本地另建立了不复制模型数据的日志入口 `logs/rsl_rl/g1_amp_get_up_no_keybody_robust_separate/a6_parent_flat_19998/model_19998.pt`，训练器解析后仍指向原始 downloaded checkpoint。

## G−/G+ 服务器配对预检（待执行）

预检使用同一父 checkpoint、seed 42、512 环境和各 50 次追加更新。每次更新采集 24 个控制步，因此每个环境累计采集 1200 步（24 s），已经跨过至少一个 10 s episode；预检只验收训练链路、数值稳定性和日志完整性，不用来挑选正式 checkpoint。预检结束后，正式实验必须再次从原始 `model_19998.pt` 启动。

从本地 WSL 增量同步代码；该命令不删除服务器文件，并排除日志、虚拟环境和 checkpoint：

```bash
cd /home/ray1/code
rsync -av --itemize-changes \
  --exclude 'logs/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pt' \
  -e 'ssh -p 20643' \
  G1Recovery_AMP/ \
  czr@47.243.212.218:/home/czr/code/G1Recovery_AMP/
```

服务器上先校验父模型；期望 SHA256 为 `4280efe9de1d6772e33f93847208896c4f7126e62630fc95b1cfeafe72e9f47c`：

```bash
cd /home/czr/code/G1Recovery_AMP
sha256sum \
  logs/rsl_rl/g1_amp_get_up_no_keybody_robust_separate/2026-09-30_01-04-15_low_height_progress_from_14999_gpu3_5000/model_19998.pt
```

G− 与 G+ 命令如下。可分别放入两个 tmux 会话；若 GPU 3/4 不是同一型号，则改为在同一块 GPU 上顺序运行。

```bash
cd /home/czr/code/G1Recovery_AMP
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH="$PWD/src:$PWD" \
MPLCONFIGDIR=/tmp/matplotlib-g1-a6-gminus \
/home/czr/code/G1Recovery_mjlab/.venv/bin/train \
  Mjlab-Recovery-AMP-A6-GMinus-Unitree-G1-Dev \
  --env.scene.num-envs 512 \
  --agent.seed 42 \
  --agent.max-iterations 50 \
  --agent.resume True \
  --agent.load-run '^2026-09-30_01-04-15_low_height_progress_from_14999_gpu3_5000$' \
  --agent.load-checkpoint '^model_19998[.]pt$' \
  --agent.run-name a6_gminus_preflight_seed42_512x50 \
  --enable-nan-guard True
```

```bash
cd /home/czr/code/G1Recovery_AMP
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH="$PWD/src:$PWD" \
MPLCONFIGDIR=/tmp/matplotlib-g1-a6-gplus \
/home/czr/code/G1Recovery_mjlab/.venv/bin/train \
  Mjlab-Recovery-AMP-A6-GPlus-Unitree-G1-Dev \
  --env.scene.num-envs 512 \
  --agent.seed 42 \
  --agent.max-iterations 50 \
  --agent.resume True \
  --agent.load-run '^2026-09-30_01-04-15_low_height_progress_from_14999_gpu3_5000$' \
  --agent.load-checkpoint '^model_19998[.]pt$' \
  --agent.run-name a6_gplus_preflight_seed42_512x50 \
  --enable-nan-guard True
```

两组必须同时记录：是否出现 NaN/Inf、episode 长度、三类场景完成/无效终止计数、无效原因、每项 task/style reward、PPO/AMP loss、随机化抽样范围和物理安全峰值。只有两组都跨过完整 episode 且日志有限，才进入正式 4096 环境预算。

预检已完成。两组均从 SHA256 为 `4280efe9...e9f47c` 的同一 `model_19998.pt` 完整恢复，在 512 环境中追加 50 次更新并保存 `model_20047.pt`。最后 10 次更新中，G−/G+ 的 episode 长度均值分别为 `423.4/427.8` 控制步；completion 加权积分均值为 `0.00723/0.01282`，invalid-plate 统计为 `0.231/0.205`。两组所有 unsafe physics 指标均为 0，PPO value/surrogate 与 AMP discriminator loss 均保持有限。G+ 的三项稠密奖励均实际非零，证明引导链路已经进入训练数据。

为避免只看训练聚合值，进一步在 play 名义动力学下以相同 seed 123、16 环境运行完整 500 控制步。两组均 16/16 正常 time-out，无未知终止或安全阈值越界；8 个板场景中，G− 在自由板场景脱困 1 个，G+ 脱困 2 个，两组导向板均为 0。G+ 的 geometry progress、clearance、completion 和 separation 均明显非零，但持续用力、脚部运动与单关节卡滞成本略高。两组都没有环境进入最终阶段 3。因此预检结论是“接口、数值与引导触发通过，存在早期正向信号”，不能把 2 比 1 当作成功率结论或据此调整奖励。

## 恢复原生 AMP 混合（2026-10-01）

一次 512 环境、2000 次更新的探索实验曾把 A6 的
`style_reward_scale` 从原生值 `50` 降到 `1`，并在板下未脱困时把
task/style 混合从 `0.5/0.5` 改成 `1.0/0.0`。该实验的 G− 最终
`plate_completion=0`，G+ completion 在中途达到小峰值后下降；两组策略
判别分数均接近 `-0.96`，而动作标准差分别升至约 `1.44/1.54`。名义回放
对应表现为少动或不动、自由板下乱蹬以及导向板下缺少有效进展。

这条门控会移除 AMP 在受困阶段的原生运动先验，不符合外部方法保留自身
机制的适配边界。因此 `model_21997.pt` 只保留为失败诊断产物，不作为后续
父模型。当前实现已恢复所有场景统一的 `style_reward_scale=50` 和
`task_style_lerp=0.5`，并删除板下 task-only 接线；G−/G+ 的约定差异仍只由
共同任务的 `alpha` 与三项稠密脱困奖励构成。下一轮必须重新从同一份干净的
`model_19998.pt` 开始配对短训。
