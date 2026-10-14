# HoST G1 23 DoF 平地起身：mjlab 复现工作区

本目录克隆自 [InternRobotics/HoST](https://github.com/InternRobotics/HoST)，基线提交为 `70bb580949a336a920833700e4b5dc3bf7fe87ce`。原始论文是 [HoST: Learning Humanoid Standing-up Control across Diverse Postures](https://arxiv.org/html/2502.08378v2)。`legged_gym/` 和 `rsl_rl/` 保留官方 Isaac Gym 实现，新增的 `src/g1recovery_host/` 与 `scripts/` 是基于同级 `../mjlab` 的本地 23 DoF 移植。这里的目标是官方初始倒地姿态下的平地起身，不包含实机部署。

## 当前状态与复现边界

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| 下载官方源码与核对论文 | 完成 | 上述提交；官方 `g1_ground` 配置和代码 |
| 29 → 23 关节、轴向、限位对齐 | 完成 | `python3 tools/check_g1_23dof_alignment.py` 通过 |
| mjlab 23 DoF 模型、观测与控制 | 完成基础运行 | 三种模型源均为 23 关节/动作、76×6=456 观测，40 步控制边界 smoke 通过 |
| 四奖励组、四 critic PPO | 完成基础运行 | CPU 和 GPU 各完成一次最小参数更新，写出 checkpoint |
| 完整 10 秒 episode 与重置 | 完成基础运行 | 2 环境连续 520 步，记录 2 次超时重置 |
| 大规模训练与无辅助评估 | **完成 mjlab 平地起身复现；未达到论文原数值** | 选定 v2 第 2099 次检查点；独立留出种子 60–64 为 96.96% ± 0.83 个百分点（每种子 250 回合） |

**已在 mjlab 中复现平地起身行为，但不是论文数值或物理环境的严格复现。** 论文报告的平地结果为 99.5 ± 0.4%（5 次、每次 250 个测试 episode），是作者结果，不是本地结果。本机为 RTX 3060 Laptop 6GB；论文使用 4096 个并行 Isaac Gym 环境、12,000 次更新。当前选定权重谱系到第 2099 次更新，每次 512×50 个控制样本，累计约 5,376 万样本，为论文约 24.576 亿样本的 2.19%；中途更换过物理、奖励和优化器设置，不能把全部更新视为单一同条件实验。MuJoCo/Warp 与 PhysX 的接触动力学、碰撞几何和随机化仍不同，且 `--height-guard` 是本地新增奖励，故这里的成功率不能称为论文原环境的严格数值复现。

## 目录与调用关系

```text
G1Recovery_HoST/
├── legged_gym/legged_gym/envs/g1/g1_config_ground.py   # HoST 原配置；本地清理未驱动关节键
├── legged_gym/legged_gym/envs/base/host_ground.py      # 原观测、课程、奖励语义参照
├── legged_gym/resources/robots/g1/g1_23dof.urdf       # 官方 23 DoF 关节、轴向、限位参照
├── rsl_rl/                                              # 原四 critic PPO 参照
├── src/g1recovery_host/model.py                         # 可选 mjlab 29 锁定 6 关节或直接加载 HoST 23 DoF URDF
├── src/g1recovery_host/action.py                        # 23 维显式 PD、30 步无驱动、β 与辅助力
├── src/g1recovery_host/task.py                          # 平地场景、观测、重置、四奖励组、课程
├── src/g1recovery_host/ppo.py                           # 一 actor/四 critic 的 PPO 与分组 GAE
├── scripts/smoke_host.py                                # 连续仿真与控制边界核验
├── scripts/train_host.py                                # 训练与 checkpoint
├── scripts/eval_host.py                                 # 无辅助评估与 JSON 指标
├── scripts/benchmark_host.py                            # 多种子 5×250 回合统计
├── scripts/record_host.py                               # 单回合离屏 MP4 录制
├── tools/check_g1_23dof_alignment.py                    # 静态关节对齐核验
├── artifacts/host_g1_getup_seed50.mp4                   # 成功回合的可视化演示
├── logs/host_ground_mjlab/                              # 检查点、源码快照、评估 JSON
└── pyproject.toml                                       # 新适配包的依赖声明
```

数据流为 `model.py → mjlab Scene → task.py 的 76 维观测×6 帧 → ppo.py actor → action.py 的关节力矩 → mjlab 物理步 → task.py 四组奖励 → ppo.py 四 critic/GAE → train_host.py checkpoint`。`model.py`、`task.py`、`action.py`、`ppo.py` 是本地权威源码；`logs/`、`__pycache__/` 和 Warp 缓存是生成文件。同级 `../mjlab/.venv` 是现有隔离环境，不属于本项目源码，也没有复制到本目录。

## 23 DoF 对齐依据

对照 `../G1Recovery_AMP/src/g1recovery_amp/g1_model.py` 的 29 关节定义和 `../mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`。移除的六个是 `waist_roll_joint`、`waist_pitch_joint`，以及左右腕的 `wrist_pitch_joint`、`wrist_yaw_joint`。剩下 23 个关节的名称、顺序、轴向、角度限位与 HoST URDF 一致：HoST 0–12 → AMP 0–12，HoST 13–17 → AMP 15–19，HoST 18–22 → AMP 22–26。**29 维 AMP checkpoint 不能直接作为 23 维 HoST 策略**，因为动作维度、观测和奖励任务都不同。

`--model-source mjlab`（默认）把 mjlab 的 29 DoF MJCF 锁定六关节，得到 23 个关节与电机，质量约 33.34 kg。`--model-source host_urdf` 直接编译仓库自带的官方 23 DoF URDF，质量 34.13386 kg；两者差约 2.3%。`--model-source host_urdf_capsule` 基于同一官方 URDF，把 4 个肩部碰撞圆柱转换为胶囊，模仿原 Isaac Gym 配置的 `replace_cylinder_with_capsule=True`。头高分别取 MJCF 的 `head_collision` 和官方 URDF 的 `keyframe_head_link`。官方 URDF 经 MuJoCo 导入时，必须把 URDF 的碰撞几何设为可接触，并禁用 mjlab 默认针对另一套 G1 几何名的碰撞覆盖规则，否则机器人会穿地。mjlab 的电机列表按电机类别排列，并非关节顺序；控制项明确使用关节顺序，正则奖励使用映射到关节空间的力矩 `qfrc_actuator`，避免错位相乘。三种模型即使关节配置一致，也有不同的惯量、几何和接触响应，checkpoint 记录模型来源、物理版本和奖励变体，评估时默认要求匹配。物理机的电机顺序、方向、限位和增益仍须实机逐项核对，本代码不应直接上机。

## 环境和方法对应

- 物理步长 0.005 s，控制降采样 4 倍，策略频率 50 Hz；训练 episode 10 s。
- 每帧观测按 HoST 顺序包含机体角速度 3、投影重力 3、绝对关节角 23、关节速度 23、执行动作 23、动作缩放 β 1，共 76；按时间排列 6 帧，共 456。
- 起始 30 个控制步无关节驱动力；之后力矩为 `clip(Kp × β × clip(action,-100,100) − Kd × 关节速度, ±电机上限)`。官方 `LeggedRobotCfg` 的 `clip_actions=100`，不应把高斯探索动作提前裁到 ±1。`β` 初始 1.0；按官方源码，只有本次重置环境的**回合结束头高均值**超过 0.9 m 时，`β` 才减 0.02（最低 0.25）。竖直辅助力初始总量 200 N：官方 URDF 的 `torso_link` 与 `keyframe_torso_link` 各承担 100 N，成功后各减 20 N，即总量每次减 40 N，最低 0；没有额外 keyframe 链接的 mjlab 模型由一个躯干链接承担总力。辅助力只在驱动阶段且躯干趋近直立时施加。先前本地版本误按“回合内最高头高”撤辅助；`v2_512_seed42` 训练因此可能过早降低了辅助，不能与修正后训练直接等同。
- 奖励分为 `task`、`regu`、`style`、`target`。任务组是直立容差和头高容差之积，不乘步长；其他组乘 0.02 s。风格组包含双脚踝高度差近似实现的脚掌平行地面项。新训练默认 `paper_style_v2` / `height_guard_style_v2`：胫骨姿态和双脚间距按原 HoST 的踝俯仰链接计算。早期检查点使用踝横滚链接；`--legacy-style` 可完整恢复其 `paper` / `height_guard_v1` 奖励。四个 critic 分别估计组回报，各组优势标准化后按 `[2.5, 0.1, 1, 1]` 合并。策略使用裁剪 PPO、L2C2（让相邻状态的策略和价值预测保持平滑）、5 轮、4 小批次、初始学习率 1e-3；自然超时回合按原 RSL-RL 代码把 `γ×当前价值估计` 加入奖励后再计算优势，避免时间截断被当成失败终止。本移植把 actor 和每个 critic 的梯度分别限幅，以免 MuJoCo 中较大的正则项价值梯度压小 actor 更新；这是针对物理移植的优化差异，不是论文原封不动的算法。`--adaptive-kl` 可启用官方 PPO 所用的 KL 自适应学习率，默认关闭以保持旧实验的固定学习率行为；切换调度方式时须配合 `--reset-optimizer`，检查点会记录调度方式。
- 可选 `--height-guard` 把任务奖励乘以 `exp[-8×max(骨盆高度−0.95m, 0)²]`，抑制 MuJoCo 中把机器人弹到 1–2 m 高处仍获得直立/头高奖励的行为。它是**单独标记的物理适配实验**，不是论文原奖励；不加此标志就是原奖励版本。
- 重置包含 HoST 的初始关节角比例/偏移扰动和根部 XY ±1 m 扰动。当前**尚未**移植完整的摩擦、恢复系数、质量/质心、马达增益、感知噪声与延迟随机化；脚掌平行项用两个脚踝链接的高度差近似。Warp 对部分胶囊体/网格接触配对只提供一个接触点。因此当前任务是可运行的研究移植，而非逐项严格等价的物理实验。

## 运行、继续与评估

从本目录运行，复用 `../mjlab/.venv`（mjlab 1.6、MuJoCo 3.11、PyTorch 2.9 CUDA 12.8）。`PYTHONPATH` 指向本地新增包；`MPLCONFIGDIR` 把 matplotlib 缓存写入 `/tmp`。静态检查无需虚拟环境。

```bash
cd /home/ray1/code/G1Recovery_HoST
python3 tools/check_g1_23dof_alignment.py
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/smoke_host.py --device cpu --steps 40
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/smoke_host.py --device cpu --model-source host_urdf --steps 40
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/smoke_host.py --device cpu --model-source host_urdf_capsule --steps 40
```

先用少量环境做 GPU 预检，再逐渐增加 `--num-envs`，观察显存、步速和奖励。RTX 3060 6GB 不适合直接套用论文的 4096 环境规模。默认训练配置是 64 环境、50 步/环境/更新、12,000 次更新；大规模训练会持续很久并产生 checkpoint，日志/模型仅写入本项目的 `logs/host_ground_mjlab/`。

```bash
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/train_host.py \
  --device cuda:0 --num-envs 64 --steps-per-env 50 --iterations 12000 \
  --output logs/host_ground_mjlab
```

下述命令复现已完成的旧版物理适配实验。它从第 399 次权重迁移；`--iterations 800` 指总更新到 800，因此再做 400 次更新。`--legacy-style` 用于与当时的奖励版本一致；去掉此标志会启动修正后的 v2 奖励，不能得到同一个实验。

```bash
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/train_host.py \
  --device cuda:0 --model-source host_urdf_capsule --height-guard --legacy-style \
  --num-envs 512 --steps-per-env 50 --iterations 800 --epochs 5 --minibatches 4 \
  --resume logs/host_ground_mjlab/host_urdf_paper_512_seed42/model_00399.pt \
  --allow-model-transfer --output logs/host_ground_mjlab/host_urdf_capsule_guard_512_seed42
```

以第 799 次权重继续的 KL 自适应实验保留课程 β/辅助力，重建 Adam 优化器，并写入新的目录。`--iterations 1200` 表示从第 800 次更新继续至第 1199 次；两个实验的训练结果不要混合成一个同条件均值。

```bash
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/train_host.py \
  --device cuda:0 --model-source host_urdf_capsule --height-guard --legacy-style \
  --num-envs 512 --steps-per-env 50 --iterations 1200 --epochs 5 --minibatches 4 \
  --adaptive-kl --reset-optimizer \
  --resume logs/host_ground_mjlab/host_urdf_capsule_guard_512_seed42/model_00799.pt \
  --output logs/host_ground_mjlab/host_urdf_capsule_guard_adaptive_512_seed42
```

第 1199 次之后在同目录继续至第 1799 次，保留优化器和课程。当前源码默认 v2 风格奖励，故续训旧检查点必须显式指定 `--legacy-style`；该开关的输出已与旧训练源码快照在连续 40 个状态上逐项一致。`run_manifest.json` 记录了两段运行及各自源码哈希。

```bash
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/train_host.py \
  --device cuda:0 --model-source host_urdf_capsule --height-guard --legacy-style \
  --num-envs 512 --steps-per-env 50 --iterations 1800 --epochs 5 --minibatches 4 \
  --adaptive-kl \
  --resume logs/host_ground_mjlab/host_urdf_capsule_guard_adaptive_512_seed42/model_01199.pt \
  --output logs/host_ground_mjlab/host_urdf_capsule_guard_adaptive_512_seed42
```

选定的可用权重来自第 1799 次旧风格检查点，再以修正后的 v2 脚部风格奖励微调 400 次。模型与施力物理版本没有变；`--allow-model-transfer` 明确接受奖励版本迁移，`--keep-curriculum-on-transfer` 保留 512 环境各自的 β 与辅助力，Adam 动量则重新初始化。初始学习率降至 `1e-4`，KL 自适应调度继续开启。第 2099 次检查点是验证集上表现最好的交付权重；第 2199 次完整训练检查点也保留供对照。

```bash
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/train_host.py \
  --device cuda:0 --model-source host_urdf_capsule --height-guard \
  --num-envs 512 --steps-per-env 50 --iterations 2200 --epochs 5 --minibatches 4 \
  --adaptive-kl --learning-rate 0.0001 \
  --resume logs/host_ground_mjlab/host_urdf_capsule_guard_adaptive_512_seed42/model_01799.pt \
  --allow-model-transfer --keep-curriculum-on-transfer \
  --output logs/host_ground_mjlab/host_urdf_capsule_guard_stylev2_512_seed42
```

训练被中断后，`--resume` 会恢复策略、优化器、课程缩放和更新编号；仿真环境会从新 episode 重建，所以不是逐采样点精确恢复。旧 checkpoint 没有课程数据时会从同目录 `metrics.jsonl` 找到对应更新的课程均值。`--reset-curriculum` 可在继续权重/优化器训练时把 `β` 和辅助力恢复到初值；`--reset-optimizer` 可单独清零优化器动量。运行目录中保留 checkpoint 即可回退到较早模型。`--iterations` 是训练目标的**总更新编号**，不是本次再增加多少次。模型、施力物理版本或奖励变体不同的 checkpoint 会被默认拒绝；须明确加 `--allow-model-transfer` 才能迁移权重，并自动从初始课程参数和新优化器开始。对于 `height_guard_v1` 等旧检查点，继续同一奖励必须指定 `--legacy-style`；不指定时默认 v2，属于新实验，需要明确迁移。只有奖励变体变化而模型/物理版本相同时，才可同时使用 `--allow-model-transfer --keep-curriculum-on-transfer`，继承每环境课程参数；跨模型或跨物理版本不允许保留课程。上述迁移已完成训练和多种子评估。

官方平地评估关闭拉力，把动作缩放设为 0.25，持续 5 秒；成功定义为根部高度曾超过 0.7 m，之后不曾低于 0.5 m。本地评估入口沿用这个高度判据，默认使用部署时的**确定性**策略均值。用 50 环境×5 episode 得到一次 250 样本估计；重复不同种子才能与论文的均值和标准差比较。`--stochastic` 仅供训练分布诊断，不代表正式部署表现。

```bash
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/eval_host.py \
  --device cuda:0 --model-source host_urdf_capsule \
  --checkpoint logs/host_ground_mjlab/host_urdf_capsule_guard_stylev2_512_seed42/model_02099.pt \
  --num-envs 50 --episodes 5 --steps 250 --seed 60 \
  --output logs/host_ground_mjlab/host_urdf_capsule_guard_stylev2_512_seed42/holdout_02099_seed60_64/seed_60.json
```

复现论文的五次统计与独立留出验证：

```bash
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/benchmark_host.py \
  --device cuda:0 --model-source host_urdf_capsule \
  --checkpoint logs/host_ground_mjlab/host_urdf_capsule_guard_stylev2_512_seed42/model_02099.pt \
  --seeds 60,61,62,63,64 --num-envs 50 --episodes 5 --steps 250 \
  --output-dir logs/host_ground_mjlab/host_urdf_capsule_guard_stylev2_512_seed42/holdout_02099_seed60_64
```

录制一个无辅助的成功回合；命令会同步写出同名 JSON，其中包含高度判据和最终高度：

```bash
MPLCONFIGDIR=/tmp/mpl-host PYTHONPATH="$PWD/src" ../mjlab/.venv/bin/python scripts/record_host.py \
  --device cuda:0 --model-source host_urdf_capsule \
  --checkpoint logs/host_ground_mjlab/host_urdf_capsule_guard_stylev2_512_seed42/model_02099.pt \
  --seed 50 --steps 250 --render-stride 2 --require-success \
  --output artifacts/host_g1_getup_seed50.mp4
```

旧训练的 `probe64_seed42/model_00499.pt` 和 `v2_512_seed42/model_00999.pt` 分别在 250 个无辅助回合里得到 **0/250**；`v2_512_seed42/model_00799.pt` 在 β=1、辅助力 200 N 的诊断条件下有 248/250 回合曾站高，但随后全部跌倒，仍为 **0/250**。旧训练在动作裁剪、脚掌平行奖励、撤辅助判据、关节空间力矩对应和超时价值补偿上与现在的代码不同，所以这些 checkpoint 是失败基线，不是可交付的起身策略。修正前的短暂官方 URDF 探针保留在 `logs/host_ground_mjlab/host_urdf_preclip_512_seed42/` 和 `host_urdf_pretimeout_512_seed42/`。`host_urdf_paper_512_seed42/` 跑到第 399 次更新时还将总辅助力集中施加在一个躯干链接；该目录中的源码快照保留了当时行为，新代码已改按原仓库分给两个链接。

官方 URDF 胶囊模型与高度抑制变体的已完成评估如下。表中“曾站高”只要求某时刻骨盆超过 0.7 m；成功还要求之后从未低于 0.5 m。最后 1 秒持续高于 0.7 m 是额外诊断指标，不是论文成功定义。

| 检查点 | 诊断条件 | 曾站高 | 论文判据成功 | 平均最长连续站高 |
| --- | --- | ---: | ---: | ---: |
| `model_00599.pt` | β=0.919，辅助力 47 N，250 回合 | 250/250 | 0/250 | 0.87 s |
| `model_00699.pt` | β=0.881，辅助力 0 N，250 回合 | 250/250 | 0/250 | 0.78 s |
| `model_00799.pt` | **β=0.25，辅助力 0 N，250 回合** | **0/250** | **0/250** | 0 s |
| `model_00799.pt` | β=0.839，辅助力 0 N，250 回合 | 250/250 | 0/250 | 0.87 s |
| 自适应 `model_00899.pt` | β=0.79，辅助力 0 N，250 回合 | 250/250 | 0/250 | 1.12 s |
| 自适应 `model_00999.pt` | β=0.718，辅助力 0 N，250 回合 | 250/250 | 0/250 | 1.16 s |
| 自适应 `model_00999.pt` | β=0.5，辅助力 0 N，250 回合 | 250/250 | 3/250 | 0.76 s |
| 自适应 `model_01199.pt` | β=0.5，辅助力 0 N，250 回合 | 250/250 | 0/250 | 1.22 s |

第 799 次接近训练 β 的诊断中有 105/250 回合在终点高于 0.7 m，但全部曾在起身后跌破 0.5 m，仅 5/250 回合最后 1 秒持续高于 0.7 m。第 999 次在 β=0.5 的 3/250 次原判据“成功”中，并不能据此认定稳定站立；该组总共只有 1/250 回合最后 1 秒持续高于 0.7 m。第 1199 次同 β 下原判据回到 0/250，平均连续站高升至 1.22 秒，最后 1 秒持续站高为 17/250，且与原判据成功的交集为 0。第 1199 次检查点的 512 个环境 β 最小约 0.28、中位数约 0.56，没有环境到达部署 β=0.25；辅助力已全部归零。这些早期诊断说明“曾站高”不足以证明可用起身，后续继续训练并修正脚部风格奖励。

## 完成的无辅助评估

下面各组均使用确定性动作、β=0.25、辅助力 0 N、每回合 250 步（5 秒）。先以未用于训练的种子 45 比较检查点，再按五个种子各 250 回合统计；选定 2099 次权重后，还使用未参与选权重的种子 60–64 做留出验证。表中百分比为五个种子成功率的均值 ± 样本标准差，单位为百分点。

| 权重/评估集 | 论文高度判据成功 | 成功且最后 1 秒持续高于 0.7 m |
| --- | ---: | ---: |
| 旧风格 1799 次，种子 45 | 217/250（86.8%） | 210/250（84.0%） |
| 旧风格 1799 次，种子 50–54 | 1052/1250（84.16% ± 3.24） | 999/1250（79.92% ± 3.03） |
| v2 1899 次，种子 45 | 180/250（72.0%） | 178/250（71.2%） |
| v2 1999 次，种子 45 | 229/250（91.6%） | 216/250（86.4%） |
| **v2 2099 次，种子 45** | **241/250（96.4%）** | **229/250（91.6%）** |
| **v2 2099 次，种子 50–54（选权重用）** | **1209/1250（96.72% ± 1.15）** | **1167/1250（93.36% ± 1.66）** |
| v2 2199 次，种子 50–54 | 1188/1250（95.04% ± 1.51） | 1163/1250（93.04% ± 1.04） |
| **v2 2099 次，种子 60–64（独立留出）** | **1212/1250（96.96% ± 0.83）** | **1167/1250（93.36% ± 1.54）** |

留出集逐种子成功数为 241、245、244、242、240（各 250 回合）。完整数据分别在 `logs/host_ground_mjlab/host_urdf_capsule_guard_adaptive_512_seed42/benchmark_01799_seed50_54/summary.json`、`logs/host_ground_mjlab/host_urdf_capsule_guard_stylev2_512_seed42/benchmark_02099_seed50_54/summary.json`、`benchmark_02199_seed50_54/summary.json` 和 `holdout_02099_seed60_64/summary.json`；同目录有每种子 JSON。视频 `artifacts/host_g1_getup_seed50.mp4` 是 seed 50 的 5 秒成功回合，伴随 JSON 记录最终骨盆高 0.738 m、最高 0.763 m、最长连续站高 3.32 秒。视频中可观察到腿部交叉、双臂姿态不对称；高度判据成功并不等于姿态符合实机安全要求。

与论文作者在 Isaac Gym 报告的 99.5% ± 0.4% 相比，本地留出集差约 2.54 个百分点；由于物理引擎、碰撞和随机化、训练预算以及新增高度抑制奖励不同，这只是同任务的 mjlab 仿真复现，不是原实验的严格重跑，也未经真实 G1 验证。当前训练只覆盖官方初始倒地姿态和水平地面，不证明俯卧、多种受扰姿态或复杂地形的成功率。若要扩展，优先补全 HoST 的域随机化及多姿态重置，重新训练后再用从未参与选权重的新种子评估；真实机器人还需独立核对电机映射、增益、限位与紧急停机机制。

每个训练目录的 `run_manifest.json` 与 `source_snapshot/` 留存了训练命令参数、源码哈希和可回查代码；检查点和 JSON 可用于继续训练或重新评估。官方原实现保留在 `legged_gym/` 与 `rsl_rl/`，仅对 `legged_gym/legged_gym/envs/g1/g1_config_ground.py`、`legged_gym/legged_gym/utils/helpers.py` 做了本地清理；`git diff` 可查看。若要回到官方基线，可单独撤销这两处本地修改；新 mjlab 适配层和训练产物在独立路径，可保留作对照。
