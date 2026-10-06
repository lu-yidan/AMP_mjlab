# HoST 23→29 DoF 最小迁移：进度、失败经验与当前结论（2026-10-05）

## 1. 结论摘要

当前不应启动 4096 长训，也不应以本轮模型进入 A6。

在多轮全网络 PPO 和奖励修补持续破坏 23 DoF 恢复先验后，本分支从
`codex/recover-study-host@965b5b2` 建立了一条最小迁移线：保留 23 个共有输出、共享
MLP 和 actor normalizer，只训练 29 DoF 资产新增的六个输出行。该实现和预检均已通过，
但 1024 环境、200 更新后的严格零辅助评估仍失败：四姿态连续 1 秒/5 秒保持全部为 0，
宽松恢复也几乎完全丢失。说明“冻结 23 DoF 闭环，只训练六个新增输出”本身仍不足以
保证闭环等价；下一步必须先定位 23→29 的观测、动作行和机器人动力学是否真正保持同构，
不能继续靠增加奖励或延长训练解决。

## 2. 可复现资产和谱系

| 项目 | 值 |
|---|---|
| 基线提交 | `965b5b2` (`codex/recover-study-host`) |
| 当前分支 | `run/host29-minimal-host-20261005` |
| 23→29 actor 初始化 | `/home/sjw/AMP_mjlab-host29-control-audit/artifacts/host23_lift29_20261004/actor_init.pt` |
| 初始化 SHA256 | `f1616efa1eb125a6fa39a5f8ee529a6e97cfdca976aad3d1c2416892e4c993e2` |
| 静止贴地 reset bank | `/home/sjw/AMP_mjlab-host29-upright-v2/outputs/native_rest_v2/train.npz` |
| bank SHA256 | `7648180fa2a2ef0a8f5d297d013d96d9bf798380d8ad95fc17a21a726bc31840` |
| 本轮最终模型 | `logs/host29_minimal_added6_pilot1024_20261005/model_200_final.pt` |
| 模型 SHA256 | `f272fe30773f7281de3515b4ae372f8bde4b6d7556e31930779b0630bc0a8256` |
| 环境/更新 | 1024 环境，200 更新，20 s episode |
| 辅助 | 无 teacher、无外力、无 AMP 判别器 |

`native_rest_v2` 只是本地原生平地派生 reset 池，不是独立最终测试集，也不得替代历史
A6 三份 bank。

## 3. 当前最小迁移实现

- `src/tasks/host_recovery/minimal29.py`
  - 直接复用 HoST/普通 PPO 环境；静止贴地、初速度为 0、六帧静态历史、首步立即动作。
  - 23 个共有轴保留增量位置控制。
  - `waist_roll`、`waist_pitch` 和左右腕 `pitch/yaw` 使用显式 home + `action * 0.25`。
- `scripts/train_host29_minimal.py`
  - actor 与 actor normalizer 从 23→29 映射模型继承；critic/optimizer 重新初始化。
  - 共享隐藏层和 23 个共有输出行冻结，只允许六个新增输出行更新。
  - 固定学习率 `1e-5`，2 epochs，探索标准差固定为 `0.03`。
  - 训练结束逐张量断言共享隐藏层和共有输出行完全不变。
- `scripts/eval_host29_minimal.py`
  - 四姿态各 256 次、20 秒、确定性、零辅助。
  - 同时记录宽松恢复、严格 1 秒/5 秒保持、躯干/骨盆直立、膝角、脚宽、前后错脚、
    关节/基座/足端速度和动作变化。

## 4. 已通过的工程验收

- 128 环境 × 1000 步完整 20 秒 episode：出现真实 timeout，状态、观测和奖励有限。
- 128 环境 × 2 更新梯度反例：只有六个新增输出行改变，共享 MLP 和 23 行逐元素不变。
- 4096 环境 × 100 步 GPU 预检通过。
- 1024 环境 × 200 更新完成；无 OOM、NaN/Inf，`dof_vel_out=0`、`base_vel_out=0`。
- 上游 `target_feet_height_var` 对单元素维度调用无偏方差会产生 degrees-of-freedom
  warning；本轮数值断言未发现非有限 reward，但该项仍需单独修复和回归，不能视为已解决。

## 5. 本轮严格评估结果

| 初态 | 宽松恢复 | 严格保持 1 s | 严格保持 5 s | 躯干高度 | torso upright | 最大膝角 | 脚宽 | joint RMS | 异常终止 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prone | 3.52% | 0% | 0% | 0.182 | 0.244 | 2.684 | 0.394 | 1.116 | 0% |
| supine | 0% | 0% | 0% | 0.147 | 0.173 | 2.510 | 0.467 | 0.914 | 0% |
| left side | 0.78% | 0% | 0% | 0.128 | 0.069 | 2.547 | 0.480 | 0.907 | 0% |
| right side | 0% | 0% | 0% | 0.129 | -0.047 | 2.426 | 0.508 | 0.915 | 0% |

注：右侧一行的严格保持 5 秒为 0%；表格保留原始量纲（m、rad、rad/s）。所有方向均远低于
严格标准：相对足高 0.75 m、pelvis/torso upright 0.95、膝角 0.30 rad、脚宽
0.18–0.35 m、前后错脚 0.12 m、joint RMS 0.5 rad/s。

## 6. 已证伪方案与失败经验

### 6.1 奖励局部最优不是单一权重问题

- 早期配方允许低位姿态持续获得 recovery 奖励，而 pose/torso/quiet 长期为 0。
- 将正奖励锁存在越过高度阈值后，会出现“短暂越界、随后掉回低位、永久领奖”的套利。
- 只将成本锁存在越界后可消除该套利，但仍无法把低位 actor 带出恢复盆地。
- 膝/脚宽的有界正奖励在膝角 0.8–1.1 rad、脚宽 0.14–0.16 m 时仍不够区分；
  增加 barrier 也会通过共享 PPO 梯度破坏已有恢复动作。
- 按方向重加权没有解决共享 actor 干扰，反而同时损伤容易方向和困难方向。

### 6.2 六个新增轴存在两种不同的控制语义

- 对原生 29 DoF actor，动作应为 `current_position + increment`；旧实现把六轴写成
  `action * 0.25` 的绝对零附近目标，确属控制约定错误。
- 对 23→29 迁移 actor，源模型根本没有六个输出；`current_position + 0` 只提供阻尼，
  无法让新增腰/腕回中。因此迁移适配需要明确的 `home + increment`。
- 腰/腕统一使用 0.94 动作尺度过大；拆分为 0.25 后控制更合理，但仍未解决闭环恢复。
- 先前还发现并联腰/踝电机的速度-扭矩曲线与双电机建模不一致。本最小分支直接继承
  `flat29`，尚未单独证明该修复是否已包含；这是进入下一轮前必须显式核对的动力学项。

### 6.3 全网络 PPO 会灾难性遗忘恢复先验

- lift23 + 350 N 混合动力学、较高探索和普通 PPO 使 supine/right 恢复率明显下降，
  且站姿塌缩为折腰、深屈膝、高速度。
- `stance2` 的 checkpoint 150 明显优于 200/250/300，说明失败不是训练时长不足；
  继续训练会恶化困难方向。
- `stance3` 方向重加权后，supine 恢复从 100% 降到 94.53%，高度从 0.775 降到
  0.721，膝角从 0.778 恶化到 0.977；right 恢复从 92.19% 降到 89.06%，膝角
  从 1.053 恶化到 1.072。
- `stance4` 即使加入严格非平均 barrier，四方向严格保持仍全部为 0；最终宽松恢复
  98.05/98.44/100/89.84%，但膝角仍为 0.909/0.851/0.794/1.070 rad，脚宽仅
  0.159/0.158/0.144/0.149 m。

### 6.4 `AMP_mjlab` 不代表当前策略正在使用 AMP 奖励

- 仓库名是 `AMP_mjlab`，但当前 HoST checkpoint 的 `agent.yaml` 使用普通 PPO。
- 当前配置没有 `AMPPPO`、判别器或 `amp_reward_coef`。
- `style_*` 是手工姿态/运动约束，不是 AMP 判别器风格奖励；不能按“去掉 AMP 奖励”
  将其整体删除。

## 7. 当前判断与下一步门槛

### 7.1 23→29 控制契约核查结果

`scripts/audit_host23_to29_contract.py` 对实际运行配置完成了逐项核查，证据保存在
`artifacts/host29_minimal_20261005/contract_audit.json`：

- 观测映射正确：23 DoF 的 456 个历史输入按名称映射到 564D 输入；新增的 108 列在
  actor 第一层权重中严格为 0。六帧静态历史误差为 0。
- 动作行映射正确：23 个共有输出映射到
  `[0..12, 15..19, 22..26]`，新增六轴为 `[13,14,20,21,27,28]`；初始化时新增
  输出权重和偏置严格为 0。
- PD target 语义已按预期执行：共有轴为 `current + action*0.25`，新增轴为
  `home + action*0.25`。
- 两份 MJCF 的共有关节轴、关节范围、共有刚体质量与惯量完全一致；问题不在共有刚体
  几何或惯量。
- **共有执行器动力学不一致**：23 DoF 的所有共有轴使用普通
  `BuiltinPositionActuatorCfg`；29 DoF 将其替换为带速度–扭矩曲线和摩擦的
  `UnitreeActuator`。这改变了原策略实际看到的闭环。
- 两个 hip pitch 从 23 DoF 的 `Kp=40.179, Kd=2.558, 88 Nm` 组换成了
  `Kp=99.098, Kd=6.309, 139 Nm` 组，刚度、阻尼、armature 和限幅全部变化。
- 共有 ankle 在 23 DoF 中是普通双电机等效 `50 Nm`；29 DoF 虽声明 effort limit
  50 Nm，却仍使用单电机曲线 `Y1=24.8/Y2=31.9` 以及额外摩擦，等效动态能力与
  23 DoF 不同。新增 waist 也存在同样的单/双电机曲线不一致。
- 其余共有 hip/knee/arm 虽保持标称 Kp/Kd，大多也新增了速度相关限幅和摩擦。
- 未训练的映射 actor 在当前 29 DoF 配置上的零辅助恢复仅为
  prone/supine/left/right `7.42/0/0/0%`，严格保持全 0；因此失败在任何 PPO 更新前
  就已存在。200 更新只训练新增六行后为 `3.52/0/0.78/0%`，没有修复闭环。

结论：观测排列和动作行映射不是当前主因；主因是把 23 DoF actor 放进了不等价的
29 DoF 执行器闭环。仅冻结网络或调整 reward 不可能恢复原策略行为。

### 7.2 最后一次 29 DoF 尝试与回退条件

本分支已把最后一次 29 DoF 尝试限定为控制匹配：29 DoF 的 23 个共有关节恢复为与
23 DoF 完全相同的 Builtin PD 分组、Kp/Kd、armature 和 effort limit；六个新增轴
保留独立 home 控制。先评估未训练 actor，再做有上限短训。若四方向零辅助恢复和
严格站姿仍不成功，停止 29 DoF 路线，按用户要求改用已验证的 23 DoF HoST 策略进行
压板/A6 训练。

已落地到当前分支：

- `src/assets/robots/unitree_g1/g1_constants_bp.py` 将 29 DoF 共有轴执行器分组
  改回与 23 DoF 相同：`hip_pitch/hip_yaw/waist_yaw` 使用
  `Kp=40.179, Kd=2.558, 88 Nm`，`hip_roll/knee` 使用
  `Kp=99.098, Kd=6.309, 139 Nm`；其余共有轴保持 23 DoF Builtin PD。
- `src/tasks/host_recovery/flat29.py` 改用 `g1_constants_bp.get_g1_robot_cfg()`，
  因此 29 DoF 不再使用带速度-扭矩曲线/摩擦的 `UnitreeActuator`。
- `scripts/audit_host23_to29_contract.py` 改为比较 `g1_constants_bp`，并对 23 个
  共有轴的 class/Kp/Kd/effort/armature 做硬断言，未匹配会直接失败。
- 23 DoF 回退资产单独归档到 `archives/23dof_host_a6_20261005/`，包含 MJCF、
  动作/观测维度和 A6 配置。

### 7.3 最终控制匹配结果与停止 29 DoF

`scripts/audit_host23_to29_contract.py` 已对实际加载的 29 DoF matched 配置通过，
23 个共有轴的 class/Kp/Kd/effort/armature 不再有任何不匹配。再用未训练的
23→29 映射 actor 做 1024 环境 x 200 更新短训，并在同一四姿态零辅助严格评估下复测。

源 actor 零辅助宽松恢复为 prone/supine/left/right
`0/12.89/0.78/1.56%`；200 更新后为 `0/6.25/0.39/1.56%`。四方向严格 1 秒/5 秒
保持仍全部为 0，异常终止仍为 0。因此控制闭环匹配本身不足以恢复 23 DoF 行为，
29 DoF 路线停止。

后续改用已验证可恢复的 23 DoF HoST 基线进行压板/A6 训练；23 DoF 的 MJCF、
动作/观测维度和 A6 配置已单独存档，禁止继续把 29 DoF 资产或失败模型带入该路径。
本轮结果证明，仅冻结 23 DoF actor 参数仍不能证明运行闭环等价。执行优先级为：

1. 对相同静止 bank，在训练前后逐步回放并逐项对齐 23 与 29 的 564D 观测语义、历史排列、
   关节名称到 action row 的映射、action scale、PD 目标和实际关节力矩。
2. 逐关节核验实际加载 MJCF、actuator gear/forcerange、Kp/Kd、速度-扭矩限制、惯量和摩擦；
   特别核对并联腰/踝模型与六个新增轴。
3. 先评估“未训练的 23→29 home 适配 actor”，再与每个训练 checkpoint 做相同 seed 的
   paired rollout，定位性能是在第一个动作、normalizer、控制映射还是 PPO 更新后丢失。
4. 修复 `target_feet_height_var` 的单元素方差定义并添加数值回归。
5. 只有四方向零辅助恢复不退化，且膝、脚宽、真实躯干直立、速度和连续保持呈一致改善，
   才能启动每卡 4096 的长训；严格验收前不得启动 A6。

## 8. 复现入口

先复核共有执行器契约，再用未训练的 23→29 映射 actor 做最终控制匹配短训：

```bash
cd /home/sjw/AMP_mjlab-host29-minimal
export PYTHONPATH=. TORCHDYNAMO_DISABLE=1 MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=0 /home/sjw/AMP_mjlab/.venv/bin/python \
  scripts/audit_host23_to29_contract.py \
  --checkpoint /home/sjw/AMP_mjlab-host29-control-audit/artifacts/host23_lift29_20261004/actor_init.pt \
  --bank /home/sjw/AMP_mjlab-host29-upright-v2/outputs/native_rest_v2/train.npz \
  --output logs/host29_matched_contract_20261005/contract_audit.json

CUDA_VISIBLE_DEVICES=0 /home/sjw/AMP_mjlab/.venv/bin/python \
  scripts/train_host29_minimal.py \
  --checkpoint /home/sjw/AMP_mjlab-host29-control-audit/artifacts/host23_lift29_20261004/actor_init.pt \
  --bank /home/sjw/AMP_mjlab-host29-upright-v2/outputs/native_rest_v2/train.npz \
  --num-envs 1024 --iterations 200 --rollout-steps 100 \
  --log-dir logs/host29_matched_shared_actuators_pilot1024_20261005

CUDA_VISIBLE_DEVICES=0 /home/sjw/AMP_mjlab/.venv/bin/python \
  scripts/eval_host29_minimal.py \
  --checkpoint logs/host29_matched_shared_actuators_pilot1024_20261005/model_200_final.pt \
  --bank /home/sjw/AMP_mjlab-host29-upright-v2/outputs/native_rest_v2/train.npz \
  --output logs/host29_matched_shared_actuators_eval200_20261005
```

旧的最小六输出失败实验仍可从以下命令复现：

```bash
CUDA_VISIBLE_DEVICES=0 /home/sjw/AMP_mjlab/.venv/bin/python \
  scripts/train_host29_minimal.py \
  --checkpoint /home/sjw/AMP_mjlab-host29-control-audit/artifacts/host23_lift29_20261004/actor_init.pt \
  --bank /home/sjw/AMP_mjlab-host29-upright-v2/outputs/native_rest_v2/train.npz \
  --num-envs 1024 --iterations 200 --rollout-steps 100 \
  --log-dir logs/host29_minimal_added6_pilot1024_20261005

CUDA_VISIBLE_DEVICES=0 /home/sjw/AMP_mjlab/.venv/bin/python \
  scripts/eval_host29_minimal.py \
  --checkpoint logs/host29_minimal_added6_pilot1024_20261005/model_200_final.pt \
  --bank /home/sjw/AMP_mjlab-host29-upright-v2/outputs/native_rest_v2/train.npz \
  --output logs/host29_minimal_added6_eval200_20261005
```
