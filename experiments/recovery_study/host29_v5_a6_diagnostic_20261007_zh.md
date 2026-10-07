# 29 DoF directional/stability v5 与 23 DoF A6 阶段记录（2026-10-07）

## 结论摘要

- 29 DoF 肘屈曲 v4 已完成四姿态正式评测：held、ever、final standing 均为
  100%，`final_joint_speed_rms` 分别为 2.05、2.20、2.15、2.09 rad/s。
- 29 DoF directional/stability v5 正在从 v4 做低扰动续训。本文记录代码、
  参数和中间状态，不把中间 checkpoint 宣称为最终成果。
- 23 DoF A6 G-/G+ 重训已按用户要求停止。G- 保留到 `model_5000.pt`，
  G+ 保留到 `model_5500.pt`。两组均未形成完整的脱困到站立动作链。
- A6 诊断录像已覆盖 3 场景 x 4 方向，并包含垂直板俯卧/仰卧程序化案例。
  视频、checkpoint 和 TensorBoard 数据不提交 Git，只记录可复现路径。

## 29 DoF directional/stability v5

训练入口：

```text
scripts/train_host29_directional.py
  --checkpoint logs/host29_elbowflex_v4_20261007/model_15999.pt
  --log-dir logs/host29_directional_stable_v5_4096_20261007
  --num-envs 4096
  --updates 3000
```

运行策略：

- GPU4，零拉力；
- actor 和 actor observation normalizer 从 v4 迁移；
- critic、critic normalizer 和 optimizer 重新初始化；
- actor observation normalizer 冻结；
- learning rate `3e-5`，entropy coefficient `0.001`；
- action noise std `0.05`，上限 `0.08`；
- 64 环境、100 步有限数值预检已通过。

新增的站起后门控奖励：

| 奖励 | 权重 | 目的 |
| --- | ---: | --- |
| 大臂与小臂同向 | 0.04 | 减少肘部异常折叠 |
| 小臂与手部同向 | 0.04 | 保持腕部与前臂方向一致 |
| 腿段竖直 | 0.08 | 站起后保持大腿、小腿朝下 |
| 腿部静止 | 0.06 | 用有界指数项降低站立抖动 |
| 腿关节速度 | -0.0012 | 温和惩罚站起后的腿部速度 |

2026-10-07 记录时训练位于约 `2522/3000`，最新周期 checkpoint 为
`model_2500.pt`，进程仍正常运行。最终结论必须等待
`model_3000_final.pt` 的四姿态 256 环境正式评测与录像。

相关实现：

- `src/tasks/host_recovery/directional29.py`
- `src/tasks/host_recovery/mdp/rewards.py`
- `scripts/train_host29_directional.py`

## 23 DoF A6 停止状态

A6 从未使用 29 DoF 的 `host29_elbowflex_v4_20261007/model_15999.pt`。
两组实际源模型都是原生 23 DoF：

```text
/home/sjw/AMP_mjlab-host29-control-audit/experiments/host_recovery/
static_diverse_elbow_v2/model_11999.pt
SHA256 2eea6c6b37127143e008c335ba787338956871b8e76e9b9613018c320e2120a7
```

训练入口在载入后逐参数断言 actor 与源 checkpoint 一致，launch manifest
同时记录 `actor_exact_at_launch=true`。这里的“V4”仅表示 actor-only 迁移、
fresh critic/optimizer 的第四版训练方案，不是 29 DoF v4 模型名。

停止的训练：

| 组别 | GPU | 最后保存 checkpoint | 停止时训练进度 |
| --- | ---: | --- | ---: |
| G- | 0 | `logs/a6_23_train_gminus_4096_20261007_retry/model_5000.pt` | 约 5405/10000 |
| G+ | 2 | `logs/a6_23_train_gplus_4096_20261007_retry/model_5500.pt` | 约 5470/10000 |

两条 Python 训练进程均已终止，A6 心跳自动化已暂停。不得从这两个
checkpoint 直接恢复原参数训练。

## A6 平地能力对照

使用同一 `a6_env_cfg`、关闭 dynamics/push/invalid-plate termination、固定
flat-only、64 环境、1150 步、seed 42 的确定性 mean-policy 对照：

| checkpoint | 平地主成功 |
| --- | ---: |
| 原始 23 DoF `model_11999.pt` | 52/64 (81.25%) |
| A6 G- `model_0.pt` | 51/64 (79.69%) |
| A6 G- `model_5000.pt` | 0/64 |
| A6 G+ `model_5500.pt` | 0/64 |

源模型与 `model_0` 的能力基本一致，证明动作/观测维度和 actor 初始化没有
被换成 29 DoF，也没有在启动时被任意改写。平地能力是在后续 PPO 更新中
被抹掉的，因此问题属于训练目标、采样和优化配置，而不是源模型选择错误。

## A6 视频观察

G-：

- 平地俯卧和左侧案例几乎完全静止；
- 平地仰卧、右侧及自由板左侧可撑起上身，但停在坐姿或跪坐；
- 板下案例通常保持蜷缩，没有持续水平净空，也没有双脚承重站立。

G+：

- 平地案例出现抬腿、举臂、蜷缩等动作，但没有形成支撑转换；
- 自由板案例会局部抬板、顶板或翻滚，未形成有效脱困；
- 垂直板自然与程序化案例均长期停留在板下。

因此失败不是单纯“训练轮数不足”。策略已经收敛到少动或局部移动板子的
低风险动作，而不是“手撑地 -> 收腿 -> 双脚承重 -> 站起”的目标序列。

## A6 根因

### 1. 训练终止条件截断了绝大多数恢复轨迹

`a6_invalid` 在板场景满足任一条件即终止：

```text
penetration depth < -0.02 m
plate force > 1500 N
episode step > 25 and no previous plate contact
```

第三项只给策略 0.5 秒建立或维持板接触。训练日志中的平均 episode 长度仅
约 68--73 control steps，即 1.36--1.46 秒，远低于配置的 23 秒。该训练
条件也比最终报告使用的 5 cm/5000 N 异常阈值严格，造成训练与验收不一致。

### 2. 静止项在躺卧阶段形成正奖励局部最优

当 standing gate 为零时，`feet_quiet` 直接为 1；低速度、低角速度、低关节
速度和低动作变化也持续获得正奖励，同时 action rate、action acceleration、
joint speed、torque 和 power 继续受罚。视频中的长期静止与该奖励结构一致。

### 3. G+ 在未脱困板下进一步削弱核心恢复奖励

G+ 对板下未 escape 环境把九项正任务统一乘以 `0.05`，负成本不缩放。
这会把动作成本相对放大约 20 倍，稠密几何奖励又可通过局部顶板获得，因而
出现“动板但不起身”的策略。

### 4. 探索噪声相对动作尺度过大

A6 续训固定 action noise std 为 `0.8`，而 action scale 为 `0.25`。尽管
actor 从已验证的 23 DoF checkpoint 精确迁移，训练采样仍被高噪声主导，
容易在新 critic 尚未稳定时抹掉原有起身技能。

## 诊断产物

远端目录：

```text
logs/evaluations/a6_23_stopped5000_gminus_video_diagnostic_20261007
logs/evaluations/a6_23_stopped5500_gplus_video_diagnostic_20261007
```

本地下载目录：

```text
C:\Users\26390\Desktop\mjlab\a6_23_video_diagnostic
```

归档 SHA256：

```text
G- videos: 706ec0315f66b96409ff38e14d6d66a47a13353e58491fec3b1208218ef11872
G+ videos and contact sheets: d9731d6d8acca568318b1c6347aeb52fa30121880f63c21051db7b08a63bac25
```

## 下一轮前必须修正

1. 将训练 invalid 条件与正式异常定义对齐，并取消 25 步无接触硬终止；
2. 让 quiet/stillness 奖励仅在明确的站起 gate 后生效；
3. 重新平衡 G+ 的 `0.05` alpha、几何奖励和未缩放成本；
4. 把迁移续训噪声降至与动作尺度匹配的范围，并先跑短周期回归；
5. 每个候选 checkpoint 先验证平地四姿态不退化，再进入板场景训练。
