# 29-DoF HoST 平地起身：腰、手臂与手腕姿态修复

## 结论

原始平地专家 `model_02999.pt` 能起身，但学到的腰 pitch、两侧手腕 pitch 和左肘动作会持续推向关节限位。六个新增关节的 MJCF、限位和电机参数均已配置；问题来自姿态奖励在多关节误差下饱和，以及相对关节动作缺乏回到站姿的直接学习信号。原始逐关节证据见 [HOST29_ARM_WRIST_DIAGNOSIS.md](HOST29_ARM_WRIST_DIAGNOSIS.md)。

修复后的无回放约束策略：

`logs/downloaded/host_ground29_pose_v6_teacher_fromv5_256_seed42_200/model_00199.pt`

SHA-256：`32a15165cdb5e012a13e875e0283506566ba92ea095f6115483a5414851c2a7e`。本地和服务器副本哈希一致。该策略仍为 **29 DoF**；训练时执行的始终是策略输出，回放也无需 `--stand-pose-guard`。

## 改动与原因

| 版本 | 改动 | 独立回放结论 |
| --- | --- | --- |
| 旧平地专家 v2 | 14 个手臂关节误差合成一个指数奖励；腰只受较弱姿态项约束 | 腰 pitch −0.532 rad、双手腕 pitch 约 −1.62 rad，贴下限 |
| v3 | 按关节计算不饱和 Huber 姿态误差，加入上肢软限位惩罚并限制探索标准差 | 手臂有所改善，腰仍贴下限 |
| v4 | 仅给腰和上肢增加残差动作分支，双腿仍由原策略控制 | 手臂姿态分数约 0.98，腰仍 −0.529 rad |
| v5 | 腰三轴残差幅度由 2 提到 5，电机/物理参数不变 | 允许更大动作，但策略仍给腰 pitch 约 −1.12 的错误动作 |
| v6 | 在稳定站立阶段，用 PD 站姿控制器给腰三轴提供辅助训练目标，损失权重 0.05 | 腰和手腕摆正，保持起身成功率 |

v6 的训练目标只在真实回合步数 ≥90、骨盆高度 >0.7 m 且骨盆直立时启用。教师根据当前关节位置和速度算出腰部目标动作；环境仍执行**策略采样动作**，没有把教师动作送入模拟器。第 199 次训练结束时教师损失约 0.37。`stand_guard.py` 的独立回放约束只用于旧模型的诊断，不是 v6 回放的必要组成部分。

重要实现关系：

- [task.py](src/g1recovery_host/task.py) 定义不饱和姿态奖励；站姿来自 [model.py](src/g1recovery_host/model.py) 中参考 NPZ 最后一帧，经左右对称和平脚处理。
- [ppo.py](src/g1recovery_host/ppo.py) 定义上肢残差网络及腰部教师损失；残差只影响动作索引 12–28，双腿索引 0–11 不变。
- [train_host.py](scripts/train_host.py) 生成教师标签、执行 PPO、记录 `teacher_loss`，并把 `pose_v6` 写入 checkpoint。
- [eval_host.py](scripts/eval_host.py)、[diagnose_host29_arms.py](scripts/diagnose_host29_arms.py) 和 [play_host.py](scripts/play_host.py) 根据 checkpoint 的版本标签加载相同的策略结构。

训练沿用旧 3000 次平地专家，经 v3 → v4 → v5 → v6 四个独立短试验，每个短试验为 256 环境、200 次更新、每环境每次 50 步。每次切换目标时保留策略和课程状态、重新初始化优化器；最终 `beta=0.25`，辅助起身力为 0 N。最终 v6 checkpoint 的 `parent_sha256` 指向 v5 第 199 次文件，不是直接指向旧第 2999 次文件。

## 验证

评估均使用确定性动作、不启用姿态回放约束、辅助力 0 N、`beta=0.25`、每回合 500 控制步：

| 测试 | 结果 |
| --- | ---: |
| seed 42，64 环境 × 2 回合 | 起身 128/128；最后一秒保持 >0.7 m：128/128 |
| seed 50，64 环境 × 2 回合 | 起身 128/128；最后一秒保持 >0.7 m：128/128 |
| seed 42/50，各 8 环境、后 150 步 | 两次均为 1200/1200 帧站立；手臂姿态分数约 0.978，腰姿态分数约 0.9993 |
| 本机 CPU、seed 123、2 环境 | 后 150 步 300/300 帧站立；手臂分数 0.9781，腰分数 0.9993 |

第 199 次策略在 seed 50 的站立后段，腰 pitch 平均绝对误差 0.0138 rad，左/右手腕 pitch 为 0.0036/0.0031 rad；腰、肩、肘和手腕均未贴限位。左右肩 roll 仍分别偏离站姿参考约 0.115/0.093 rad，应以 Viser 观察外观是否符合具体展示要求。两个脚踝 roll 仍分别贴在一侧限位，这是旧策略延续的足部问题，本修复没有处理；真实机器人部署前需另行检查脚部接触和关节安全。

完整证据在下载目录的 `eval_199_64x2_seed{42,50}.json`、`diag_199_8env_seed{42,50}.json`、`diag_local_2env_seed123.json`、`metrics.jsonl` 和 `train.log`。
本机 Viser 入口已成功加载 v6 模型和 29 维环境，并运行到人为设置的 18 秒超时；网页中的外观仍需人工观察。

## 本地回放和复核

从 `/home/ray1/code/G1Recovery_HoST` 执行：

```bash
cd /home/ray1/code/G1Recovery_HoST
MPLCONFIGDIR=/tmp/mpl-host29-v6-play PYTHONPATH="$PWD/src" \
  /home/ray1/code/mjlab/.venv/bin/python scripts/play_host.py \
  --task flat29 --device cpu --episode-seconds 10 \
  --checkpoint logs/downloaded/host_ground29_pose_v6_teacher_fromv5_256_seed42_200/model_00199.pt
```

终端会打印 Viser 网址；若从远程服务器回放，需要把远程 8080 端口转发到本机。该命令没有 `--stand-pose-guard`，因此看到的是策略本身的姿态。

```bash
MPLCONFIGDIR=/tmp/mpl-host29-v6-eval PYTHONPATH="$PWD/src" \
  /home/ray1/code/mjlab/.venv/bin/python scripts/eval_host.py \
  --task flat29 --device cpu --num-envs 8 --episodes 2 --steps 500 \
  --checkpoint logs/downloaded/host_ground29_pose_v6_teacher_fromv5_256_seed42_200/model_00199.pt
```

原始 v2 checkpoint 完全保留，可随时用它和本文件中的 v6 checkpoint 做无约束对照。A6 脱困任务尚未针对 v6 残差分支重新训练或验证；从 v6 迁移到 A6 时应保留残差网络，不能只复制原 tanh actor。
