# LU Yidan：几何脱困引导对比工作分支

- 工作分支：`codex/lu-yidan-amp-guidance`。
- 基于 LU Yidan 原 `lu_dev`（`cbaac51375a15787fbeb8e483f614d5982c4cd58`），合入 `CZRay` 的独立恢复子项目（`b8325f8ee54af5ed8bcb4783e5adfb8351b66335`）作为可运行基础。
- 原 `lu_dev`、`CZRay` 和 `main` 不写入本工作的提交。后续实验脚本和配置只在此工作分支修改。

## 比较目的

比较各恢复学习器是否能从相似的几何脱困引导思路中获益。不要求 AMP 与 SMP 逐项复制完整奖励、系数、观测或控制结构；各方法可以保留自己的风格奖励、辅助奖励和控制设置，并记录实际实现差异。

评测场景、机器人初始姿态分配、板子随机参数及 SR3 成功判据保持一致。方法内部的有／无引导对比应尽量保持其余训练条件一致。若移除了整个引导组合而非严格三项奖励，要准确说明所关闭的内容。

## 已有记录

现有 `model_22899.pt` 属于有几何引导版本；原任务启用了 geometry progress、clearance 和 separation，并保留额外奖励。该 checkpoint 对应的实测 SR3 为 Guided 425/512（83.0%）、Free-R 506/512（98.8%）。此前尚未定位无几何引导 checkpoint；2026-10-05 已在同学工作空间找到历史配对训练及平地父模型，见下文。不能通过在推理时关掉 reward 代替训练消融。

当前 `guidance="minus"` 还会关闭 no-progress 和 prone-lateral 两项辅助引导。若采用该配置，需将其定义为整个引导组合的消融，不要声称只关闭三项几何项。

## 评测脚本

- `scripts/evaluate_success_protocols.py`：已有诊断和共享 SR3 评测。
- `scripts/evaluate_free_sr3.py`、`scripts/free_r_protocol.py`：读取冻结的 Free-R 参数，使用原生 AMP 模型评测。
- `scripts/sr3_protocol.py`：20 秒内开始连续站立、持续 3 秒，观察至 23 秒。

完整的历史评测数据、参数文件和重现说明位于论文仓库 `G1_Recovery_Below_Block/evidence/amp_free_r_20261005`。这些数据不因移动工作分支而改变。

## 2026-10-05：历史来源与正式配对训练

历史文件只读，来源 `/home/czr/code/G1Recovery_AMP`。完整父模型路径、哈希、原始配置与同学的迁移记录保存在 `provenance/flat_parent_19998/`。

- 平地父模型 `model_19998.pt`：从原平地 14999 续训 5000 次低位起身进展任务；未训练压板。它可直接用于平地或压板零样本泛化评测，不能替代“同样压板训练、无几何引导”的消融。
- 找到历史 `2026-10-01_14-02-39_a6_gminus_ampgate_from19998_512x2000_seed42` 和 `2026-10-01_14-02-46_a6_gplus_ampgate_from19998_512x2000_seed42`。两者具有同一平地父模型，但采用旧版 AMP 系数和引导配置；未按当前 SR3 重新评测，不填写新表格。
- 新配对由 `scripts/train_luyidan_guidance.py` 构造；两组均加载平地父模型的 actor、critic、优化器、AMP 判别器，适配计数从零开始。训练阶段不使用向上辅助力。

| 项目 | 两组共同设置 |
|---|---|
| 服务器 / GPU | 宁波联通；off GPU 4，on GPU 5 |
| 初始化 | flat_model_19998.pt，SHA256 4280efe9de1d6772e33f93847208896c4f7126e62630fc95b1cfeafe72e9f47c |
| 预算 | 4096 环境，24 步/更新，20000 更新，每 500 更新保存 |
| 种子 / 学习率 | 20261005 / 固定 2.25e-5 |
| 场景 | 平地 30%，上下滑动板 50%，自由板 20% |
| 压板 reset | 全部低位；四朝向均匀；自然低位 75%、程序化 25% |
| 平地 reset | 低/中/高 40/40/20% |
| episode / 控制 | 10 s；物理 2 ms × decimation 10 |
| 随机化 | actor 噪声、push、质量/惯量、Kp/Kd、0–10 ms 延迟等保留 |
| AMP 与辅助奖励 | 原生 AMP 与任务衰减设置保留；completion、no-progress、prone-lateral、Q/L 等两组一致 |

唯一干预：on 保留 geometry progress / clearance / separation 权重 .45 / .08 / .10；off 将这三项设为零。没有调用会同时关闭其他辅助奖励的旧 `guidance="minus"` 工厂。

4096 环境、2 更新的两组预检均完成，初始 qpos/qvel、scene/direction/source/stage 完全一致；损失有限、actor 权重确实更新。配置测试验证两组除三项权重外相同。正式实验只构成一个配对种子；增加训练种子时两组都增加，不能只选择有利种子。

启动：`bash scripts/run_luyidan_guidance_job.sh off 4 logs/flat19998_guidance_20k_20261005`，on 使用 GPU 5。该脚本训练完成后自动评测最终 `model_19999.pt`。

## 统一评测与表格填入规则

`evaluation/sr3_guidance_v1/` 固定初始姿态、标签与 Free-R 参数，manifest 记录哈希。2048 环境包括 1024 平地、512 Guided、512 Free-R。SR3：20 s 内开始连续站立并保持 3 s，观察至 23 s；头高 ≥1.15 m、直立度 ≥.90、双足载荷各 >20 N。不增加安静站立门槛，不使用历史无效压板筛选。

新配对两组都评测后填入配对结果。已有 22899 的结果保留为此前选定策略的记录；不要把它和新 off 隐含为同一预算的训练对照。SR3 数值必须来自保存的逐次试验结果。父模型评测如果采用适配后的 90% 力矩上限，需说明该控制条件；若要评价它原生能力，则使用父模型原生控制上限。

正式运行已启动（2026-10-05，源代码 `fa8822a`）：

- 无三项几何引导：GPU 4，job PID 1020860，W&B https://wandb.ai/tabletennis/amp-guidance-luyidan/runs/yuqgl0st 。
- 有三项几何引导：GPU 5，job PID 1020861，W&B https://wandb.ai/tabletennis/amp-guidance-luyidan/runs/65qbrvrs 。
- 根目录 `/root/workplace/amp-luyidan-guidance/G1Recovery_AMP/logs/flat19998_guidance_20k_20261005`，两个目录分别为 `off` 和 `on`。
- 已验证正式训练完成首个更新，损失有限、actor 权重变化、初始与第零次更新 checkpoint 均保存；W&B online 初始化成功。这里不代表训练完成或成功率已测出。
