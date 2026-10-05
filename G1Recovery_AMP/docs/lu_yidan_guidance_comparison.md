# LU Yidan：几何脱困引导对比工作分支

- 工作分支：`codex/lu-yidan-amp-guidance`。
- 基于 LU Yidan 原 `lu_dev`（`cbaac51375a15787fbeb8e483f614d5982c4cd58`），合入 `CZRay` 的独立恢复子项目（`b8325f8ee54af5ed8bcb4783e5adfb8351b66335`）作为可运行基础。
- 原 `lu_dev`、`CZRay` 和 `main` 不写入本工作的提交。后续实验脚本和配置只在此工作分支修改。

## 比较目的

比较各恢复学习器是否能从相似的几何脱困引导思路中获益。不要求 AMP 与 SMP 逐项复制完整奖励、系数、观测或控制结构；各方法可以保留自己的风格奖励、辅助奖励和控制设置，并记录实际实现差异。

评测场景、机器人初始姿态分配、板子随机参数及 SR3 成功判据保持一致。方法内部的有／无引导对比应尽量保持其余训练条件一致。若移除了整个引导组合而非严格三项奖励，要准确说明所关闭的内容。

## 已有记录

现有 `model_22899.pt` 属于有几何引导版本；原任务启用了 geometry progress、clearance 和 separation，并保留额外奖励。该 checkpoint 对应的实测 SR3 为 Guided 425/512（83.0%）、Free-R 506/512（98.8%）。尚无无几何引导 checkpoint；不能通过在推理时关掉 reward 代替训练消融。

当前 `guidance="minus"` 还会关闭 no-progress 和 prone-lateral 两项辅助引导。若采用该配置，需将其定义为整个引导组合的消融，不要声称只关闭三项几何项。

## 评测脚本

- `scripts/evaluate_success_protocols.py`：已有诊断和共享 SR3 评测。
- `scripts/evaluate_free_sr3.py`、`scripts/free_r_protocol.py`：读取冻结的 Free-R 参数，使用原生 AMP 模型评测。
- `scripts/sr3_protocol.py`：20 秒内开始连续站立、持续 3 秒，观察至 23 秒。

完整的历史评测数据、参数文件和重现说明位于论文仓库 `G1_Recovery_Below_Block/evidence/amp_free_r_20261005`。这些数据不因移动工作分支而改变。
