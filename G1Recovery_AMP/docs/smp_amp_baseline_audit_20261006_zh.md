# SMP / AMP 无三项几何引导基线核对

## 已执行

按用户要求停止 AMP 两组正式训练及其父级训练后自动评测脚本。通过已登记 PID、命令行和进程组校验后发送 SIGTERM；核验四个训练/启动进程均已退出，GPU 4/5 释放。未停止其他用户任务，未删除 checkpoint。停止记录见 `../results/amp_stopped_20261006.json`。

最新进度记录约 off 11901、on 12351 更新；最后完整 checkpoint 分别 11500、12000。10k 配对结果完整保留。

## 已确认的配置差异

| 项目 | SMP C20 | 当前 AMP 配对 |
|---|---|---|
| 平地 / Guided / Free 训练比例 | 50 / 25 / 25% | 30 / 50 / 20% |
| 无三项几何引导仍有额外 no-progress / prone-lateral 引导 | 未采用 AMP 这两项辅助项 | 保留 −0.15 / +0.20 |
| 风格和任务结合 | task × SMP score | AMP 原生任务/风格混合，受阻风格缩放 0.05 |
| 父模型 | SMP R2 | AMP flat model_19998 |
| 加载训练状态 | actor 继承，critic/optimizer 新建 | actor/critic/optimizer/AMP 判别器加载 |
| 当前用于表格的无引导 checkpoint | seed20261022，19999 | seed20261005，10000 |

这些差异真实存在，但不能仅由配置推定各项造成了多少成功率变化。尤其不能将整个差距归因于轮次、SMP 算法或某一个 reward。

SMP 完整策略现有 Guided/Free-R 是 88.5%/98.6%，AMP 有引导10k是88.3%/98.8%；完整策略的成功率接近。需要补强的是 SMP 无三项几何引导基线的学习质量及配对实验的可复现性。

## 历史核对

论文仓库 `evidence/sr3_policy_inventory_20261005` 中 C20_no_geometry 三个 seed 的19999在标称 Guided 上是0/512、95/512、0/512（本组历史评测）。当前表格复测所选 seed 20261022 为97/512。两次结果属于不同运行，保留各自来源，不混合。该历史 inventory 的 Free 为标称条件，不可填写到 Free-R。

因此没有证据表明换一个已评测20k seed就能接近AMP；也没有证据证明更早 checkpoint 必然不好。先检查同预算早期 checkpoint 有无退化，不能只将差异归因于20k过长。

## 建议的有限开发流程（本次未启动新的 SMP 训练）

1. 先对已有5k/10k SMP有/无几何引导做配对验证，固定待检查的种子及预算。诊断受阻状态的 SMP 分数、未乘风格前任务分数、停滞时长，区分“尝试但失败”与“停在局部最优”。
2. 构建加强的 SMP 训练设置：平地/Guided/Free 30/50/20%，保留质量、惯量、电机、延迟、push和原reset姿态来源；从同一个R2父模型开始，先以10k为预算，3k/5k/10k记录验证。
3. AMP 的 no-progress / prone-lateral 是可迁移的候选辅助项；如采用，两组均加，三项核心几何奖励仍一组开、一组关。把它视为基线开发，不称为只改场景比例的单因素消融。系数需核对实际量纲和奖励幅值，不直接照抄常数。
4. 如果诊断确认低 SMP 分数抑制了必要的受阻探索，再单独测试受阻阶段的风格门控下限；不要与上述改动一起全部叠加，也不直接取消风格约束或放开硬件力矩上限。
5. 验证集选配置后，两组使用同一选点规则；最终统一测试并保留全部结果。收益不预先保证，不能为了缩小差距选择性隐藏种子。

论文中可把 Table IV 定位为 SMP 内部机制消融，Table V 定位为几何引导跨学习器适用性。若经过可靠对照后 AMP 仍更好，应如实说明适用范围；方法的几何引导贡献不要求 SMP 框架本身在所有场景优于AMP。

## 代码和来源

- SMP `smp-a6-paper/outputs/a6_confirmatory_review_20261003_20k/seed20261022/C20_no_geometry/launch.json`：场景计数、初始化方式和种子。
- SMP `src/smp/rl/rewards.py::task_smp_product`：风格乘法门控。
- AMP `src/g1recovery_amp/tasks/recovery/__init__.py::_low_torque_cap_only_env_cfg`：场景与受阻辅助机制。
- AMP `src/g1recovery_amp/tasks/recovery/a6_env_cfg.py`：no-progress、prone-lateral及风格缩放常数。
- AMP `src/g1recovery_amp/tasks/recovery/rl/discriminator.py::blend_reward`：任务/风格混合。
