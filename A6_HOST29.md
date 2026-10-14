# HoST G1 29DoF：平地适应与历史 A6 脱困

本地实现采用 HoST 四 critic PPO、六帧 proprioception 历史、相对关节目标、30 步无驱动期和辅助力课程；29DoF 机器人来自 mjlab 的 Unitree G1 MJCF。原有 23DoF 平地代码和 checkpoint 保留，用于权重迁移及历史比较。A6 不是 HoST 原论文自带任务，而是把 [SMP 历史 A6 提交](https://github.com/lu-yidan/smp/tree/1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2)的场景、重置和共同 reward 接到 HoST 训练器。

当前 A6 场景把导向板和自由板统一固定为 6 kg，重置时不再随机化质量或惯量；模型物理版本仍为 `split_torso_force_v2_a6_29_v2`。当前 G+ 试验版本为 `historical_a6_G+_29dof_pose_v6_v8_edge_curriculum_v1`（冻结平地教师时追加 `_frozen_flat_teacher_v1`）；G− 仍为 v3，v7 保留作历史对照。历史 A6 v1 checkpoint 是在从 4–6 kg 渐扩至 4–12 kg 的质量课程下训练的；固定 6 kg 的旧奖励 v2 checkpoint 尚无板抬升重置和受困侧移奖励。v8 保持平地 29DoF 的模型物理版本、29 动作、564 维观测不变，但重置分布和板下侧移奖励不同，必须使用独立输出目录。

2026-10-11 更新了**当前代码的失效 reset 判据**：板与机器人接触力取每个 20 ms 控制步内十个 2 ms 物理子步的峰值，任一峰值 **>5000 N** 即无效，不再使用旧的 >1500 N 连续三步或 >2500 N 单步判据。导向绿板在电机接管后、机器人碰撞体仍投影在板下时，单步内抬升峰值 **>0.05 m** 仍无效；自由板没有此抬升失效项。两种板的接触距离 `<−0.02 m` 均**不再触发无效 reset**，但诊断仍记录穿透深度。超过 25 控制步仍从未接触板、通用异常速度和回合超时保持原规则。新 checkpoint 写入 `a6_reset_variant=plate_force_5000_guided_lift_005_no_depth_v2`；直接 `--resume` 旧 A6 checkpoint 会报判据版本不一致，若明确要跨规则迁移须使用 `--allow-model-transfer` 并重新建立优化器。旧 checkpoint 可以回放，但当前脚本会提示其按**新**失效规则运行。下文已完成试训和评估均来自旧 reset 判据，不能当作当前配置的训练效果；其中的穿透重置描述仅为历史记录。

## 当前新增奖励：离板前限制双膝蜷缩

29DoF G1 的平地站姿双膝参考角均为 `0.30 rad`，物理关节范围约 `−0.087～2.880 rad`。新增双膝惩罚在偏离站姿超过 `0.30 rad` 时开始生效，即允许约 `±17.2°`，仍小于要求的 `20°`；完全伸直至 `0 rad` 不扣分，向前弯曲超过约 `0.60 rad` 才开始扣分。每侧超限量 `e=max(0, |q−0.30|−0.30)` 的成本现为 `e/(e+0.25)+0.20e`，两侧相加。第一项对刚超限敏感，线性项让深度蜷缩仍有明确的伸膝收益；在物理关节范围内成本有限。`target` 奖励率以系数 `−1.0` 扣除此成本；mjlab 再按控制周期 `0.02 s` 缩放一次。膝盖伸直到容许带内即不扣分；长训中仍须监测是否出现主动触发无效 reset 的捷径。

新项对导向绿板和自由板都启用，平地为零。从回合开始直到**首次**满足连续 25 控制步离板判定，权重为 1；首次判定后锁存关闭，即使之后再次接触板也不重新开启。旧 G+ 绿板覆盖时的 `0.45 rad` 膝弯曲项仍保留，两项相加；G− 也获得新的共同惩罚。训练日志新增 `a6_guided_pre_escape_knee_curl`、`a6_free_pre_escape_knee_curl` 及对应每控制步扣分，诊断逐步记录成本。新 checkpoint 的 reward 版本统一追加 `_preescape_knee_v2`，旧 `model_01999.pt` 不会因回放新代码而改变动作；如从它续训，需按 reward 跨版本迁移并重建优化器。此项是软约束，是否最终把膝角稳定限制在 20° 内仍须用独立回合评估确认。

量级核对：旧 v8 策略的同一批受困轨迹（电机接管后、离板前且未失效）按新公式离线重算，自由板仰卧每步平均扣 `0.0295`，导向绿板俯卧扣 `0.0407`，导向绿板仰卧扣 `0.0501`。同批轨迹中，去掉膝盖项后的 `target` 组每步均值分别约 `+0.0019`、`−0.0132`、`−0.0309`，所以膝盖项并非被其他目标项淹没。在 `2.5 rad` 的深弯膝位置，单膝伸直 `0.1 rad` 可降低成本约 `0.0257`，相当于接下来每控制步少扣约 `0.00051`；旧曲线只有约 `0.00011`，新曲线的边际信号约强 4.7 倍。这是奖励计算和信号量级的证据，不等于训练后动作一定改善。自由板单次失效重置当前扣 `2` 分、导向绿板 G+ 扣 `15` 分，若长时间困住，密集膝罚仍可能诱发提前重置；应同时观察失效率和净空率，不能只看膝角下降。

验证：CPU 32 环境 40 步烟测、从平地 `00999` 的 CPU/GPU 3 各一轮 PPO、CPU 四视角回放入口均通过；服务器预检两种板的成本指标约为 1.35/1.37。把旧 v8 策略在新奖励下回放 120 步时，自由板仰卧于第 95 步首次满足离板判定：判定前 94 步中 86 步成本为正；第 107 步左膝仍有 `0.5867 rad`、超过新阈值，但成本已锁存为零。绿板俯卧和仰卧未在该单回合内离板，平均成本约 1.29/1.63；平地为零。该对照验证的是奖励和开关逻辑，不代表旧策略动作已改善。

## 当前探索性长训：G+ v8，1000 次 PPO 更新

用户选择在 v8 的绿板短试训未成功后，按当前奖励配置进行探索性长训。服务器物理 GPU 3 从 SHA256 `0fa09f84bfc1eed56526804c0e0d47c76c69a09c1a64fb1922e8f233b28aaa0a` 的 29DoF 平地 `model_00999.pt` 独立分叉，同一文件作冻结教师；4096 环境、每环境 24 控制步、seed 42、5 epochs、4 minibatches。执行编号 1000–1999 共 1000 次更新，因此 `--iterations 2000`；板边偏移初始 0.25 m，保持 100 次更新、随后 300 次线性收回。新输出目录为 `logs/host_a6_29_v8_nodepth5000_from0999_frozen_Gplus_4096_seed42_1000_20261011/`，启动命令封装在 `scripts/launch_a6_v8_nodepth_1000.sh`，每 100 次更新保存一次，目标末轮为 `model_01999.pt`。此实验没有沿用失败短试训的 `01199` 学生。

服务器 GPU 3 上已通过 32 环境单轮 PPO 预检，checkpoint 元数据确认新 reset 版本、G+ v8 奖励、父代 999 和同一冻结教师；正式 4096 环境训练的前 10 次更新指标均为有限值。预检和启动状态只证明训练链路可运行，不能证明绿板脱困。应在 `01099`、`01399` 和 `01999` 等 checkpoint 上分别评估绿板俯卧与仰卧的**完整 episode**净空率、抬板失效率、腿伸直、上肢支撑和站姿保持；旧 v8 200 次更新的绿板 0/8 结果仍是明显风险。

```bash
cd /home/czr/code/G1Recovery_HoST
bash scripts/launch_a6_v8_nodepth_1000.sh  # 仅首次启动；输出目录已存在时会拒绝重复启动
OUT=logs/host_a6_29_v8_nodepth5000_from0999_frozen_Gplus_4096_seed42_1000_20261011
tail -f "$OUT/train.log"
tail -n 1 "$OUT/metrics.jsonl"
```

## 当前 G+ v8：先在近板边学横移，再逐渐回到板中央

v7 的 256 环境、200 次更新及八种子评估表明，绿板俯卧和仰卧仍是 0/8 脱困，主要在尝试起身时抬板超过 5 cm 被正确重置；末轮平地稳定站姿也从 8/8 退到 6/8。v8 不沿用失败的 `01199` 学生，而从同一 29DoF 平地 `00999` 创建新学生和固定教师。只对绿板，重置时把 6 kg 板沿初始身体左侧轴平移，令机器人更靠近反方向板边；板的高度按平移后的机器人覆盖范围重新计算。板移到身体左侧时，靠近的出口在身体右侧，因此受困侧移、手支撑推移和停滞计时只认可朝右侧的新进展。v8 试训时自由板、平地、接触力/穿透/5 cm 抬板重置、25 控制步净空判据均未改动；当前代码的板力及穿透判据另按上文更新。不启用 `--a6-edge-curriculum` 时，板边课程和方向奖励仍为旧 v7 行为。

课程默认在迭代 1000–1099 保持 0.25 m 板偏移，随后 300 次更新线性收回，到迭代 1399 板居中。调度锚点与参数写入 checkpoint，续训必须使用相同参数，不能在恢复时重新开始易阶段。训练日志新增 `a6_guided_edge_target_m` 和实际重置偏移均值；由于旧回合会跨迭代存活，目标偏移和实际均值短时不必完全相同。固定种子评估需分别做 `--guided-offset-m 0.25` 的易阶段和默认 0 m 的原始居中场景；200 次更新只覆盖课程前半段，不能把易阶段成功率当作原始场景成功率。

本地已通过 32 环境旧配置与偏移配置烟测、0.25 m 的 96 步仿真、从 `00999` 出发的 32 环境 × 96 步完整单次 PPO，以及从 v8 checkpoint 的续训和四视角回放入口验证。单轮 checkpoint 仅用于检查训练链路，未证明脱困。初始策略的 seed 42 首回合诊断：板偏移 0.25 m 时俯卧未脱困但 300 步内未抬板失效，仰卧仍在第 48 步无效；这是选择近板边课程的依据，也是后续必须单独跟踪仰卧的原因。

## v8 独立短试训：已完成，板边课程仍未解决绿板脱困

2026-10-11 在服务器物理 GPU 3 从平地 `model_00999.pt` 独立训练 256 环境 × 96 步、200 次 PPO 更新（累计迭代 1000–1199）。GPU 预检验证 29 动作、564 维观测、6 kg 板、0.25 m 实际重置偏移；单次 PPO checkpoint 记录 `parent_iteration=999`、`optimizer_reset_on_transfer=True`、课程锚点 1000 和同一冻结平地教师。训练前服务器源码备份为 `logs/pre_v8_source_20261011_1.tar.gz`。实验目录为 `logs/host_a6_29_v8_edge25_from0999_frozen_Gplus_256_seed42_probe200/`，末轮 `model_01199.pt` SHA256 为 `b03b5304068ea8ee7cc864238fcd82cd686cec643b1429529cc5046f56a81d5f`，本地副本与服务器一致。200 行 `metrics.jsonl` 完整；末轮课程目标偏移 0.1667 m，实际各回合均值 0.1681 m，因为未结束的回合仍使用此前偏移。

对 `01099` 和 `01199` 分别在 0.25 m 偏移板与原始居中板做固定种子 42–49 的四场景首回合评估；每回合最多 300 控制步、无辅助力，脱困要求连续 25 步净空。结果如下，括号内为无效结束次数，平地稳定阈值为末步骨盆高 ≥0.70 m 且躯干直立度 ≥0.95：

| checkpoint 与板位置 | 绿板俯卧脱困（无效） | 绿板仰卧脱困（无效） | 朝近侧出口最大位移均值：俯卧 / 仰卧 | 平地稳定 |
| --- | ---: | ---: | ---: | ---: |
| `01099`，偏移 0.25 m | 0/8（8/8） | 0/8（8/8） | 0.0696 / 0.0223 m | 6/8 |
| `01099`，居中 | 0/8（5/8） | 0/8（8/8） | 0.0439 / 0.0102 m | 7/8 |
| `01199`，偏移 0.25 m | 0/8（6/8） | 0/8（8/8） | 0.0579 / 0.0188 m | 8/8 |
| `01199`，居中 | 0/8（6/8） | 0/8（7/8） | 0.0339 / 0.0411 m | 8/8 |

末轮居中俯卧 6 次无效中 5 次出现 >5 cm 抬板，仰卧 7 次无效中 6 次出现 >5 cm 抬板；偏移 0.25 m 的绿板也没有任何首回合脱困。训练中绿板俯卧已脱困步占比在四个 50 轮窗口由约 0.0082/0.0082 降为 0.0019/0，仰卧约 0.0464/0.0506/0.0360/0.0341；蜷腿指标约 0.642→0.659，手撑侧移指标约 0.065→0.065。逐步占比不能替代独立回合成功率，但与评估一致地说明该试训没有稳定形成朝出口移动的动作。**不建议从 `01199` 直接续训，也不建议把此配置放大到 4096 环境长训。**下一轮应首先解决 actor 只有本体历史观测、没有板边位置/接触阶段显式信息的问题，并引入可行的横移示范或更容易产生真实完成轨迹的课程；若增加 actor 输入维度，必须显式迁移平地父代的第一层权重，不能直接加载旧 564 维策略。完整逐种子时序与汇总位于本地 `logs/downloaded/host_a6_29_v8_edge25_from0999_frozen_Gplus_256_seed42_probe200/`。

服务器训练命令（`--iterations 1200` 是累计轮数，从父代 999 开始只更新 200 次）：

```bash
cd /home/czr/code/G1Recovery_HoST
PARENT=logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt
OUT=logs/host_a6_29_v8_edge25_from0999_frozen_Gplus_256_seed42_probe200
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/mpl-host-a6-v8-probe \
PYTHONPATH="$PWD/src" /home/czr/code/G1Recovery_mjlab/.venv/bin/python \
  scripts/train_host.py --task a6 --arm G+ --a6-pose-v6 \
  --continue-from29 "$PARENT" --frozen-flat-teacher "$PARENT" \
  --flat-teacher-weight 0.5 --a6-edge-curriculum \
  --a6-edge-start-offset-m 0.25 --a6-edge-hold-updates 100 \
  --a6-edge-anneal-updates 300 --reset-optimizer --adaptive-kl \
  --learning-rate 0.00005 --device cuda:0 --num-envs 256 \
  --steps-per-env 96 --iterations 1200 --epochs 5 --minibatches 4 \
  --seed 42 --log-every 10 --save-every 50 --output "$OUT"
```

单种子评估使用 `scripts/diagnose_a6_escape_sequence.py --checkpoint "$OUT/model_01199.pt" --teacher-checkpoint "$PARENT" --device cuda:0 --seed 42 --steps 300 --guided-offset-m 0.25 --output "$OUT/eval_01199/edge025/seed42.json"`；省略 `--guided-offset-m` 即原始居中板。四组 32 个 JSON 可用 `python3 scripts/summarize_a6_probe.py "$OUT" --output "$OUT/evaluation_summary.json"` 汇总。

## 历史 G+ v7：让旧起身先验真正让位于板下脱困

v6 `model_01999.pt` 的 seed 42、无辅助力、固定四场景单回合诊断：绿板俯卧 300 步无脱困，双膝从第 31 步接近伸直逐渐弯到第 300 步约 2.75/2.27 rad；手接地占比均值约 0.55，但骨盆总平面位移最大 0.157 m、**真正沿板短边的侧移最大仅 0.057 m**，无进展指标均值 0.77。俯卧并非整个上肢不出力：左肩外展力矩平均已达其 25 Nm 上限的约 81%，左腕俯仰约达 5 Nm 上限的 54%，但力没有转化为有效横移。绿板仰卧在第 62 步因穿透失效，双膝在结束前接近关节上限，沿板短边最大只移 0.031 m；其肩肘关节平均上限占用多在约 20%，更像缺少起始抬臂/撑地动作。四场景单种子不是总体成功率，但足以确认“接触存在，没转化为横向推移”以及腿持续回蜷；单看骨盆平面位移会高估脱困进展。

根因之一在 `ppo.py`：HoST 四个奖励组各自计算并标准化优势，之后才以 `(2.5, 0.1, 1, 1)` 合成 actor 更新。`style` 单独成组，若只在原始 reward 中乘 0.05，这个倍率大部分会被本组标准差除掉，因此旧平地风格仍能推动板下动作。v7 在标准化**之后**，对尚未确认脱困的板场景样本将 `task`/`style` 的 actor 优势分别再乘 0.05；平地和确认脱困后恢复 1.0，`target`/`regu` 保持 1.0。原始 reward 的 0.05 仍用于 critic 学习正确的阶段价值；这并非冻结 actor，所有 29 个关节仍由同一个 PPO 更新。冻结 `00999` 教师的模仿损失依然只作用于平地和已脱困样本。

奖励侧只改变 G+ 的导向板受困阶段：新增双膝超过 1.2 rad、双髋大幅折叠的有界惩罚 `−1.20`；手部传感器提供真实地面接触力，手有载荷且骨盆沿初始侧向**向外运动**时给 `+1.20`，单纯支撑只给 `+0.15`，原有刷新最大侧移时的手撑奖励仍为 `+1.50`。侧向位移的持续奖励由 `+1.50` 降为 `+0.50`，防止移动几厘米后躺着不动刷分；导向板无进展成本由 `−0.40` 提到 `−1.50`。移动直腿与躯干侧移奖励不再因停滞计时达到上限而直接归零。几何进展在双腿极度蜷曲时保留 20% 基础信号，其余 80% 随伸腿恢复，打破初态无梯度死锁。双臂朝头姿态分也由旧的 `cos < −0.6` 完全无分，改成从 `cos = −1` 到目标 `+0.6` 连续上升，仰卧刚开始反向伸臂时即可获得渐进反馈；左右两臂仍取较差侧。手部反作用力必须带来身体侧移才拿到主要推移分，仍保留抬板/超力/穿透无效判据。

G+ v7 保持 actor 564 维、29 动作、六帧观测、四 critic 网络、6 kg 板和 25 步净空判据，因此 29DoF 平地 `model_00999.pt` 可作为父代重新起训，v6 `model_01999.pt` 也可显式跨版本迁移；两者应是独立实验，不能把 v6 的 1000 轮直接当成 v7 的成效。新日志记录 `a6_obstructed_advantage_fraction`、`a6_guided_hand_drive`、`a6_guided_hand_plant`、`a6_guided_overcurl`。验收时按每种姿态分别统计**独立回合**的 25 步净空完成率、无效结束率、膝角峰值、手接地载荷与骨盆侧移；单步 `escaped_fraction` 不能替代回合成功率。小环境预检只证明信号可计算；200 轮训练后的固定种子效果见下文。

本地验证：最终版 32 环境 40 步 CPU 烟测、静态检查和编译通过；从 SHA256 `0fa09f84...` 的平地 00999 父代加同一冻结教师做 32 环境 × 96 步、`5 epochs × 4 minibatches` 的单轮 PPO，生成 `/tmp/host_a6_v7_final_preflight/model_01000.pt`。平均 KL 约 0.0157，末学习率约 `2.22e−5`；新增手推移、手支撑、过度蜷腿指标均非零，受困优势样本占比 0.50，checkpoint 记录 `optimizer_reset_on_transfer=True`。该 checkpoint 已通过 CPU 四场景回放入口验证。把**原 v6 策略动作保持不变**在最终 v7 奖励下重新打分，俯卧第 101–300 步的平均 target 控制步奖励从约 `−0.0133` 降为 `−0.0498`；仰卧的双臂朝头姿态分均值由 0.039 升为 0.142，证明初始反向姿态得到了渐进分。绿板俯卧仍未脱困，说明评分快速识别了卡住与蜷腿，但并不表示新策略已有行为改善。

## v7 独立短试训：已完成，绿板仍未脱困

2026-10-11 在服务器物理 GPU 3 用下列命令从平地 `model_00999.pt` 建立 v7 学生，同时固定该文件作教师，256 环境、每环境 96 步，完成迭代 1000–1199 共 200 次。父代/教师 SHA256 为 `0fa09f84bfc1eed56526804c0e0d47c76c69a09c1a64fb1922e8f233b28aaa0a`，末轮 `model_01199.pt` SHA256 为 `bb18b5f9137b3f626a831b517dc42cd98b7fd8b5b9a7a2b2a03dd8eecc15104e`。checkpoint 元数据确认 `parent_iteration=999`、`optimizer_reset_on_transfer=True`、29 DoF 和冻结教师哈希；200 行 `metrics.jsonl` 完整。服务器原 v6 脚本备份在 `logs/pre_v7_source_20261011_1.tar.gz`。本地保留末轮 checkpoint 和诊断数据于 `logs/downloaded/host_a6_29_v7_from0999_frozen_Gplus_256_seed42_probe200/`。

固定种子 42–49、四场景、无辅助力、每场景仅统计第一个 300 步回合，与旧 v6 `model_01999.pt` 对照如下；“脱困”表示满足 25 个连续净空控制步，“无效”主要由导向板被抬高超过 5 cm 触发。侧移是每回合最大侧移的八种子均值，不是累计里程。

| 策略 | 俯卧绿板脱困 / 无效 | 仰卧绿板脱困 / 无效 | 俯卧 / 仰卧平均最大侧移 | 平地末步稳定直立 |
| --- | ---: | ---: | ---: | ---: |
| 旧 v6 `01999` | 0/8 · 5/8 | 0/8 · 6/8 | 0.0816 / 0.0846 m | 8/8 |
| v7 `01149` | 0/8 · 5/8 | 0/8 · 7/8 | 0.0607 / 0.0847 m | 8/8 |
| v7 `01199` | 0/8 · 5/8 | 0/8 · 7/8 | 0.0595 / 0.0487 m | 6/8 |

平地稳定直立的操作性阈值为末步骨盆高度 ≥0.70 m 且躯干直立度 ≥0.95；末轮 seed 42 和 44 分别只有 0.4316/0.8974、0.14581/0.563。冻结教师提供软约束，不会阻止学生 PPO 把平地动作训坏。绿板逐步脱困占比在四个 50 轮窗口只约 0.0077、0.0081、0.0089、0.0069，过度蜷腿指标也未持续下降。因此**不建议直接放大到 4096 环境长训，也不建议从 `01199` 接着训**。下一版应先做受困初态的板边距离课程，让横移成功轨迹在训练早期可见；再检查是否需要把板边相对方向加入 actor 观测。改观测维度会影响平地父代权重迁移，必须单独处理；旧 v7 的 200 轮不能证明新设计有效。逐种子的原始评估和汇总见本地 `logs/downloaded/host_a6_29_v7_from0999_frozen_Gplus_256_seed42_probe200/evaluation_summary.json`。下列命令保留作本次实验复现记录：

```bash
cd /home/czr/code/G1Recovery_HoST
PARENT=logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt
OUT=logs/host_a6_29_v7_from0999_frozen_Gplus_256_seed42_probe200
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/mpl-host-a6-v7-probe \
PYTHONPATH="$PWD/src" /home/czr/code/G1Recovery_mjlab/.venv/bin/python \
  scripts/train_host.py --task a6 --arm G+ --a6-pose-v6 \
  --continue-from29 "$PARENT" --frozen-flat-teacher "$PARENT" \
  --flat-teacher-weight 0.5 --reset-optimizer --adaptive-kl \
  --learning-rate 0.00005 --device cuda:0 --num-envs 256 \
  --steps-per-env 96 --iterations 1200 --epochs 5 --minibatches 4 \
  --seed 42 --log-every 10 --save-every 50 --output "$OUT"
```

本次在绿板下未形成可靠侧移，后续应先改 reset 难度课程，再用同一八种子首回合协议验证；单纯继续提高奖励系数不保证改变动作。

## 历史 G+ v6：把动作引导送到蜷腿与不动的状态

本次使用的 v5 `model_01999.pt` 从 29DoF 平地 `model_00999.pt` 完整继承 PPO，并保留同一 `00999` 冻结教师，训练了 A6 1000 轮；checkpoint 里的 `parent_iteration=999`、`transfer_source_iteration=None` 验证了这条谱系。在 seed 42 固定四场景、无辅助力、200 步诊断中，绿板俯卧全程未脱困，仰卧第 61 步因抬板超过 5 cm 失效；俯卧原有指数直腿分均值 0.1004、移动直腿分 0.0125，仰卧分别为 0.0288、0.0041。v5 把 G 和 dense 覆盖进展乘以指数直腿分，腿越蜷时越难拿到学会脱困所需的几何引导；原侧移是刷新历史记录才给分，手臂没有独立的板下奖励。

v6 仅改 G+ 的 target 奖励和诊断指标。G−、29 轴驱动、四 critic PPO、6 kg 板、25 步净空、5 cm 抬板与接触力失效判据均不变。奖励仍按控制步 `dt=0.02 s` 乘一次。

- 在绿板受困且电机接管后，根据肩到肘的向量在躯干坐标系的朝头方向余弦给双臂 `+0.60` 权重；两臂取较差一侧，避免一只手动了就掩盖另一只。目标余弦为 0.60，方向由实际连杆位置计算，而非仅看肩关节角。此姿态奖励不要求手先触地，脱困后自动关闭；实现参考 AMP 的 `trapped_shoulders_headward`。另外，只有手接地且躯干沿初始左右轴刷新历史最佳距离时，手臂推移项才按每新增 1 cm 得最高 `+1.50`；因此静止摆臂不能拿到推移分。
- 躯干沿重置时左右轴的**当前位置**离初始点 2–35 cm 时提供渐进分 `+1.50`，只有持续推进、停滞惩罚尚未升满时保持；俯卧/仰卧刷新侧移纪录的单步权重由 `+0.30` 提高至 `+1.00`。这两项不以直腿或手触地为前提，后退会丢失持续分，原地不动会触发无进展惩罚。无进展的导向板权重由 `−0.25` 调至 `−0.40`。
- 另加线性膝髋伸展分：膝超过 0.45 rad、髋 pitch 偏离 −0.10 rad 超过 0.75 rad 才逐渐扣分，最坏一侧膝/髋分别占 70%/30%；即使双膝已蜷到约 2.8 rad，分值及改进梯度也不会像旧指数项一样几乎消失。受困时持续姿态分权重 `+0.80`，有侧移且未停滞时的移动直腿项改用这条线性分、权重 `+0.80`；单步净伸展权重 `+0.50`，重新蜷腿仍得负分。旧指数分保留在覆盖/腿部离板门控中，以防纯蜷腿骗取几何进展。
- 因为增加了持续的姿态与动作引导，绿板无效结束惩罚由每秒率 `−500` 增至 `−750`，每次触发的控制步实际为 `−15`；不能用主动顶板重置来轻易逃避板下的持续成本。`metrics.jsonl` 另记躯干持续侧移、双臂朝头、线性腿部姿态、手撑侧移四项，旧版指标不能与新版按相同奖励解释。

这是奖励设计修正，**并不表示 v6 已学会脱困**。需要在新的输出目录里从 v5 `model_01999.pt` 迁移学生策略，继续保留 `model_00999.pt` 作为冻结的平地教师；`--iterations` 是总编号上限，例如设 `3000` 才是从第 2000 轮起再训 1000 轮。旧 v5 checkpoint 可用新代码诊断奖励信号，但其动作不会因代码修改而改变。正式训练前应在 GPU 主机用 32 环境单轮 PPO 预检；训练后重点比较绿板俯卧/仰卧的独立 episode 净空率、抬板失效次数、膝角、躯干侧移和双臂方向。

本地验证：32 环境 × 80 控制步 CPU 仿真通过，静态检查与编译通过；从 v5 `model_01999.pt` + 冻结平地 `model_00999.pt` 迁移，32 环境 × 64 步、1 次 PPO 更新生成 `/tmp/host_a6_v6_final_preflight_seed42/model_02000.pt`，记录了直接来源 v5、优化器重建和教师 SHA256，首轮平均 KL 约 0.00251。该预检 checkpoint 也完成固定四场景 50 步无辅助力回放；这不足以评估策略效果。用**未训练的 v5 策略**在 v6 代码下重放，绿板俯卧手接地均值 0.85、旧 G 路径进展仅 0.0002，而新增线性腿姿 0.399、手撑侧移 0.0299；绿板仰卧线性腿姿 0.1174、手撑侧移 0.0204，但仍在第 61 步抬板失效。新信号能到达旧策略的困难状态，实际改善仍需训练与独立评估。

```bash
--resume <v5 model_01999.pt> --allow-model-transfer --keep-curriculum-on-transfer \
  --a6-pose-v6 --frozen-flat-teacher <flat model_00999.pt> \
  --flat-teacher-weight 0.5 --reset-optimizer --learning-rate 0.00005
```

## 历史 G+ v5：持续伸腿，防止蜷腿骗取板边奖励

冻结 `00999` 教师的 v4 G+ `model_01999.pt` 在 seed 42 固定四场景、无辅助力诊断中，绿板俯卧第 75 步、仰卧第 62 步均因抬板超过 5 cm 无效结束；俯卧膝角峰值 2.82 rad，仰卧双膝接近 2.9 rad，均未完成水平脱困。仰卧时旧“腿部离板比例”曾达到 0.7，但腿仍明显蜷缩。该四场景单种子结果说明奖励有可利用的捷径，不能代表完整训练分布的成功率。

- v4 的“刷新历史最好膝角才给一次分”改为每控制步结算的双膝/双髋姿态分。膝角超过 0.45 rad、髋 pitch 偏离站姿 −0.10 rad 超过 0.75 rad 时，持续直腿分下降，弯曲成本上升；最弯的一侧也计入成本，防止只伸一条腿。当前步较上一步伸展得分，重新蜷腿则得负分，避免反复伸屈刷奖励。
- 绿板下原先可由蜷腿触发的覆盖改善奖励 G、dense 乘以腿部直度；足部离板和腿部覆盖改善还需躯干离初始位置超过 2 cm，15 cm 时门控全开。俯卧/仰卧躯干侧移奖励保留，避免直腿但不移动。
- G+ 绿板受困期的膝弯曲成本权重为 −0.90、髋弯曲 −0.35、移动中的持续直腿最高 +1.00、单步净伸展 +0.35；持续直腿分随躯干侧移从 15% 增到 100%，停滞惩罚满额时降到 0，避免学会“直腿等在原地”。腿部覆盖改善/离板项保持 +0.25/+0.12，但现在受直度和躯干位移门控。无效结束惩罚由每秒率 −100 提到 −500，按 0.02 s 控制步实际为 −10，降低主动顶板触发重置来逃避姿态成本的收益。G−、平地、自由板的腿部引导和物理配置不变。
- 新增 `a6_guided_leg_straightness`、`a6_guided_hip_bend`、`a6_guided_moving_straight` 训练指标；旧指标 `a6_guided_knee_extension` 现为有正负号的逐步净伸展，`a6_guided_knee_bend` 也改为归一化成本，均不能直接与 v4 数值比较。旧 v4 checkpoint 可用新版脚本回放与诊断，但只有训练 v5 策略才会改变动作。

验证：32 环境 80 步 CPU 仿真通过；从 v4 `model_01999.pt` 跨奖励迁移，保留 `00999` 冻结教师做 32 环境、64 步单轮 PPO，成功生成 v5 `model_02000.pt`，首轮 KL 为 0.00249，四场景回放入口检查通过。将**同一个旧策略**在 v5 奖励下重放，绿板仰卧的腿部覆盖改善平均值由 0.0968 降至 0.0042、腿部离板奖励由 0.0468 降至 0.0017；其失效步仍为 62，俯卧仍为 75。这个对照只验证蜷腿得分捷径被抑制，并非新策略已经学会脱困；仍需长训及独立成功率、膝角、抬板指标评估。

如需从 v4 G+ `model_01999.pt` 保留已学学生策略与 `00999` 冻结教师继续试训 v5，使用 `--resume <v4 checkpoint> --allow-model-transfer --keep-curriculum-on-transfer --a6-pose-v6 --frozen-flat-teacher <flat model_00999.pt> --flat-teacher-weight 0.5 --reset-optimizer --learning-rate 0.00005`，并使用新的输出目录和大于 2000 的累计 `--iterations`。奖励版本切换时会继承学生 actor、四个 critic 与课程值，但重建 Adam 状态；不要对同一输出目录做原样恢复。

## 历史 G+ v4：腿部与仰卧水平脱困引导

对冻结 `00999` 教师的 v3 G+ 3000 轮结果，九个高分 checkpoint 在 seed 42 固定四场景回放中都能从自由板仰卧脱困，但绿板俯卧、仰卧均未完成 25 步水平净空；多次在抬板约 5 cm 时无效重置。训练日志原来的 `a6_escaped_fraction` 混合两种板，不能代表绿板成功率。v4 只改 G+ 的奖励和日志，不改 29 轴模型、6 kg 板、物理判据、25 步完成条件或四 critic PPO 结构。旧 checkpoint 可以用新版回放脚本诊断，但奖励需要续训才会改变动作。

- 绿板下把初始起身阶段中的姿态、头部竖直速度、头高和直立四项正奖励关掉；剩余平稳项与 HoST style 仍乘 `0.05`。冻结平地教师模仿损失仍为 0，脱困后各项恢复。这样旧阶段 0 偏好的 `0.8–1.8 rad` 屈膝不再在绿板下获得起身奖励。
- 膝盖采用 `0.45 rad` 软上限：受困时最弯的一侧超过该角度才扣分 `−0.20`，每次刷新双膝中最坏角度的伸展纪录获得 `+0.20`。这只作用于仍被绿板覆盖且电机已接管的样本，不锁死膝关节；脱困后仍用原起身阶段目标。
- 用两侧小腿、连杆支架和足部共 18 个碰撞体在板平面上的投影，奖励新的边缘覆盖改善 `+0.25` 与持续的腿部离板比例 `+0.12`。纯竖直抬板不会改变投影，因此不能拿到这两项奖励。
- 俯卧和仰卧都奖励沿初始身体左右方向刷新躯干侧移纪录，各 `+0.30`；路径进展、覆盖改善、躯干远离板中心分别提高到 `+0.65/+0.12/+0.15`。无侧移进展的惩罚从受困 `0.20 s` 开始，`0.90 s` 达满值 `−0.25`；卡死×停滞 `−0.12`。抬板从 `0.5 cm` 开始扣分，满刻度到 `5 cm` 权重 `−0.45`，硬重置阈值不变。
- `metrics.jsonl` 新增绿板与自由板分别统计的已脱困状态占比、绿板俯卧/仰卧占比、绿板无效结束次数/占比，以及膝弯曲、伸展、腿部覆盖和仰卧侧移指标。已脱困占比仍是 rollout **采样时刻**的状态，不等于独立 episode 成功率；无效次数从重置前的终止标记读取。正式筛选 checkpoint 需要固定种子逐场景评估。

若从 v3 冻结组 `model_03999.pt` 试训 v4，使用 `--resume <v3 checkpoint> --allow-model-transfer --keep-curriculum-on-transfer --a6-pose-v6 --frozen-flat-teacher <flat model_00999.pt> --flat-teacher-weight 0.5 --reset-optimizer --learning-rate 0.00005`，并设新输出目录和大于 4000 的累计 `--iterations`。这沿用学生网络和课程值、重新建立 Adam 状态；不应把 v3 和 v4 指标直接当作同一奖励下的连续曲线。

## 历史 v3 修订：板下先侧移、超载或顶板即结束

以现有固定 6 kg G+ 策略回放时，仰卧导向板曾在未水平脱离时被顶高约 5.5 cm；俯卧则能在板下卡住。v3 在每次重置时保存躯干初始 XY、身体初始左右方向以及板初始高度。G+ 的俯卧导向板场景按躯干沿初始左右方向的**新增最佳位移**给 `+0.20`，板下整体水平分离进展从 `+0.01` 调到 `+0.10`；受困躯干每新增 2 cm 位移才清除“无进展”计时器，持续 0.5–1.5 s 不前进时惩罚逐渐升到 `−0.15`，同时持续高负载/无位移再扣 `−0.10`。侧移计算使用初始身体坐标，避免机器人转身后改变奖励方向。板下 G+ 的阶段起身项和 HoST 姿态风格项都乘 `0.05`，使旧平地站起动作不再是板下的主要正奖励；脱困后恢复原权重。29 轴的关节、电机和 PPO 网络结构未变。

安全判据按 10 个 2 ms 物理子步的接触历史取最大力、最深穿透和最高板位移；板力连续 3 个控制步超过 1500 N 或任一控制步超过 2500 N、穿透深于 2 cm，均结束回合。导向板在电机启动后仍覆盖躯干且被抬高超过 5 cm，也结束回合；轻微抬板从 1 cm 起逐渐扣分。无效结束额外给一次 `−100/s × 0.02 s = −2`，防止策略把主动重置当作脱困捷径。导向板的完成判据改为**水平离开板边至少 4 cm**且连续 25 个控制步无接触；自由板仍沿用原来的 2.5 cm 三维净空判据。历史 v2 策略在 v3 环境回放可用于诊断，但必须重新训练才能把这些奖励学进 PPO。以下旧命令及结果对应历史版本，启动 v3 时应使用新的输出目录。

本地用旧固定 6 kg G+ `model_01999.pt` 做 seed 42 四场景 200 步诊断：仰卧导向板在第 62 步板抬升子步峰值 5.07 cm，立即无效结束；俯卧导向板仍未逃出，躯干离初始位置最大约 21.68 cm，后期停止推进，无进展量达到 1.0。平地未触发板无效规则。这是旧策略在新环境中的**判据与奖励验证**，不是新策略成功率。32 环境 60 步烟测、旧 A6 策略到 v3 的 96 步单轮 PPO 更新和四场景 checkpoint 回放入口均通过；训练记录中冻结教师样本占比约 33.9%。用旧 checkpoint 末尾约 `2.56e−4` 的学习率初始化，首轮平均 KL 约 `0.083`；从 `5e−5` 初始化则约 `0.0035`，更接近 PPO 的 `0.01` 目标。建议正式奖励迁移使用后者起步并观察后续 KL，迁移参数为 `--resume <旧 A6 checkpoint> --allow-model-transfer --keep-curriculum-on-transfer --a6-pose-v6 --frozen-flat-teacher <平地 00999 checkpoint> --flat-teacher-weight 0.5 --learning-rate 0.00005`。这会继承旧学生策略/四 critic 权重及课程值，奖励改变时重建 Adam 优化器，且仍保留冻结的平地教师。

按正式 `5 epochs × 4 minibatches` 做的 32 环境、96 步单轮 CPU 预检也完成：起始学习率 `5e−5`，首轮平均 KL `0.0169`，自适应调度将末尾学习率降至 `1e−5`，冻结教师占比仍为 33.9%；新指标 `a6_prone_lateral_gain`、`a6_no_progress`、`a6_guided_lift_cost` 均写入 `metrics.jsonl`。这一轮只证明训练链路可运行，不能证明 4096 环境 GPU 容量或训练后的脱困成功率；正式训练仍应监控 KL、无效回合及四场景回放。

跨奖励 checkpoint 另外记录 `transfer_source_sha256`、`transfer_source_iteration`、`transfer_source_reward_variant` 和 `optimizer_reset_on_transfer`，从而区分直接来源的旧 A6 学生与更早的平地父代；本地试训文件验证旧学生迭代为 1999，直接来源 SHA256 为 `9cec94c5cf7108438d32f3354a22513de453384e1f0bf9cc63fbc4f5950d2f9c`。奖励迁移沿用学生 actor 与四个 critic 权重，但 Adam 动量会重新建立。

## 历史实验：固定平地 0999 教师，A6 学生续训 1000 轮

这次从同一个 `model_00999.pt` 建立两个网络：学生通过 `--continue-from29` 继承完整 PPO 权重、Adam 状态及课程，从迭代 1000 更新到 1999；教师通过 `--frozen-flat-teacher` 读取该文件，固定参数，不参与优化。平地及已确认脱离板的有效控制步使用教师的全部 29 轴平均动作作参考；仍在板下的动作只由学生 PPO 探索。教师损失按每个样本的 29 轴平方误差求和，乘以 `--flat-teacher-weight 0.5`。这是一个新实验，教师损失不能替代独立的平地站姿与脱困成功率评估。代码会核对教师与父代文件的 SHA256，并在续训时核对教师 SHA256 和权重；日志记录 `frozen_teacher_loss`、`frozen_teacher_fraction`。

本次净空完成判据为连续 25 控制步（0.5 秒），与先前 15 步实验不同。以下在服务器项目根目录运行；物理 GPU 3/4 分别训练 G+/G−，每组新增 1000 轮，两个输出目录均需是新的：

```bash
cd /home/czr/code/G1Recovery_HoST
PARENT=logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt
sha256sum "$PARENT"
for SPEC in 'G+ 3 Gplus' 'G- 4 Gminus'; do
  read -r ARM GPU TAG <<< "$SPEC"
  OUT="logs/host_a6_29_frozen0999_${TAG}_4096_seed42_1000"
  mkdir -p "$OUT"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/mpl-host-a6-frozen-${TAG}" \
    PYTHONPATH="$PWD/src" /home/czr/code/G1Recovery_mjlab/.venv/bin/python \
    scripts/train_host.py --task a6 --arm "$ARM" --a6-pose-v6 \
    --continue-from29 "$PARENT" --frozen-flat-teacher "$PARENT" \
    --flat-teacher-weight 0.5 --device cuda:0 --num-envs 4096 \
    --steps-per-env 24 --iterations 2000 --epochs 5 --minibatches 4 \
    --adaptive-kl --seed 42 --log-every 10 --save-every 100 --output "$OUT" \
    > "$OUT/train.log" 2>&1 < /dev/null &
  echo "$ARM PID=$! log=$OUT/train.log"
done
```

期望父代 SHA256 为 `0fa09f84bfc1eed56526804c0e0d47c76c69a09c1a64fb1922e8f233b28aaa0a`。运行前需将本地的 `scripts/train_host.py`、`scripts/play_a6_host29.py`、`src/g1recovery_host/ppo.py`、`src/g1recovery_host/a6_reward.py` 上传到服务器相同相对路径。若从本实验中途断点恢复，使用相同输出目录和 `--resume <本实验 checkpoint> --a6-pose-v6 --frozen-flat-teacher "$PARENT" --flat-teacher-weight 0.5 --iterations 2000`，不要同时传 `--continue-from29`。

## 历史路线：从 29DoF 姿态修复专家完整续训 A6

平地父代是 `logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt`，SHA256 `0fa09f84bfc1eed56526804c0e0d47c76c69a09c1a64fb1922e8f233b28aaa0a`，`iteration=999`。`--continue-from29` 要求其为 `flat29`/`unitree_g1_29`/`height_guard_style_v2_29dof_pose_v6`，并严格载入全部 29 轴 actor、上半身残差分支、四个 critic、探索方差、Adam 动量与学习率、beta 与辅助力课程；下一次迭代编号为 1000。与只取 actor 的 `--init-from29` 不同，此模式是真正的**同一套四 critic PPO 参数续训**。A6 的奖励、接触和物理子步发生变化，故不是平地仿真的逐位恢复，首轮价值损失可能偏高。G+ 与 G− 从同一父代独立分叉，分别是一个在平地/导向板/自由板混合场景上训练的 PPO；两组之间不共享更新后的参数。

继续学习脱困时，两组均保留同一站姿约束：只有平地站稳或板已脱离后，才轻量惩罚手臂、弱腕和腰轴偏离 v6 站姿，并启用腰轴教师损失。脱困途中不启用这些新增约束。当时的奖励版本标识为 `historical_a6_G+_29dof_pose_v6_v2` 或 `historical_a6_G-_29dof_pose_v6_v2`；旧 A6 checkpoint 不可直接当作这个版本的同任务恢复。

正式训练在服务器项目根目录运行。下面每组执行 1000–2999 共 2000 轮、每轮 4096×24 个控制步，GPU 3/4 分开运行；`CUDA_VISIBLE_DEVICES` 将指定的物理卡映射为进程内的 `cuda:0`。两组输出目录必须不同：

```bash
cd /home/czr/code/G1Recovery_HoST
PARENT=logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt
for SPEC in 'G+ 3 Gplus' 'G- 4 Gminus'; do
  read -r ARM GPU TAG <<< "$SPEC"
  OUT="logs/host_a6_29_posev6_${TAG}_from0999_4096_seed42_2000"
  mkdir -p "$OUT"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR="/tmp/mpl-host-a6-${TAG}" \
    PYTHONPATH="$PWD/src" /home/czr/code/G1Recovery_mjlab/.venv/bin/python \
    scripts/train_host.py --task a6 --arm "$ARM" --continue-from29 "$PARENT" \
    --device cuda:0 --num-envs 4096 --steps-per-env 24 --iterations 3000 \
    --epochs 5 --minibatches 4 --adaptive-kl --seed 42 \
    --log-every 10 --save-every 100 --output "$OUT" \
    > "$OUT/train.log" 2>&1 < /dev/null &
  echo "$ARM PID=$! log=$OUT/train.log"
done
```

断点继续时传 `--resume <A6 checkpoint> --a6-pose-v6 --adaptive-kl`，不要再传 `--continue-from29`。`--iterations` 始终表示目标总迭代编号；父代的 1000 轮不计入 A6 的 2000 轮预算。可用 `scripts/play_a6_host29.py --checkpoint <A6 checkpoint> --four-view --device cpu` 回放当前 4 场景。

2026-10-10 已在服务器 GPU 3/4 启动上述两组训练，初始进程号分别为 G+ `1485189`、G− `1485381`。父代的服务器文件与本地文件 SHA256 相同。正式启动前，CPU 上通过 32 环境关节/数据审计与 G+ 一轮 PPO 更新及回放入口校验；服务器 GPU 3/4 上两组各通过 32 环境、120 步一轮 PPO 更新，GPU 3 上通过 4096 环境容量预检。训练日志中的 `a6_escaped_fraction` 是板场景**采样时刻**处于已脱离状态的占比，不是完整 episode 脱困成功率；正式效果仍须独立评估。下方其余旧训练命令为历史路径，当前续训请用本节。

## 模型、参数和数据谱系

29 个关节依次为 12 腿、3 腰、7 左臂、7 右臂。相对原 HoST 23DoF 模型增加 `waist_roll_joint`、`waist_pitch_joint`、左右 `wrist_pitch_joint`、`wrist_yaw_joint`。新增轴的连杆惯量、碰撞形状与关节限位由 [mjlab 的 G1 MJCF](../mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml) 加载；模型共 29 个铰链关节、36 维浮动基座 qpos、总质量约 33.341 kg。Unitree [29DoF 机器人说明](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md)列出 3 腰轴和每臂 7 轴；[Unitree 29DoF MJCF](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/g1_29dof.xml)给出腰 roll/pitch 50 N·m、腕 pitch/yaw 5 N·m 的电机力矩范围。新增轴的反射惯量及 PD 使用 [mjlab G1 电机常量](../mjlab/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py)：腰 roll/pitch 每轴 `armature=0.00721945`, `kp≈28.50`, `kd≈1.814`, `effort=50`；四个腕 pitch/yaw 各 `armature=0.00425`, `kp≈16.78`, `kd≈1.068`, `effort=5`。原有 23 轴维持 HoST 本地控制参数。

数据流：`model.py` 的 29DoF G1 → `a6_geometry.py` 的三场景 → `a6_reset.py` 的 36 维初态 → `task.py` 的 94×6=564 维 actor 观测 → `ppo.py` 的 29 动作 actor → `action.py` 的电机力矩 → 500 Hz 物理步 → `a6_reward.py` 的接触/历史/奖励 → `train_host.py` 的四 critic PPO。`assets/a6/source/` 是冻结的 SMP 原始 reset bank；`assets/a6/host29/` 是按当前 29DoF 关节限位验证并生成的训练数据；`logs/` 是生成的 checkpoint 和曲线。

原始 bank 的 SHA256 见 `assets/a6/host29/manifest.json`。主 bank 4096 条中有 933 条姿态在当前 MJCF 关节硬限位外，生成文件只将越界角截到限位，最大改动 0.04026 rad；自然中位/后期和程序化低位 bank 无截断。转换脚本会拒绝 SHA 不符的输入。

23DoF 父代 checkpoint `logs/host_ground_4096_seed42_3000/model_02999.pt` 的 SHA256 为 `65a6b52146fb497d6e159bad9e7d8aa78fa46fa91b8bd0d228ca11ea94707112`。`a6_transfer.py` 按关节名和每一帧观测位置映射旧 23 轴；新增六轴初始 actor 输出均值为零。`--continue-from` 将 actor、四个 critic、探索方差、Adam 动量和步数、beta 与辅助力课程全部迁移，并从迭代 3000 开始；父代最后的 `beta=0.25`、辅助力 `0 N`、Adam 学习率约 `0.000170859375`。原训练模型为 23DoF URDF，新任务为 29DoF MJCF，几何、惯量、接触与 reward 都发生变化，所以**这仍是跨模型/任务续训，不是字节完全相同的仿真恢复**。`--init-from` 则只迁移 actor，两种模式不能混用。

## 场景和 reward

4096 环境固定为平地 2048、仅能上下滑动的导向板 1024、六自由度自由板 1024。低/中/近站立共 2868/820/408；低位四方向各 717，低位源约 75% 自然、25% 程序化。板尺寸 0.90×0.64×0.07 m；当前质量固定 6 kg，底面高于覆盖机器人碰撞几何最高点 2 mm。控制 50 Hz、物理 500 Hz、10 个物理子步、episode 10 s。此处保留 HoST 的前 30 步无驱动和 6 帧观测规则；SMP 的初始静态历史填充不适用于 HoST，属于明确的方法差异。

| 原生 HoST 项 | A6 处理 | 共同项及权重（每秒奖励率） | critic |
| --- | --- | --- | --- |
| `task_reward_height_guard` | 替换 | 阶段姿态 .22、头速度 .18、高度 .10、直立 .15、脚/基座/角速度/关节/动作安静 .08/.07/.07/.06/.07；G+ 受困时乘 α=.05，G− α=1 | `task` |
| `regularization_reward` | 替换，避免重复扣费 | action rate −.0015、加速度 −.0012、关节加速度 −5e−8、力矩 −1e−6、超速 −.02、超功率 −2e−6、头超速 −1、持续负载 −.05、软关节限位 −.10、Q −.03、L −.20 | `regu` |
| `style_reward` | 保留 HoST 姿态先验；G+ 受困时乘 α=.05 | 原 HoST style 公式 | `style` |
| `target_reward` | 替换 | 两组共用 completion +.60、板超力 −.03、无效结束 −100；仅 G+ 有手支撑低位水平进展 G +.45、dense clearance +.08、separation +.10、俯卧侧移 +.20、无进展 −.15、卡关节滞留 −.10、抬板 −.20 | `target` |

A6 的 `regu` 组对 29 轴动作变化、关节加速度、力矩、超速、超功率、软关节限位和持续高负载扣分。`--a6-pose-v6` 另在平地站稳或确认脱困后惩罚手臂、腕、腰偏离参考站姿；在板下不启用这项站姿偏差惩罚。旧平地 `regularization_reward` 在 A6 中被替换，避免两套关节成本重复计算。这些成本没有直接限制板的竖直位移。

所有上述权重由 mjlab reward manager **只乘一次** `dt=.02`；训练器从分组内部率张量读取时也显式乘 `.02` 供 PPO 使用。G 依据水平覆盖的历史最佳改善，头高≤0.90 m、手触地、曾接触且未完成/无效时才支付；竖直抬板不获得 G。v3 的导向板 completion 要求水平净空 ≥0.04 m 且无接触连续 25 控制步；自由板保留原三维净空 ≥0.025 m。无效板包括子步穿透 `<−0.02 m`、连续 3 控制步峰值力 `>1500 N`、单步峰值力 `>2500 N`、导向板在电机启动后仍覆盖躯干且抬升 `>0.05 m`，或 25 步后从未接触。Q/L 用物理子步载荷和逐关节 25 步运动窗口。G−/G+ 的场景、reset、共享任务/成本、completion/force、四 critic 结构和训练预算均相同；G+ 另外启用侧移、滞留、抬板奖励项及受困姿态门控。

## 运行顺序

从项目根目录运行，`PY` 指向已安装 mjlab 的虚拟环境 Python。本地示例：

```bash
cd /home/ray1/code/G1Recovery_HoST
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/mpl-host-a6
export WARP_CACHE_PATH=/tmp/warp-host-a6
PY=/home/ray1/code/mjlab/.venv/bin/python
$PY scripts/prepare_a6_banks.py --source-root assets/a6/source --dof 29
$PY scripts/smoke_a6_host29.py --device cpu --num-envs 32 --steps 501
```

若按用户指定，从旧平地 `model_02999.pt` 继续 2000 轮并同时引入 A6 场景和 reward，应对 G+、G− 分别用 `--continue-from`；`--iterations 5000` 是**总迭代编号上限**，会执行 3000–4999 共 2000 轮。以下为单卡模板，两组必须使用不同输出目录和物理 GPU：

```bash
CUDA_VISIBLE_DEVICES=3 $PY scripts/train_host.py --task a6 --arm 'G+' \
  --continue-from logs/host_ground_4096_seed42_3000/model_02999.pt \
  --device cuda:0 --num-envs 4096 --steps-per-env 24 --iterations 5000 \
  --epochs 5 --minibatches 4 --adaptive-kl --seed 42 \
  --save-every 100 --output logs/host_a6_29_Gplus_full_from2999_seed42
```

此模式继承父代优化器学习率，因此命令中的 `--learning-rate` 不会覆盖它；若明确希望换学习率，需加 `--reset-optimizer`，但那就不再继承 Adam 状态。先前 G+/G− 两组 `model_01999.pt` 使用的是 actor-only 迁移，不属于上述完整续训，不能混作同一实验。`--continue-from` 模式已在本地 CPU 上通过 G+/G− 各一轮 A6 PPO 更新及 G+ 一轮 checkpoint 恢复；4096 环境 GPU 容量曾在同一 A6 任务下通过，但**新模式本身尚未在服务器 GPU 上预检或长训**。

另一条路线是先做 29DoF 平地适应：从旧 23DoF actor 权重出发，平地能力达到要求后用 `eval_host.py --task flat29` 检查；此路线后续仍使用 actor-only 的 A6 初始化，与上面指定的完整续训实验不同：

```bash
$PY scripts/train_host.py --task flat29 \
  --init-from logs/host_ground_4096_seed42_3000/model_02999.pt \
  --device cuda:0 --num-envs 4096 --steps-per-env 50 --iterations 3000 \
  --adaptive-kl --learning-rate 0.001 --output logs/host_ground29_seed42
$PY scripts/eval_host.py --task flat29 \
  --checkpoint logs/host_ground29_seed42/model_02999.pt --device cuda:0
```

将**同一个**合格的 29DoF 平地 checkpoint 传给两组 A6；以下以第 2999 次文件为例，实际应替换为经独立验证选定的文件。每组正式预算为 4096×24×10000 控制步、每 500 次保存：

```bash
for ARM in 'G-' 'G+'; do
  $PY scripts/train_host.py --task a6 --arm "$ARM" \
    --init-from29 logs/host_ground29_seed42/model_02999.pt \
    --device cuda:0 --num-envs 4096 --steps-per-env 24 --iterations 10000 \
    --epochs 5 --minibatches 4 --adaptive-kl --learning-rate 0.001 \
    --save-every 500 --output "logs/host_a6_29_${ARM}_seed42"
done
```

正式训练前先用 `--num-envs 32 --steps-per-env 40 --iterations 1 --epochs 1 --minibatches 2` 在目标 GPU 主机完成预检，并做至少一次真实策略回放。本地 CPU 已完成 32 环境×501 步仿真、G+/G− 各一次 40 步 PPO 更新、29DoF 平地→A6 actor 初始化，以及 G+ 一次跨完整 10 秒 episode 的 501 步 PPO 更新。2026-10-10 在服务器 RTX 4090 上，29DoF 平地通过 32 和 4096 环境各一次 PPO 更新，A6 G+ 也通过 32 和 4096 环境各一次 PPO 更新；这些容量预检的 checkpoint 位于服务器 `/tmp/host29_*_preflight_*`，**不是已训练策略**。后续还需合格的 29DoF 平地父代、四方向起身评估、A6 脱困成功率评估及基于合格父代的正式对照训练。下述 2000 轮 A6 结果直接从 23DoF 父代迁移，属于探索试验。

### 2026-10-10 训练状态与 Viser 回放

先前在服务器 GPU 1 启动的 `flat29` 平地适应已按用户要求终止，未作为 A6 父代。随后 G+ 和 G− 从**同一个 23DoF 平地 checkpoint**做 23→29 actor 迁移，分别在 GPU 3/4 以 `seed=42`、4096 环境、每轮 24 控制步训练 2000 轮。服务器上两组 `metrics.jsonl` 均为 2000 行，最终 checkpoint 分别为 `logs/host_a6_29_Gplus_4096_seed42_2000/model_01999.pt` 和 `logs/host_a6_29_Gminus_4096_seed42_2000/model_01999.pt`；元数据标记为 `task=a6`、`model_source=unitree_g1_29`、迭代 1999。两组末轮记录的辅助力均约为 200 N，因此训练完成不代表无辅助力脱困成功。仍需四方向、导向板和自由板的定量评估。

本地 Viser 回放入口是 `scripts/play_a6_host29.py`。它从 checkpoint 读取 G+/G− 及训练时动作系数和辅助力。`--four-view` 只建立四个回放世界并同时显示：自由板仰卧、绿色导向板俯卧、绿色导向板仰卧、平地仰卧。也可用 `--scene flat|guided|free` 与 `--direction supine|prone|left_side_down|right_side_down` 在原有 32 个固定测试世界中选择一个；`--assist-force-n 0` 可单独检查无辅助力行为。下载 checkpoint 至本地项目的 `logs/downloaded/` 后，例如：

```bash
cd /home/ray1/code/G1Recovery_HoST
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/mpl-host-a6-play
/home/ray1/code/mjlab/.venv/bin/python scripts/play_a6_host29.py \
  --checkpoint logs/downloaded/host_a6_29_Gplus_model_01999.pt \
  --four-view --device cpu
```

打开终端显示的 Viser 地址，通常为 `http://localhost:8080`。`--validate-only` 仅检查 checkpoint、场景选择与一次仿真步，不启动网页回放。
