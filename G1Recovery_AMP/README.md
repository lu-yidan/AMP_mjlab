# G1Recovery_AMP

这个目录是我维护的独立 Unitree G1 起身训练子项目。我使用 [mjlab](https://github.com/mujocolab/mjlab) 搭建 MuJoCo 仿真环境，使用 PPO 学习“完成起身”的任务，并用 AMP（对抗式动作先验）引导动作风格接近参考动作。这里同时保留了平地恢复和压板约束下的恢复实验代码；它不会自动替换仓库根目录已有的行走/恢复任务。

这是研究与实验项目，不是已经验证可直接部署到真机的控制器。本目录包含代码、测试和当前任务必需的 NPZ 数据；**不包含训练日志、视频或模型 checkpoint**，因此克隆后可以检查数据与重新训练，但不能直接回放我以前训练出的策略。

## 项目怎样工作

1. `motion_assets.py` 选出默认训练使用的 8 段起身动作；`commands.py` 读取这些动作，提供重置起点和 AMP 参考样本。本目录只上传这 8 个当前使用的动作 NPZ；完整清单见 `DATA_FILES.txt`。
2. `env_cfg.py` 组装 G1 机器人、仿真、动作、观测、奖励、终止和随机扰动；压板约束实验在独立的环境配置中扩展场景与重置方式。
3. 每个控制周期，策略根据自身观测输出 29 个关节位置目标。环境推进仿真，计算起身任务奖励；AMP 判别器比较策略动作与参考动作的短时序列，提供动作风格信号。
4. `rl/algorithm.py` 把任务奖励和风格信号接入 PPO 更新；评测脚本用固定 checkpoint 检查起身与物理状态，而不是把训练奖励当作成功率。

这条链路可以简写为：`参考动作/初始姿态 → 仿真环境 → 观测 → 策略动作 → 仿真与奖励 → PPO + AMP 更新`。

## 目录与模块

```text
G1Recovery_AMP/
├── pyproject.toml                 # 依赖、Python 版本和 mjlab 任务入口
├── DATA_FILES.txt                 # 本子项目实际上传的 10 个数据文件
├── src/g1recovery_amp/           # 项目实现
│   ├── data/motions/get_up/       # 起身动作 NPZ
│   ├── data/reset_banks/a6/       # 约束场景的初始姿态数据
│   └── tasks/recovery/            # 环境、训练与任务逻辑
├── scripts/                      # 预览、诊断、训练冒烟检查和评测
├── tests/                        # 单元与接口测试
└── README.md                     # 本子项目说明
```

### 入口、机器人和数据

| 文件 | 作用 |
| --- | --- |
| `src/g1recovery_amp/__init__.py`、`tasks/__init__.py`、`tasks/recovery/__init__.py` | 加载并注册任务，使 mjlab 的 `train`、`play` 等入口能找到环境和训练配置。 |
| `g1_model.py` | 定义 G1 的 29 个关节顺序、动作尺度及模型映射；使用 mjlab 提供的机器人模型资源。 |
| `motion_assets.py` | 列出默认参与训练的 8 个动作文件及相同采样权重，是“当前训练用哪些动作”的权威清单。 |
| `motion_conversion.py` | 把旧格式动作转换成项目可读取的 NPZ；转换工具不是训练时每步都会执行的模块。 |
| `a6_reset_assets.py` | 读取并校验两个初始姿态数据文件的哈希、形状与数值，防止错误数据进入仿真。 |
| `data/motions/get_up/` | 8 个当前默认训练使用的动作 NPZ。 |
| `data/reset_banks/a6/` | 两个初始姿态 NPZ。它们提供不同倒地姿势与阶段的起点，不是 AMP 判别器使用的连续示范动作。 |

### 仿真环境与任务逻辑

`tasks/recovery/` 中的配置文件负责“装配环境”，`mdp/` 中的函数负责“每一步具体怎么算”。前者选择后者，训练入口再加载装配好的环境。
`mdp/__init__.py` 汇总环境项的公开接口；`rl/__init__.py` 汇总学习算法的公开接口，本身不执行训练。

| 文件 | 作用 |
| --- | --- |
| `env_cfg.py` | 平地起身环境的总配置：机器人、仿真步长、动作、观测、奖励、终止条件与随机扰动。 |
| `a6_env_cfg.py`、`a6_scene.py` | 约束场景配置与压板实体；在平地任务之外提供可控的受困实验场景。 |
| `mdp/commands.py` | 管理参考动作与起身辅助力；负责动作抽样、时间位置和环境重置时的参考状态。 |
| `mdp/observations.py` | 整理策略、价值网络及 AMP 判别器所需的观测。策略输入与训练时的额外信息分开。 |
| `mdp/rewards.py` | 平地起身任务的奖励与惩罚，例如朝目标姿态和完成站立的进展。 |
| `mdp/events.py` | 重置或训练过程中的扰动与参数随机化。 |
| `mdp/curriculums.py` | 随训练进度调整难度，例如逐渐减少起身辅助。 |
| `mdp/metrics.py` | 记录关节加速度、接触等健康指标，供诊断使用；指标本身不等于奖励。 |
| `mdp/recorders.py` | 保存自动重置前的最后观测，避免 AMP 学习时误把下一回合首帧接到上一回合末尾。 |
| `mdp/a6_resets.py` | 从初始姿态数据中抽取起点、写入机器人状态，并为约束场景放置压板。 |
| `mdp/a6_actions.py`、`mdp/a6_dynamics.py` | 重置后的动作接管保护，以及质量、摩擦、电机等动力学参数变化。 |
| `mdp/a6_plate_state.py` | 跟踪压板接触、净空、脱困完成和无效物理状态。 |
| `mdp/a6_rewards.py`、`mdp/a6_shared_tasks.py`、`mdp/a6_shared_costs.py` | 约束场景的脱困/恢复奖励、共同任务项与运动成本。 |
| `mdp/a6_reward_scaling.py` | 按是否仍受困调整部分任务、风格和成本项的相对权重。 |

### 学习算法

| 文件 | 作用 |
| --- | --- |
| `rl/rl_cfg.py` | 网络结构、PPO 超参数与训练运行配置。 |
| `rl/runner.py` | 把本项目的环境和动作数据接到训练运行器。 |
| `rl/algorithm.py` | 在 PPO 更新中组合任务奖励与 AMP 风格信号，管理相关训练状态。 |
| `rl/discriminator.py` | 判别短时动作序列更像参考动作还是策略生成动作，输出风格学习信号。 |
| `rl/buffer.py` | 保存判别器训练要用的策略样本与参考样本。 |
| `rl/optimizer.py` | 执行判别器参数更新。 |

### 脚本与测试

| 脚本 | 用途 |
| --- | --- |
| `scripts/preview_motion.py` | 单独预览 NPZ 动作，不加载策略，也不运行训练。 |
| `scripts/smoke_recovery_env.py` | 短步数检查环境能否创建、重置和推进。 |
| `scripts/smoke_amp_training.py` | 进行极短 AMP/PPO 更新，检查训练接口是否接通；不是正式训练。 |
| `scripts/check_legacy_checkpoint.py` | 检查已有 checkpoint 能否按当前模型结构加载。 |
| `scripts/check_a6_flat_reset.py`、`scripts/check_a6_three_scene_reset.py` | 检查初始姿态映射、场景分配和压板放置。 |
| `scripts/check_a6_flat_rollout.py` | 在约束场景中做短时非训练 rollout，观察数值与终止行为。 |
| `scripts/evaluate_recovery_checkpoint.py` | 固定一个 checkpoint，测量起身表现与诊断指标。 |

`tests/` 对关节映射、动作数据、观测、奖励、扰动、AMP 更新和场景接口做回归检查。测试通过只说明这些接口符合预期，不能替代训练成功率评测或真机验证。

## 安装与验证

项目需要 Python 3.13，以及与 `pyproject.toml` 兼容的 mjlab、MuJoCo、PyTorch 和 RSL-RL 环境。此子目录的依赖配置将 `mjlab` 指向 `../../mjlab`；下面的命令假设 `AMP_mjlab` 和 `mjlab` 两个仓库并排放置，且已按 mjlab 的说明建立虚拟环境。本子项目没有独立的 `uv.lock`，因此全新机器上的依赖解析仍需验证。

```text
工作目录/
├── mjlab/
└── AMP_mjlab/
    └── G1Recovery_AMP/
```

```bash
cd /path/to/AMP_mjlab/G1Recovery_AMP
uv pip install --python ../../mjlab/.venv/bin/python --no-deps --editable .
PYTHONPATH="$PWD/src:$PWD" ../../mjlab/.venv/bin/python -m pytest -q tests
```

`--no-deps` 只安装本项目的可编辑入口，不会替你安装或更新 mjlab 的 GPU 依赖。在整理上传副本时，完整单元测试结果为 `182 passed, 16 subtests passed`；这不是新机器安装成功或起身成功率的保证。

## 运行示例

预览一段参考动作：

```bash
PYTHONPATH="$PWD/src:$PWD" ../../mjlab/.venv/bin/python scripts/preview_motion.py \
  --motion-file src/g1recovery_amp/data/motions/get_up/fallAndGetUp1_subject1_1060_1150.npz
```

从头启动一个短训练，先验证链路，再按硬件能力增大环境数和训练轮数：

```bash
PYTHONPATH="$PWD/src:$PWD" ../../mjlab/.venv/bin/train \
  Mjlab-Recovery-AMP-Flat-Unitree-G1-Dev \
  --env.scene.num-envs 512 \
  --agent.max-iterations 100 \
  --agent.run-name first_smoke_512x100
```

训练输出写入本子目录的 `logs/`，该目录被 `.gitignore` 排除。回放和评测需要另外提供对应版本的 checkpoint；此分支不附带旧训练权重。历史实验记录未随此最小可运行子项目上传，阅读参数时请以代码和该次运行保存的配置为准。

## 来源与边界

本子项目使用 mjlab/MuJoCo、RSL-RL 与 G1 模型资源，并整理了既有起身动作及初始姿态数据。我负责此子项目的组织、适配、实验和维护；依赖项目及原始素材的权利仍属于各自权利人。此子项目没有单独声明开源 License，也不应被当作安全认证或真机部署说明。
