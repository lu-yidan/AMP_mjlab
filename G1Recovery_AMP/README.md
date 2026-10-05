# G1Recovery_AMP

这个目录是我维护的独立 Unitree G1 起身训练子项目。我使用 [mjlab](https://github.com/mujocolab/mjlab) 搭建 MuJoCo 仿真环境，使用 PPO 学习“完成起身”的任务，并用 AMP（对抗式动作先验）引导动作风格接近参考动作。这里同时保留了平地恢复和压板约束下的恢复实验代码；它不会自动替换仓库根目录已有的行走/恢复任务。

这是研究与实验项目，不是已经验证可直接部署到真机的控制器。本目录包含代码、测试、当前任务必需的 NPZ 数据，以及一个经过限幅续训的 `model_22899.pt`。训练日志、视频和其他 checkpoint 不随仓库上传。

## 当前实验版本

- `checkpoints/model_22899.pt`：从平地与压板联合恢复策略 `model_21900.pt` 继续训练 1000 轮（512 个并行环境）的权重；不是从零训练的模型。
- 对应任务：`Mjlab-Recovery-AMP-LowTorque-CapOnly-90-Unitree-G1-Dev`。与父任务相比，只把原为 139 N·m 的髋外展和膝关节两组仿真电机上限乘以 0.90，即 125.1 N·m；奖励、观测和动作定义保持不变。这样可以单独考察限幅的影响，不能理解成全部关节扭矩都降低了 10%。
- 这份 checkpoint 的 SHA-256 是 `3a685513dabc0f0680eda2471b4353d2ed08aa363bed307396fea44baad88568`。下载后可用 `sha256sum checkpoints/model_22899.pt` 校验文件完整性。
- 在已有的 2048 次、seed 42、10 秒评测中，整体基础站立成功数从父节点的 1833/2048 变为 1878/2048；上下滑动板基础站立从 307/512 变为 350/512。但上下滑动板与自由板的脱困数分别从 406/512 降为 401/512、412/512 降为 401/512。两份旧评测没有逐条初态配对证明；这些数字不能当作严格配对实验或真机安全性的结论。32 个固定初态的逐关节诊断还发现部分腕、肩关节长时间触限，必须继续评估。

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
├── checkpoints/model_22899.pt     # 唯一随仓库上传的限幅续训权重
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
| `a6_env_cfg.py`、`a6_scene.py` | 约束场景配置与压板实体；另提供只降低 139 N·m 电机上限的可选任务配置。 |
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
| `scripts/record_a6_scene_video.py` | 用同一个 checkpoint，分别录制平地、上下滑动板仰卧/俯卧、自由板场景；默认 1080p、关闭阴影。 |

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

`--no-deps` 只安装本项目的可编辑入口，不会替你安装或更新 mjlab 的 GPU 依赖。此版本在本地完整单元测试结果为 `183 passed, 16 subtests passed`；这不是新机器安装成功或起身成功率的保证。

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

训练输出写入本子目录的 `logs/`，该目录被 `.gitignore` 排除。历史实验记录未随仓库上传，阅读参数时请以代码和该次运行保存的配置为准。下列回放和录像均使用随仓库上传的 `22899` 权重，必须选用相应的限幅任务；不要把它当成父节点 `21900` 的配置来评估。

回放四个并行场景（首次使用前先完成上面的可编辑安装）：

```bash
cd /path/to/AMP_mjlab/G1Recovery_AMP
WANDB_MODE=disabled PYTHONPATH="$PWD/src:$PWD" ../../mjlab/.venv/bin/play \
  Mjlab-Recovery-AMP-LowTorque-CapOnly-90-Unitree-G1-Dev \
  --agent trained \
  --checkpoint-file "$PWD/checkpoints/model_22899.pt" \
  --motion-file "$PWD/src/g1recovery_amp/data/motions/get_up/fallAndGetUp1_subject1_1060_1150.npz" \
  --num-envs 4 --device cuda:0 --viewer viser
```

逐场景录制 10 秒视频（视频只写到被忽略的 `videos/`，不会自动进入 Git）：

```bash
cd /path/to/AMP_mjlab/G1Recovery_AMP
for scene in flat guided_supine guided_prone free; do
  WANDB_MODE=disabled PYTHONPATH="$PWD/src:$PWD" ../../mjlab/.venv/bin/python \
    scripts/record_a6_scene_video.py \
    --task-id Mjlab-Recovery-AMP-LowTorque-CapOnly-90-Unitree-G1-Dev \
    --checkpoint-file "$PWD/checkpoints/model_22899.pt" \
    --scene "$scene" --seed 42 --device cuda:0 \
    --output-dir "$PWD/videos/model_22899"
done
```

`flat` 是无压板对照，`guided_supine` 和 `guided_prone` 是上下滑动板下的仰卧、俯卧，`free` 是自由板。它们用于观察行为，不替代批量成功率或电机安全评估。

## 来源与边界

本子项目使用 mjlab/MuJoCo、RSL-RL 与 G1 模型资源，并整理了既有起身动作及初始姿态数据。我负责此子项目的组织、适配、实验和维护；依赖项目及原始素材的权利仍属于各自权利人。此子项目没有单独声明开源 License，也不应被当作安全认证或真机部署说明。
