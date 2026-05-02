# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

G1 AMP motion control project: a single unified policy learns both locomotion (walk/run) and fall-recovery for the Unitree G1 humanoid robot. Built on **mjlab** (simulation framework) + a local fork of **rsl_rl** (RL training library). Deployment integration lives in a separate repo (`ccrpRepo/wbc_fsm`).

## Setup

```bash
conda activate mjlab   # Python 3.11 environment required
cd AMP_mjlab
python -m pip install -e .
# Also install the local rsl_rl fork:
cd rsl_rl && python -m pip install -e . && cd ..
```

### Optional: Apply mjlab Patch

Enables `history_ordering="time"` in observation configs (required by this project). Without it, remove all `history_ordering` fields.

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  $(python -c "import mjlab; import os; print(os.path.dirname(mjlab.__file__))")/managers/observation_manager.py
```

## Common Commands

```bash
# List available tasks
python scripts/list_envs.py --keyword AMP

# Train (flat terrain, adjust num-envs as needed)
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096

# Train (rough terrain)
python scripts/train.py Unitree-G1-AMP-Rough --env.scene.num-envs=4096

# Play/evaluate a checkpoint
python scripts/play.py Unitree-G1-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt

# Convert motion CSVs to NPZ
python scripts/csv_to_npz.py --help

# Analyze base motion data
python scripts/analyze_base_motion.py
```

Training logs are saved to `logs/rsl_rl/g1_amp_locomotion/<timestamp>/`. ONNX export runs automatically during training (every save) and play.

## Architecture

### Two-Package Structure

- **`src/`** — Task definitions, robot configs, MDP logic (mjlab-specific)
- **`rsl_rl/`** — Local fork of the RL library; extends upstream with AMP support

### Training Flow

`scripts/train.py` → `ManagerBasedRlEnv` (mjlab env) → `RslRlVecEnvWrapper` → `AMPOnPolicyRunner` (in `src/tasks/amp_loco/rl/runner.py`) → `AmpOnPolicyRunner` (in `rsl_rl/`) → `AMPPPO` algorithm

### Task Config Layers (for G1 AMP)

1. **`src/tasks/amp_loco/amp_env_cfg.py`** — `make_amp_env_cfg()`: base factory returning a `ManagerBasedRlEnvCfg` with all MDP terms wired up, but robot-specific parameters left empty (body names, motion dirs, etc.)
2. **`src/tasks/amp_loco/config/g1/env_cfgs.py`** — `g1_amp_rough_env_cfg()` / `g1_amp_flat_env_cfg()`: fills in G1-specific values; flat inherits from rough
3. **`src/tasks/amp_loco/config/g1/rl_cfg.py`** — `g1_amp_ppo_runner_cfg()`: returns `RslRlAmpRunnerCfg` with AMP hyperparameters and discriminator config

### Key Design: Unified Locomotion + Recovery

- Motion data split: `WalkandRun/` (velocity tracking) and `Recovery/` (fall-and-get-up) under `src/assets/motions/g1/amp/`
- `delay_reset_env_ratio=0.4` — 40% of envs do not reset immediately on termination; instead they receive a recovery window with initial states sampled from recovery clips
- AMP discriminator (`rsl_rl/modules/discriminator.py`) regularizes motion style across both skill sets
- Reward blending: `amp_task_reward_lerp=0.75` (task) vs `amp_reward_coef=0.1` (discriminator)

### Observation Groups

Three observation groups are used:
- `actor` — noisy, history_length=4, `history_ordering="time"` (requires mjlab patch)
- `critic` — clean, same history, includes full body pose
- `amp` — no history, body pos/ori/vel in anchor frame (fed to discriminator)

### rsl_rl Fork Modules

- `rsl_rl/algorithms/amp_ppo.py` — `AMPPPO`: PPO + discriminator gradient + replay buffer
- `rsl_rl/runners/amp_on_policy_runner.py` — `AmpOnPolicyRunner`: manages discriminator, AMP data loader, preloaded transitions
- `rsl_rl/utils/motion_loader.py` — `AMPLoader`: loads NPZ motion files and samples reference transitions
- `rsl_rl/modules/discriminator.py` — MLP discriminator network

### ONNX Export

`AMPOnPolicyRunner.save()` automatically exports `policy.onnx` alongside each checkpoint. The export wraps the actor + obs normalizer so the ONNX model takes raw observations directly (no normalization needed in deployment).

## Motion Data

Raw CSV files live in `motion_data_csv/amp/`. Convert to NPZ with `scripts/csv_to_npz.py`, then place under:
- `src/assets/motions/g1/amp/WalkandRun/` — locomotion clips
- `src/assets/motions/g1/amp/Recovery/` — fall-recovery clips

NPZ files in those directories are auto-loaded by the task config at training startup.

## Training Curve Note

Around 20k iterations, the policy often suddenly acquires fall-recovery behavior, causing abrupt jumps in multiple logged metrics. This is expected behavior, not a training failure.
