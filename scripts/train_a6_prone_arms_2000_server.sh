#!/usr/bin/env bash
set -euo pipefail

project=/home/czr/code/G1Recovery_HoST
cd "$project"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-prone-arms}"
export PYTHONPATH="$project/src"
num_envs="${A6_NUM_ENVS:-4096}"
steps_per_env="${A6_STEPS_PER_ENV:-96}"
iterations="${A6_ITERATIONS:-3400}"
log_every="${A6_LOG_EVERY:-10}"
save_every="${A6_SAVE_EVERY:-100}"
output="${A6_OUTPUT:-logs/host_a6_v10_prone_arms_from1599_4096_seed42_1800}"
learning_rate="${A6_LEARNING_RATE:-0.00005}"
epochs="${A6_EPOCHS:-5}"
resume="${A6_RESUME:-logs/host_a6_v9_arm_footprint_from1399_4096_seed42_2000/model_01599.pt}"

exec /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/train_host.py \
  --task a6 --arm G+ --device cuda:0 \
  --resume "$resume" \
  --allow-model-transfer --keep-curriculum-on-transfer \
  --a6-pose-v6 \
  --frozen-flat-teacher logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt \
  --a6-edge-curriculum --a6-staged-escape-v5 --a6-focused-guided \
  --a6-safe-actions-v1 --a6-progress-aux-v1 --a6-aligned-edge-v1 \
  --a6-lateral-actions-v1 --a6-lift-guard-v1 --a6-knee-straight-prior-v1 \
  --a6-footprint-edge-v1 --a6-footprint-overlap-m 0.03 \
  --a6-arm-extraction-v1 --a6-prone-full-arms-v1 \
  --adaptive-kl --learning-rate "$learning_rate" --seed 42 \
  --num-envs "$num_envs" --steps-per-env "$steps_per_env" --iterations "$iterations" \
  --epochs "$epochs" --minibatches 4 --log-every "$log_every" \
  --save-every "$save_every" \
  --output "$output"
