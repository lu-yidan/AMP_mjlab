#!/usr/bin/env bash
set -euo pipefail

project=/home/czr/code/G1Recovery_HoST
cd "$project"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-v26-allside}"
export PYTHONPATH="$project/src"

resume="${A6_RESUME:-logs/preflight_a6_v24_lrcap5e5_from2949_4096_seed42_100/model_02999.pt}"
output="${A6_OUTPUT:-logs/host_a6_v26_allside_from2999_to3999_4096_seed42}"
flat_teacher="${A6_FLAT_TEACHER:-logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt}"
guided_teacher="${A6_GUIDED_TEACHER:-logs/host_a6_v17_episode15_long_from2399_4096_seed42_to3399/model_02599.pt}"

if [[ -e "$output/metrics.jsonl" ]]; then
  echo "Training metrics already exist in $output; choose a new A6_OUTPUT to avoid duplicated iterations" >&2
  exit 2
fi

exec /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/train_host.py \
  --task a6 --arm G+ --device cuda:0 \
  --resume "$resume" --a6-pose-v6 \
  --frozen-flat-teacher "$flat_teacher" --flat-teacher-weight 0.5 \
  --frozen-supine-teacher "$guided_teacher" --supine-teacher-weight 0.2 \
  --a6-edge-curriculum --a6-staged-escape-v5 --a6-focused-guided \
  --a6-reset-mix diverse_v3 \
  --a6-safe-actions-v1 --a6-progress-aux-v1 --a6-aligned-edge-v1 \
  --a6-lateral-actions-v1 --a6-lift-guard-v1 --a6-knee-straight-prior-v1 \
  --a6-footprint-edge-v1 --a6-footprint-overlap-m 0.03 \
  --a6-arm-extraction-v1 --a6-prone-full-arms-v1 \
  --a6-supine-full-arms-v1 --a6-postescape-prone-replay-v1 \
  --a6-prone-recovery-v1 --a6-phase-history-reset-v1 \
  --a6-guided-teacher-v1 --a6-episode-seconds 15 \
  --adaptive-kl --learning-rate 0.00001 --max-learning-rate 0.00005 \
  --entropy-coef 0.001 --action-std-max 0.5 --seed 42 \
  --num-envs "${A6_NUM_ENVS:-4096}" --steps-per-env "${A6_STEPS_PER_ENV:-96}" \
  --iterations "${A6_ITERATIONS:-4000}" --epochs 1 --minibatches 4 \
  --log-every 10 --save-every 50 --output "$output"
