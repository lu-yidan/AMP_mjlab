#!/usr/bin/env bash
set -euo pipefail

# Train a stronger 612D board-aware teacher on centered guided-side worlds.
# This checkpoint is for supervision only: deployment still uses a 738D body
# and motor actor with no plate observations or privileged action switch.
cd /home/czr/code/G1Recovery_HoST
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/mpl-host-a6-privileged-side-v38
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
resume=logs/host_a6_v26_allside_from2999_to3999_4096_seed42/model_03999.pt
flat=logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt
guided=logs/host_a6_v17_episode15_long_from2399_4096_seed42_to3399/model_02599.pt
out="${A6_OUTPUT:-logs/host_a6_privileged_side_v38_from3999_seed42_4500}"
if [[ -e "$out/metrics.jsonl" ]]; then
  echo "Existing v38 output: $out" >&2
  exit 2
fi

exec "$python" scripts/train_host.py \
  --task a6 --arm G+ --device cuda:0 \
  --resume "$resume" --allow-model-transfer --reset-optimizer \
  --keep-curriculum-on-transfer --a6-edge-restart-offset-m 0.0 \
  --a6-pose-v6 --frozen-flat-teacher "$flat" --flat-teacher-weight 0.5 \
  --frozen-supine-teacher "$guided" --supine-teacher-weight 0.2 \
  --a6-edge-curriculum --a6-staged-escape-v5 --a6-focused-guided \
  --a6-reset-mix guided_side_specialist_v1 \
  --a6-safe-actions-v1 --a6-progress-aux-v1 --a6-aligned-edge-v1 \
  --a6-lateral-actions-v1 --a6-lift-guard-v1 --a6-knee-straight-prior-v1 \
  --a6-footprint-edge-v1 --a6-footprint-overlap-m 0.03 \
  --a6-arm-extraction-v1 --a6-prone-full-arms-v1 \
  --a6-supine-full-arms-v1 --a6-postescape-prone-replay-v1 \
  --a6-prone-recovery-v1 --a6-phase-history-reset-v1 \
  --a6-guided-teacher-v1 --a6-episode-seconds 15 \
  --a6-side-directional-progress-v1 --a6-side-center-bidirectional-v1 \
  --adaptive-kl --learning-rate 0.00001 --max-learning-rate 0.00005 \
  --entropy-coef 0.001 --action-std-max 0.5 --seed 42 \
  --num-envs 4096 --steps-per-env 96 \
  --iterations "${A6_ITERATIONS:-4500}" --epochs 1 --minibatches 4 \
  --log-every 10 --save-every 25 --output "$out"
