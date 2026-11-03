#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-proprio-v1}"
export PYTHONPATH="$project/src"

python_bin="${A6_PYTHON:-/home/czr/code/G1Recovery_mjlab/.venv/bin/python}"
flat_expert="${A6_FLAT_EXPERT:-logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt}"
output="${A6_OUTPUT:-logs/host_a6_proprio_v1_Gplus_4096_seed42_2000}"

if [[ ! -f "$flat_expert" ]]; then
  echo "Missing 564D flat29 expert: $flat_expert" >&2
  exit 2
fi
if [[ -e "$output/metrics.jsonl" ]]; then
  echo "Existing training output: $output; set a new A6_OUTPUT" >&2
  exit 2
fi

init_args=(--init-from29 "$flat_expert")
if [[ -n "${A6_RESUME:-}" ]]; then
  if [[ ! -f "$A6_RESUME" ]]; then
    echo "Missing proprioceptive checkpoint: $A6_RESUME" >&2
    exit 2
  fi
  init_args=(--resume "$A6_RESUME")
fi

exec "$python_bin" scripts/train_host.py \
  --task a6 --arm G+ --device cuda:0 \
  "${init_args[@]}" --a6-pose-v6 \
  --frozen-flat-teacher "$flat_expert" --flat-teacher-weight 0.5 \
  --a6-proprio-actor-v1 --a6-edge-curriculum --a6-staged-escape-v5 \
  --a6-focused-guided --a6-reset-mix diverse_v3 \
  --a6-progress-aux-v1 --a6-aligned-edge-v1 \
  --a6-knee-straight-prior-v1 --a6-footprint-edge-v1 \
  --a6-footprint-overlap-m 0.03 --a6-arm-extraction-v1 \
  --a6-prone-recovery-v1 --a6-bidirectional-recovery-v1 \
  --a6-episode-seconds 15 \
  --adaptive-kl --learning-rate 0.00001 --max-learning-rate 0.00005 \
  --entropy-coef 0.001 --action-std-max 0.5 --seed 42 \
  --num-envs "${A6_NUM_ENVS:-4096}" --steps-per-env "${A6_STEPS_PER_ENV:-96}" \
  --iterations "${A6_ITERATIONS:-2000}" --epochs 1 --minibatches 4 \
  --log-every 10 --save-every 50 --output "$output"
