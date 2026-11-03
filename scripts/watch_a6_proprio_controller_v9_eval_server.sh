#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
export PYTHONPATH="$project/src"
export CUDA_VISIBLE_DEVICES="${A6_EVAL_GPU:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-v9-eval}"

python_bin="${A6_PYTHON:-/home/czr/code/G1Recovery_mjlab/.venv/bin/python}"
train_output="${A6_TRAIN_OUTPUT:-logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500}"
eval_output="${A6_EVAL_OUTPUT:-logs/host_a6_proprio_controller_v9_eval_center}"
controller="${A6_PROPRIO_RECOVERY_CONTROLLER:-artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt}"
offset="${A6_EVAL_OFFSET:-0}"
seed="${A6_EVAL_SEED:-48}"
read -r -a iterations <<< "${A6_EVAL_ITERS:-00324 00374 00424 00499}"

if [[ ! -f "$controller" ]]; then
  echo "Missing controller: $controller" >&2
  exit 2
fi
mkdir -p "$eval_output"
for iteration in "${iterations[@]}"; do
  checkpoint="$train_output/model_${iteration}.pt"
  result="$eval_output/model_${iteration}.json"
  if [[ -f "$result" ]]; then
    echo "Already evaluated: $result"
    continue
  fi
  while [[ ! -f "$checkpoint" ]]; do
    sleep 5
  done
  echo "Evaluating $checkpoint with offset=$offset seed=$seed"
  "$python_bin" scripts/eval_a6_batch.py \
    --checkpoint "$checkpoint" \
    --proprio-recovery-controller "$controller" \
    --output "$result" --device cuda:0 --seed "$seed" \
    --steps 750 --per-scene 128 --all-directions \
    --guided-offset-m "$offset" \
    > "$eval_output/model_${iteration}.log" 2>&1
  echo "Finished $result"
done
