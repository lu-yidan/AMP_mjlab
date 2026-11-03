#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
train_output="${A6_TRAIN_OUTPUT:-logs/host_a6_proprio_v1_Gplus_4096_seed42_2000}"
eval_output="${A6_EVAL_OUTPUT:-logs/host_a6_proprio_v1_eval}"
python_bin="${A6_PYTHON:-/home/czr/code/G1Recovery_mjlab/.venv/bin/python}"
export PYTHONPATH="$project/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-proprio-eval}"
export CUDA_VISIBLE_DEVICES="${A6_EVAL_GPU:-6}"
mkdir -p "$eval_output"

for iteration in 00049 00199 00499 00999 01499 01999; do
  checkpoint="$train_output/model_${iteration}.pt"
  while [[ ! -s "$checkpoint" ]]; do
    if ! pgrep -f "scripts/train_host.py.*--output $train_output" > /dev/null; then
      echo "Training stopped before $checkpoint appeared" >&2
      exit 1
    fi
    sleep 30
  done
  sleep 5
  "$python_bin" scripts/audit_a6_proprio.py --checkpoint "$checkpoint"
  "$python_bin" scripts/eval_a6_batch.py \
    --checkpoint "$checkpoint" \
    --output "$eval_output/model_${iteration}.json" \
    --device cuda:0 --seed 42 --steps 750 --per-scene 128 \
    --all-directions --knee-diagnostics \
    > "$eval_output/model_${iteration}.log" 2>&1
  "$python_bin" scripts/summarize_a6_proprio_eval.py \
    "$eval_output/model_${iteration}.json" \
    > "$eval_output/model_${iteration}.summary.txt"
  cat "$eval_output/model_${iteration}.summary.txt"
done
