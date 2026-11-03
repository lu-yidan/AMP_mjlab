#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
train_output="${A6_TRAIN_OUTPUT:-logs/host_a6_proprio_hardmix_v1_Gplus_4096_seed42_300}"
eval_output="${A6_EVAL_OUTPUT:-logs/host_a6_proprio_hardmix_v1_eval}"
python_bin="${A6_PYTHON:-/home/czr/code/G1Recovery_mjlab/.venv/bin/python}"
export PYTHONPATH="$project/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-proprio-hardmix-eval}"
export CUDA_VISIBLE_DEVICES="${A6_EVAL_GPU:-6}"
mkdir -p "$eval_output"

for iteration in ${A6_EVAL_ITERS:-00149 00199 00299}; do
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
  for entry in '0:offset000' '0.25:offset025'; do
    offset="${entry%%:*}"
    suffix="${entry#*:}"
    "$python_bin" scripts/eval_a6_batch.py \
      --checkpoint "$checkpoint" \
      --output "$eval_output/model_${iteration}_${suffix}.json" \
      --device cuda:0 --seed 42 --steps 750 --per-scene 128 \
      --all-directions --knee-diagnostics --guided-offset-m "$offset" \
      > "$eval_output/model_${iteration}_${suffix}.log" 2>&1
    "$python_bin" scripts/summarize_a6_proprio_eval.py \
      "$eval_output/model_${iteration}_${suffix}.json" \
      > "$eval_output/model_${iteration}_${suffix}.summary.txt"
    cat "$eval_output/model_${iteration}_${suffix}.summary.txt"
  done
done
