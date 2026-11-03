#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
train_output="${A6_TRAIN_OUTPUT:-logs/host_a6_proprio_recovery_v5_Gplus_4096_seed42_150}"
eval_output="${A6_EVAL_OUTPUT:-logs/host_a6_proprio_recovery_v5_postescape_eval}"
bank="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_proprio_v4_center256.npz}"
python_bin="${A6_PYTHON:-/home/czr/code/G1Recovery_mjlab/.venv/bin/python}"
export PYTHONPATH="$project/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-proprio-recovery-eval}"
export CUDA_VISIBLE_DEVICES="${A6_EVAL_GPU:-5}"
mkdir -p "$eval_output"

for iteration in ${A6_EVAL_ITERS:-00074 00099 00149}; do
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
    --output "$eval_output/model_${iteration}_postescape.json" \
    --device cuda:0 --seed 42 --steps 750 --per-scene 128 \
    --all-postescape --postescape-bank "$bank" --postescape-live-obs-override \
    > "$eval_output/model_${iteration}_postescape.log" 2>&1
  "$python_bin" scripts/summarize_a6_proprio_eval.py \
    "$eval_output/model_${iteration}_postescape.json" \
    > "$eval_output/model_${iteration}_postescape.summary.txt"
  cat "$eval_output/model_${iteration}_postescape.summary.txt"
done
