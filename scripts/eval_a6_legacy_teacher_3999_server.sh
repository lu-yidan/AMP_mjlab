#!/usr/bin/env bash
set -euo pipefail

cd /home/czr/code/G1Recovery_HoST
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/mpl-host-a6-legacy-3999
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
checkpoint="${A6_LEGACY_CHECKPOINT:-logs/host_a6_v26_allside_from2999_to3999_4096_seed42/model_03999.pt}"
out="${A6_LEGACY_EVAL_DIR:-artifacts/a6_proprio_phase_detector/online/legacy_teacher_3999}"
mkdir -p "$out"

for mode in guard noguard; do
  args=()
  if [[ "$mode" == noguard ]]; then
    args=(--disable-lift-guard)
  fi
  for seed in 44 48; do
    "$python" scripts/eval_a6_batch.py \
      --checkpoint "$checkpoint" --all-directions --guided-offset-m 0 \
      --per-scene 64 --steps 750 --seed "$seed" --device cuda:0 \
      "${args[@]}" --output "$out/${mode}_seed${seed}.json" \
      >"$out/${mode}_seed${seed}.log" 2>&1
  done
done
