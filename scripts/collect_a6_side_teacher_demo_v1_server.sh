#!/usr/bin/env bash
set -euo pipefail

cd /home/czr/code/G1Recovery_HoST
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/mpl-host-a6-side-demo-v1
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
student=logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt
teacher=logs/preflight_a6_v24_lrcap5e5_from2949_4096_seed42_100/model_02999.pt
out=artifacts/a6_proprio_phase_detector/side_demo_v1
mkdir -p "$out"

for seed in 42 50; do
  "$python" scripts/collect_a6_guided_side_teacher_demo.py \
    --student "$student" --teacher "$teacher" \
    --seed "$seed" --per-side 256 --steps 750 --sample-stride 5 \
    --device cuda:0 --output "$out/teacher_center_seed${seed}.npz" \
    >"$out/teacher_center_seed${seed}.log" 2>&1
done
