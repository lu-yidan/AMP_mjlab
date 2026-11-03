#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
export PYTHONPATH="$project/src"
export MPLCONFIGDIR=/tmp/mpl-host-a6-side-scene-v1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
checkpoint=logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt
teacher=logs/preflight_a6_v24_lrcap5e5_from2949_4096_seed42_100/model_02999.pt
out=artifacts/a6_proprio_phase_detector/side_scene_v1
mkdir -p "$out"

for seed in 42 48; do
  "$python" scripts/collect_a6_phase_dataset.py \
    --checkpoint "$checkpoint" --teacher "$teacher" \
    --balanced-side --shuffle-layout --steps 150 \
    --guided-offset-m 0 --seed "$seed" --device cuda:0 \
    --output "$out/v17_seed${seed}.npz" \
    >"$out/v17_seed${seed}.log" 2>&1
done

"$python" scripts/diagnose_a6_scene_from_proprio.py \
  --train "$out/v17_seed42.npz" --valid "$out/v17_seed48.npz" \
  --posture side --epochs 40 --device cuda:0 \
  --output "$out/side_v17_seed48.json" \
  --save-model "$out/side_v17_seed48.pt" \
  >"$out/side_v17_seed48.log" 2>&1
