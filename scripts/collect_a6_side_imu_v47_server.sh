#!/usr/bin/env bash
set -euo pipefail

cd /home/czr/code/G1Recovery_HoST
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/tmp/mpl-host-a6-side-imu-v47
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
out=artifacts/a6_proprio_phase_detector/side_scene_imu_v47
mkdir -p "$out"

for seed in 42 48; do
  "$python" scripts/collect_a6_phase_dataset.py \
    --checkpoint logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt \
    --teacher logs/preflight_a6_v24_lrcap5e5_from2949_4096_seed42_100/model_02999.pt \
    --balanced-side --shuffle-layout --steps 150 \
    --mirror-right-initial-side-main --capture-imu-accel \
    --guided-offset-m 0 --seed "$seed" --device cuda:0 \
    --output "$out/v17_mirror_imu_seed${seed}.npz" \
    >"$out/v17_mirror_imu_seed${seed}.log" 2>&1
done

"$python" scripts/diagnose_a6_scene_from_proprio.py \
  --train "$out/v17_mirror_imu_seed42.npz" \
  --valid "$out/v17_mirror_imu_seed48.npz" \
  --posture side --min-control-step 31 --max-control-step 45 \
  --include-imu-accel --epochs 60 --device cuda:0 \
  --output "$out/side_v17_mirror_imu_early31_45_seed48.json" \
  >"$out/side_v17_mirror_imu_early31_45_seed48.log" 2>&1
