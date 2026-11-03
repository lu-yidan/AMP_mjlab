#!/usr/bin/env bash
set -euo pipefail

cd /home/czr/code/G1Recovery_HoST
export CUDA_VISIBLE_DEVICES="${A6_GPU:-6}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-side-shadow}"
export PYTHONPATH="$PWD/src"
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
output="artifacts/a6_proprio_phase_detector/online/side_shadow_v1_seed${A6_SEED:-44}.json"
mkdir -p "$(dirname "$output")"

exec "$python" scripts/eval_a6_batch.py \
  --checkpoint logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt \
  --proprio-prone-specialist logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt \
  --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
  --proprio-supine-flat-gate artifacts/a6_proprio_phase_detector/recovery_gate_v1/supine_flat_recovery_gate_unweighted_v1.pt \
  --proprio-postescape-flat-teacher logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt \
  --all-directions --guided-offset-m 0 --steps 750 --per-scene 64 \
  --shadow-side-flexible-exit --seed "${A6_SEED:-44}" --device cuda:0 --output "$output"
