#!/usr/bin/env bash
set -euo pipefail

threshold="${1:?expected prone controller threshold}"
gpu="${2:?expected physical GPU index}"
case "$threshold" in 0.98|0.99) ;; *) echo "threshold must be 0.98 or 0.99" >&2; exit 2;; esac
case "$gpu" in 0|1|2|3|4|5|6) ;; *) echo "GPU index must be 0..6" >&2; exit 2;; esac
cd /home/czr/code/G1Recovery_HoST
export CUDA_VISIBLE_DEVICES="$gpu"
export MPLCONFIGDIR=/tmp/mpl-host-prone-threshold
export PYTHONPATH="$PWD/src"
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
for seed in 44 48; do
  output="artifacts/a6_proprio_phase_detector/online/prone_threshold_v1/v17_499_v13_449_t${threshold}_seed${seed}.json"
  mkdir -p "$(dirname "$output")"
  "$python" scripts/eval_a6_batch.py \
    --checkpoint logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt \
    --proprio-prone-specialist logs/host_a6_matching_replay_v13_from374_seed42_450/model_00449.pt \
    --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
    --diagnostic-controller-threshold "$threshold" \
    --proprio-supine-flat-gate artifacts/a6_proprio_phase_detector/recovery_gate_v1/supine_flat_recovery_gate_unweighted_v1.pt \
    --proprio-postescape-flat-teacher logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt \
    --guided-offset-m 0 --steps 750 --per-scene 64 \
    --seed "$seed" --device cuda:0 --output "$output" > "${output%.json}.log" 2>&1
done
