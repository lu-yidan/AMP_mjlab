#!/usr/bin/env bash
set -euo pipefail

weight="${1:?expected free-negative weight}"
gpu="${2:?expected physical GPU index}"
case "$weight" in 5|10) ;; *) echo "weight must be 5 or 10" >&2; exit 2;; esac
case "$gpu" in 0|1|2|3|4|5|6) ;; *) echo "GPU index must be 0..6" >&2; exit 2;; esac

cd /home/czr/code/G1Recovery_HoST
export MPLCONFIGDIR=/tmp/mpl-host-weighted-gate
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$PWD/src"
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python

for seed in 44 48; do
  output="artifacts/a6_proprio_phase_detector/online/free_specialist_v17/model_00499_gate_weight${weight}_seed${seed}.json"
  "$python" scripts/eval_a6_batch.py \
    --checkpoint logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt \
    --proprio-prone-specialist logs/host_a6_matching_replay_v13_from374_seed42_450/model_00449.pt \
    --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
    --proprio-supine-flat-gate "artifacts/a6_proprio_phase_detector/scene_id_v2/supine_safe_free_negative${weight}.pt" \
    --proprio-postescape-flat-teacher logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt \
    --guided-offset-m 0 --steps 750 --per-scene 64 \
    --seed "$seed" --device cuda:0 --output "$output" > "${output%.json}.log" 2>&1
done
