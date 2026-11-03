#!/usr/bin/env bash
set -euo pipefail

cd /home/czr/code/G1Recovery_HoST
export CUDA_VISIBLE_DEVICES="${A6_GPU:-4}"
export MPLCONFIGDIR=/tmp/mpl-host-guided-prone-gate
export PYTHONPATH="$PWD/src"
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
streak="${A6_GUIDED_PRONE_STREAK:-10}"
threshold="${A6_GUIDED_PRONE_THRESHOLD:-0.99}"
min_steps="${A6_GUIDED_PRONE_MIN_STEPS:-40}"
max_steps="${A6_GUIDED_PRONE_MAX_STEPS:-120}"
gate_path="${A6_GUIDED_PRONE_GATE_PATH:-artifacts/a6_proprio_phase_detector/scene_id_prone_v1/prone_scene_seed42_to44.pt}"
gate_name="${A6_GUIDED_PRONE_GATE_NAME:-ordered}"
for seed in 44 48; do
  output="artifacts/a6_proprio_phase_detector/online/guided_prone_gate_v1/v17_499_v18_474to549_${gate_name}_t${threshold}_k${streak}_${min_steps}to${max_steps}_seed${seed}.json"
  mkdir -p "$(dirname "$output")"
  "$python" scripts/eval_a6_batch.py \
    --checkpoint logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt \
    --proprio-prone-specialist logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt \
    --proprio-guided-prone-specialist logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00549.pt \
    --proprio-guided-prone-gate "$gate_path" \
    --guided-prone-threshold "$threshold" --guided-prone-consecutive "$streak" \
    --guided-prone-min-steps "$min_steps" --guided-prone-max-steps "$max_steps" \
    --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
    --proprio-supine-flat-gate artifacts/a6_proprio_phase_detector/recovery_gate_v1/supine_flat_recovery_gate_unweighted_v1.pt \
    --proprio-postescape-flat-teacher logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt \
    --all-directions --guided-offset-m 0 --steps 750 --per-scene 64 \
    --seed "$seed" --device cuda:0 --output "$output" > "${output%.json}.log" 2>&1
done
