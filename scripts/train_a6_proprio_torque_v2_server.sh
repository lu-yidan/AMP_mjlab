#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-proprio-torque-v2}"
export PYTHONPATH="$project/src"

python_bin="${A6_PYTHON:-/home/czr/code/G1Recovery_mjlab/.venv/bin/python}"
flat_expert="${A6_FLAT_EXPERT:-logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt}"
teacher="${A6_TEACHER:-logs/preflight_a6_v24_lrcap5e5_from2949_4096_seed42_100/model_02999.pt}"
output="${A6_OUTPUT:-logs/host_a6_proprio_torque_v2_Gplus_4096_seed42_100}"

for checkpoint in "$flat_expert" "$teacher"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 2
  fi
done
if [[ -e "$output/metrics.jsonl" ]]; then
  echo "Existing training output: $output; set a new A6_OUTPUT" >&2
  exit 2
fi

init_args=(--init-from29 "$flat_expert")
force_args=()
if [[ "${A6_TEACHER_FORCED_QUARTER:-0}" == 1 ]]; then
  force_args=(--a6-teacher-forced-quarter)
fi
bank_args=()
if [[ -n "${A6_PROPRIO_POSTESCAPE_BANK:-}" ]]; then
  if [[ ! -f "$A6_PROPRIO_POSTESCAPE_BANK" ]]; then
    echo "Missing postescape bank: $A6_PROPRIO_POSTESCAPE_BANK" >&2
    exit 2
  fi
  bank_args=(--a6-proprio-postescape-bank "$A6_PROPRIO_POSTESCAPE_BANK")
fi
controller_args=()
if [[ -n "${A6_PROPRIO_RECOVERY_CONTROLLER:-}" ]]; then
  if [[ ! -f "$A6_PROPRIO_RECOVERY_CONTROLLER" ]]; then
    echo "Missing proprio recovery controller: $A6_PROPRIO_RECOVERY_CONTROLLER" >&2
    exit 2
  fi
  controller_args=(--a6-proprio-recovery-controller "$A6_PROPRIO_RECOVERY_CONTROLLER")
fi
prone_exit_args=()
if [[ "${A6_PRONE_FLEXIBLE_EXIT_V1:-0}" == 1 ]]; then
  prone_exit_args=(--a6-prone-flexible-exit-v1)
fi
supine_recovery_args=()
if [[ "${A6_SUPINE_POSTESCAPE_FLAT_TEACHER_V1:-0}" == 1 ]]; then
  supine_recovery_args=(--a6-supine-postescape-flat-teacher-v1)
fi
scene_phase_args=()
if [[ "${A6_SCENE_PHASE_REWARD_V1:-0}" == 1 ]]; then
  scene_phase_args=(--a6-scene-phase-reward-v1)
fi
free_plate_args=()
if [[ "${A6_FREE_PLATE_MOTION_REWARD_V1:-0}" == 1 ]]; then
  free_plate_args=(--a6-free-plate-motion-reward-v1)
fi
flat_prone_args=()
if [[ "${A6_FLAT_PRONE_COMPACT_V1:-0}" == 1 ]]; then
  flat_prone_args=(--a6-flat-prone-compact-v1)
fi
side_bridge_args=()
if [[ "${A6_SIDE_BRIDGE_CURRICULUM_V1:-0}" == 1 ]]; then
  side_bridge_args=(--a6-side-bridge-curriculum-v1)
fi
if [[ "${A6_SIDE_STAGED_BRIDGE_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-side-staged-bridge-v1)
fi
if [[ "${A6_SIDE_LIFT_ANTICIPATION_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-side-lift-anticipation-v1)
fi
if [[ "${A6_SIDE_DIRECTIONAL_PROGRESS_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-side-directional-progress-v1)
fi
if [[ "${A6_SIDE_CENTER_BIDIRECTIONAL_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-side-center-bidirectional-v1)
fi
if [[ "${A6_SIDE_CENTER_ONLY_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-side-center-only-v1)
fi
if [[ "${A6_DISTILL_GUIDED_SIDE_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-distill-guided-side-v1)
fi
if [[ "${A6_TEACHER_FORCE_GUIDED_SIDE_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-teacher-force-guided-side-v1)
fi
side_rollin_args=()
if [[ -n "${A6_SIDE_ROLLIN_MAIN:-}" ]]; then
  side_rollin_args=(
    --a6-side-rollin-main "$A6_SIDE_ROLLIN_MAIN"
    --a6-side-rollin-min-step "${A6_SIDE_ROLLIN_MIN_STEP:-31}"
    --a6-side-rollin-max-step "${A6_SIDE_ROLLIN_MAX_STEP:-45}"
    --a6-side-student-scale "${A6_SIDE_STUDENT_SCALE:-0.5}"
  )
fi
if [[ "${A6_SIDE_PROGRESS_WEIGHTED_DISTILL_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-side-progress-weighted-distill-v1)
fi
if [[ "${A6_SIDE_FOCUS_ADVANTAGE_V1:-0}" == 1 ]]; then
  side_bridge_args+=(--a6-side-focus-advantage-v1)
fi
supine_boost_args=()
if [[ "${A6_GUIDED_SUPINE_TEACHER_BOOST_V1:-0}" == 1 ]]; then
  supine_boost_args=(--a6-guided-supine-teacher-boost-v1)
fi
free_anchor_args=()
if [[ "${A6_FREE_PLATE_ANCHOR_V1:-0}" == 1 ]]; then
  free_anchor_args=(--a6-free-plate-anchor-v1)
fi
parent_teacher_args=()
if [[ -n "${A6_FROZEN_PREESCAPE_PARENT:-}" ]]; then
  parent_teacher_args=(--a6-frozen-preescape-parent "$A6_FROZEN_PREESCAPE_PARENT")
fi
recovery_args=()
if [[ "${A6_STRONG_RECOVERY_V1:-0}" == 1 ]]; then
  recovery_args=(--a6-strong-recovery-v1)
fi
if [[ "${A6_RECOVERY_BRIDGE_V2:-0}" == 1 ]]; then
  recovery_args+=(--a6-recovery-bridge-v2)
fi
if [[ "${A6_RECOVERY_TEACHER_V1:-0}" == 1 ]]; then
  recovery_args+=(--a6-recovery-teacher-v1)
fi
if [[ -n "${A6_POSTESCAPE_STRIDE:-}" ]]; then
  recovery_args+=(--a6-postescape-stride "$A6_POSTESCAPE_STRIDE")
fi
transfer_args=()
edge_restart_args=()
if [[ -n "${A6_EDGE_RESTART_OFFSET_M:-}" ]]; then
  edge_restart_args=(--a6-edge-restart-offset-m "$A6_EDGE_RESTART_OFFSET_M")
fi
if [[ "${A6_ALLOW_MODEL_TRANSFER:-0}" == 1 ]]; then
  transfer_args+=(--allow-model-transfer)
fi
if [[ "${A6_RESET_OPTIMIZER:-0}" == 1 ]]; then
  transfer_args+=(--reset-optimizer)
fi
if [[ "${A6_KEEP_CURRICULUM_ON_TRANSFER:-0}" == 1 ]]; then
  transfer_args+=(--keep-curriculum-on-transfer)
fi
if [[ -n "${A6_RESUME:-}" ]]; then
  if [[ ! -f "$A6_RESUME" ]]; then
    echo "Missing torque-v2 checkpoint: $A6_RESUME" >&2
    exit 2
  fi
  init_args=(--resume "$A6_RESUME")
fi

exec "$python_bin" scripts/train_host.py \
  --task a6 --arm G+ --device cuda:0 \
  "${init_args[@]}" "${transfer_args[@]}" "${edge_restart_args[@]}" --a6-pose-v6 \
  --frozen-flat-teacher "$flat_expert" --flat-teacher-weight "${A6_FLAT_TEACHER_WEIGHT:-2.0}" \
  --a6-proprio-actor-v1 --a6-proprio-torque-v2 --a6-hard-mix-v1 \
  --a6-privileged-action-teacher "$teacher" \
  "${force_args[@]}" "${bank_args[@]}" "${recovery_args[@]}" "${controller_args[@]}" \
  "${prone_exit_args[@]}" "${supine_recovery_args[@]}" "${scene_phase_args[@]}" "${free_plate_args[@]}" "${flat_prone_args[@]}" "${side_bridge_args[@]}" "${side_rollin_args[@]}" "${supine_boost_args[@]}" "${free_anchor_args[@]}" "${parent_teacher_args[@]}" \
  --a6-distill-weight "${A6_DISTILL_WEIGHT:-0.5}" \
  --a6-edge-curriculum --a6-staged-escape-v5 \
  --a6-focused-guided --a6-reset-mix "${A6_RESET_MIX:-diverse_v3}" \
  --a6-progress-aux-v1 --a6-aligned-edge-v1 \
  --a6-knee-straight-prior-v1 --a6-footprint-edge-v1 \
  --a6-footprint-overlap-m 0.03 --a6-arm-extraction-v1 \
  --a6-prone-recovery-v1 --a6-bidirectional-recovery-v1 \
  --a6-episode-seconds 15 \
  --adaptive-kl --learning-rate "${A6_LEARNING_RATE:-0.00001}" \
  --max-learning-rate "${A6_MAX_LEARNING_RATE:-0.00005}" \
  --entropy-coef 0.001 --action-std-max 0.5 --seed 42 \
  --num-envs "${A6_NUM_ENVS:-4096}" --steps-per-env "${A6_STEPS_PER_ENV:-96}" \
  --iterations "${A6_ITERATIONS:-100}" --epochs 1 --minibatches 4 \
  --log-every 10 --save-every 25 --output "$output"
