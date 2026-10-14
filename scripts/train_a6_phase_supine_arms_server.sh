#!/usr/bin/env bash
set -euo pipefail

project=/home/czr/code/G1Recovery_HoST
cd "$project"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-postescape-replay}"
export PYTHONPATH="$project/src"
num_envs="${A6_NUM_ENVS:-4096}"
steps_per_env="${A6_STEPS_PER_ENV:-96}"
iterations="${A6_ITERATIONS:-3400}"
log_every="${A6_LOG_EVERY:-10}"
save_every="${A6_SAVE_EVERY:-50}"
output="${A6_OUTPUT:-logs/host_a6_v15_phase_supine_arms_restart0125_from2219_4096_seed42}"
learning_rate="${A6_LEARNING_RATE:-0.00001}"
max_learning_rate="${A6_MAX_LEARNING_RATE:-0.01}"
action_std_max="${A6_ACTION_STD_MAX:-0.5}"
epochs="${A6_EPOCHS:-1}"
resume="${A6_RESUME:-logs/host_a6_v14_phase_recovery_from2019_4096_seed42/model_02219.pt}"
supine_teacher="${A6_SUPINE_TEACHER_CHECKPOINT:-logs/host_a6_v10_prone_arms_from1599_4096_seed42_1800/model_01699.pt}"
supine_teacher_weight="${A6_SUPINE_TEACHER_WEIGHT:-0.5}"
if [[ -s "$output/metrics.jsonl" && -z "${A6_RESUME:-}" ]]; then
  echo "Existing run at $output; set A6_RESUME to its latest checkpoint before continuing" >&2
  exit 2
fi
edge_restart_args=()
if [[ -n "${A6_EDGE_RESTART_OFFSET:-}" ]]; then
  edge_restart_args=(--a6-edge-restart-offset-m "$A6_EDGE_RESTART_OFFSET")
fi
reset_optimizer_args=()
if [[ "${A6_RESET_OPTIMIZER:-0}" == 1 ]]; then
  reset_optimizer_args=(--reset-optimizer)
fi
recovery_args=()
if [[ "${A6_BIDIRECTIONAL_RECOVERY:-0}" == 1 ]]; then
  recovery_args=(--a6-bidirectional-recovery-v1)
fi
adaptive_kl_args=()
if [[ "${A6_ADAPTIVE_KL:-1}" == 1 ]]; then
  adaptive_kl_args=(--adaptive-kl)
fi
guided_teacher_args=()
if [[ "${A6_GUIDED_TEACHER:-0}" == 1 ]]; then
  guided_teacher_args=(--a6-guided-teacher-v1)
fi

exec /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/train_host.py \
  --task a6 --arm G+ --device cuda:0 \
  --resume "$resume" "${reset_optimizer_args[@]}" \
  --allow-model-transfer --keep-curriculum-on-transfer \
  --a6-pose-v6 \
  --frozen-flat-teacher logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt \
  --frozen-supine-teacher "$supine_teacher" \
  --supine-teacher-weight "$supine_teacher_weight" \
  "${edge_restart_args[@]}" \
  --a6-edge-curriculum --a6-staged-escape-v5 --a6-focused-guided \
  --a6-safe-actions-v1 --a6-progress-aux-v1 --a6-aligned-edge-v1 \
  --a6-lateral-actions-v1 --a6-lift-guard-v1 --a6-knee-straight-prior-v1 \
  --a6-footprint-edge-v1 --a6-footprint-overlap-m 0.03 \
  --a6-arm-extraction-v1 --a6-prone-full-arms-v1 \
  --a6-supine-full-arms-v1 \
  --a6-postescape-prone-replay-v1 \
  --a6-prone-recovery-v1 \
  "${recovery_args[@]}" \
  "${guided_teacher_args[@]}" \
  --a6-phase-history-reset-v1 \
  --a6-episode-seconds "${A6_EPISODE_SECONDS:-10}" \
  "${adaptive_kl_args[@]}" --learning-rate "$learning_rate" \
  --max-learning-rate "$max_learning_rate" \
  --entropy-coef 0.001 --action-std-max "$action_std_max" --seed 42 \
  --num-envs "$num_envs" --steps-per-env "$steps_per_env" --iterations "$iterations" \
  --epochs "$epochs" --minibatches 4 --log-every "$log_every" \
  --save-every "$save_every" \
  --output "$output"
