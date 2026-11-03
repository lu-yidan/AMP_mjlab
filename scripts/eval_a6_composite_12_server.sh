#!/usr/bin/env bash
set -euo pipefail

cd /home/czr/code/G1Recovery_HoST
export CUDA_VISIBLE_DEVICES="${A6_GPU:-3}"
export MPLCONFIGDIR=/tmp/mpl-host-composite-12
export PYTHONPATH="$PWD/src"
python=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
checkpoint="${A6_MAIN_CHECKPOINT:-logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt}"
prone="${A6_PRONE_CHECKPOINT:-logs/host_a6_matching_replay_v13_from374_seed42_450/model_00449.pt}"
side_args=()
if [[ -n "${A6_SIDE_CHECKPOINT:-}" ]]; then
  side_args=(--proprio-side-specialist "$A6_SIDE_CHECKPOINT")
fi
if [[ -n "${A6_LEFT_SIDE_CHECKPOINT:-}" ]]; then
  side_args+=(--proprio-left-side-specialist "$A6_LEFT_SIDE_CHECKPOINT")
fi
if [[ "${A6_MIRROR_LEFT_SIDE:-0}" == 1 ]]; then
  side_args+=(--mirror-left-side-specialist)
fi
if [[ "${A6_MIRROR_RIGHT_MAIN:-0}" == 1 ]]; then
  side_args+=(--mirror-right-initial-side-main)
fi
if [[ "${A6_SIDE_SOFT_GATE_V1:-0}" == 1 ]]; then
  side_args+=(
    --side-soft-gate-v1
    --side-soft-gate-low "${A6_SIDE_SOFT_GATE_LOW:-0.25}"
    --side-soft-gate-high "${A6_SIDE_SOFT_GATE_HIGH:-0.85}"
  )
fi
if [[ -n "${A6_GUIDED_SIDE_GATE:-}" ]]; then
  if [[ -z "${A6_SIDE_CHECKPOINT:-}" ]]; then
    echo "A6_GUIDED_SIDE_GATE requires A6_SIDE_CHECKPOINT" >&2
    exit 2
  fi
  side_args+=(
    --proprio-guided-side-gate "$A6_GUIDED_SIDE_GATE"
    --guided-side-threshold "${A6_GUIDED_SIDE_THRESHOLD:-0.95}"
    --guided-side-consecutive "${A6_GUIDED_SIDE_CONSECUTIVE:-10}"
    --guided-side-ema-alpha "${A6_GUIDED_SIDE_EMA_ALPHA:-1.0}"
    --guided-side-min-steps "${A6_GUIDED_SIDE_MIN_STEPS:-35}"
    --guided-side-max-steps "${A6_GUIDED_SIDE_MAX_STEPS:-120}"
  )
  if [[ -n "${A6_GUIDED_SIDE_EARLY_GATE:-}" ]]; then
    side_args+=(
      --proprio-guided-side-early-gate "$A6_GUIDED_SIDE_EARLY_GATE"
      --guided-side-early-max-step "${A6_GUIDED_SIDE_EARLY_MAX_STEP:-45}"
    )
  fi
  if [[ -n "${A6_GUIDED_RIGHT_SIDE_GATE:-}" ]]; then
    side_args+=(
      --proprio-guided-right-side-gate "$A6_GUIDED_RIGHT_SIDE_GATE"
      --guided-right-side-min-steps "${A6_GUIDED_RIGHT_SIDE_MIN_STEPS:-35}"
      --guided-right-side-max-steps "${A6_GUIDED_RIGHT_SIDE_MAX_STEPS:-120}"
    )
  fi
fi
if [[ -n "${A6_GUIDED_PRONE_GATE:-}" || -n "${A6_GUIDED_PRONE_CHECKPOINT:-}" ]]; then
  if [[ -z "${A6_GUIDED_PRONE_GATE:-}" || -z "${A6_GUIDED_PRONE_CHECKPOINT:-}" ]]; then
    echo "Guided-prone gate and checkpoint must be supplied together" >&2
    exit 2
  fi
  side_args+=(
    --proprio-guided-prone-gate "$A6_GUIDED_PRONE_GATE"
    --proprio-guided-prone-specialist "$A6_GUIDED_PRONE_CHECKPOINT"
    --guided-prone-threshold "${A6_GUIDED_PRONE_THRESHOLD:-0.95}"
    --guided-prone-consecutive "${A6_GUIDED_PRONE_CONSECUTIVE:-5}"
  )
fi
name="${A6_EVAL_NAME:-v17_499_v13_449_gate_old}"
steps="${A6_STEPS:-750}"
pulse_args=()
if [[ -n "${A6_RECOVERY_BLEND_STEPS:-}" ]]; then
  pulse_args=(--recovery-pulse-blend-steps "$A6_RECOVERY_BLEND_STEPS")
fi
if [[ -n "${A6_DIAG_FLAT_EARLY_PULSE_STEP:-}" ]]; then
  pulse_args+=(--diagnostic-flat-prone-early-pulse-step "$A6_DIAG_FLAT_EARLY_PULSE_STEP")
fi
if [[ -n "${A6_DIAG_FREE_EARLY_PULSE_STEP:-}" ]]; then
  pulse_args+=(--diagnostic-free-prone-early-pulse-step "$A6_DIAG_FREE_EARLY_PULSE_STEP")
fi
if [[ -n "${A6_PRONE_FLAT_GATE:-}" ]]; then
  pulse_args+=(
    --proprio-prone-flat-gate "$A6_PRONE_FLAT_GATE"
    --prone-flat-pulse-step "${A6_PRONE_FLAT_PULSE_STEP:-50}"
    --prone-flat-threshold "${A6_PRONE_FLAT_THRESHOLD:-0.95}"
    --prone-flat-consecutive "${A6_PRONE_FLAT_CONSECUTIVE:-2}"
  )
fi
if [[ -n "${A6_PRONE_FREE_GATE:-}" ]]; then
  pulse_args+=(
    --proprio-prone-free-gate "$A6_PRONE_FREE_GATE"
    --prone-free-pulse-step "${A6_PRONE_FREE_PULSE_STEP:-100}"
    --prone-free-threshold "${A6_PRONE_FREE_THRESHOLD:-0.99}"
    --prone-free-consecutive "${A6_PRONE_FREE_CONSECUTIVE:-3}"
  )
fi
if [[ "${A6_DIAG_FLAT_AFTER_PRONE_PULSE:-0}" == 1 ]]; then
  pulse_args+=(--diagnostic-flat-expert-after-prone-pulse)
fi
if [[ -n "${A6_SIDE_NEUTRAL:-}" ]]; then
  side_args+=(--side-neutral-through-step "$A6_SIDE_NEUTRAL")
fi
if [[ -n "${A6_SIDE_ACTION_SCALE:-}" ]]; then
  side_args+=(--side-action-scale "$A6_SIDE_ACTION_SCALE")
fi
if [[ -n "${A6_SIDE_LATERAL_GAIN:-}" ]]; then
  side_args+=(--side-lateral-gain "$A6_SIDE_LATERAL_GAIN")
fi
if [[ -n "${A6_DIAG_GUIDED_NONSIDE_LATERAL_GAIN:-}" ]]; then
  side_args+=(--diagnostic-guided-nonside-lateral-gain "$A6_DIAG_GUIDED_NONSIDE_LATERAL_GAIN")
fi
if [[ -n "${A6_DIAG_GUIDED_NONSIDE_SAFE_SCALE:-}" ]]; then
  side_args+=(--diagnostic-guided-nonside-safe-scale "$A6_DIAG_GUIDED_NONSIDE_SAFE_SCALE")
fi
if [[ -n "${A6_SIDE_SAFE_THROUGH:-}" ]]; then
  side_args+=(--initial-side-joint-safe-through-step "$A6_SIDE_SAFE_THROUGH")
fi
if [[ -n "${A6_SIDE_SAFE_SCALE:-}" ]]; then
  side_args+=(--initial-side-joint-safe-scale "$A6_SIDE_SAFE_SCALE")
fi
if [[ -n "${A6_SIDE_HANDOFF_STEPS:-}" ]]; then
  side_args+=(--side-handoff-steps "$A6_SIDE_HANDOFF_STEPS")
fi
if [[ "${A6_SIDE_SAFE_GATE_ONLY:-0}" == 1 ]]; then
  side_args+=(--side-safe-only-when-guided-gate)
fi
if [[ -n "${A6_SIDE_SAFE_PROBE_THROUGH:-}" ]]; then
  side_args+=(--side-safe-probe-through-step "$A6_SIDE_SAFE_PROBE_THROUGH")
fi
if [[ -n "${A6_LEFT_SIDE_EXPERT_PROBE_THROUGH:-}" ]]; then
  side_args+=(--left-side-specialist-probe-through-step "$A6_LEFT_SIDE_EXPERT_PROBE_THROUGH")
fi

for seed in ${A6_SEEDS:-44 48}; do
  output="artifacts/a6_proprio_phase_detector/online/composite_12/${name}_seed${seed}.json"
  mkdir -p "$(dirname "$output")"
  "$python" scripts/eval_a6_batch.py \
    --checkpoint "$checkpoint" \
    --proprio-prone-specialist "$prone" \
    "${side_args[@]}" \
    --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
    "${pulse_args[@]}" \
    --proprio-supine-flat-gate artifacts/a6_proprio_phase_detector/recovery_gate_v1/supine_flat_recovery_gate_unweighted_v1.pt \
    --proprio-postescape-flat-teacher logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt \
    --all-directions --guided-offset-m 0 --steps "$steps" --per-scene 64 \
    --seed "$seed" --device cuda:0 --output "$output" > "${output%.json}.log" 2>&1
done
