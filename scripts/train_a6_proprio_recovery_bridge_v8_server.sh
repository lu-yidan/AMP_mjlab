#!/usr/bin/env bash
set -euo pipefail

# Start recovery from early, phase-correct teacher states while allowing the
# confirmed postescape knee-fold and torso-tip needed for the first push-up.
export A6_RECOVERY_BRIDGE_V2=1
export A6_RECOVERY_TEACHER_V1=1
export A6_POSTESCAPE_STRIDE=2
export A6_STRONG_RECOVERY_V1=1
export A6_TEACHER_FORCED_QUARTER=1
export A6_DISTILL_WEIGHT="${A6_DISTILL_WEIGHT:-5.0}"
export A6_FLAT_TEACHER_WEIGHT="${A6_FLAT_TEACHER_WEIGHT:-10.0}"
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_legacy_v24_phase_mix10_15_20_center768.npz}"
export A6_RESUME="${A6_RESUME:-logs/host_a6_proprio_recovery_curriculum_v7_Gplus_4096_seed42_200/model_00199.pt}"
export A6_ALLOW_MODEL_TRANSFER=1
export A6_RESET_OPTIMIZER=1
export A6_KEEP_CURRICULUM_ON_TRANSFER=1
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_proprio_recovery_bridge_v8b_Gplus_4096_seed42_300}"
export A6_ITERATIONS="${A6_ITERATIONS:-300}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_torque_v2_server.sh"
