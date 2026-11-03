#!/usr/bin/env bash
set -euo pipefail

export A6_TEACHER_FORCED_QUARTER=1
export A6_DISTILL_WEIGHT="${A6_DISTILL_WEIGHT:-5.0}"
export A6_FLAT_TEACHER_WEIGHT="${A6_FLAT_TEACHER_WEIGHT:-10.0}"
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_proprio_v4_center256.npz}"
export A6_RESUME="${A6_RESUME:-logs/host_a6_proprio_torque_strong_v4_Gplus_4096_seed42_50/model_00049.pt}"
export A6_ALLOW_MODEL_TRANSFER=1
export A6_RESET_OPTIMIZER=1
export A6_KEEP_CURRICULUM_ON_TRANSFER=1
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_proprio_recovery_v5_Gplus_4096_seed42_150}"
export A6_ITERATIONS="${A6_ITERATIONS:-150}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_torque_v2_server.sh"
