#!/usr/bin/env bash
set -euo pipefail

# Correct the v52 scale mismatch: physical teacher and student actions both
# match the v49 side controller's 0.5 post-scale of v33's raw safe-action label.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_teacher_all_v33_gpu4_from549_seed42_700/model_00624.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_rollin_v54_from624_seed42_725}"
export A6_ITERATIONS="${A6_ITERATIONS:-725}"
export A6_RESET_MIX=guided_side_specialist_v1
export A6_SIDE_CENTER_ONLY_V1=1
export A6_TEACHER_FORCE_GUIDED_SIDE_V1=0
export A6_DISTILL_WEIGHT="${A6_DISTILL_WEIGHT:-5.0}"
export A6_SIDE_ROLLIN_MAIN="${A6_SIDE_ROLLIN_MAIN:-logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt}"
export A6_SIDE_ROLLIN_MIN_STEP=31
export A6_SIDE_ROLLIN_MAX_STEP=45
export A6_SIDE_STUDENT_SCALE=0.5

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_teacher_v31_server.sh"
