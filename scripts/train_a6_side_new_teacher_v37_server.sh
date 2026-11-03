#!/usr/bin/env bash
set -euo pipefail

# Matched v33 experiment: only the training-only 612D teacher changes from
# its 2999 checkpoint to the stronger 3999 checkpoint.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_axis_v29_from499_seed42_650/model_00549.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_new_teacher_v37_from549_seed42_700}"
export A6_ITERATIONS="${A6_ITERATIONS:-700}"
export A6_TEACHER=logs/host_a6_v26_allside_from2999_to3999_4096_seed42/model_03999.pt

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_teacher_all_v33_server.sh"
