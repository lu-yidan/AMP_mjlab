#!/usr/bin/env bash
set -euo pipefail

# Left-side-only fine-tuning. Initial positive IMU body-Y gravity selects this
# checkpoint during evaluation; the right side keeps the v33 parent.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_teacher_all_v33_gpu4_from549_seed42_700/model_00624.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_guided_left_v36_from624_seed42_775}"
export A6_ITERATIONS="${A6_ITERATIONS:-775}"
export A6_RESET_MIX=guided_left_side_specialist_v1
export A6_SIDE_CENTER_ONLY_V1=1
export A6_SIDE_PROGRESS_WEIGHTED_DISTILL_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_teacher_v31_server.sh"
