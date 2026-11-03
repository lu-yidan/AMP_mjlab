#!/usr/bin/env bash
set -euo pipefail

# Matched with v33: same parent, centred guided sides and teacher control.
# Only safe-progress weighting of imitation examples changes.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_axis_v29_from499_seed42_650/model_00549.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_progress_teacher_v35_from549_seed42_700}"
export A6_ITERATIONS="${A6_ITERATIONS:-700}"
export A6_SIDE_PROGRESS_WEIGHTED_DISTILL_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_teacher_all_v33_server.sh"
