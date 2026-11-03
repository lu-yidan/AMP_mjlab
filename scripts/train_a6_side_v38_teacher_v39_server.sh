#!/usr/bin/env bash
set -euo pipefail

# Matched v37 comparison: keep the 738D proprio student, centred guided-side
# reset, PPO settings, and parent fixed; replace only the privileged teacher.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_axis_v29_from499_seed42_650/model_00549.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_v38teacher_v39_from549_seed42_700}"
export A6_ITERATIONS="${A6_ITERATIONS:-700}"
export A6_TEACHER=logs/host_a6_privileged_side_v38_from3999_seed42_4500/model_04499.pt

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_teacher_all_v33_server.sh"
