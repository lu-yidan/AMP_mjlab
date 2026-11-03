#!/usr/bin/env bash
set -euo pipefail

# Same parent and proprio actor as v31. All guided-side imitation and PPO
# samples use a centered plate, eliminating the hidden offset label conflict.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_axis_v29_from499_seed42_650/model_00549.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_center_teacher_v32_from549_seed42_700}"
export A6_ITERATIONS="${A6_ITERATIONS:-700}"
export A6_SIDE_CENTER_ONLY_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_teacher_v31_server.sh"
