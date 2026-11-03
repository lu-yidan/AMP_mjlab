#!/usr/bin/env bash
set -euo pipefail

# Retest the guided-side curriculum after fixing the side-lying planar axis.
# Resume a proprio-only 738D actor before the old, unstable side curriculum.
export A6_RESUME="${A6_RESUME:-logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_axis_v29_from499_seed42_750}"
export A6_ITERATIONS="${A6_ITERATIONS:-750}"
export A6_EDGE_RESTART_OFFSET_M="${A6_EDGE_RESTART_OFFSET_M:-0.25}"
export A6_RESET_MIX="${A6_RESET_MIX:-guided_side_specialist_v1}"
export A6_SIDE_STAGED_BRIDGE_V1=1
export A6_SIDE_DIRECTIONAL_PROGRESS_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_bridge_v22_server.sh"
