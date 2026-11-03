#!/usr/bin/env bash
set -euo pipefail

# Expose genuinely centred guided-side resets (25% of side starts). The
# side-specific reward accepts either exit direction only when centred.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_axis_v29_from499_seed42_650/model_00549.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_center_v30_from549_seed42_700}"
export A6_ITERATIONS="${A6_ITERATIONS:-700}"
export A6_EDGE_RESTART_OFFSET_M="${A6_EDGE_RESTART_OFFSET_M:-0.25}"
export A6_RESET_MIX="${A6_RESET_MIX:-guided_side_specialist_v1}"
export A6_SIDE_DIRECTIONAL_PROGRESS_V1=1
export A6_SIDE_CENTER_BIDIRECTIONAL_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_free_plate_v12_server.sh"
