#!/usr/bin/env bash
set -euo pipefail

# Same v22 parent and bridge curriculum. Only the guided-side lift reward
# changes, so this run isolates whether early lift discouragement helps.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_bridge_v22_from499_seed42_650/model_00524.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_lift_v23_from524_seed42_650}"
export A6_ITERATIONS="${A6_ITERATIONS:-650}"
export A6_EDGE_RESTART_OFFSET_M=0.225
export A6_SIDE_LIFT_ANTICIPATION_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_bridge_v22_server.sh"
