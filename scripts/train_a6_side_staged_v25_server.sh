#!/usr/bin/env bash
set -euo pipefail

# Same actor/reward as v22, with a physical intermediate plate position:
# side tiers 1.0, 0.75, 0.5 times the edge offset; no centred side starts yet.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_bridge_v22_from499_seed42_650/model_00574.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_staged_v25_from574_seed42_725}"
export A6_ITERATIONS="${A6_ITERATIONS:-725}"
export A6_EDGE_RESTART_OFFSET_M=0.225
export A6_SIDE_STAGED_BRIDGE_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_bridge_v22_server.sh"
