#!/usr/bin/env bash
set -euo pipefail

# Same guided-side scene quotas and rewards as v27. PPO assigns extra actor
# gradient weight to the medium/hard guided-side target advantage only.
export A6_RESUME="${A6_RESUME:-logs/host_a6_guided_side_v27_from624_seed42_800/model_00649.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_focus_v28_from649_seed42_825}"
export A6_ITERATIONS="${A6_ITERATIONS:-825}"
export A6_EDGE_RESTART_OFFSET_M=0.225
export A6_SIDE_FOCUS_ADVANTAGE_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_guided_side_v27_server.sh"
