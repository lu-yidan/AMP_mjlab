#!/usr/bin/env bash
set -euo pipefail

# Side actor with 2560/4096 worlds devoted to fixed-board side starts. The
# actor input remains 738D proprio; board labels are training-only reset data.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_staged_v25_from574_seed42_725/model_00624.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_guided_side_v27_from624_seed42_800}"
export A6_ITERATIONS="${A6_ITERATIONS:-800}"
export A6_EDGE_RESTART_OFFSET_M=0.225
export A6_RESET_MIX=guided_side_specialist_v1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_staged_v25_server.sh"
