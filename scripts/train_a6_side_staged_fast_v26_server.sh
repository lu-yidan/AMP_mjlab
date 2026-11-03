#!/usr/bin/env bash
set -euo pipefail

# Continue the same v25 staged-side distribution and reward from its 624th
# checkpoint. Adaptive KL may now raise the learning rate above 5e-5.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_staged_v25_from574_seed42_725/model_00624.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_staged_fast_v26_from624_seed42_750}"
export A6_ITERATIONS="${A6_ITERATIONS:-750}"
export A6_EDGE_RESTART_OFFSET_M=0.225
export A6_LEARNING_RATE=0.00005
export A6_MAX_LEARNING_RATE=0.0002

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_staged_v25_server.sh"
