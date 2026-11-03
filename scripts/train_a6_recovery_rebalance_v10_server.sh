#!/usr/bin/env bash
set -euo pipefail

# New branch from the fully trained v9 model. The actor remains proprio-only;
# simulator escape flags select training labels and rewards, never runtime input.
export A6_RESUME="${A6_RESUME:-logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500/model_00374.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_recovery_rebalance_v10_from374_seed42_400}"
export A6_ITERATIONS="${A6_ITERATIONS:-400}"
export A6_RESET_MIX="${A6_RESET_MIX:-diverse_v4_recovery}"
export A6_PRONE_FLEXIBLE_EXIT_V1="${A6_PRONE_FLEXIBLE_EXIT_V1:-1}"
export A6_SUPINE_POSTESCAPE_FLAT_TEACHER_V1="${A6_SUPINE_POSTESCAPE_FLAT_TEACHER_V1:-1}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_controller_v9_server.sh"
