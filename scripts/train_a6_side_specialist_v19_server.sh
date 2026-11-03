#!/usr/bin/env bash
set -euo pipefail

# 75-update diagnostic branch; no checkpoint or optimizer state is overwritten.
export A6_RESUME="${A6_RESUME:-logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500/model_00374.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_specialist_v19_from374_seed42_450}"
export A6_ITERATIONS="${A6_ITERATIONS:-450}"
export A6_RESET_MIX=side_specialist_v1
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_v9_controller_center_512.npz}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_free_plate_v12_server.sh"
