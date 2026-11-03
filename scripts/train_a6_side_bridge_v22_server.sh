#!/usr/bin/env bash
set -euo pipefail

# Continue the 738D side actor from the saved v21 policy. Restart the plate at
# the easier 25 cm offset; the six-gate curriculum now also checks both
# half-offset side directions before moving inward. Other postures keep the
# original 2:1:1 difficulty mix.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_curriculum_v21_from399_seed42_550/model_00499.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_bridge_v22_from499_seed42_650}"
export A6_ITERATIONS="${A6_ITERATIONS:-650}"
export A6_EDGE_RESTART_OFFSET_M="${A6_EDGE_RESTART_OFFSET_M:-0.25}"
export A6_RESET_MIX="${A6_RESET_MIX:-side_specialist_v1}"
export A6_SIDE_BRIDGE_CURRICULUM_V1=1
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_v9_controller_center_512.npz}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_free_plate_v12_server.sh"
