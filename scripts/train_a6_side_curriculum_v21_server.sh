#!/usr/bin/env bash
set -euo pipefail

# Resume v20 after its first saved checkpoint, restarting the easy offset so
# both side directions must succeed before the guided plate moves inward.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_curriculum_v20_from374_seed42_525/model_00399.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_curriculum_v21_from399_seed42_550}"
export A6_ITERATIONS="${A6_ITERATIONS:-550}"
export A6_EDGE_RESTART_OFFSET_M=0.25
export A6_RESET_MIX=side_specialist_v1
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_v9_controller_center_512.npz}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_free_plate_v12_server.sh"
