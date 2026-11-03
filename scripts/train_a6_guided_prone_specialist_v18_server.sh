#!/usr/bin/env bash
set -euo pipefail

# Continue the most promising matched postescape replay actor for 150 updates.
# This candidate is selected only for initial-prone resets from body IMU.
export A6_RESUME="${A6_RESUME:-logs/host_a6_matching_replay_v13_from374_seed42_450/model_00449.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_guided_prone_specialist_v18_from449_seed42_600}"
export A6_ITERATIONS="${A6_ITERATIONS:-600}"
export A6_RESET_MIX=guided_prone_specialist_v1
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_v9_controller_center_512.npz}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_free_plate_v12_server.sh"
