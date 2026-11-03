#!/usr/bin/env bash
set -euo pipefail

# Isolate the replay-bank distribution against the v10 control group.
export A6_RESUME="${A6_RESUME:-logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500/model_00374.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_matching_replay_v13_from374_seed42_450}"
export A6_ITERATIONS="${A6_ITERATIONS:-450}"
export A6_RESET_MIX="${A6_RESET_MIX:-diverse_v4_recovery}"
export A6_PRONE_FLEXIBLE_EXIT_V1=1
export A6_SUPINE_POSTESCAPE_FLAT_TEACHER_V1=1
export A6_PROPRIO_POSTESCAPE_BANK=assets/a6/host29/escaped_prone_v9_controller_center_512.npz

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_controller_v9_server.sh"
