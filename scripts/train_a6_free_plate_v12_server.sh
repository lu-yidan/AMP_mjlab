#!/usr/bin/env bash
set -euo pipefail

# Isolate movable-board motion against v11, with the same parent and reset mix.
export A6_RESUME="${A6_RESUME:-logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500/model_00374.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_free_plate_v12_from374_seed42_450}"
export A6_ITERATIONS="${A6_ITERATIONS:-450}"
export A6_RESET_MIX="${A6_RESET_MIX:-diverse_v4_recovery}"
export A6_SCENE_PHASE_REWARD_V1=1
export A6_FREE_PLATE_MOTION_REWARD_V1=1
export A6_PRONE_FLEXIBLE_EXIT_V1=1
export A6_SUPINE_POSTESCAPE_FLAT_TEACHER_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_controller_v9_server.sh"
