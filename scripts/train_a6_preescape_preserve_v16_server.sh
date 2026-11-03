#!/usr/bin/env bash
set -euo pipefail

# Contrast v14's destructive shared-actor transfer with frozen preescape labels.
parent=logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500/model_00374.pt
export A6_RESUME="${A6_RESUME:-$parent}"
export A6_FROZEN_PREESCAPE_PARENT="${A6_FROZEN_PREESCAPE_PARENT:-$parent}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_preescape_preserve_v16_from374_seed42_450}"
export A6_ITERATIONS="${A6_ITERATIONS:-450}"
export A6_RESET_MIX="${A6_RESET_MIX:-diverse_v4_recovery}"
export A6_PRONE_FLEXIBLE_EXIT_V1=1
export A6_SUPINE_POSTESCAPE_FLAT_TEACHER_V1=1
export A6_GUIDED_SUPINE_TEACHER_BOOST_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_controller_v9_server.sh"
