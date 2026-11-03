#!/usr/bin/env bash
set -euo pipefail

# Continue the same 738D actor / four-critic PPO with a fixed proprio-only
# recovery controller. Its forced pulse steps are excluded from PPO ratio loss;
# all other steps update the one student actor normally.
export A6_PROPRIO_RECOVERY_CONTROLLER="${A6_PROPRIO_RECOVERY_CONTROLLER:-artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt}"
export A6_RESUME="${A6_RESUME:-logs/host_a6_proprio_recovery_bridge_v8b_Gplus_4096_seed42_300/model_00299.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500}"
export A6_ITERATIONS="${A6_ITERATIONS:-500}"
export A6_ALLOW_MODEL_TRANSFER=1
export A6_RESET_OPTIMIZER=1
export A6_KEEP_CURRICULUM_ON_TRANSFER=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_recovery_bridge_v8_server.sh"
