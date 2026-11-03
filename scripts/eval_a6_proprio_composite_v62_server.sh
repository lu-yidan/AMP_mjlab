#!/usr/bin/env bash
set -euo pipefail

export A6_PRONE_FREE_GATE=artifacts/a6_proprio_phase_detector/scene_id_prone_v1/prone_scene_shuffled_seed42_to44.pt
export A6_PRONE_FREE_PULSE_STEP=100
export A6_PRONE_FREE_THRESHOLD=0.99
export A6_PRONE_FREE_CONSECUTIVE=3
export A6_EVAL_NAME="${A6_EVAL_NAME:-v62_prone_scene_pulses}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_a6_proprio_composite_v61_server.sh"
