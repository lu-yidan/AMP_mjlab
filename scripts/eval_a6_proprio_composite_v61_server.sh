#!/usr/bin/env bash
set -euo pipefail

export A6_PRONE_FLAT_GATE=artifacts/a6_proprio_phase_detector/scene_id_prone_v1/prone_scene_shuffled_seed42_to44.pt
export A6_PRONE_FLAT_PULSE_STEP=50
export A6_PRONE_FLAT_THRESHOLD=0.99
export A6_PRONE_FLAT_CONSECUTIVE=3
export A6_EVAL_NAME="${A6_EVAL_NAME:-v61_sidegain110_flatpulse50}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_a6_proprio_composite_v58_server.sh"
