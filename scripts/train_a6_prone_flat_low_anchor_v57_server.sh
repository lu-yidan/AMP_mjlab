#!/usr/bin/env bash
set -euo pipefail

# The v18-to-flat reset transfer exposes a large initial flat-teacher error.
# Keep its pose cue weak so PPO feedback can control the first updates.
export A6_RESUME="${A6_RESUME:-logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_prone_flat_low_anchor_v57_from474_seed42_525}"
export A6_ITERATIONS="${A6_ITERATIONS:-525}"
export A6_FLAT_TEACHER_WEIGHT=0.1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_prone_flat_v53_server.sh"
