#!/usr/bin/env bash
set -euo pipefail

# v61 plus a strict proprio-only free-board prone recovery pulse.
exec bash "$(dirname "${BASH_SOURCE[0]}")/play_a6_proprio_composite_v61.sh" \
  --proprio-prone-free-gate artifacts/a6_proprio_phase_detector/scene_id_prone_v1/prone_scene_shuffled_seed42_to44.pt \
  --prone-free-pulse-step 100 --prone-free-threshold 0.99 \
  --prone-free-consecutive 3 "$@"
