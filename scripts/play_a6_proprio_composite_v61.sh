#!/usr/bin/env bash
set -euo pipefail

# v58 side escape gain plus a strict proprio-only flat-prone early pulse.
exec bash "$(dirname "${BASH_SOURCE[0]}")/play_a6_proprio_composite_v58.sh" \
  --proprio-prone-flat-gate artifacts/a6_proprio_phase_detector/scene_id_prone_v1/prone_scene_shuffled_seed42_to44.pt \
  --prone-flat-pulse-step 50 --prone-flat-threshold 0.99 \
  --prone-flat-consecutive 3 "$@"
