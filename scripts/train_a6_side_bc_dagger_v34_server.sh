#!/usr/bin/env bash
set -euo pipefail

# Recover from success-only BC distribution shift: student drives three
# quarters of guided-side worlds, receives online teacher corrections there,
# and is optimized by PPO. One quarter remains teacher-driven for coverage.
export A6_RESUME="${A6_RESUME:-artifacts/a6_proprio_phase_detector/side_success_bc_v1/from_v33_624.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_bc_dagger_v34_from624_seed42_725}"
export A6_ITERATIONS="${A6_ITERATIONS:-725}"
export A6_SIDE_CENTER_ONLY_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_teacher_v31_server.sh"
