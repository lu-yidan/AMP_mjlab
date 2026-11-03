#!/usr/bin/env bash
set -euo pipefail

export A6_TEACHER_FORCED_QUARTER=1
export A6_DISTILL_WEIGHT="${A6_DISTILL_WEIGHT:-5.0}"
export A6_FLAT_TEACHER_WEIGHT="${A6_FLAT_TEACHER_WEIGHT:-10.0}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_proprio_torque_strong_v4_Gplus_4096_seed42_50}"
export A6_ITERATIONS="${A6_ITERATIONS:-50}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_torque_v2_server.sh"
