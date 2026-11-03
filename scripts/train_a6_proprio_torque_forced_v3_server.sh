#!/usr/bin/env bash
set -euo pipefail

export A6_TEACHER_FORCED_QUARTER=1
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_proprio_torque_forced_v3_Gplus_4096_seed42_100}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_torque_v2_server.sh"
