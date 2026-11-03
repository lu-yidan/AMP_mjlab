#!/usr/bin/env bash
set -euo pipefail

export A6_STRONG_RECOVERY_V1=1
export A6_RESUME="${A6_RESUME:-logs/host_a6_proprio_recovery_v5_Gplus_4096_seed42_150/model_00099.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_proprio_recovery_strong_v6_Gplus_4096_seed42_200}"
export A6_ITERATIONS="${A6_ITERATIONS:-200}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_proprio_recovery_v5_server.sh"
