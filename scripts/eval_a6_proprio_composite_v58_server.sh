#!/usr/bin/env bash
set -euo pipefail

export A6_SIDE_LATERAL_GAIN=1.10
export A6_EVAL_NAME="${A6_EVAL_NAME:-v58_side_lateral_gain110}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_a6_proprio_composite_v49_server.sh"
