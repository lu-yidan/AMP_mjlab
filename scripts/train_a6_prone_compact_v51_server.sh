#!/usr/bin/env bash
set -euo pipefail

# A proprio-only prone candidate with many flat/free starts. The flat-prone
# travel cost is training-only; the actor still sees the same 738D history.
export A6_RESUME="${A6_RESUME:-logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_prone_compact_v51_from474_seed42_625}"
export A6_ITERATIONS="${A6_ITERATIONS:-625}"
export A6_RESET_MIX=prone_compact_v1
export A6_FLAT_PRONE_COMPACT_V1=1
export A6_DISTILL_WEIGHT="${A6_DISTILL_WEIGHT:-1.0}"
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_v9_controller_center_512.npz}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_free_plate_v12_server.sh"
