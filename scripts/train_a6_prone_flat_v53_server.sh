#!/usr/bin/env bash
set -euo pipefail

# Flat-prone candidate. Route it only after an independent proprio scene
# classifier is validated; keep the v18 expert for guided/free prone starts.
export A6_RESUME="${A6_RESUME:-logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_prone_flat_v53_from474_seed42_550}"
export A6_ITERATIONS="${A6_ITERATIONS:-550}"
export A6_RESET_MIX=prone_compact_v1
export A6_FLAT_PRONE_COMPACT_V1=0
export A6_DISTILL_WEIGHT="${A6_DISTILL_WEIGHT:-1.0}"
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_v9_controller_center_512.npz}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_free_plate_v12_server.sh"
