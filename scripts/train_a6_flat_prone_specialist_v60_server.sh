#!/usr/bin/env bash
set -euo pipefail

# Test a separate flat-prone actor. The frozen ground teacher stays present
# for pose metadata but has negligible weight during the distribution shift.
export A6_RESUME="${A6_RESUME:-logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_flat_prone_specialist_v60_from474_seed42_525}"
export A6_ITERATIONS="${A6_ITERATIONS:-525}"
export A6_RESET_MIX=flat_prone_specialist_v1
export A6_FLAT_TEACHER_WEIGHT=0.01
export A6_DISTILL_WEIGHT="${A6_DISTILL_WEIGHT:-1.0}"
export A6_PROPRIO_POSTESCAPE_BANK="${A6_PROPRIO_POSTESCAPE_BANK:-assets/a6/host29/escaped_prone_v9_controller_center_512.npz}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_free_plate_v12_server.sh"
