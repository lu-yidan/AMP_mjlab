#!/usr/bin/env bash
set -euo pipefail

# Student actor remains 738D proprio-only; the 612D plate-aware teacher is
# queried only during training. Its side-lying action scaling is baked into
# 29-D imitation labels, so evaluation needs no privileged action switch.
export A6_RESUME="${A6_RESUME:-logs/host_a6_side_axis_v29_from499_seed42_650/model_00549.pt}"
export A6_OUTPUT="${A6_OUTPUT:-logs/host_a6_side_teacher_v31_from549_seed42_700}"
export A6_ITERATIONS="${A6_ITERATIONS:-700}"
export A6_DISTILL_GUIDED_SIDE_V1=1

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_a6_side_center_v30_server.sh"
