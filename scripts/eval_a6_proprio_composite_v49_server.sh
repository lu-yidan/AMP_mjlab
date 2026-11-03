#!/usr/bin/env bash
set -euo pipefail

export A6_MAIN_CHECKPOINT="${A6_MAIN_CHECKPOINT:-logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt}"
export A6_PRONE_CHECKPOINT="${A6_PRONE_CHECKPOINT:-logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt}"
export A6_SIDE_CHECKPOINT="${A6_SIDE_CHECKPOINT:-logs/host_a6_side_teacher_all_v33_gpu4_from549_seed42_700/model_00624.pt}"
export A6_GUIDED_SIDE_GATE=artifacts/a6_proprio_phase_detector/side_scene_canonical_v48/side_early31_45_seed48.pt
export A6_GUIDED_RIGHT_SIDE_GATE=artifacts/a6_proprio_phase_detector/side_scene_mirror_v44/side_v17_mirror_seed48.pt
export A6_GUIDED_SIDE_THRESHOLD=0.90
export A6_GUIDED_SIDE_CONSECUTIVE=2
export A6_GUIDED_SIDE_EMA_ALPHA=0.2
export A6_GUIDED_SIDE_MIN_STEPS=31
export A6_GUIDED_SIDE_MAX_STEPS=45
export A6_GUIDED_RIGHT_SIDE_MIN_STEPS=35
export A6_GUIDED_RIGHT_SIDE_MAX_STEPS=120
export A6_MIRROR_LEFT_SIDE=1
export A6_MIRROR_RIGHT_MAIN=1
export A6_SIDE_SAFE_THROUGH=500
export A6_SIDE_SAFE_SCALE=0.5
export A6_SIDE_SAFE_GATE_ONLY=1
export A6_EVAL_NAME="${A6_EVAL_NAME:-v49_reproduction}"
export A6_SEEDS="${A6_SEEDS:-68 72}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_a6_composite_12_server.sh"
