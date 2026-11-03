#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "Usage: $0 {v12|v13|v14|v15|v16} SEED GPU_ID" >&2
  exit 2
fi
kind=$1
seed=$2
gpu=$3
case $kind in
  v12) checkpoint=logs/host_a6_free_plate_v12_from374_seed42_450/model_00449.pt ;;
  v13) checkpoint=logs/host_a6_matching_replay_v13_from374_seed42_450/model_00449.pt ;;
  v14) checkpoint=logs/host_a6_supine_boost_v14_from374_seed42_450/model_00449.pt ;;
  v15) checkpoint=logs/host_a6_free_plate_anchor_v15_from374_seed42_450/model_00449.pt ;;
  v16) checkpoint=logs/host_a6_preescape_preserve_v16_from374_seed42_450/model_00449.pt ;;
  *) echo "Unknown ablation: $kind" >&2; exit 2 ;;
esac
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
out="artifacts/a6_proprio_phase_detector/online/scene_phase_pair_v11/${kind}_449_seed${seed}.json"
mkdir -p "$(dirname "$out")"
export CUDA_VISIBLE_DEVICES="$gpu"
export MPLCONFIGDIR=/tmp/mpl-host-a6-ablation
export PYTHONPATH="$project/src"
exec /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/eval_a6_batch.py \
  --checkpoint "$checkpoint" \
  --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
  --device cuda:0 --seed "$seed" --per-scene 64 --steps 750 \
  --guided-offset-m 0 --output "$out"
