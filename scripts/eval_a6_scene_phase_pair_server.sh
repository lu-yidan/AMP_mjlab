#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
  echo "Usage: $0 {control|v11} {399|424|449} {44|48} GPU_ID" >&2
  exit 2
fi
group=$1
iteration=$2
seed=$3
gpu=$4
if [[ $group != control && $group != v11 ]]; then
  echo "Group must be control or v11" >&2
  exit 2
fi
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
checkpoint="logs/host_a6_scene_phase_${group}_from374_seed42_450/model_00${iteration}.pt"
output="artifacts/a6_proprio_phase_detector/online/scene_phase_pair_v11/${group}_${iteration}_seed${seed}.json"
mkdir -p "$(dirname "$output")"
export CUDA_VISIBLE_DEVICES="$gpu"
export MPLCONFIGDIR=/tmp/mpl-host-phase-eval
export PYTHONPATH="$project/src"
exec /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/eval_a6_batch.py \
  --checkpoint "$checkpoint" \
  --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
  --device cuda:0 --seed "$seed" --per-scene 64 --steps 750 \
  --guided-offset-m 0 --output "$output"
