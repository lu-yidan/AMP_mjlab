#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "Usage: $0 THRESHOLD SEED GPU_ID" >&2
  exit 2
fi
threshold=$1
seed=$2
gpu=$3
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
tag=${threshold/./}
out="artifacts/a6_proprio_phase_detector/online/controller_threshold_v9/t${tag}_seed${seed}.json"
mkdir -p "$(dirname "$out")"
export CUDA_VISIBLE_DEVICES="$gpu"
export MPLCONFIGDIR=/tmp/mpl-host-controller-threshold
export PYTHONPATH="$project/src"
exec /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/eval_a6_batch.py \
  --checkpoint logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500/model_00374.pt \
  --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
  --diagnostic-controller-threshold "$threshold" \
  --device cuda:0 --seed "$seed" --per-scene 64 \
  --steps 750 --guided-offset-m 0 --output "$out"
