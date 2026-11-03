#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
  echo "Usage: $0 {actor|observation|controller} SEED GPU_ID" >&2
  exit 2
fi
mode=$1
seed=$2
gpu=$3
case $mode in
  actor) extra=(--proprio-oracle-actor-reset-only) ;;
  observation) extra=(--proprio-oracle-observation-reset-only) ;;
  controller) extra=(--proprio-oracle-controller-reset-only) ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
out="artifacts/a6_proprio_phase_detector/online/phase_handoff_probe/${mode}_seed${seed}.json"
mkdir -p "$(dirname "$out")"
export CUDA_VISIBLE_DEVICES="$gpu"
export MPLCONFIGDIR=/tmp/mpl-host-phase-handoff
export PYTHONPATH="$project/src"
exec /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/eval_a6_batch.py \
  --checkpoint logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500/model_00374.pt \
  --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
  "${extra[@]}" --device cuda:0 --seed "$seed" --per-scene 64 \
  --steps 750 --guided-offset-m 0 --output "$out"
