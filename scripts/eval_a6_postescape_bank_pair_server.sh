#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ( $1 != legacy && $1 != v9 ) ]]; then
  echo "Usage: $0 {legacy|v9} SEED GPU_ID [restore]" >&2
  exit 2
fi
kind=$1
seed=$2
gpu=$3
restore_args=()
suffix=""
if [[ ${4:-} == restore ]]; then
  restore_args=(--restore-postescape-bank-history)
  suffix=_history
fi
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
if [[ $kind == legacy ]]; then
  bank=assets/a6/host29/escaped_prone_legacy_v24_phase_mix10_15_20_center768.npz
else
  bank=assets/a6/host29/escaped_prone_v9_controller_center_512.npz
fi
out="artifacts/a6_proprio_phase_detector/online/bank_mismatch_v9/${kind}${suffix}_seed${seed}.json"
mkdir -p "$(dirname "$out")"
export CUDA_VISIBLE_DEVICES="$gpu"
export MPLCONFIGDIR=/tmp/mpl-host-bank-v9-eval
export PYTHONPATH="$project/src"
exec /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/eval_a6_batch.py \
  --checkpoint logs/host_a6_proprio_controller_v9_Gplus_4096_seed42_500/model_00374.pt \
  --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
  --all-postescape --postescape-bank "$bank" "${restore_args[@]}" \
  --device cuda:0 --seed "$seed" --per-scene 64 --steps 500 \
  --guided-offset-m 0 --output "$out"
