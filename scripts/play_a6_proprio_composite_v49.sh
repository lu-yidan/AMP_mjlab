#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${A6_PYTHON:-}" ]]; then
  python_bin="$A6_PYTHON"
elif [[ -x /home/ray1/code/mjlab/.venv/bin/python ]]; then
  python_bin=/home/ray1/code/mjlab/.venv/bin/python
elif [[ -x /home/czr/code/G1Recovery_mjlab/.venv/bin/python ]]; then
  python_bin=/home/czr/code/G1Recovery_mjlab/.venv/bin/python
else
  echo "Set A6_PYTHON to the Python in a mjlab virtual environment" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-host-a6-proprio-v49-play}"

exec "$python_bin" scripts/play_a6_host29.py \
  --checkpoint logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt \
  --proprio-prone-specialist logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt \
  --proprio-side-specialist logs/host_a6_side_teacher_all_v33_gpu4_from549_seed42_700/model_00624.pt \
  --proprio-guided-side-gate artifacts/a6_proprio_phase_detector/side_scene_canonical_v48/side_early31_45_seed48.pt \
  --proprio-guided-right-side-gate artifacts/a6_proprio_phase_detector/side_scene_mirror_v44/side_v17_mirror_seed48.pt \
  --guided-side-threshold 0.90 --guided-side-consecutive 2 \
  --guided-side-ema-alpha 0.2 --guided-side-min-steps 31 --guided-side-max-steps 45 \
  --guided-right-side-min-steps 35 --guided-right-side-max-steps 120 \
  --mirror-left-side-specialist --mirror-right-initial-side-main \
  --side-safe-through-step 500 --side-safe-scale 0.5 \
  --proprio-recovery-controller artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt \
  --proprio-supine-flat-gate artifacts/a6_proprio_phase_detector/recovery_gate_v1/supine_flat_recovery_gate_unweighted_v1.pt \
  --flat-expert-checkpoint logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt \
  --assist-force-n 0 --episode-seconds 15 --device "${A6_DEVICE:-cuda:0}" "$@"
