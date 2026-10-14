#!/usr/bin/env bash
# Launch 1000 G+ A6 updates from the 29-DoF flat expert on physical GPU 3.
set -euo pipefail

cd "$(dirname "$0")/.."

parent=logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt
output=logs/host_a6_29_v8_nodepth5000_from0999_frozen_Gplus_4096_seed42_1000_20261011
python_bin=${HOST29_PYTHON:-/home/czr/code/G1Recovery_mjlab/.venv/bin/python}
gpu=${HOST29_GPU:-3}
expected_parent_sha=0fa09f84bfc1eed56526804c0e0d47c76c69a09c1a64fb1922e8f233b28aaa0a

actual_parent_sha=$(sha256sum "$parent")
actual_parent_sha=${actual_parent_sha%% *}
if [[ "$actual_parent_sha" != "$expected_parent_sha" ]]; then
  echo "Flat expert SHA256 mismatch: $actual_parent_sha" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "Output directory already exists: $output" >&2
  exit 1
fi
if ! grep -q 'plate_force_5000_guided_lift_005_no_depth_v2' src/g1recovery_host/a6_reward.py; then
  echo "A6 reset source is not the verified no-depth version" >&2
  exit 1
fi

mkdir -p "$output"
nohup env CUDA_VISIBLE_DEVICES="$gpu" \
  MPLCONFIGDIR=/tmp/mpl-host-a6-v8-nodepth-1000 \
  PYTHONPATH="$PWD/src" \
  "$python_bin" scripts/train_host.py \
  --task a6 --arm G+ --a6-pose-v6 \
  --continue-from29 "$parent" --frozen-flat-teacher "$parent" \
  --flat-teacher-weight 0.5 --a6-edge-curriculum \
  --a6-edge-start-offset-m 0.25 --a6-edge-hold-updates 100 \
  --a6-edge-anneal-updates 300 --reset-optimizer --adaptive-kl \
  --learning-rate 0.00005 --device cuda:0 --num-envs 4096 \
  --steps-per-env 24 --iterations 2000 --epochs 5 --minibatches 4 \
  --seed 42 --log-every 10 --save-every 100 \
  --output "$output" > "$output/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$output/train.pid"
echo "Started G+ A6 training: PID=$pid GPU=$gpu output=$output"
