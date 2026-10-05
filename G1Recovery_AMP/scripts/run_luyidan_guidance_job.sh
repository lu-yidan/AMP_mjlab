#!/usr/bin/env bash
# Detached job: train a fixed-budget policy, then evaluate the final checkpoint.
set -euo pipefail
amp_arm=${1:?Usage: run_luyidan_guidance_job.sh off|on 4|5 RUN_ROOT}
amp_gpu=${2:?GPU index required}
amp_run_root=${3:?Unique relative output root required}
case "$amp_arm" in off|on) ;; *) exit 2;; esac
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="$amp_gpu" PYTHONPATH=src:scripts MUJOCO_GL=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NETRC=/root/workplace/amp-runtime/wandb.netrc
export WANDB_MODE=online
amp_train_dir="$amp_run_root/$amp_arm"
.venv/bin/python -u scripts/train_luyidan_guidance.py --arm "$amp_arm" --num-envs 4096 \
  --updates 20000 --seed 20261005 --out "$amp_train_dir" --wandb online
.venv/bin/python -u scripts/evaluate_free_sr3.py --checkpoint "$amp_train_dir/model_19999.pt" \
  --sr3 --horizon 23 --num-envs 2048 --seed 42 \
  --frozen-reset evaluation/sr3_guidance_v1/sr3_initial.npz \
  --free-parameters evaluation/sr3_guidance_v1/free_initial_parameters.npz \
  --out "$amp_run_root/${amp_arm}_final_sr3"
