# 29-DoF flat-ground HoST expert

This task trains **from random weights** on the full Unitree G1 29-DoF MuJoCo
model. It is a new expert; the earlier `model_02999.pt` was trained with the
23-DoF HoST URDF and cannot be a native 29-DoF flat expert.

## Model and controls

`src/g1recovery_host/model.py` uses mjlab's Unitree G1 29-DoF MJCF and the
joint order in `G1Recovery_mjlab/src/g1recovery_mjlab/g1_model.py`. The six
additional joints are waist roll/pitch and left/right wrist pitch/yaw. Their
MJCF limits are ±0.52 rad for waist roll/pitch and ±1.61443 rad for wrist
pitch/yaw. `action.py` keeps HoST's relative-action torque controller. For the
new axes, it uses mjlab's 5020/4010 motor inertias and PD constants, 50 Nm
waist effort and 5 Nm wrist effort. The reference recovery project instead
uses a position-action controller with its own engineering PD gains, so its
gain numbers are not copied into HoST's different control law.

## Standing target

The arm angles are copied, by joint name, from frame 429 (the last frame) of
`/home/ray1/code/G1Recovery_mjlab/src/g1recovery_mjlab/data/final_motion/29dof/Final_ce1_29dof.npz`,
SHA-256 `eab294889e210e8f0f245bab1efd1a7746bbfee79dbd0263f4ff36ff66c7ff30`.
That motion's metadata reports the two hand centers about 0.10 m forward,
0.22 m to either side and 0.20 m below the torso in the final frame. The
reference legs have small asymmetric residuals and waist yaw 0.038 rad. The
flat expert therefore uses the symmetric HoST leg target and a centered
three-axis waist, while preserving the reference's symmetric, down-at-sides
arm pose. The separate 29-DoF target reward is active only after pelvis height
exceeds 0.65 m; HoST's head-height/upright, style, regularization and
assistance-curriculum rewards remain in place.

## Fresh long run

From the project root on a GPU host with the mjlab environment installed:

```bash
mkdir -p logs/host_ground29_standref_4096_seed42_3000
MPLCONFIGDIR=/tmp/mpl-host29-flat-long CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$PWD/src" \
  /home/czr/code/G1Recovery_mjlab/.venv/bin/python scripts/train_host.py \
  --task flat29 --device cuda:0 --adaptive-kl --learning-rate 0.001 \
  --seed 42 --num-envs 4096 --steps-per-env 50 --iterations 3000 \
  --epochs 5 --minibatches 4 --log-every 10 --save-every 100 \
  --output logs/host_ground29_standref_4096_seed42_3000
```

There is intentionally no `--resume`, `--init-from` or `--continue-from`.
Checkpoints are `model_00099.pt`, `model_00199.pt`, ..., `model_02999.pt`.
`CUDA_VISIBLE_DEVICES=3` selects physical GPU 3; the process sees it as
`cuda:0`. Change the physical index if that GPU is occupied.
Inspect `metrics.jsonl` during training and evaluate the final model before
using it as the A6 parent: a completed run alone does not establish reliable
get-up or the desired standing style.
