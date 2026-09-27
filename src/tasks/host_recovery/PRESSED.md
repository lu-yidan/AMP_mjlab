# Chest/pelvis loaded recovery baseline

Task: `Unitree-G1-HoST-Pressed`. Independent of `Unitree-G1-HoST-StandUp`.

First stage uses supine ground starts and one **free, movable 12 kg box**
(0.40 x 0.44 x 0.12 m), centered over chest or pelvis with equal probability.
The box is placed after reset kinematics, 2 cm above the selected collision
surface, then moves only through gravity and contact. The 120-step passive
settling phase is inherited from HoST. No upward assistance is enabled.
`HOST_LOAD_TARGET=chest|pelvis|mixed` and `HOST_LOAD_MASS=1..40` configure
separate experiments; no automatic mass curriculum is claimed.

## Reward provenance and scope

Reference: [tholin-1007/smp](https://github.com/tholin-1007/smp), commit
`0e67286fe7df77a73740d237ef36b109136552b6`,
`src/smp/rl/tasks/getup/mdp/rewards.py` and `getup_env_cfg.py`.

This baseline reimplements its scalar height and upward-head-speed shaping:
`0.3 exp(-max(1.1-head_z,0)^2)` plus
`0.7 exp(-100 max(0.25-head_vz,0)^2)` below 0.9 m.
Upward-speed shaping is attenuated by `exp(-contact_force/30)` while loaded.
The added head site is 0.43 m above the torso origin. HoST's existing rewards,
including elbow deviation, remain in place. Added terms have weights 1 (SMP
task shapes), 2 (release/separation), and 2 (released upright standing).

Release shaping uses horizontal target-body/box separation, clipped at 0.45 m,
and the same contact gate. It activates only after step 130 and only for
episodes which actually sustained >20 N torso/pelvis contact near motor
activation (steps 100..130). This prevents passive box fall-off from earning
release success. `loaded_fraction` must be checked, not assumed to equal 1.
`released_standing` also requires head height >1.0 m and gravity_z <-0.8;
it is an instantaneous diagnostic, not a timed success guarantee.

**This is SMP-inspired task shaping, not a diffusion-prior implementation.**
The reference prior uses a 29-DoF feature definition; the current robot has
23 DoFs. Its checkpoint is not compatible without explicit feature adaptation
and validation. Learned constraint inference, recoverability prediction,
active information gathering, depth, and complex support terrain remain future
experiments. A simple free box may be moved/lifted as well as escaped laterally.

Policy input remains the existing 456-dimensional deployable proprioceptive
history. Exact box poses/contact readings enter only rewards and metrics, not
actor or critic observations. The contact gate is a training-time heuristic,
not a deployed learned feasibility estimator.

## Run

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
CKPT=/absolute/path/to/model_11999.pt \
LOG_DIR=logs/rsl_rl/g1_host_pressed/pressed_chest_pelvis_12kg_v1 \
HOST_LOAD_MASS=12 HOST_LOAD_TARGET=mixed NUM_ENVS=2048 MAX_ITER=6000 \
.venv/bin/python scripts/train_pressed.py
```

The output directory must not already exist. Optimizer and iteration counter
start fresh; policy, value weights, and normalization are initialized from the
explicit checkpoint. Do not overwrite the baseline or restart an existing run.

## Initial physics check

Using `model_9500.pt`, 16 mixed-target environments, no fine-tuning:
all environments experienced >20 N near motor activation; mean contact force
at step 120 was about 69 N. At step 600, 25% met the instantaneous released
standing criterion. Rewards remained finite for 600 steps. This demonstrates
a nontrivial loaded task; it is not a final success-rate claim or a safety test.
Use `scripts/probe_pressed.py` for a small repeatable diagnostic.

A 64-environment, 5-iteration PPO smoke run also completed, including checkpoint
and ONNX export. Its 250 steps do not cover a full 20-second episode/reset or
establish learning quality; monitor full-run loaded fraction and success.
