# Supine SMP reward fine-tune

`Unitree-G1-HoST-SupineSmp` starts G1 supine on flat ground. It has no
external load and no auxiliary pull force. The actor and action space remain
identical to the HoST baseline. This task adds SMP get-up's head-height
(`0.3`) and upward-head-velocity (`0.7`) reward shapes while keeping HoST's
standing, style, regularization, and target rewards. It does **not** load the
SMP diffusion prior; SMP's pretrained model is for a different G1 joint set.

The head is a massless site on the torso at `(0, 0, 0.43)`. Height and velocity
are measured there. The new `smp_head_world_height` and `upright_standing`
metrics make recovery quality visible alongside reward. Checkpoint evaluation
must still inspect actual motion and posture before hardware use.

Reference: `tholin-1007/smp` commit
`0e67286fe7df77a73740d237ef36b109136552b6`,
`src/smp/rl/tasks/getup/mdp/rewards.py` and `getup_env_cfg.py`.

Run on GPU 0 from the static-diverse baseline:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_ENVS=4096 MAX_ITER=6000 \
CKPT=/absolute/path/to/model_11999.pt \
LOG_DIR=/absolute/path/to/a/new/run \
.venv/bin/python scripts/train_supine_smp.py
```
