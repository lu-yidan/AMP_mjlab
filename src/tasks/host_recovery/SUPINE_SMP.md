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
ACTION_NOISE_STD=0.1 \
FREEZE_OBS_NORMALIZER=1 FREEZE_ACTION_STD=1 \
LEARNING_RATE=1e-5 ENTROPY_COEF=0 \
CKPT=/absolute/path/to/model_11999.pt \
LOG_DIR=/absolute/path/to/a/new/run \
.venv/bin/python scripts/train_supine_smp.py
```

`ACTION_NOISE_STD` replaces the checkpoint's exploration standard deviation
after loading. The first 4096-environment attempt with the inherited 0.5
standard deviation failed: training episode upright rate stayed zero and a
deterministic evaluation of `model_500.pt` stood in 0/64 environments, while
the initial checkpoint stood in 64/64. A 64-environment, 25-iteration smoke
with `ACTION_NOISE_STD=0.1` reported upright standing in 42.27% of its first
completed episodes. This is a diagnostic, not final policy validation.

Further diagnosis found that the checkpoint's actor observation normalizer
had counted only 204,800 samples. One 4096-environment training iteration
raised that count to 409,600 and shifted its mean by up to 0.38, while the
largest actor weight change was below 0.0002. A conservative 4096-environment
smoke without normalizer freeze still lost recovery by its first checkpoint.
`FREEZE_OBS_NORMALIZER=1` keeps the pretrained observation scale fixed during
fine-tuning; the flag must be validated by checkpoint evaluation.
