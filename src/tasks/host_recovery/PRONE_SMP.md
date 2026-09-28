# Prone SMP-reward fine-tune

`Unitree-G1-HoST-ProneSmp` uses the same 23-DoF actor, flat ground, zero
assistance, and SMP-inspired head-height/upward-velocity task rewards as the
supine experiment. Only the reset posture changes to prone. It does not use
SMP's diffusion prior. The experiment starts from the completed diverse
posture HoST checkpoint, not from the failed final supine fine-tune.

The supine run kept recovery at iteration 500 but then lost it despite frozen
observation normalization and low action noise. For this prone comparison,
save checkpoints more often and evaluate them independently; training reward
alone is not evidence that the robot stands up.

```bash
CUDA_VISIBLE_DEVICES=0 NUM_ENVS=4096 MAX_ITER=6000 SAVE_INTERVAL=250 \
ACTION_NOISE_STD=0.1 FREEZE_ACTION_STD=1 FREEZE_OBS_NORMALIZER=1 \
LEARNING_RATE=1e-5 ENTROPY_COEF=0 \
CKPT=/absolute/path/to/static_diverse_elbow_v2/model_11999.pt \
LOG_DIR=/absolute/path/to/a/new/prone/run \
.venv/bin/python scripts/train_prone_smp.py
```
