# Supine SMP reward fine-tune, v3 (2026-09-28)

This is an **unsuccessful long-run experiment**, archived to make the result
reproducible. `Unitree-G1-HoST-SupineSmp` starts the 23-DoF G1 supine on flat
ground, with no load and no assistance. It adds the SMP get-up task's head
height (weight 0.3) and upward head velocity (weight 0.7) rewards to the HoST
reward stack. It does not use SMP's diffusion motion prior.

The run used 4096 environments for 6000 PPO iterations (1,228,800,000
environment steps), starting from `static_diverse_elbow_v2/model_11999.pt`.
The action noise standard deviation was frozen at 0.1, the observation
normalizer was frozen, the learning rate was 1e-5, and entropy coefficient
was zero. `source.json`, `env.yaml`, and `agent.yaml` record the configuration.
The source code is in this repository's `sjw` branch.

| Checkpoint | At step 500: torso-height >0.6 m | At step 1000 | Outcome |
| --- | ---: | ---: | --- |
| `model_500.pt` | 64/64 | 64/64 and upright | Best measured checkpoint |
| `model_1000.pt` | 13/64 | 21/64 height criterion | Degraded |
| `model_1500.pt` | 0/64 | 0/64 | Failed |
| `model_4500.pt` | 0/64 | 0/64 | Failed |
| `model_5999.pt` | 0/64 | 0/64 | Final checkpoint failed |

These are deterministic, 64-environment, 1000-step tests from the supine
start with zero pull force. The early `model_500` reached standing by step
500 and remained there at step 1000; its upper-joint velocity RMS while
standing was 2.40467 rad/s across 11 upper joints. This statistic and the
video do not establish natural motion, elbow safety, or hardware readiness.
The `best_500.mp4` video shows a side-supported rise to standing;
`final_5999.mp4` shows the final policy rising partway and falling back.
`policy_5999.onnx` is the export of the **failed final policy**.

`eval_*.log` contains the full evaluation output; compressed TensorBoard
events and training log preserve the learning curve. The complete original
run, including every intermediate checkpoint, is stored locally at
`C:/Users/26390/Desktop/mjlab/training_data/supine_4096_frozen_norm_v3`.
Each of the 34 downloaded run/log/model/video files was checked against a
remote SHA256 digest in its local `download_manifest.json`. The GitHub archive
keeps the best and final checkpoints rather than duplicating all 13 models.
`SHA256SUMS` lists hashes of the curated files.
