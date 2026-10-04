# Prone SMP reward fine-tune, v1 (2026-09-28)

This is an **unsuccessful long-run fine-tune**, preserved alongside its useful
early checkpoint. `Unitree-G1-HoST-ProneSmp` resets the 23-DoF G1 prone on flat
ground, with no external load or assistance. It adds SMP-style head-height
(weight 0.3) and upward head-velocity (weight 0.7) rewards to HoST. It does
not use SMP's diffusion motion prior.

The run used 4096 environments for 6000 PPO iterations (1,228,800,000
environment steps), starting from `static_diverse_elbow_v2/model_11999.pt`.
Checkpoints were saved every 250 iterations. Action noise standard deviation
was frozen at 0.1, the observation normalizer was frozen, the learning rate
was 1e-5, and entropy coefficient was zero. `source.json`, `env.yaml`, and
`agent.yaml` record the configuration. Source code is in the `sjw` branch.

| Checkpoint | At step 500: torso height >0.6 m | At step 1000 | Outcome |
| --- | ---: | ---: | --- |
| `model_250.pt` | 64/64 | 64/64 and upright | Best measured checkpoint |
| `model_500.pt` | 60/64 | 56/64 height criterion | Degraded |
| `model_1000.pt` | 33/64 | 44/64 height criterion | Degraded |
| `model_1500.pt` | 23/64 | 24/64 height criterion | Degraded |
| `model_2500.pt` | 18/64 | 20/64 height criterion | Degraded |
| `model_5750.pt` | 0/64 | 0/64 | Failed |
| `model_5999.pt` | 0/64 | 0/64 | Final checkpoint failed |

These are deterministic 64-environment, 1000-step tests from a prone start
with zero pull force. The early `model_250` reached standing by step 500 and
remained upright at step 1000. Its upper-joint velocity RMS while standing
was 2.35029 rad/s across 11 upper joints. This metric and the video do not
establish natural motion, elbow safety, or hardware readiness. The final
policy does not rise. `policy_5999.onnx` exports the **failed final policy**.

`eval_*.log` contains full independent evaluations. Compressed TensorBoard
events and the training log preserve the learning curve. The complete
original run with every intermediate checkpoint is stored locally at
`C:/Users/26390/Desktop/mjlab/training_data/prone_4096_frozen_norm_v1`.
Its `download_manifest.json` records the remote and local SHA256 values for
each downloaded file. The GitHub archive retains the best and final models;
other checkpoints remain in that local raw copy. `SHA256SUMS` covers the
curated files, and `.gitattributes` prevents Git newline conversion.
