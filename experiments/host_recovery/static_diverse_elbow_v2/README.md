# Static diverse get-up with elbow penalty (2026-09-27)

G1 23-DoF HoST task, commit `c1f2df8`, 4096 environments, 12,000 PPO iterations.
Four lying start postures are sampled uniformly at measured settled heights.
No upward assistance. The elbow hyperextension penalty is enabled.

`model_11999.pt` is the final checkpoint; `model_9500.pt` is the evaluated
mid-training reference. `policy.onnx` is the exported final policy.
`env.yaml` and `agent.yaml` are the recorded training configuration.
The compressed event file and training log preserve the full learning curve.
Four MP4s show one rollout per start posture. `eval_11999.txt` has the
zero-assistance, 64-environment-per-posture quantitative evaluation.
At step 1000, all four groups meet the upright and >0.6 m torso criteria
(64/64 each). This does not establish real-world safety or natural joint pose.

All intermediate checkpoints and uncompressed data are preserved locally at
`C:/Users/26390/Desktop/mjlab/training_data/2026-09-27_18-31-18_static_diverse_elbow_v2`.
The manifest records a remote/local SHA256 match for every downloaded file.
GitHub carries the final checkpoint and mid-training reference rather than
25 redundant intermediate checkpoints.
