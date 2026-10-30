# AMP-C controlled comparisons (2026-10-30)

Branch: `lyd-c-controlled-ablations`. These experiments start from scratch and
must not overwrite/relabel historical A–E. They are intended to separate shared
versus independent discriminator networks from recovery sampling effects.

| Code | Networks | Recovery experts | Recovery sampling |
|---|---|---|---|
| F | One shared conditional network and one optimizer | Same SMP bank as C | C phase/direction balancing and expert matching |
| G | Two independent networks | Same SMP bank as C | Uniform replay within the recent/history pools; uniform bank transitions |
| H | Two independent networks | Same SMP bank as C | Original C phase/direction balancing and expert matching |

F retains the two mode-specific running normalizers, replay partitions, mode
routing and locomotion expert-condition matching. Recovery inputs get three zero
condition values to use the locomotion-width network. Only network/optimizer
sharing changes against H; F is a controlled shared-discriminator variant, NOT
legacy AMP-A or an unconditional mixed-motion AMP baseline. Report this explicitly
in a paper. Each mode has the same fresh-sample budget in every arm; the shared
network receives both modes' optimizer steps (up to 40 total versus 20 per network
in the dual arms). Parameter counts differ. This is a fixed-data/update-budget
comparison, not a matched-parameter-count comparison.

G disables both inverse-frequency recovery replay sampling and phase/direction
expert matching. The expert bank, transition support and history-age restriction
remain the same. It therefore tests the sampling package, not its individual parts.
H retains C's architecture, normalization and update arithmetic; a regression test
compares its initialization, reward and update against the original StableMultiAMP.

All arms: 4096 environments, 24 steps, 100000 iterations, checkpoints every 1000,
seeds 42/43/44, identical stratified resets and delayed termination, Tracking
curriculum to 3 m/s, original task rewards and 0.75/0.25 task/style blend. No added
L4 recovery guidance. All use the same SMP recovery expert bank (94 clips, 16336
transitions), existing locomotion references and training reset bank. Recovery-bank
SHA256: `525d90307cf738b2051aad840abf5bb36ebe8460ad4cf7c0f1a3635b8d8a2823`.

All new arms fail immediately on non-finite rollout inputs/rewards, PPO batches,
losses or gradient norms and discriminator losses/gradient norms. They do not
silently skip PPO updates. MuJoCo's NaN guard is also enabled. These checks prevent
an invalid run from being counted as successful training; they do not establish
or repair the historical AMP-A failure's root cause. Historical tasks retain their
old behavior. New F uses C's separate discriminator optimizer and raw-feature
running moments, avoiding the legacy single AMP update path.

## Execution

Run `scripts/launch_amp_controlled.py --preflight --source-commit <sha> --output
<new-directory>` for three 20-iteration 4096-environment smoke runs. Verify their
checkpoints, zero skipped updates and finite metrics before launching without
`--preflight`. Formal launch uses GPUs 0/1 for F, 2/3 for G, 4/5 for H. Seeds 42/43
start concurrently; seed 44 queues behind seed 42 for each arm. GPU 6 remains free.
A failed worker does not start its queued seed. Launch manifests are exclusive and
record source SHA, data hashes, planned runs, PIDs, commands, W&B URLs and exit codes.

## Evaluation plan (not yet results)

Use the frozen `logs/eval100k_single/validation.npz`, identical commands, seeds,
physics, timeout and success thresholds. At matched 20k/60k/100k checkpoints report:
recovery success by source/stage/direction, time to stable standing, command-axis
MAE and unintended drift/yaw, upright fraction and post-recovery stability. Retain
failures in denominators. Report all three seed results plus mean and spread;
do not pool trials across seeds as if they were independent trained policies.
Use the old successful 100k single Tracking checkpoint as a historical reference,
not a same-data ablation. Naturalness videos are qualitative evidence only.

The existing fixed-bank recovery and command tests do not establish recovery from
in-motion perturbations or confined/rough-terrain capability. Those require a
separate preregistered perturbation protocol and must not be claimed from this run.
