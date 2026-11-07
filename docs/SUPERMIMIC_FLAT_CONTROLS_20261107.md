# Flat SuperMimic controls — 2026-11-07

Branch: `lyd-supermimic-flat-speed-controls`. Server: `192.168.11.91:17535`.
Remote checkout: `/root/workplace/lyd-supermimic-flat-20261107`.

All three arms start from scratch, 4096 environments, seed 42, maximum 40k
iterations, save every 1000. Actor has 384 proprioceptive/history inputs and no
terrain perception. Flat tasks have no terrain scan in the critic or world model.
WM width is 2048 x 3, imagination horizon 32, 10 WM batches per iteration,
300 warmup iterations, EMA off. Rewards, expert bank and stratified reset are
unchanged across arms. These are shared conditional versus independent dual
AMP networks, not legacy unconditional single AMP.

| Full experiment name | GPU | Task | Real steps/iteration |
|---|---:|---|---:|
| SuperMimic-Flat-SharedAMP-SpeedCurriculum-2p5Steps-WM2048 | 0 | F-Shared | 2.5 |
| SuperMimic-Flat-SharedAMP-SpeedCurriculum-5Steps-WM2048 | 1 | F-Shared | 5 |
| SuperMimic-Flat-DualAMP-SpeedCurriculum-2p5Steps-WM2048 | 2 | H-Full | 2.5 |

## Speed coverage correction

The old TrackingCommand used minimum physical control-step counters 120000 and
360000 for advancing speed stages. A 40k run with 2.5 steps only collects 100000
such steps, so its first promotion was impossible. This is not evidence that the
world model fundamentally cannot learn high speed.

New arms disable automatic competence-gated advancement and use an explicitly
scheduled coverage curriculum: stage 0 until iteration 5000, stage 1 until 15000,
stage 2 thereafter. Forward speed caps are 1, 2, 3 m/s. Other mode limits follow
the existing LIMITS table. Recovery still gates actual commands to zero until
standing stabilizes. The schedule increases exposure; it does not certify that
previous-stage competence was achieved. Resampling applies the new limits to
new targets. No global changes to historical task behavior (automatic curriculum
still defaults to true). W&B logs curriculum stage and proportions of actual
commands above 1/2 m/s and target commands above 2 m/s.

## Execution

`scripts/experiments/launch_flat_supermimic_20261107.py` runs an 8-iteration
4096-environment smoke check per arm before launching its full training. Each
output includes process.json, stdout.log, config.json and exit.json on completion.
Full runs use W&B tabletennis/lyd-multi-amp. Do not run the launcher twice against
existing directories.

## BlinkMimic

Native upstream pinned source d264658d6c9ce61fb52f9f0f8985a369f3ccc409 is in
`/root/workplace/blinkmimic-d264658`. Its reference-motion tracking benchmark is
separate from these three velocity-command/recovery tasks. Exact frozen uv.lock
installation, robot conversion, environment checks, native main-preset seed0
training and 4-replica checkpoint evaluation are required before claiming fidelity.
`scripts/experiments/run_native_blinkmimic.py` records this process to W&B.
The reference-conditioned native Actor must not be relabeled as our blind,
velocity-command Actor. Same-task BlinkMimic integration remains separate work.
