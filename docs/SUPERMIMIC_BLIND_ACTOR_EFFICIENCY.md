# Blind-actor SuperMimic efficiency experiments — 2026-11-04

Branch: `lyd-supermimic-efficiency-blind-actor`.
Implementation: `d61bbd5`. All new execution uses `192.168.11.91:17535`.

## Execution update (16:22, server time)

The upstream locked-wheel download remained pending after roughly 50 minutes.
The original gate orchestrator and its waiting installer were stopped; the
original download process was left running. The four validated AMP-runtime
ablations were dispatched independently on GPUs 2–5, with no changes to their
training recipes. This supersedes the native-first execution dependency below.
Native fidelity remains unverified and must not be inferred from these runs.
Remote `records/ablation-jobs.json` records exact commands and PIDs.

## Observation contract

Our task Actor keeps the existing **384-dimensional proprioceptive input**.
No height scan, depth, terrain class, or terrain map is added to the Actor.
Terrain information stays in the privileged Critic/world-model inputs.
The trainer asserts Actor width and absence of the height-scan observation term.
The reward, motion bank, reset distribution, terrain, and Actor architecture stay
fixed across the four efficiency variants below.

## Separate upstream fidelity benchmark

Source: https://github.com/fuyichen0725/blinkmimic at
`d264658d6c9ce61fb52f9f0f8985a369f3ccc409`.
Use the upstream frozen `uv.lock`, including PyTorch 2.9.0+cu128 and Warp 1.16.0,
and the released `main` preset, seed 0, 3500 iterations.
Data are the upstream 40 LAFAN1 G1 CSV clips and G1 robot assets from
`lvhaidong/LAFAN1_Retargeting_Dataset`; the downloaded data directory contains
`download_manifest.json` with source paths, byte counts and SHA256 hashes.

This is reference-conditioned motion tracking, **not** our velocity-command,
blind-terrain locomotion and recovery task. Its convergence time cannot be
directly advertised as our task's convergence time. One seed is a fidelity
screen, not a reproduction of the upstream multi-seed statistical claim.

Before expensive ablations, require environment sanity checks and evaluation of
all saved native checkpoints with 4 replicas per clip. Gate: mean normalized
return >= 0.82 and zero failed clips for 3 consecutive saved checkpoints.
Failure stops the pipeline for investigation; it does not silently switch recipe.

## Controlled ablations

All: fresh initialization, 4096 environments, seed 42, maximum 40k iterations,
checkpoint every 1000 iterations, same shared AMP discriminator and terrain.

| Full experiment name | GPU | Sole change from control |
|---|---:|---|
| SuperMimic-BlindActor-SharedAMP-Terrain-Control-24Steps-WM2048 | 2 | Control: 24 real steps, WM width 2048 |
| SuperMimic-BlindActor-SharedAMP-Terrain-EMA0995-24Steps-WM2048 | 3 | Evaluation-only policy EMA, decay 0.995 |
| SuperMimic-BlindActor-SharedAMP-Terrain-UTD04-2p5Steps-WM2048 | 4 | Average 2.5 real steps/update (alternates 2 and 3) |
| SuperMimic-BlindActor-SharedAMP-Terrain-24Steps-WM1024 | 5 | WM width 1024 |

EMA smooths parameters only; normalizer buffers are copied directly. Live policy
continues collection/optimization. Checkpoints retain both live and EMA weights;
inference defaults to EMA when present, and supports `use_ema=False` for comparison.
No Muon, custom CUDA graph, or dynamics parameterization change is bundled here.

The 2.5-step condition receives 9.6x fewer real transitions per iteration than
the 24-step condition. Therefore equal iteration is not equal sample budget.
Report elapsed wall time, real environment transitions and task quality jointly.
Use the established matched locomotion, recovery, and terrain evaluation protocol;
include recovery smoothness and compare both EMA/live policies for the EMA run.

## Validation and operations

Two EMA unit tests pass; numeric terrain verification passes. All four variants
completed 6 iterations with 4096 environments and actual WM/policy updates.
These checks establish execution correctness, not learning quality.

Remote root: `/root/workplace/lyd-supermimic-efficiency-20261104`.
Native root: `/root/workplace/blinkmimic-d264658`.
Orchestrator: `scripts/experiments/blind_supermimic_pipeline.py`.
It installs pinned dependencies, runs native sanity/training/evaluation, then
launches the four ablations only if the native gate passes. It refuses occupied
GPUs and duplicate execution. All training logs go to W&B `tabletennis/lyd-multi-amp`.
Native training CSV is bridged to W&B without changing upstream source.

Read `records/pipeline-status.json` for the current stage; `records/ablation-jobs.json`
exists only after ablation dispatch. Each run has its own stdout and checkpoints.
At preparation time the native dependency download was still pending; no native
success or full ablation launch is claimed by this document.
