# Step 1: 29-joint flat recovery baseline

This is a from-scratch baseline using the existing HoST port: single-critic PPO,
not the complete paper algorithm (multi-critic and objective regularizers are
not implemented). No 23-joint checkpoint is padded or resumed.

## Configuration

- Entry point: `scripts/train_host29.py`; configuration: `src/tasks/host_recovery/flat29.py`.
- Repository G1 29-joint robot; 29 incremental joint actions.
- Actor/critic observation: 94 values per frame, six frames, total 564.
- All 17 upper-body joints enter posture and upper-body velocity terms.
- Four equiprobable initial orientations, root height 0.5 m, gravity settling.
- Physics step 0.002 s, decimation 10, control step 0.02 s.
- Existing HoST reward and curricula retained; no A6 dense guidance added.
- Torch compile disabled. Each run exports env.yaml, agent.yaml and launch.json.

## Preflight and formal command

```bash
CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 .venv/bin/python scripts/train_host29.py --num-envs 32 --iterations 2 --smoke --log-dir logs/host29_preflight_NEW
CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 .venv/bin/python scripts/train_host29.py --num-envs 4096 --iterations 12000 --seed 42 --log-dir logs/host29_flat_NEW
```

Log directories must be new. The script is single-GPU: selecting GPUs 0,1 does
not make it distributed. A second seed can run as a separate process on GPU 1.
Do not stop other users' GPU jobs; formal shared-GPU use needs confirmation.

The smoke test sets the live event manager reset term (not the copied outer
configuration), checks reset quaternion alignment, and checks finite observations
and rewards for all four postures through a full episode before PPO updates.
The earlier preflight without quaternion assertions establishes only generic
rollout finiteness, not fixed-posture coverage.

## Remaining scientific gates

Successful smoke/PPO updates do not establish recovery performance. Evaluate
per-orientation unassisted recovery, hold duration, failure rates, and actuator
costs before using this as a common initialization. Check checkpoint reload.

The repository G1 asset is not yet certified equivalent to the historical A6
deployment asset (including actuator limits). Before A6 adaptation, align robot
dynamics, reset banks, sensors, substep statistics, historical buffers and
success/invalid criteria, then fork paired guidance-off/on runs from one frozen
checkpoint. Do not report this initial flat run as a historical A6 reproduction.

## 2026-09-30 failed-run audit

The first 29-joint run was stopped after checkpoint 5000: deterministic,
unassisted evaluation had zero success in all four postures and the curriculum
remained at action scale 1.0 / pull force 200 N. Its checkpoints are retained as
negative evidence and must not initialize A6 experiments.

Comparison against the official HoST source identified three silent semantic
differences, corrected in commit `66a6333`:

- the official `unactuated_timesteps=30` remains 30 policy steps because HoST
  multiplies by `0.02 / control_dt`; the port had incorrectly divided by the
  physics step and used 120 policy steps;
- HoST already folds the 0.02 s control period into constraint weights while
  leaving the task group unscaled, so mjlab reward-manager dt scaling is disabled;
- official exploration starts at standard deviation 0.8 without an upper clamp;
  the generic vendored policy had silently capped it at 0.5. The cap is now an
  optional policy parameter and remains unchanged for non-HoST configurations.

The corrected run is a fresh start, not a continuation of the failed policy.
Checkpoints 500 and 1000 are mandatory four-posture evaluation gates before the
run is allowed to continue unattended to 12000.
