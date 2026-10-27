# AMP-E: AMP-C + MBRL SuperMimic

Branch: `lyd-amp-c-supermimic`, based on `lyd-amp-recovery-stability` at c861f3a.
The imported library comes from `/home/luyd/workspace/MBRL_supermimic` commit
`eb7a82bedad9e9e25a93cff3318d24e4a9f88cb2`, clean source snapshot; attribution
and licenses are in `third_party/mbrl_supermimic/`. This experiment does not use
or restart the previous `src/supermimic` vibed-based experiments.

## Intended comparison

AMP-C uses PPO. AMP-E retains C's real MDP and replaces the policy optimizer
with SuperMimic (world model + BPTT actor + TD(lambda) critic). Both start from
scratch, with 4096 environments and 24 real steps per iteration, seed 42,
100000 iterations, checkpoints every 1000. Existing A-D jobs are independent.

Retained: C's 60% locomotion / 10% late / 10% middle / 20% low reset groups,
low-pose direction quotas, 250-step delayed termination for recovery groups,
TrackingCommand recovery hysteresis, measured velocity curriculum ending at
3 m/s, motor-order 29 actions, per-joint action scales, 4-frame observations,
domain randomization and pushes. No additional recovery guidance reward.

Recovery experts: `logs/amp_stability/smp_recovery_train.npz`, SHA256
`525d90307cf738b2051aad840abf5bb36ebe8460ad4cf7c0f1a3635b8d8a2823`.
94 retained SMP clips / 16336 transitions. Same phase/direction balanced
replay and same locomotion expert dataset as C. Two discriminators train ONLY
on real simulator transitions, at most 20 minibatches per iteration, with C's
fresh-sample reuse cap. During imagination they are frozen but input gradients
flow through both states to the actor. Reward stays .75 task + .25 style.

## SuperMimic settings and explicit differences

- 32-step (0.64 s) imagined horizon; deterministic 512/256/128 actor and critic,
  exploration noise 0.1. This is the library's MLP interface, not the paper's
  motion encoder/decoder. Actor initialization/exploration differ from PPO.
- World model 2048/2048/2048, bf16, 10 updates x 2048 x 8-step windows.
  300 world-model-only warmup iterations; cosine learning rate over 100k.
- Replay holds 128 vectorized steps, with exact stored actor/critic histories
  and command/termination side state. This is smaller than upstream's default
  1000 to bound memory with 4096 environments and 4-frame histories.
- Hard termination preserves C's counter-based death rule. The physical state,
  recovery hold, delayed termination counter, command ramp and observation
  histories advance together in imagination; timeouts bootstrap.
- World-model context reads actual randomized physical parameters. Reset
  boundaries are captured before reset; push/teleport transitions are excluded
  from world-model fitting. Future random pushes/resampling are not imagined.
- Joint acceleration cost, foot-slip cost, and contact self-collision cost are
  world-model auxiliary targets. Slip is learned *before* command gating and
  gated by the imagined command. Cost predictions cannot become positive
  bonuses. These are learned estimates, not exact contact physics.
- Command target is held fixed over the imagined horizon. Locomotion expert
  condition/support are matched once at rollout start (effective command for
  locomotion, target command for a potential recovery exit), then held fixed.
  Command smoothing continues, so this is an approximation during ramps.
  Real discriminator collection uses C's exact per-step command matching.
- mjlab's reward/termination reads some kinematics one physics substep before
  the final state used for the next observation. The numerical gate separately
  reports formula agreement at the actual reward snapshot and residual error
  when evaluated using the end-of-step state. Do not confuse the two.

Equal iterations match real transition count, not optimizer compute or wall
clock. Compare recovery success, recovery time, signed velocity/yaw errors and
real sample count alongside GPU hours; imagined reward alone is not success.

## Validation and usage

```
PYTHONPATH=<mjlab runtime>:. python scripts/train_amp_supermimic.py \
  --num-envs 256 --verify-only --output logs/amp_e_verify
PYTHONPATH=<mjlab runtime>:. python scripts/train_amp_supermimic.py \
  --num-envs 4096 --iterations 3 --smoke --output logs/amp_e_smoke
PYTHONPATH=<mjlab runtime>:. python scripts/train_amp_supermimic.py \
  --num-envs 4096 --iterations 100000 --sim-steps 24 --wandb \
  --output logs/amp_e_100k
```

Smoke changes only warmup to 1 and log interval to 1, retaining full network,
world-model and imagined-horizon sizes. `--verify-only` disables observation
noise and pushes, compares observations/AMP features, weighted reward terms,
termination decisions, and central-difference state gradients. It does not
measure learned world-model prediction accuracy or policy performance.

Checkpoints contain actor, critic, world model, all optimizers and observation
normalizers, plus both discriminators and their optimizer states. The current
launcher supports scratch training only; resume requires restoring all of those
and environment-side curriculum state, not just loading an actor.

W&B logs upstream `wm/*`, `policy/*`, `critic/*`, real episode statistics,
C's environment tracking/recovery metrics and `Discriminator/*`. The final
launch record and validation evidence are kept under `reports/`.
