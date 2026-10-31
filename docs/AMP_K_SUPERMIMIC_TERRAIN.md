# K-SuperMimic-Shared-Terrain

Branch: `lyd-supermimic-terrain`. Built from `lyd-terrain-critic-scan`.

K keeps I/F's controlled shared discriminator (one network and optimizer,
mode-specific normalization and replay), reset groups, experts, command curriculum,
rewards, action order/scales and terrain. It uses the existing MBRL SuperMimic
implementation instead of PPO. Actor observation remains 384-dimensional and
proprioceptive; critic observation is 1612-dimensional (four frames including
187 height samples). No explicit reference motion is supplied to the actor.

The world model receives physical randomization context and a local terrain
scan plus its validity mask. Every model step queries a static collision-terrain
raster at its own predicted position and yaw. Replay stores the original tile
origin, rather than implicitly reusing the current environment's origin. The
origin is used for querying, not fed as a terrain-ID feature to the MLP. Body
height in the model input is ground-relative. The map is saved in the model
state dictionary; its geometry fingerprint is recorded in the run config.

The 5 cm raster is generated from MuJoCo group-0 collision geometry with downward
rays. Bilinear interpolation provides pose gradients. It smooths stair edges;
it is not exact differentiable contact physics. The real simulator still uses
physical ray casts. Missing/below-surface rays have explicit validity bits.
Outside the raster, samples use border padding and are marked invalid. The
first pilot supports static height-field-like terrain only, not bridges,
overhangs, movable obstacles or reconstructed depth maps.

## Sensor cache correction

A verification run found up to 0.2813 m critic scan error: rewards/commands read
RayCastData before sense(), which filled the per-step sensor cache. Ray
postprocessing replaced the tensors but did not invalidate that earlier cache.
The critic then observed the prior scan, including pre-reset poses.
`FreshTerrainRayCastSensor` invalidates its cache after postprocessing. This fix
is shared by I/J/K configurations on this branch. Paused remote I/J source
snapshots have not been replaced, and the old queues remain paused.

Rewards and termination still execute before the next sense() in mjlab. Thus
ground-relative decisions can use an earlier scan. Verification reports both
agreement at the actual reward snapshot and differences when using the final
state; these are separate from map interpolation errors.

## Verification and launch

```sh
python scripts/train_amp_supermimic.py --terrain --num-envs 32 --verify-only --output logs/amp_k_verify
python scripts/train_amp_supermimic.py --terrain --num-envs 32 --iterations 3 --smoke --no-compile --output logs/amp_k_smoke
```

Numerical verification includes observations, each task reward, termination,
state gradients, and an off-platform scan audit across all generated tiles.
The terrain audit allows explicitly bounded interpolation errors, separately
from the stricter observation/reward checks on real trajectories. Passing these
gates does not establish learned world-model accuracy or task performance.

Planned full training: scratch, seed 42, 4096 environments, 24 real steps per
iteration, 100k iterations, save every 1000, W&B `tabletennis/lyd-multi-amp`.
World model and imagination settings retain E's 2048x3 model, 8-step model fit,
32-step imagined horizon and 300 warmup iterations. Compare real samples and
GPU-hours, not iterations alone. Do not launch until numerical and full-size
smoke gates pass. Existing jobs/queues remain paused by user request.

References: InstinctMJ's SuperMimic branch for terrain-aware imagination,
SuperMimic-perceptive for pose-dependent map queries, and local MBRL_supermimic
as the existing training core. No external checkpoint is imported.

## Initial validation results (2026-11-01)

Five unit/regression tests pass. The 32-env numerical gate with off-platform
queries offset by (0.013, 0.017) m passes: actor error 1.43e-6, critic error
8.39e-6, AMP error 4.91e-5, maximum task-reward error 1.86e-8, no termination
mismatches at the reward snapshot, gradient finite-difference error 1.77e-6.
Across 23,936 off-platform rays: mean error 0.51 mm, p99 14.54 mm, maximum
16.98 mm, no validity mismatches. These are geometry/MDP checks only.

The separate exactly grid-aligned audit is retained as
`reports/amp_k_grid_edge_audit_20261101.json`: up to 6.99% validity mismatch
in a pose batch. A 13/17 mm horizontal displacement removes those mismatches,
consistent with ray/mesh boundary degeneracy. The differentiable raster does
not reproduce these exact-edge ray misses. This limitation is not erased by
the passing offset audit; learned dynamics and terrain tracking still require
real-simulator evaluation. Both reports are retained.

K opts into fail-fast checks for non-finite world-model, actor and critic
updates, in addition to the controlled discriminator checks.

Deployment preflight: the first archive exposed a pre-existing packaging bug:
`.gitignore`'s `env/` rule excluded the vendored task protocol. Commit `4b7abd6`
adds the source and an explicit ignore exception. Server source is isolated at
`/root/workplace/lyd-supermimic-terrain-4b7abd6`. Full-size preflight uses GPU 0;
old jobs remain SIGSTOP-paused (and still hold memory). GPUs 5/6 are not used.
