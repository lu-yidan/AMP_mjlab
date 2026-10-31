# I/J terrain critic pilot

Branch: `lyd-terrain-critic-scan`. Reference implementation inspected read-only:
`/home/luyd/workspace/instinctMj-cra/src/instinct_mj/tasks/shadowing/perceptive/perceptive_env_cfg.py`
(height scanner and critic `height_scan` term). Uses the native mjlab raycast sensor
and height observation; does not import or modify InstinctMJ.

- I-Shared-TerrainCritic: F shared discriminator, balanced recovery sampling.
- J-Dual-TerrainCritic: H independent discriminators, balanced recovery sampling.
- Identical terrain, expert data, reset quotas, task rewards and speed curriculum.
- Critic-only 17 x 11 yaw-aligned rays, 0.1 m spacing, 1.6 x 1.0 m footprint,
  30 m range, clip [-20,20], no observation noise. Pelvis mounting and geom group 0
  exclude robot visual/collision meshes (groups 2/3). Reports relative ray heights.
- Existing four-frame critic history is retained: 864 -> 1612 dimensions. Actor
  remains 384, AMP remains 195. Old actor inference interface remains the same;
  old full actor/critic checkpoints cannot be loaded as-is into the larger critic.
- Pilot terrain: flat, pyramid slopes, inverted pyramid slopes, pyramid stairs,
  inverted pyramid stairs (20% requested proportion each). Slopes: 0.05–0.2 gradient;
  steps: 2.5–7.5 cm. 8 m tiles with 3 m central platforms, 3 x 10 tiles.
  Inverted tiles spawn in the central depression, so outward travel climbs; regular
  pyramid tiles spawn on top, so outward travel descends. Fixed difficulty distribution,
  not a terrain curriculum. Existing speed curriculum still targets up to 3 m/s.
- Stratified resets remain on the central flat platforms with tile-origin height.
  This deliberately does NOT implement arbitrary rough-surface recovery resets.
- Recovery gate, phase geometry, low-root termination and root-height reward use
  height above the central ray's ground estimate instead of world/tile height.
  Head height is measured against the pelvis ground reference, not a separate ray
  under the head. This approximation must be reconsidered at sharp terrain edges.
- Missing central rays fall back to the tile origin. Audit counts misses; this is
  not evidence that misses cannot occur during long running/falls. mjlab computes
  command/reward before the next `sense()` call, so these decisions can use the
  preceding sensor sample; the critic receives the refreshed scan. Do not claim
  exact instantaneous surface clearance from this pilot.

## Validation

`PYTHONPATH=/tmp/amp-eval100k-runtime:. /tmp/amp-eval100k-venv/bin/python
scripts/audit_terrain_critic.py` verifies 187 rays, finite observations, dimensions,
platform height and displaced-ground response. The script runs locally on CUDA0.

Short-training command (replace I with J for the dual control):

```sh
MUJOCO_GL=egl PYTHONPATH=/tmp/amp-eval100k-runtime:. \
/tmp/amp-eval100k-venv/bin/python scripts/train.py \
  Unitree-G1-AMP-I-Shared-TerrainCritic \
  --env.scene.num-envs 512 --agent.num-steps-per-env 24 \
  --agent.max-iterations 20 --agent.amp-num-preload-transitions 2048 \
  --agent.logger tensorboard --agent.run-name terrain-critic-512-preflight \
  --agent.seed 42 --agent.resume False --gpu-ids '[0]'
```

This is integration and numerical validation, not a trained terrain-navigation
result or evidence of dual-discriminator superiority. All tests use the local GPU;
existing server training and reserved GPUs are untouched. Before a long run, test
4096-environment memory/throughput, arbitrary-surface reset clearance, scan misses
and a disturbance/recovery/resume-command evaluation. Actor is blind to terrain;
privileged critic observations help training, but do not provide deployment-time
terrain perception or anticipation.

## Formal queue: 2026-11-01

Deployed immutable source `c3572be` to server
`/root/workplace/lyd-terrain-critic-c3572be`. Archive SHA256:
`f81a4686bcc152b2741042d7c1ec5743857f849514ab87eb95904b2d1e391b67`.
Scheduler PID at launch: `297594`; records under `records/terrain-20261101`.

- I-Shared-TerrainCritic: seed 42, GPU 1, waits for F43 completion.
- J-Dual-TerrainCritic: seed 42, GPU 0, waits for F42 and F44 completion.
- Both: from scratch, 4096 environments, 24 rollout steps, 100000 iterations, checkpoint every 1000.
- Each runs a 20-iteration full-size preflight first; any failed run or non-finite batch skip prevents formal launch.
- Formal runs use W&B project `tabletennis/lyd-multi-amp`, group `amp-terrain-critic-20261101`. URLs are written to the remote manifest only at launch.
- Queue does not use GPUs 5 or 6, and checks its assigned GPU has no compute processes before starting.

Verified scheduler alive and both jobs queued. This is not evidence of formal training having started. Local launch snapshot: `reports/amp_terrain_launch_20261101.json`.
