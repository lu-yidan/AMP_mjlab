# HoST historical A6 migration

## Status

As of 2026-10-07, the native 23-DoF A6 G-/G+ retries have been stopped at
updates 5000 and 5500 for diagnosis. Both runs remained numerically healthy,
but diagnostic videos showed no complete escape-to-stand sequence. The main
failure mechanisms are the training-only early invalid-plate termination,
quiet-motion local optima, the G+ constrained-task multiplier, and excessive
exploration noise. Do not resume either run unchanged.

The 29-DoF flat line is separate. The elbow-flexion v4 checkpoint passed the
four-posture zero-assistance evaluation; directional/stability v5 is an
in-progress continuation and has not yet passed final evaluation. See
`../host29_v5_a6_diagnostic_20261007_zh.md` for the exact run state, parameters,
artifact paths, and video findings.

This directory is the HoST-side migration boundary for the historical A6
stage-3 paired experiment. It is not the existing flat `Unitree-G1-HoST`
baseline and must not reuse `prone_smp_v1` or `supine_smp_v3` as an A6 claim.

The protocol requires:

- 50% flat, 25% vertical guided plate, and 25% free rigid plate;
- low/middle/near-standing reset strata of 70/20/10;
- four balanced low postures, with 75% natural and 25% procedural low states;
- the exact three reset banks and SHA256 values in `a6_protocol.yaml`;
- 0.02 s control steps, 0.005 s physics steps, ten physics substeps;
- common task, execution cost, Q/L cost, and escape terms from the A6 table;
- paired `G-` and `G+` runs from the same native HoST checkpoint.

Run the asset gate before wiring the runtime scene:

```powershell
python scripts/validate_a6_host_assets.py --root .
```

The next runtime step is to add the three-scene entities, reset-bank sampler,
per-environment geometry/history buffers, and the `G-`/`G+` HoST task
registrations. Until the banks and historical plate geometry are present, the
branch must remain preflight-only.

## Mapping from the historical entry point

In the historical SMP entry point, `build_config(..., arm="A6")` first calls
the D-series builder, then adds the path reset, hand-contact sensor,
per-substep load metric, standing metric, and the A6 reward arms. A6 belongs
to all three sets `G_ARMS`, `Q_ARMS`, and `L_ARMS`, so its effective additions
are:

```text
G: plate geometry progress (+0.45), dense clearance (+0.08)
Q: quiet feet cost (-0.03)
L: joint stall cost (-0.20)
```

The historical entry point also performs these checks before training:

1. export the resolved environment and agent configuration;
2. load the actor while keeping a fresh critic and optimizer;
3. restore the fixed reference normalizer;
4. audit reset counts, contact distance, zero initial velocity, and bank SHA;
5. audit randomized dynamics and a partial reset that must not touch other envs;
6. emit a launch manifest containing checkpoint/code/bank hashes;
7. run nominal and stress validation from saved checkpoints.

For HoST the checkpoint contract is different from SMP: preserve the native
HoST runner, critic/reward grouping, action constraints, and auxiliary
curriculum unless the experiment record explicitly says otherwise. The
SMP-specific actor-only loader and 93D/960D shape assertions must not be
copied into the HoST adapter.

## A6 path-state translation

The historical A6 path module confirms the following semantics:

- `A6` is in `G_ARMS`, `Q_ARMS`, and `L_ARMS`.
- Geometry uses the conservative world-XY projection of a rotating free
  plate. There is no above-board exemption.
- `G` is an incremental improvement over each environment's historical best
  coverage/clearance, gated by low body height, prior obstacle contact, and
  not-yet-escaped/non-invalid state. Hand support is required for A6.
- `Q` uses a continuous posture-only gate (`height` and uprightness); it does
  not require a contact or resettable timer.
- `L` is accumulated per joint from high actuator-load, low-joint-speed
  intervals. A moving unrelated joint must not suppress another joint's stall.
- Reset must clear best geometry, initial coverage, progress/clearance, gate,
  support, load accumulators, joint-history buffers, and the per-substep tick.

The HoST adapter will translate these to `_a6_*` state and use its own reset,
robot-bound, actuator and contact APIs. It must not call the SMP functions
`ra.reset`, `mt.robot_bounds`, or read `_ra_*`/`_mt_*` buffers directly.

## R2 transfer state needed by HoST

The companion R2 transfer module defines the non-obstacle state that must be
ported together with A6:

- flat environments are split into low/middle/near-standing groups with
  nominal `40%/40%/20%` allocation inside each direction; the global study
  remains `70%/20%/10%` after obstacle scenes are included;
- middle and near-standing states are sampled from the weighted natural
  curriculum bank, while the low subset keeps the 75% natural / 25%
  procedural source split;
- `prime_static_history` is applied after natural-state injection so history
  observations match the sampled state;
- load is accumulated on every physics substep, not only every control step;
  per-joint timers are reset when the normalized actuator load falls below
  `0.7`, and the stall term uses both load duration and low joint speed;
- torque, speed, power, high-load time, and longest high-load duration are
  tracked per joint for evaluation; a single stalled ankle must not disappear
  into a mean over all joints;
- standing convergence uses a hold gate over height, uprightness, foot force,
  and non-foot contact, then records pose, foot quietness, base quietness, and
  action slew as separate task components;
- planar escape credit is finite and based on confirmed progress over a rolling
  ten-control-step window; it is not an unbounded distance bonus.

For HoST, these buffers must be allocated per environment and partially reset
only for the requested `env_ids`. The HoST implementation must use its own
`physics_dt`, `decimation`, actuator-force field, action target, feet/contact
sensor names, and native reset-bank loader; copying the SMP `_ra_*` layout
without those API substitutions would make the preflight numbers invalid.

## Plate, completion, invalidity, and gating

The balanced-low implementation supplies the concrete plate lifecycle:

- park both plates away from the world, forward the simulation, and compute
  the robot's highest covered collision geometry;
- place the board axis-aligned at the robot root XY and at
  `top + half_thickness + 0.002 m`;
- use half-size `(0.45, 0.32, 0.035)`, i.e. full size
  `0.90 x 0.64 x 0.07 m`;
- use a mocap body for the vertically guided plate and a free joint for the
  free plate;
- ramp free-plate mass from `4 kg` to `4..12 kg` with
  `min(common_step_counter / 100000, 1)` and rescale inertia with mass;
- update contact, maximum force, and minimum signed contact distance from the
  selected plate sensor every control step.

The historical completion/invalidity rules are:

```text
completion candidate = prior contact AND no current plate contact
                       AND clearance >= 0.025 m
completion = candidate for 15 consecutive control steps
invalid = penetration depth < -0.02 m
          OR plate force > 1500 N
          OR episode step > 25 without prior contact
```

The pasted balanced-low code uses an `above-board` geometry exemption and a
`.05` multiplier on the task product while an obstacle episode is constrained.
That exact geometry exemption is not historical A6 and must be removed in the
HoST adapter. The `.05` multiplier is the implementation form of the A6
`alpha=.05` gate before confirmed escape; after confirmation and on flat
episodes alpha is `1`, and negative costs must never be scaled by alpha.

The escape terms remain separate: geometry progress, dense clearance,
completion, separation progress, and excess plate-force cost. Completion is a
state transition/hold criterion, not a one-shot visual bonus; separation is
allowed to continue under the historical A6 rule after escape.

## Scene geometry boundary

The shared geometry module fixes the actual A6 scene boundary:

- each MuJoCo world is an independent patch with no cross-environment contact;
- the active scene strata are only `flat`, `vertical_plate`, and `free_plate`;
- `BOXES` is empty, so stairs, slopes, stair edges, and slope edges must not be
  added to this experiment;
- scene quotas are exactly `50% / 25% / 25%`, with all four directions balanced
  within each quota;
- the free plate is a free-jointed box with half-size
  `(0.45, 0.32, 0.035)`, friction `(1.2, 0.01, 0.001)`, and the historical
  visual/solver settings;
- the terrain sensor target is retained as a parked, unreachable sphere;
  `scene_spec` must also install the deployment collision/contact pattern.

The HoST port must therefore add both plate entities and the parked terrain
sensor target to the scene construction. Adding only a reward function to the
existing plane scene would not be an A6 migration.

## R2/V33 common task and cost layer

The ordered-recovery transfer code defines the common layer that must be
adapted before adding A6's obstacle terms. It uses a per-environment four-stage
state machine, selected from the current physical height/uprightness on reset
rather than trusting the reset label:

```text
stage 0 -> 1: z >= 0.55, upright >= 0.55, knees >= 0.8, |vz| <= 0.16
stage 1 -> 2: z >= 0.78, upright >= 0.72, knees >= 0.6, |vz| <= 0.18
stage 2 -> 3: z >= 1.08, upright >= 0.85, feet load > 20, |vz| <= 0.12
```

The first two transitions hold for 10 control steps; the final transition
holds for 25. A fall (`z < 0.65` and upright `< 0.45`) returns the environment
to stage 0. The task components and weights are:

```text
stage_pose .22       head_velocity .18       height .10
upright .15          feet_quiet .08          base_quiet .07
angular_quiet .07    joint_quiet .06         action_quiet .07
```

Stage targets are phase-dependent: nominal upward velocity targets are
`0.06/0.08/0.10/0 m/s`, with relaxed targets
`0.10/0.15/0.20/0 m/s`. Near the target height, the velocity target is
attenuated. The height/upright smooth gate is used for foot, base, and
quietness terms; reset labels alone do not make a state eligible for a hold.

The common negative cost layer is sampled on every physics substep and reduced
over the control interval:

```text
action rate       -0.0015    action acceleration  -0.0012
joint acceleration -5e-8      torque               -1e-6
joint overspeed   -0.02       joint overpower      -2e-6
head overspeed    -1.0        sustained effort     -0.05
```

The HoST adapter must provide equivalent native fields for actuator torque,
joint acceleration, joint power, action target history, and per-joint effort
timers. These costs are common costs and remain active in both G- and G+;
alpha only gates the positive task product and must not attenuate them.

## Dynamics and reset contract

The R2/A6 dynamics module adds a second reset layer on top of the reset bank:

- body mass is scaled coherently for three groups: lower limbs, upper subtree,
  and pelvis/waist; a fourth audit column remains fixed at one;
- body inertia is scaled by the same group factor as mass, preserving the
  inertia-to-mass ratio;
- actuator gains use six groups: waist, hip, knee, ankle, arm, and wrist;
  each group's `Kp` and `Kd` share one sampled multiplier;
- the added dynamics range grows from `+/-10%` to `+/-20%` over the first
  2000 updates;
- 25% of episodes remain nominal, with no added mass/gain perturbation and
  zero command delay;
- non-nominal command delay is sampled as `0, 2, 4, 6, 8, 10 ms` and applied
  per environment through a six-entry action ring buffer;
- effort/force limits are not scaled.

The reset implementation must restore only the requested `env_ids`, then
recompute the corresponding model mass/inertia and actuator gains before the
robot reset. Evaluation fixes the obstacle mass to `6 kg`; training uses the
curriculum mass. A HoST preflight must verify actual model fields and action
targets without advancing simulation: mass/inertia, all six gain groups,
unchanged force caps, and both zero-delay and 10 ms-delay behavior.

The current HoST baseline has friction, encoder-bias, and COM randomization,
but does not yet have this grouped mass/inertia, Kp/Kd, or delayed-action
adapter. These are runtime requirements, not documentation-only metadata.

## Reward ledger

The 27-term three-stage index is recorded in
`reward_ledger.yaml`. It is an audit/readability layer, not a second source of
truth. Runtime code and the frozen expanded environment YAML remain the
authoritative numeric sources. In particular:

- Stage II costs ramp with `min(update / 500, 1)`;
- soft joint limits do not ramp;
- Stage III uses full costs (`eta=1`);
- `alpha` is one in Stages I/II and follows the Stage III standing gate;
- Q, L-prime, and all obstacle interaction terms are Stage III additions;
- the common `0.02 s` scaling is applied exactly once.

## Expanded resolved reference

The four runtime-facing sections from the supplied historical resolved
configuration are preserved in `a6_expanded_reference.yaml`: expanded reward
terms, startup/reset/interval/step events, termination predicates, and MuJoCo
simulation settings. It is an audit artifact only. In particular, its SMP
function paths are provenance and are not importable HoST implementations.

Two historical values require an explicit migration decision before launch:

- the reference uses `physics_dt=0.002`, `decimation=10`; the current HoST
  baseline uses `0.005`, `4` while keeping the same `0.02 s` control period;
- the reference starts GSI with `compile_model=true`; the current smoke test
  contract requires torch compilation disabled.

## Initialization manifest

The exact historical starting state is recorded in
`a6_initialization_manifest.json`. It fixes the A6 actor checkpoint and SHA,
fresh critic/optimizer status, common critic SHA, 4096-environment reset
counts, all three reset-bank hashes, seed, event order, and code SHA. The
scene counts are `[2048, 1024, 1024, 0]` for flat, vertical guided plate,
free plate, and the unused fourth scene slot; the low/middle/near-standing
counts are `[2868, 820, 408]`.
