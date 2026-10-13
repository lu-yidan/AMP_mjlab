# HoST historical A6 migration

## Status

### V35 paired A6 launch (2026-10-12)

The relaxed posture gate accepted the 29-DoF V35 update-400 checkpoint as the
new A6 source. Its SHA256 is
`abe5850708a1da9a6fd3167af10d30b976dbc168128177bcd598f2b37e140da3`.
Four-posture evaluation held and finished standing in 100% of environments
with zero termination. Compared with V25, yaw and horizontal motion improved
slightly at update 400. Video remains visibly leaned with one raised arm, but
it is stationary after recovery and shows no knee walking, repeated stepping,
or sustained spin, which is sufficient for the user-approved A6 transition.

Both G- and G+ passed a 64-environment, 510-control-step preflight, including a
complete 10-second training episode and reset. Paired formal training is now
running from the same source checkpoint:

- G-: `logs/a6_29_v35_gminus_4096_u10000_20261012` on GPU0;
- G+: `logs/a6_29_v35_gplus_4096_u10000_20261012` on GPU1.

Both runs use 4096 environments, 10000 updates, learning rate `2e-5`, fixed
action-noise standard deviation `0.05`, and checkpoints every 250 updates.
This deliberately replaces the historical adapter default noise `0.8`, which
previous diagnostics showed could erase flat-ground recovery. The two runs
retain separate fresh critics/optimizers and the frozen inherited actor
normalizer. Early checkpoint audits must verify flat-ground retention before
the long runs are treated as useful A6 adaptations.

### Formal-aligned V2 branch (2026-10-13)

The first V35 G-/G+ pair remained numerically healthy through update 1000 and
retained 100% HoST flat-ground held/final standing, but both formal 256-env
train-bank evaluations remained at `0/256` primary success and `1/256`
geometric escape. The flat diagnostic converged to about `1.09 m` head height,
below the formal `1.15 m` threshold. This is not flat-skill collapse; it is a
training/evaluation mismatch and a sparse local optimum.

Three concrete causes motivate a separate method instead of changing the two
healthy long runs in place. First, their inherited action scale begins near
`0.932` and remains about `0.81` at update 1000, while formal evaluation uses
exactly `0.25`. Second, G+ multiplies all common recovery tasks by `0.05`
under an uncleared plate, suppressing the height/upright signal where it is
needed most. Third, training terminates `invalid_plate` episodes after 25
steps without registered contact, while formal evaluation intentionally does
not apply that early abort. The existing smooth height reward is also almost
saturated at `1.09 m`, so it provides little gradient toward the hard
`1.15 m` success boundary.

V2 is therefore a formal-protocol-aligned curriculum, not another noise or
weight-only retry. A direct fixed-`0.25` preflight was rejected because the
V35 source never entered the new `1.04 m` height-margin reward region from
all-low resets. V2 instead starts at the source checkpoint's exact `0.932`
action scale and uses the historical performance-triggered decay at an
accelerated `0.02` per successful reset, bridging to the formal `0.25` target
while the policy can adapt. It keeps the V35 actor as an immutable backbone
and trains only a bounded final output layer. Training uses all-low formal
reset allocation, nominal dynamics, no observation noise or push/pull
curriculum, and no training-only invalid-plate termination. G+
geometry guidance remains active, but common recovery tasks are no longer
attenuated under the plate. Two bounded smooth rewards add gradient toward the
formal height margin and the combined height/upright/two-foot-load condition;
the quiet-feet cost is reduced during this acquisition phase. Exploration is
fixed at `0.12`, with learning rate `5e-5`, four PPO epochs, and output-layer
deltas clamped to `0.05` per weight and `0.25` per bias. The original G-/G+
runs continue untouched while V2 is preflighted and trained on GPU4.

The revised 64-environment bridge preflight passed on 2026-10-13. The source
action scale was recovered exactly as `0.932387`, the target remained `0.25`,
all-low reset allocation was `48` natural plus `16` procedural scenes, and the
new height-margin and primary-stance rewards were both finite and non-zero
(`0.00375` and `0.00213` mean absolute single-step contribution). Frozen actor
normalization, fresh critic/optimizer state, bounded output-layer training, and
the removal of `invalid_plate` termination were all verified without NaN/Inf.
The formal run uses `logs/a6_29_v35_v2_bridge_4096_u2000_20261013` on GPU4;
the original G-/G+ runs remain untouched on GPUs 0/1.

V2 completed all 2000 updates without numerical or termination failures, but
its core bridge never activated: every saved checkpoint retained action scale
`0.932387`. The curriculum trigger uses the mean episode-final head height over
the resetting all-low batch; that metric plateaued near `0.564 m`, below the
configured `0.70 m` trigger, even though the high-scale flat diagnostic still
reached about `1.17 m` after standing. The final checkpoint retained HoST flat
held success (`128/128`, final standing `99.22%`) at its saved high scale, but a
formal fixed-`0.25` 256-environment evaluation produced `0/256` primary and
`0/256` escape. V2 therefore learned useful high-scale posture shaping but did
not test the intended scale transition and is not a formal A6 improvement.

V3 changes only the failed mechanism. It starts from the V2 final actor and
replaces the success-triggered curriculum with a reset-synchronized linear
schedule: hold the source scale for 500 control steps, then move from
`0.932387` to `0.25` over 17,000 control steps regardless of plate success.
This preserves the V2 all-low resets, bounded final-layer adaptation, fixed
exploration, formal reward alignment, and fresh critic/optimizer. The actor
observes the current action rescale, so the forced bridge remains identifiable;
one high-scale acquisition episode is retained before the scale starts moving.

The 64-environment V3 preflight verified the complete schedule at `0.932387`,
`0.591194`, and `0.25`, with finite non-zero height-margin and primary-stance
terms (`0.00827` and `0.00445` mean absolute single-step contribution), correct
gradient scope, and no NaN/Inf. The formal run uses
`logs/a6_29_v35_v3_forced_bridge_4096_u1600_20261013` on GPU4, saving every
100 updates. The original G-/G+ runs continue unchanged on GPUs 0/1.

The live checkpoint audit exposed a counter-origin bug that the synthetic V3
schedule check did not cover. The V2 source restores
`common_step_counter=869100`; V3 evaluated its schedule against that absolute
counter, so `model_0` remained at `0.932387` but the first training reset jumped
directly to `0.25`. From update 100 onward, height-margin and primary-stance
rewards were zero and mean reward was about `-55`, so the run did not execute
the intended bridge and is not a valid method result.

V4 keeps the V3 method unchanged but anchors elapsed bridge time to the source
counter immediately after checkpoint restore. Its schedule is therefore based
on `common_step_counter - bridge_start_counter`, preserving the 500-step hold
and 17,000-step transition even when the source carries a large historical
counter. A new 64-environment preflight anchored the restored counter at
`869100`, rechecked the start/midpoint/target scales as
`0.932387/0.591194/0.25`, and retained finite non-zero height-margin and
primary-stance contributions (`0.00829` and `0.00425`) with the expected
gradient scope and no NaN/Inf. The invalid V3 process was then stopped without
touching the original G-/G+ runs. The corrected formal run uses
`logs/a6_29_v35_v4_relative_bridge_4096_u1600_20261013`; its initial checkpoint
and first update both retained scale `0.9324`, confirming that the inherited
counter no longer skips the bridge.

V4 completed the intended bridge cleanly, reaching scales `0.8521`, `0.7518`,
`0.5711`, and `0.25` at updates 100, 200, 400, and 800. The policy nevertheless
lost recovery continuously below about `0.65`: mean reward fell from `156.7`
at update 100 to `22.7` at update 400 and `-58.9` at update 800; episode-final
head height fell from `0.546 m` to `0.374 m` and `0.112 m`. At fixed formal
scale, updates 800/1200/1600 had zero flat HoST holds and zero formal primary
success, with only `1/2/1` of 256 geometric escapes. Multiplying the inherited
actor output logits by gains from `0` through `3.73` also produced zero of 64
flat holds at scale `0.25`. The failure is therefore not bridge timing or a
single output-amplitude calibration: the bounded final layer cannot relearn
the state-feedback policy as control authority changes.

V5 changes the training hypothesis. It fixes action scale at the formal `0.25`
from the first step, trains the full actor rather than a bounded output layer,
and uses a reverse reset curriculum with 25% low, 37.5% middle, and 37.5%
near-standing states in every scene/direction cell. A bounded stage-progress
reward remains informative below the final height threshold, while the formal
height/upright/two-foot-load terms and nominal protocol alignment are retained.
This phase is intended to acquire a formal-scale controller before a later
all-low consolidation phase; it does not claim low-reset success in advance.

The 64-environment V5 preflight passed with the fixed `0.25` scale, the
expected mixed reset allocation (`16/28/20` low/middle/near at this small
batch size), non-zero gradients in both the first and final actor layers, and
finite rollout statistics without NaN/Inf. Mean absolute single-step reward
contributions were `0.00102` stage pose, `0.00179` height, `0.00285` upright,
`0.01676` stage progress, `0.0000063` height margin, `0.000000044` primary
stance, `0.000178` geometry, and `0.000324` completion. This confirms that the
new dense acquisition signal is active while the formal sparse objectives
remain measurable rather than dominating the initial update.

The original paired runs subsequently completed all 10,000 updates without a
runtime failure. Their final checkpoint SHA256 values are
`6a9fbbc14c02d2e69e9259b5906fe90007466ed4b8ff6074c3e660f487e14f1c`
for G- and
`bad69ccaad502de7e6bdac5c0f56007a30a868ae7f81832f3fd96252af7eac19`
for G+. G- retained `128/128` flat HoST holds but ended at `79.69%` standing;
G+ retained `128/128` holds but ended at `72.66%`. Their final 256-environment
train-bank evaluations produced respectively `0/256` primary with `0/256`
escape and `0/256` primary with `1/256` escape. Neither is a formal A6
improvement, so no 2048-environment or video claim is made.

V5 also completed all 1,200 updates without NaN/Inf or abnormal termination.
Its final SHA256 is
`b2add6e6557a36815a4a9588ea531afffba366e344b7d7920423d9f096b6b57f`.
Updates 100, 200, 400, 800, and 1200 all remained at `0/256` formal primary;
the final result also had `0/256` escape and zero abnormal trials. Final low
flat evaluation had `0/128` HoST hold and `0%` final standing. More
importantly, update 800 failed separately from low, middle, and near-standing
resets. Near-standing states began around `0.955 m` head height and `0.839`
uprightness but fell to roughly `0.45 m` by step 100 and `0.42 m` by step 200.
The mixed reverse curriculum therefore failed to acquire even its easiest
stratum; extending it is not justified.

Two concrete implementation interactions explain the V5 collapse. First, the
inherited frozen actor normalizer still centers its six action-scale history
channels near `0.939-0.942` with standard deviations near `0.233-0.239`; the
formal `0.25` constant is consequently presented as an out-of-distribution
value near `-2.9` in every history slot. Second, legacy non-A6 posture terms
dominate the acquisition signal: around update 770, hip-yaw and shoulder-roll
episode penalties were about `-3.13` and `-1.88`, while stage progress was
only `+0.85`, height `+0.09`, and upright `+0.15`. Output-gain scans through
`3.73`, with and without action-scale-channel recentering at inference, still
produced zero near-standing holds, so this cannot be repaired by a static
gain alone.

V6 changes the acquisition boundary rather than extending V5. It trains only
flat near-standing natural resets at fixed formal scale `0.25`, disables all
legacy non-A6 style/target and obstacle rewards during this first controller
acquisition phase, trains the full actor, recenters the six action-scale
normalizer means to `0.25`, and permits the actor normalizer to adapt. It uses
a fresh critic/optimizer, standard deviation `0.10`, learning rate `5e-5`,
four PPO epochs, and stronger bounded stage-pose, height, upright,
stage-progress, height-margin, and primary-stance rewards. A 64-environment
200-step preflight verified exact flat/near reset allocation, fixed scale,
full-actor gradients, finite rollout values, and non-zero target rewards; the
largest target contribution was stage progress at `0.0418` per step. The
formal pilot runs in
`logs/a6_29_v35_v6_near_acquire_4096_u400_20261013`. Near-standing retention
is only its first acquisition criterion: middle, low, and obstacle curricula
must still be added in later phases before any A6 success claim.

The first V6 checkpoint audit exposed a runner-level issue before this claim
could be accepted. `model_0`, `model_100`, and `model_150` all retained actor
normalizer count `307200` and identical scale-channel moments despite the
requested adaptive setting. Checkpoint ONNX export recursively called
`eval()` on the policy wrapper but restored only device placement, not the
policy and normalizer training modes. Since update zero saves a checkpoint,
the normalizer stopped adapting immediately. Near-standing evaluations at
updates 25, 100, and 150 consequently remained at `0/128` hold with head
height near `0.43 m` by step 100 and `0.21-0.22 m` at the end.

The export path now snapshots and restores both training flags in a `finally`
block. A two-update, 64-environment V7 smoke run proved that normalizer count
continued from `206400` at the first saved checkpoint to `207936` after the
next checkpoint, while the six recentered scale-channel means remained near
`0.249923`. V7 keeps the V6 acquisition method unchanged so this single bug is
isolated, and runs in
`logs/a6_29_v35_v7_near_acquire_4096_u400_20261013` on GPU1. The original V6
pilot remains running as a non-adaptive control and is not stopped or
overwritten.

V6 and V7 both completed 400 updates without NaN/Inf, abnormal termination,
or a runtime failure. Their final SHA256 values are
`ff03383d2fd23a67864d3053a0a3e3b136c70dc7d79a3a469be1ec8e66ba4ea1`
and
`2d2abc35ee49024f3e86fb3b80b122bf0e8d6169542f6c87f089b224527327b3`.
The non-adaptive V6 checkpoints at updates 0, 25, 100, 150, 200, and 400 all
had `0/128` near-standing holds and zero final standing. V7 updates 25, 50,
100, 200, and 400 also all had `0/128` holds and zero final standing. V7 did
extend the middle of the first episode slightly: at update 400, mean head
height was `0.4058/0.4352/0.4642 m` at steps 100/200/400, compared with
V6-final `0.4361/0.4102/0.3836 m`. It still ended lower at step 1149
(`0.1566 m` versus `0.2179 m`) and never acquired a standing hold.

The normalization fix itself is verified rather than rejected. V6 remained at
normalizer count `307200` with all six recentered action-scale means fixed at
`0.246667`. V7 grew from `307200` at update 0 to `2,764,800` at update 25 and
`39,530,496` at update 400; its scale-channel means converged to
`0.232256..0.234717` and standard deviations to `0.066467..0.062205`.
Skipping the 30-step unactuated window at V7-final still yielded `0/128` hold
and zero final standing, so the startup mask is not the sole blocker. The
source V2 policy at its native scale retained `127/128` near-standing holds
and `99.22%` final standing, while multiplying its output by the exact
`0.932387/0.25` ratio at formal scale, with or without scale-channel
recentering, yielded `0/128`. The failure is therefore the abrupt loss of
control authority plus the need to relearn closed-loop feedback, not one
normalizer channel, one dead window, or one static output gain.

V8 changes the acquisition strategy to a gated homotopy instead of another
fixed-scale restart. It begins from the verified V2 near-standing controller
at its native scale, uses only flat near-standing resets, trains the full
actor with a smaller `2e-5` learning rate and two PPO epochs, and lowers scale
by `0.01` only after the HoST episode-final torso-above-feet metric exceeds
`0.65 m`. The
state-observation normalizer remains frozen, while a forward hook synchronizes
only the six action-scale history means to the current runtime scale. A new
bounded height-upright retention reward dominates the lying-stage progress
loophole, whose weight is reduced eightfold. This phase must first preserve
near-standing while reaching scale `0.25`; middle, low, and obstacle stages
remain separate follow-on work.

As of 2026-10-10, the source of truth for this migration is the frozen SMP
commit `1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2`. A dedicated local checkout is
kept at `../smp-a6-baseline` on branch `baseline/a6-adaptation`; files from
later recovery/S3 branches must not be substituted into this contract.

The 29-DoF V8- and V13-based A6 pairs completed the current HoST adapter's
paired `4096 x 10000` training schedule. They did **not** complete the unified
historical-A6 final evaluation specified by the 2026-09-28 handoff. Their
2048-environment evaluations used the older 23-second diagnostic predicate
(start before 20 s and hold for 3 s), not the required 20-second SR1/SR10
protocol with physical-substep load reporting. They remain useful interface
and failure-analysis results, but must not be cited as final cross-method A6
success rates.

The V8 checkpoints are:

- G-: `logs/a6_29_v8_gminus_4096_u10000_20261008/model_10000_final.pt`;
- G+: `logs/a6_29_v8_gplus_4096_u10000_20261008/model_10000_final.pt`.

Both V8 runs use the same source checkpoint and completed 14 diagnostic videos
per group. Under the legacy diagnostic, primary success is 0/2048 for both
groups; geometric escape is 24/2048 for G- and 12/2048 for G+, with no abnormal
termination. The V13 pair is also diagnostic: primary success is 0/2048 for
both groups, with geometric escape 19/2048 for G- and 58/2048 for G+. See
`A6_29_V8_FINAL_20261009.md` and `A6_29_V13_FINAL_20261010.md` for the original
counts, hashes, video findings, and artifact paths.

The first formal launch only encountered an output-directory ownership error:
the evaluator intentionally creates its own output directory. It was relaunched
with a non-existent result directory; neither checkpoint nor protocol state was
changed.

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
- 0.02 s control steps, 0.002 s physics steps, ten physics substeps;
- common task, execution cost, Q/L cost, and escape terms from the A6 table;
- paired `G-` and `G+` runs from the same native HoST checkpoint.

Run the asset gate before wiring the runtime scene:

```powershell
python scripts/validate_a6_host_assets.py --root .
```

The runtime now contains the three-scene entities, exact-bank sampler,
per-environment geometry/history buffers, grouped dynamics, delayed action,
shared task/cost layer, and paired `G-`/`G+` registrations. New formal long
training is nevertheless gated by the migration acceptance checklist in
`HISTORICAL_A6_PROTOCOL_AUDIT_20261010_ZH.md`, especially reward-trace parity,
native flat migration verification, reset screenshots, and the unified
20-second evaluator.

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
- `G` geometry uses the conservative world-XY projection of a rotating free
  plate and has no above-board exemption. The separate completion-clearance
  calculation does retain the frozen above-board exemption.
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

The frozen balanced-low code uses an `above-board` exemption only in the
completion-clearance calculation. The HoST adapter must retain it there, while
keeping it out of `G`'s horizontal geometry progress. The `.05` multiplier is
the implementation form of the A6 `alpha=.05` gate before confirmed escape;
after confirmation and on flat episodes alpha is `1`, and negative costs must
never be scaled by alpha.

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

The current adapter implements friction, encoder-bias, COM and push
randomization together with grouped mass/inertia, grouped Kp/Kd, and the
0--10 ms delayed-action ring buffer. Formal acceptance still requires exported
samples proving the actual MuJoCo fields, force caps, nominal fraction, and
zero/10 ms action-target behavior rather than only showing configuration flags.

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

The current A6 runtime now matches `physics_dt=0.002`, `decimation=10`, and the
0.02 s control period. The frozen reference starts GSI with
`compile_model=true`, whereas the HoST adapter currently disables compilation;
that optimizer/runtime difference must be disclosed but does not change the
physical timing contract.

## Formal evaluation boundary

`scripts/eval_a6_29_formal.py` currently delegates to the legacy
`scripts/eval_a6_23_formal.py` evaluator. It observes 1150 steps (23 s), uses a
3-second standing predicate, treats 4 cm horizontal clearance as a separate
diagnostic, and does not produce the frozen 20-second SR1/SR10 or complete
physical-substep load ledger. It must therefore not be used as the final
cross-method evaluator.

The unified evaluator still needs to:

1. run exactly 1000 control steps (20 s) at 50 Hz with nominal 6 kg plates;
2. require confirmed escape plus stable standing for plate-scene SR1/SR10,
   while flat scenes require stable standing only;
3. freeze one standing checker covering height, torso uprightness, body/joint/
   foot speed, and bilateral foot support;
4. report every scene and direction separately plus an unweighted scene macro
   average;
5. retain failed trials and report time-to-rise, longest hold, hard falls,
   post-escape displacement, intervention/protection events, and threshold
   definitions;
6. sample torque, joint speed, mechanical power, and sustained load on every
   2 ms physics step, then compute per-trial peaks and cross-trial P95 values.

## Initialization manifest

The exact historical starting state is recorded in
`a6_initialization_manifest.json`. It fixes the A6 actor checkpoint and SHA,
fresh critic/optimizer status, common critic SHA, 4096-environment reset
counts, all three reset-bank hashes, seed, event order, and code SHA. The
scene counts are `[2048, 1024, 1024, 0]` for flat, vertical guided plate,
free plate, and the unused fourth scene slot; the low/middle/near-standing
counts are `[2868, 820, 408]`.
