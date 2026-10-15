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
by `0.01` only after at least 90% of the resetting cohort has an episode-final
HoST torso-above-feet metric above `0.60 m`. The
state-observation normalizer remains frozen, while a forward hook synchronizes
only the six action-scale history means to the current runtime scale. A new
bounded height-upright retention reward dominates the lying-stage progress
loophole, whose weight is reduced eightfold. This phase must first preserve
near-standing while reaching scale `0.25`; middle, low, and obstacle stages
remain separate follow-on work.

The corrected V8b run completed 1,800 updates without NaN/Inf or abnormal
termination. Its final checkpoint is
`logs/a6_29_v35_v8b_near_homotopy_4096_u1800_20261013/model_1800_final.pt`
with SHA256
`be1f6c31f4d1718dbc07e2434d4be19cb936c750269b72bfcf95b6c91630090d`.
The gate reduced scale from `0.932387` to `0.622388` by update 700, then held
that scale through update 1750 while the controller relearned. Near-standing
evaluation improved at the plateau from `89/128` held and `35.16%` final
standing at update 800 to `117/128` held and `61.72%` final standing at update
1600; update 1750 reached `115/128` held and the best final-standing rate,
`64.06%`. This is the first clear retained-control improvement below the
previous approximately `0.65` collapse boundary. A final successful training
cohort lowered scale once more to `0.612388`, but the unadapted update-1800
checkpoint fell to `108/128` held and `53.12%` final standing. V8b therefore
shows that gated homotopy works, but one successful cohort and a `0.01` step
are still too aggressive for robust transfer.

V9 resumes the full actor, critic, optimizer, and normalizers from V8b update
1750 (SHA256
`5aff2e12cf49148c4ec303fa61317c37cb69a763b28d4436d9f1b1d579711fa3`)
at scale `0.622388`. It lowers the step to `0.005`, requires two consecutive
cohorts with at least 90% episode-final head height above `0.70 m`, reduces
exploration standard deviation to `0.06`, and uses learning rate `1.5e-5`.
This continuation is intended to consolidate each lower-control plateau
before another scale change; it does not yet claim formal-scale or obstacle
success.

The V9 64-environment, 1,100-step preflight passed with finite rollout and
gradients, restored critic/optimizer state, all target rewards active, and no
unintended scale change: the final cohort had `70.31%` above the stricter
`0.70 m` gate, below the required `90%`. The 4,096-environment run is active
in `logs/a6_29_v35_v9_confirmed_homotopy_4096_u3000_20261013`; update zero
preserved scale `0.622388`, and early PPO updates are numerically healthy.

Fixed-scale near diagnostics through V9 update 800 show real consolidation
without another curriculum step. Updates 50, 100, 200, 400, and 800 achieved
respectively `111/128`, `113/128`, `118/128`, `120/128`, and `117/128` HoST
holds; final-standing rates rose from `59.38%` to `60.16%`, `63.28%`,
`63.28%`, and `69.53%`. At update 800, mean head height at step 1149 was
`0.9234 m` with uprightness `0.9115`, compared with `0.8928 m` and `0.8929`
for the fixed-scale V8b update-1750 source. The scale and all six synchronized
normalizer means remained exactly `0.622388`, so this gain is controller
learning rather than a hidden scale change. V9 remains active because the
stricter two-cohort gate has not yet been met.

The stricter gate subsequently triggered real scale migration. Update 1200
was evaluated at its checkpoint-native scale `0.617388` and retained
`113/128` holds with `60.94%` final standing, `0.9308 m` step-1149 head height,
and `0.8995` uprightness. Further confirmed cohorts lowered scale in bounded
`0.005` steps to `0.582388` by update 1550, where it then stopped decreasing.
The update-1800 diagnostic at that lower scale reached `108/128` holds but only
`43.75%` final standing, with step-1149 head height `0.7939 m` and uprightness
`0.8734`. Training remained finite with zero velocity terminations, so this is
an adaptation lag after cumulative authority reduction rather than a numerical
failure. The update-800 checkpoint remains the best fixed-scale retention
point so far; V9 continues at `0.582388` to test whether consolidation recovers
before any further scale change.

V9 completed 3,000 updates without NaN/Inf or abnormal termination. Its final
checkpoint is
`logs/a6_29_v35_v9_confirmed_homotopy_4096_u3000_20261013/model_3000_final.pt`
with SHA256
`5ea66e9fbd2dab7bda48281af9638e8aa32802087641e56f1c3fb5a1938aefb1`.
The curriculum eventually reached scale `0.567388`, but fixed-scale retention
did not recover: update 2400 at scale `0.572388` achieved `97/128` holds and
`44.53%` final standing, while the final checkpoint achieved `102/128` and
`42.19%`. Their step-1149 head heights were only `0.7801 m` and `0.7879 m`.
The failure mechanism is now observable: the V9 gate inspected only
episode-final head height. The lower-scale policies often fell soon after the
30-step motor-off window (`0.6743 m` and `0.6659 m` mean head height at step
100), then recovered by step 400 (`0.9101 m` and `0.8774 m`). Episode-final
cohorts could therefore authorize another scale reduction even though the
controller no longer retained a near-standing reset continuously.

V10 restarts from the best V9 update-800 checkpoint (SHA256
`c7693f8449819471b83a8d9fcf6af8f5ae2bc8917e4462f4a2b4678c8fd02125`) at
scale `0.622388` and replaces the endpoint-only gate with an episode-retention
gate. After the 30-step motor-off interval, each environment must keep head
height above `0.70 m` and uprightness above `0.80` for at least 80% of its
controlled steps and finish retained; at least 90% of the 4,096-environment
cohort must pass for three consecutive episodes. Scale changes remain bounded
to `0.005` and are separated by at least 14,400 control steps (600 updates).
The actor/critic/optimizer state is resumed, with standard deviation `0.05`,
learning rate `1e-5`, and two PPO epochs. A 64-environment, 1,100-step
preflight kept scale fixed, reported a `59.38%` retained cohort and `79.31%`
mean retained-step fraction, and passed finite reward/gradient checks. The
2,400-update run is active in
`logs/a6_29_v35_v10_retained_near_4096_u2400_20261013`; update zero was saved
and early PPO/GPU execution is healthy.

Through update 800, V10 correctly held scale at `0.622388`. The fixed-seed
near diagnostics at updates 400, 600, and 800 produced respectively `116/128`,
`119/128`, and `119/128` holds with final-standing rates `65.63%`, `67.19%`,
and `63.28%`; the source V9 update-800 checkpoint remains better on that last
single-time predicate (`69.53%`). The new gate audit, however, shows the
intended continuous-retention signal improving materially: the source
preflight retained only `59.38%` of environments for the required share of
controlled steps, while V10 updates 600 and 800 reached `79.69%` and `81.25%`.
Mean retained-step fraction rose from `79.31%` to `83.54%` and `84.45%`, and
episode-final head-height pass fraction reached `96.88%` and `98.44%`. The
90% cohort threshold has therefore not been met, so preventing a scale drop is
the correct behavior; V10 continues consolidating rather than launching a new
method from the noisier final-step diagnostic alone.

At update 1200, scale is still correctly fixed at `0.622388`. The same
128-environment near diagnostic improved held success to `121/128`, with a
`60.16%` final-standing rate, step-100/400/1149 mean head heights of
`0.7664/1.0470/0.9117 m`, and final uprightness `0.8953`. More importantly,
the retention audit advanced again to an `85.16%` retained cohort, while the
mean retained-step fraction remained strong at `83.30%` and the episode-final
head-height pass fraction was `96.88%`. This is the best continuous-retention
cohort so far, but it remains below the deliberately conservative 90% scale
gate, so training continues at the current scale.

V10 completed all 2,400 updates without numerical or termination failures;
the final checkpoint SHA256 is
`d40968a685792caba3673a6732888c59b42d9532d82672769e4e8ce5ca86b24d`.
Scale correctly remained `0.622388` throughout because the retention gate did
not pass. Update 1200 was the best retention checkpoint (`85.16%` cohort).
Update 1800 reached `84.38%` with `109/128` held and `60.16%` final standing;
the final checkpoint fell to `78.91%`, `107/128` held, and `58.59%` final
standing. The late decline at unchanged scale shows optimization drift rather
than a scale-transition failure, so the final checkpoint is not the source for
the next run.

V11 restarts from V10 update 1200 (SHA256
`4929accdae244812bf205c785a5cc2cd673fee3e5cf271174f38e913066304c1`) and
targets conservative retention consolidation before any further scale decay.
It raises the bounded near-retention reward weight from `4` to `6`, reduces
standard deviation from `0.05` to `0.035`, and reduces learning rate from
`1e-5` to `5e-6`; the 90% three-cohort scale gate and 600-update minimum hold
remain unchanged. A 64-environment preflight kept scale at `0.622388`, measured
an `85.94%` retained cohort and `83.43%` retained-step fraction, and verified
finite rewards and gradients. The 1,200-update run is active in
`logs/a6_29_v35_v11_retention_consolidation_4096_u1200_20261014` on GPU 0.

V11 completed 1,200 updates without numerical failures; the final checkpoint
SHA256 is
`eb727712d0c2e0976e84167c896b70473ba5480e468a4b09a81d51d09be78d25`.
The stronger all-episode retention reward did not improve the gate metric:
updates 600, 800, and 1200 reached only `70.31%`, `68.75%`, and `70.31%`
retained cohorts, versus V10 update 1200's `85.16%`. Their fixed-seed held
counts were `117/128`, `116/128`, and `113/128`; scale correctly remained
`0.622388`. This rejects further simple retention-weight increases or smaller
learning-rate/noise variants.

V12 changes the optimization scope and timing rather than reward magnitude.
It returns to V10 update 1200, freezes the actor backbone, trains only the
final output layer, restores the retention weight to `4`, and adds a bounded
early-recovery reward over the first 120 controlled steps after motor
activation. Standard deviation is `0.04` and learning rate is `1e-5`. A
64-environment preflight verified that only the output layer receives actor
gradients, the new reward is active and finite, scale stays `0.622388`, and
the rollout has no NaN/Inf. The 1,200-update run is active in
`logs/a6_29_v35_v12_early_recovery_4096_u1200_20261014` on GPU 0.

V12 completed 1,200 updates without numerical failures; final SHA256 is
`fbda08e4e3cb6e1c3a93762ca9885045caea42ab4c7df993bb5bcdd38030673c`.
The output-layer-only early-recovery hypothesis also failed to improve
continuous retention. Updates 600, 800, and 1200 reached retained cohorts of
`73.44%`, `67.97%`, and `57.81%`, all below the V10 source's `85.16%`.
Although final standing reached `67.19%` at update 600 and `67.97%` at the
final checkpoint, the gate correctly rejected those endpoint-only gains.

V13 treats adaptation as an explicitly bounded lower-body residual. It again
starts from V10 update 1200, freezes the actor backbone and all upper-body
output rows, and trains only output rows 0-12. Final-layer changes are clamped
to `0.003` per weight and `0.001` per bias; frozen-output exploration standard
deviation is `0.005`, and optimizer moments are reset. The 120-step early
recovery reward remains active, while the original retention gate is
unchanged. A 64-environment preflight verified exact frozen-row gradients,
finite lower-row gradients/rewards, bounded scale, and no NaN/Inf. The
1,200-update run is active in
`logs/a6_29_v35_v13_bounded_recovery_4096_u1200_20261014` on GPU 0.

V13 completed 1,200 updates without numerical failures; final SHA256 is
`dc8c28cf16e751c5369b8b03cd27348d3f8bd5b09bf4747d34bef89291a40f8c`.
The bounded lower-body residual did not exceed the V10 update-1200 source.
At updates 600, 800, and 1200, fixed-seed near-standing held counts were
`115/128`, `118/128`, and `118/128`, with final-standing rates of `62.50%`,
`67.97%`, and `66.41%`. Retention audits at update 800 and final measured
`71.88%` and `83.59%` retained cohorts, both below the source's `85.16%`.
Scale therefore correctly remained `0.622388`. The final checkpoint recovered
most of the source retention but did not improve it, rejecting ordinary PPO
adaptation of a tightly bounded lower-body output residual. The next attempt
must change the state distribution or supervision: use motor-on recovery
resets collected from actual early-failure trajectories, or offline action
distillation, rather than another reward-weight or output-bound variant.

V14 implements that state-distribution change from the unchanged V10
update-1200 source. A 4,096-environment source rollout captured finite failed
motor-on states at controlled steps 40-160. The versioned recovery bank
`datasets/reset_banks/a6_29_motor_recovery_v3/train.npz` contains 7,487 rows
with all four directions represented; 95.19% of the captured rows later
returned above the collection height/upright thresholds under the source
policy. Its SHA256 is
`906cb43b9d76dd11f7b53881a67cf46e862b19b2a57ade6d55168674283a7a03`.
Training samples only those recoverable rows for half of each reset cohort,
restores root and joint position/velocity, and enables motors immediately;
the other half retains the ordinary near-standing reset and passive 30-step
settling window to constrain forgetting. The actor, critic and normalizers
resume from V10, optimizer moments are reset, scale remains `0.622388`, and
the original retention gate remains authoritative.

The 64-environment, 1,100-step V14 preflight passed finite state, reward,
gradient, action-scale, reset-mixture, and immediate-motor checks. Under the
unmodified source policy, replayed recovery states improved from 0% retained
at reset (mean head height `0.489 m`) to 84.38% at 100 steps and 100% at 400
steps, with no velocity termination; ordinary near resets reached 78.13% and
100% at the same probes. The combined final retention audit was 68.75% for
this small seeded cohort, so no scale change was authorized. The formal
1,600-update V14 run is active in
`logs/a6_29_v35_v14_recovery_replay_4096_u1600_20261014` on GPU 0. This is an
acquisition experiment, not yet evidence that retention exceeds V10's 85.16%
or that formal A6 low-reset recovery succeeds.

V14 completed all 1,600 updates without numerical or runtime failure; final
SHA256 is
`bef8a7e54f63114c6890b91225f50645413ff0f686dbaf033de1475a50610511`.
The fixed-scale checkpoint sweep did not exceed the unchanged V10 update-1200
source. Retained-cohort fractions at updates 200/400/600/800/1200/final were
`81.25 / 75.00 / 78.13 / 84.38 / 81.25 / 71.88%`, versus the source's
`85.16%`; scale therefore correctly remained near `0.622388` throughout.
Update 800 is the best V14 compromise, with near held `119/128`, final standing
`64.06%`, mean retained-step fraction `85.74%`, and SHA256
`35456085b1bc5615e322cc62d0c447465eb4e6bb1752c1eda71d98bbd636a117`.

The reset-stratified audit explains the failure. Replay-state recovery at step
100 rose from `78.13%` at update 200 to `89.06%` at update 1200, and nearly all
replay states were retained by step 400. Ordinary near resets did not improve
in parallel: their step-100 retention stayed around `73-78%`, while the mixed
cohort never passed the 90% gate and fell to `71.88%` at final. V14 therefore
learned the replay distribution but did not transfer that gain into stronger
continuous retention from natural near-standing states. This rejects another
ordinary PPO extension of the same mixed replay objective. A follow-up should
use explicit teacher-action distillation or a history-aware residual trained
on paired failure/recovery trajectories, while constraining behavior on the
V10 source distribution. These results remain a 29-DoF HoST A6 adapter
diagnostic and are not formal full-method A6 success.

## V15: V37-seeded retained high-scale homotopy (2026-10-15)

V15 restarts A6 acquisition from the selected V37 update-100 stability actor,
not from V14. The immutable source is
`logs/host29_posture_v37_4096_u300_20261014/model_100.pt`, SHA256
`6317a1990393bcb603cec449042f30302b71a70a25e50a894e7af57e41945e09`.
At its native action scale `0.9323874`, the fixed-seed 128-environment near
baseline retained HoST standing in `128/128` environments and finished
standing in `98.44%`, so the quieter V37 actor is a materially stronger A6
restart point than forcing the previous `0.622388` controller through more
ordinary PPO variants.

V15 uses flat near-standing resets, nominal dynamics, no obstacle reward, a
fresh critic and optimizer, and the full inherited actor. The action scale is
allowed to fall by `0.01` only after two complete reset cohorts each reach at
least 90% episode retention, where retention requires height above `0.70 m`
and uprightness above `0.80` for at least 80% of controlled steps and at the
episode end. This high-scale bridge stops descending automatically when
continuous retention weakens; later low-scale consolidation remains a
separate stage.

The 64-environment, 1,100-step preflight passed with finite rewards, gradients,
actions and observations. It measured `98.44%` retained cohort and `93.49%`
mean retained-step fraction; after two successful cohorts the scale changed
exactly once from `0.9323874` to `0.9223874`, validating the gate and decrement.
Formal 4,096-environment training runs on GPU 0 in
`logs/a6_29_v37_v15_retained_homotopy_4096_u2400_20261015`. This remains a
29-DoF HoST A6 adapter acquisition experiment, not formal A6 success.

V15 completed all 2,400 updates without numerical or termination faults. The
scale reached `0.8123875` by update 800 and then stopped. A new uninterrupted
128-environment audit removes the historical diagnostic's 20-second reset:
update 800 retains `126/128` environments, has a `97.04%` mean retained-step
fraction, and finishes retained in `99.22%`. It is the selected V15 checkpoint
(SHA256 `4ee30c668ba7bc3ac04a575c5ebb48d85a6a01e9ddf1e7c0df1cf946f17f22bd`).
At formal action scale `0.25`, source, update 800, update 1200, update 1600, and
final all score `0/256` primary for both G- and G+; escape remains only
`0-1/256`, with zero abnormal termination. V15 is therefore meaningful
high-scale acquisition progress but not formal A6 progress. Full metrics,
protocol caveats, and the reset-time gate diagnosis are in
`A6_29_V15_FINAL_20261015.md`.

The V15 scale gate was then corrected to cache the episode-final retained
predicate before reset instead of recomputing uprightness from the newly reset
pose. V16 completed 2,400 updates from V15 update 800 without numerical or
termination failure. The corrected gate authorized exactly three `0.005`
decrements after their mandatory hold periods: action scale moved from
`0.8123875` to `0.8073874`, `0.8023873`, and finally `0.7973875`, with no jump
or premature change.

Every V16 checkpoint from update 200 through final retained the same `126/128`
environments in the uninterrupted 1,150-step audit. At the final checkpoint,
mean retained-step fraction is `96.58%`, final retained is `98.44%`, and final
head height/uprightness are `1.092 m / 0.908`. Final SHA256 is
`346064cbcc80ef34fb3d58cf1497ecfceb3f9fb1d6314e371a45957db1e6dac8`.
This proves the cache fix supports real scale migration while preserving near
retention, but `0.7974` is still far from formal action scale `0.25`; no formal
A6 success is claimed. Full results are in `A6_29_V16_FINAL_20261015.md`.

V17 continues from the V16 final checkpoint with the same full-state resume,
`0.005` decrement, three-cohort `90%` retained gate, std `0.04`, and learning
rate `7.5e-6`. Because V16 showed stable `98.44%` external retention across all
three migrations, only the redundant minimum hold is shortened from 14,400 to
2,400 control steps. The gate remains the safety stop; middle, low, and
obstacle resets remain disabled during this flat near-scale continuation.
Its 64-environment, 1,100-step preflight retained `98.44%` of the cohort,
averaged `92.07%` retained controlled steps, kept scale exactly `0.7973875`,
and passed finite reward/gradient checks. The 4,096-environment, 6,000-update
run is active in
`logs/a6_29_v37_v17_accelerated_cached_4096_u6000_20261015`.

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

For the 29-DoF G+ adapter, the vertically guided plate also exposes a bounded
robot-center separation shaping term. It measures horizontal distance from the
robot root center to the guided-plate center, is zero through `0.05 m`, rises
smoothly to one at `0.45 m`, and is enabled only after motor activation and
confirmed plate contact. Its pre-dt weight is `0.20`. This supplements rather
than replaces the historical `0.01` incremental separation-progress term: the
new signal gives the policy a persistent direction for moving its body center
off the up/down plate while remaining bounded and inactive during settling.

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
