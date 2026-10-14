# 29 DoF posture V16 anti-spin continuation

## Motivation

V15-Scratch reaches and retains standing from all four postures with no
terminations and joint-speed RMS below 1.59 in the original formal evaluation.
Video review nevertheless shows continuous turning and stance drift. A second
seed-42 evaluation added motion diagnostics and confirmed that the missing
constraint is base yaw:

| posture | standing yaw-rate RMS | standing horizontal-speed RMS |
| --- | ---: | ---: |
| prone | 1.715 rad/s | 0.412 m/s |
| supine | 0.974 rad/s | 0.220 m/s |
| left side | 1.684 rad/s | 0.404 m/s |
| right side | 1.549 rad/s | 0.373 m/s |

The inherited `target_ang_vel_xy` reward only damps body roll and pitch rates;
it does not include rotation around world vertical.

## V16 change

- source checkpoint:
  `logs/host29_posture_v15_scratch_4096_u12000_20261009/model_12000_final.pt`;
- source SHA256:
  `1d469b97195c21fd4faea288fb39f02c4a6b8b63b5476b9f49b996f6de2923ff`;
- exact actor and actor-normalizer migration, fresh critic and optimizer;
- only the final `Linear(128, 29)` actor layer remains trainable;
- fixed action scale from V15, fixed action standard deviation `0.005`, learning
  rate `2e-6`, one PPO epoch, entropy `0`;
- action-scale and pull-force curricula disabled during continuation;
- new gated dense penalties after standing: world-yaw rate `-0.75` and root
  horizontal speed `-0.40`;
- motion penalties use a `0.65 m` root-height gate; V15's measured stable root
  height is only `0.686–0.690 m`, so the posture branch's `0.72 m` gate would
  never train the anti-spin correction;
- foot horizontal speed strengthened from `-0.20` to `-0.30`, and leg joint
  velocity from `-0.0025` to `-0.0030`.

The 64-environment, 100-step preflight passed. Two diagnostic launches were
discarded before producing a candidate: the first exposed an inherited active
action-scale curriculum, and the second showed that the `0.72 m` posture gate
sat above V15's actual stable root height. The corrected run uses a null
curriculum manager, leaves V15's action scale unchanged, and gates only the
motion penalties at `0.65 m`.

## Evaluation gate

V16 must preserve held and final-standing rates of at least 99%, zero
terminations, and joint-speed RMS at most 2.20 for every posture. It must also
materially reduce both standing yaw-rate RMS and horizontal-speed RMS, with
video confirmation that continuous turning and visible leg jitter are gone.

## Staged result and V17 fallback

V16 update 125 and 250 retained 100% held/final standing with zero
terminations, but standing yaw-rate RMS improved by only about 1-3% relative
to V15. The anti-spin terms are active, yet the final-layer-only `2e-6`
update is too constrained to escape the learned turning gait.

V17 therefore restarts from the untouched V15 final checkpoint, keeps the
normalizer and action scale frozen, and trains only the last two actor linear
layers. It uses one PPO epoch, fixed standard deviation `0.01`, learning rate
`5e-6`, and stronger gated penalties: yaw rate `-1.50`, root horizontal speed
`-0.60`, foot horizontal speed `-0.40`, and leg velocity `-0.0035`. This adds
adaptation capacity without the full-actor, two-epoch update that destabilized
V14.

Detailed V15 evaluation shows the dominant actuator-level asymmetry: standing
left-hip-yaw velocity is about `3.6-5.8 rad/s`, while right-hip-yaw velocity is
about `0.6-1.1 rad/s`. V18 runs in parallel from the same untouched V15 source,
adds a gated `-0.04` penalty specifically to the two hip-yaw joint velocities,
and permits full-actor adaptation at the conservative `3e-6`, one-epoch PPO
setting. V17 and V18 are compared at matched checkpoints before selecting a
final candidate.

## Interim measurements

V16 final preserves 100% held/final standing and zero terminations, but its
standing yaw-rate RMS is `1.699/0.922/1.669/1.531 rad/s` for
prone/supine/left/right. Against V15's
`1.715/0.974/1.684/1.549 rad/s`, only supine improves materially; the other
three postures improve by about 1%. V16 is therefore a safe but insufficient
candidate.

V17 update 250 also preserves 100% held/final standing and zero terminations.
Its yaw-rate RMS improves to `1.642/0.919/1.622/1.498 rad/s`, about 4-6%, but
left-hip-yaw RMS remains `5.922/3.408/5.721/5.271 rad/s`. The untargeted
anti-spin terms reduce some root motion without correcting the asymmetric
left-hip cycle, which is why V18 adds the dedicated hip-yaw term.

V17 update 500 reaches `1.606/0.880/1.581/1.443 rad/s` yaw-rate RMS while
retaining 100% standing, but sampled video still shows continuous turning.
V18 update 250 is not better than the matched V17 checkpoint and also leaves
the left-hip-yaw cycle nearly unchanged. V19 therefore increases only the
post-standing hip-yaw velocity penalty from `-0.04` to `-0.25`, uses full
actor adaptation at `5e-6` for one PPO epoch, and checkpoints every 125 updates
for early safety evaluation.

V17 final preserves 100% held/final standing with zero terminations and lowers
yaw-rate RMS to `1.515/0.792/1.479/1.374 rad/s`, an 11-19% improvement over
V15, but this remains visibly rotational. V19 update 250 still trails V17 at a
matched training stage. V20 keeps V19's gated rewards but masks actor gradients
so only output rows 2 and 8 (left/right hip yaw) can change. This permits a
higher `2e-5` learning rate while keeping all other action outputs bitwise
anchored to V15 at launch.

## V20 result and V21 design

V20 completes 500 updates with 100% held/final standing and zero terminations,
but its standing yaw-rate RMS is `1.743/1.069/1.735/1.610 rad/s`. This is worse
than V17 in every posture. The videos also retain continuous turning and
repeated corrective steps. Isolating the two hip-yaw outputs therefore removes
too much of the coupled lateral-balance response and is not a viable endpoint.

V21 starts from the untouched V17 final checkpoint and trains only final actor
rows `1/2/5/7/8/11/12`: bilateral hip roll/yaw, bilateral ankle roll, and waist
yaw. All other action rows remain bitwise frozen. It uses one PPO epoch,
learning rate `5e-6`, fixed standard deviation `0.005`, and 500 updates with
125-update checkpoints. The yaw/root/foot penalties are increased slightly,
the hip-yaw velocity penalty is moderated to `-0.15`, and a `-0.015` velocity
penalty is added for the coupled lateral-balance joints. This retains V17's
anti-spin representation while allowing the minimum actuator group needed to
rebalance without turning.

V22 is the less-conservative parallel run from the same V17 checkpoint and
with the same reward objective. It trains the final `Linear(256, 128)` layer
and the complete `Linear(128, 29)` action layer, uses learning rate `1e-5` and
fixed standard deviation `0.01`, and keeps the earlier actor layers,
normalizer, and action scale frozen. V21 and V22 therefore isolate whether the
remaining spin needs only a coupled lateral actuator subset or broader policy
coordination.

## V21/V22 final result

Both runs completed 500 updates and the formal 256-environment, 1000-step,
seed-42 four-posture evaluation. Every posture has 100% held and final
standing, zero termination, and final joint-speed RMS below 2.20.

- V21 final checkpoint:
  `logs/host29_posture_v21_4096_u500_20261010/model_500_final.pt`, SHA256
  `3af9866275695f28394824fb07229632981f2db5a11a80c765ead83960837496`.
- V22 final checkpoint:
  `logs/host29_posture_v22_4096_u500_20261010/model_500_final.pt`, SHA256
  `8191f00e3219913206358eeaec0460f4d844e99294b616c006946bdffa2dd8ba`.

Standing yaw-rate RMS, ordered prone/supine/left/right:

| model | rad/s |
| --- | --- |
| V17 | `1.515 / 0.792 / 1.479 / 1.374` |
| V20 | `1.743 / 1.069 / 1.735 / 1.610` |
| V21 | `1.511 / 0.771 / 1.456 / 1.355` |
| V22 | `1.465 / 0.739 / 1.406 / 1.307` |

Standing horizontal-speed RMS, in the same order:

| model | m/s |
| --- | --- |
| V17 | `0.382 / 0.184 / 0.379 / 0.353` |
| V20 | `0.418 / 0.245 / 0.425 / 0.395` |
| V21 | `0.381 / 0.178 / 0.373 / 0.349` |
| V22 | `0.370 / 0.169 / 0.361 / 0.337` |

V21 changes are too small: yaw improves only 0.3-2.6% and horizontal speed
0.1-3.0% relative to V17. V22 is consistently better, reducing yaw by
3.3-6.8% and horizontal speed by 3.2-8.0%; its final joint-speed RMS is
`1.453 / 0.721 / 1.426 / 1.303`, also below V17 in three of four postures.

The video gate still fails for both candidates. Keyframes at 0/4/8/12/16 s
show the robot repeatedly alternating between front and side views, with
continued corrective stepping, lateral torso lean, and arms held close to the
torso. V22 is the best numerical candidate, but the visible continuous
rotation is only reduced, not removed. Neither V21 nor V22 should be described
as an anti-spin solution without a new design decision.

V23 continues from V22, the best numerical candidate. The review exposed a
configuration issue: V15's torso vertical/tilt/support rewards still used the
old `0.72 m` gate, while the measured standing root height is only about
`0.685 m`, so those terms were effectively inactive. V23 lowers those three
gates to `0.65 m`, adds a dense foot-alignment error in the torso frame
(forward staggering, height mismatch, and stance-width error), and strengthens
the yaw/root/foot damping moderately. It retains V22's final-two-layer actor
scope, uses learning rate `7.5e-6`, fixed standard deviation `0.008`, one PPO
epoch, and 500 updates from the V22 final checkpoint.

## V23 final result

V23 completed 500 updates. Its final checkpoint is
`logs/host29_posture_v23_4096_u500_20261010/model_500_final.pt`, SHA256
`85be65faff3002086c20f8cc10cc46a5e03e27ae43ba76f35abda695a8955925`.
The formal 256-environment, 1000-step, seed-42 evaluation preserves 100% held
and final standing with zero terminations in all four postures. Final
joint-speed RMS is `1.450 / 0.715 / 1.364 / 1.336`, safely below the 2.20
gate.

Standing yaw-rate RMS, ordered prone/supine/left/right:

| model | rad/s |
| --- | --- |
| V17 | `1.515 / 0.792 / 1.479 / 1.374` |
| V22 | `1.465 / 0.739 / 1.406 / 1.307` |
| V23 | `1.431 / 0.740 / 1.372 / 1.286` |

Standing horizontal-speed RMS, in the same order:

| model | m/s |
| --- | --- |
| V17 | `0.382 / 0.184 / 0.379 / 0.353` |
| V22 | `0.370 / 0.169 / 0.361 / 0.337` |
| V23 | `0.364 / 0.170 / 0.356 / 0.338` |

Relative to V22, yaw improves only about 1.6-2.4% in prone, left-side, and
right-side starts, while supine is unchanged. Horizontal-speed RMS improves
about 1.5-1.7% in prone and left-side starts but is unchanged or slightly
worse in supine and right-side starts. The numerical change is therefore
small rather than a qualitative gait correction.

The video gate still fails. The 0/4/8/12/16 s keyframes continue to alternate
between front, side, and rear views, demonstrating persistent turning and
corrective stepping. Feet are not kept reliably side by side, and lateral
torso lean remains visible. Activating the previously gated torso rewards and
adding the dense foot-alignment objective improved several metrics slightly,
but did not break the inherited rotational gait. V23 is stable and is the best
numerical candidate so far, but it is not an anti-spin solution.

## V24 design

V24 continues from V23 but changes the objective rather than merely increasing
the existing speed penalties. V23's relative foot geometry can be satisfied
while the whole robot rotates, so V24 adds an absolute world-heading error,
world-facing and level-foot orientation, absolute foot-floor clearance, and a
foot-position anchor captured after 0.6 s of standing. The anchor is reset per
episode and is disabled again if the robot falls below the standing gate.

For lateral upper-body bias, V24 separately centers the pelvis, torso, and full
torso/arm chain over the ankle midpoint, strongly penalises torso tilt, and
drives waist roll/pitch to zero. All reward terms carrying a post-standing gate
are changed from the inherited `0.72 m` to the measured reachable `0.65 m`;
this also activates the previously dormant upper-pose and hands-beside-hips
terms. Yaw, root translation, foot speed, leg motion, and side-by-side stance
weights are increased together so the policy cannot trade one form of stepping
for another.

V24 trains the last three actor linear layers from the untouched V23 final
checkpoint, with a fresh critic/optimizer, frozen normalizer and action scale,
fixed standard deviation `0.008`, learning rate `5e-6`, one PPO epoch, and
1000 updates. This is intentionally less conservative than V23 while retaining
the earliest actor representation.

Artifacts:

- V21 evaluation: `logs/evaluations/host29_posture_v21_final_256_seed42/evaluation.json`;
- V21 videos and contact sheet:
  `logs/evaluations/host29_posture_v21_final_256_seed42/videos/`;
- V22 evaluation: `logs/evaluations/host29_posture_v22_final_256_seed42/evaluation.json`;
- V22 videos and contact sheet:
  `logs/evaluations/host29_posture_v22_final_256_seed42/videos/`.
- V23 evaluation:
  `logs/evaluations/host29_posture_v23_final_256_seed42/evaluation.json`;
- V23 videos, keyframes, and contact sheet:
  `logs/evaluations/host29_posture_v23_final_256_seed42/videos/`.

## V24 final result and failure analysis

V24 completed 1000 updates. Its final checkpoint is
`logs/host29_posture_v24_4096_u1000_20261010/model_1000_final.pt`, SHA256
`ff554a1be6edad1c81def5b92f2ea0da720bae92ac4d2038034193643f657ca7`.
The formal 256-environment, 1000-step, seed-42 evaluation retains 100% held
and final standing with zero terminations in all four postures. Final
joint-speed RMS is `1.412 / 0.678 / 1.359 / 1.276`.

Standing yaw-rate RMS, ordered prone/supine/left/right:

| model | rad/s |
| --- | --- |
| V23 | `1.431 / 0.740 / 1.372 / 1.286` |
| V24 update 500 | `1.439 / 0.762 / 1.371 / 1.288` |
| V24 final | `1.424 / 0.733 / 1.363 / 1.278` |

Standing horizontal-speed RMS, in the same order:

| model | m/s |
| --- | --- |
| V23 | `0.364 / 0.170 / 0.356 / 0.338` |
| V24 update 500 | `0.366 / 0.177 / 0.353 / 0.335` |
| V24 final | `0.366 / 0.168 / 0.351 / 0.333` |

These changes are below 1% in most cells and are mixed for horizontal speed,
so V24 does not pass the improvement gate. Its 0/4/8/12/16-second contact
sheet is visually almost identical to V23: every posture still alternates
between front and side views, the feet repeatedly cross or stagger, and the
torso remains laterally inclined. Artifacts are under
`logs/evaluations/host29_posture_v24_final_256_seed42/`.

The failure is not evidence that still larger coefficients are needed. V24's
final return is about `-51,961`, versus `-6,019` for V23, and its value loss is
about `2.43e6`, versus `2.74e4`. The final per-episode foot-anchor and squared
side-by-side terms alone are approximately `-3,087` and `-1,032`. Surrogate
loss remains effectively zero, and the trained actor tensors move by less than
0.1% of their source norms. The oversized unbounded terms therefore overwhelm
the fresh critic while barely changing the policy.

There is also an observability mismatch. The actor receives body angular
velocity, projected gravity, joint position/velocity, previous action, and
action scale. It does not receive world yaw, root linear velocity, or world
foot position. Absolute heading, world-facing feet, and a world foot-position
anchor are therefore not directly inferable by the actor. The fixed 0.6-second
anchor can additionally capture a foot placement while the inherited policy
is still stepping. These terms add return variance rather than a usable
correction signal.

The remaining motion is strongly asymmetric: standing left-hip-yaw RMS is
`5.25 / 2.68 / 4.96 / 4.53 rad/s`, while right-hip-yaw RMS is only
`1.43 / 0.62 / 1.28 / 1.31 rad/s`. This identifies the left hip-yaw cycle,
coupled to lateral balance, as the main actionable source of turning.

## V25 design

V25 starts from V24 final but removes the partially observable absolute
heading, world-foot orientation, root-speed, and world-anchor objectives. It
replaces the squared stance loss with a linear-tailed smooth-L1 loss, replaces
unbounded torso/pelvis support costs with bounded support scores, and restores
all post-standing weights to the same order as the original dt-scaled HoST
target terms. It adds directly observable neutral lateral-joint pose,
lateral-joint stillness, selected-action magnitude, and an extra left-hip-yaw
velocity/action cost. This tests a different hypothesis rather than increasing
V24's failed coefficients.

The final two actor linear layers are trainable, with a fresh critic and
optimizer, frozen actor normalizer/action scale, fixed standard deviation
`0.01`, learning rate `2e-5`, two PPO epochs, 1000 updates, and checkpoints
every 100 updates. The preflight records each target term's mean absolute
per-step contribution and rejects any target term above 2.0 before long
training begins.

The 64-environment, 100-step preflight passed. Its largest target contribution
was the observable yaw-rate term at `1.788` mean absolute reward per step;
every other target remained below `0.67`. At training iteration 7, value loss
was about `4.59e3`, mean reward `-1,744`, and surrogate loss `0.0054`, showing
that the critic and actor are both updating without V24's scale collapse.

## V25 final result and V26 decision

V25 completed 1000 updates. Its midpoint and final checkpoints are
`logs/host29_posture_v25_4096_u1000_20261010/model_500.pt` and
`logs/host29_posture_v25_4096_u1000_20261010/model_1000_final.pt`, with SHA256
`b84fddd35d88712842aa0d11bd24b8d11ac8f6192ee037775b207c97b6f3a90a` and
`c2ea1fd01832a6171cfedf32b4cbd1785fb27b5356ae2f1c865683352c7072ad`.
Both retain 100% held/final standing and zero terminations in every posture.

V25 final metrics, ordered prone/supine/left/right, are:

- final joint-speed RMS: `0.880 / 0.729 / 0.891 / 0.829`;
- standing yaw-rate RMS: `0.524 / 0.814 / 0.621 / 0.747`;
- standing horizontal-speed RMS: `0.084 / 0.116 / 0.105 / 0.115`;
- left hip-yaw RMS: `1.228 / 1.202 / 1.541 / 1.300`;
- right hip-yaw RMS: `0.485 / 0.699 / 0.530 / 0.707`.

This is a qualitative motion improvement over V24: yaw falls by roughly
36-63%, horizontal motion by roughly 65-77%, and the 0/4/8/12/16-second frames
no longer alternate between front, side, and rear views. The value loss also
falls to about `756`, mean reward to `-1,612`, and surrogate loss remains
nonzero at `0.0368`. The trained actor changes by about 0.76% of its source
norm, so V25 successfully escapes V24's critic-scale failure.

V25 nevertheless fails the full visual gate. All four videos converge to a
stationary but biased pose: the torso remains visibly inclined relative to the
support line, and one arm is repeatedly raised while the other stays low. The
motion problem is solved much more strongly than the posture problem. The
existing `target_upper_pose_linear` averages 17 waist/arm joints, while the
hand target averages both arms; these aggregate terms allow a large one-sided
error to be diluted. Continuing to train shared hidden layers also risks
undoing the new leg/yaw solution.

V26 therefore starts from V25 final and freezes every leg and waist-yaw action
row. Only final-layer rows 13-28 (waist roll/pitch and both arms) are trainable.
It replaces the aggregate upper-pose term with separate left-arm, right-arm,
and waist targets, strengthens torso vertical/support geometry within the same
bounded reward scale, and retains V25's anti-spin objective. This isolates the
remaining visual failure without giving the optimizer access to the stable
lower-body controller.

V25 artifacts are under
`logs/evaluations/host29_posture_v25_final_256_seed42/`, including the formal
JSON, four videos, 20 keyframes, and contact sheet.

## V26 final result and V27 decision

V26 completed 800 updates from V25. Its midpoint and final checkpoints are
`logs/host29_posture_v26_4096_u800_20261011/model_400.pt` and
`logs/host29_posture_v26_4096_u800_20261011/model_800_final.pt`, with SHA256
`0a3150b0571846cfbfca587cdef76659e299af297719f9983b625148590a7440` and
`10bffe6ceefd24ceae57ee33e7814ea5d1df0f05f531cc75ff6b54f5b7fced15`.
The formal 256-environment evaluation keeps 100% held/final standing and zero
terminations in all postures. Final joint-speed RMS is
`0.937 / 0.701 / 0.927 / 0.820`; standing yaw-rate RMS is
`0.516 / 0.780 / 0.626 / 0.727`; standing horizontal-speed RMS is
`0.083 / 0.113 / 0.110 / 0.113`. Relative to V25 these changes are mixed and
small, with left-side horizontal speed worsening by about 5%.

The visual gate still fails. The final contact sheet shows nearly the same
stationary biased pose in every posture: about 29-30 degrees of torso tilt,
one arm lifted above the hip, and pronounced left/right arm asymmetry. The new
read-only posture diagnostic quantifies the failure. V25-to-V26 torso tilt
changes by less than 0.06 degrees, waist roll/pitch mean absolute error stays
near 0.529 rad, left-hand target error stays near 0.22 m, and right-hand target
error stays near 0.74-0.77 m. In the right arm, shoulder roll remains near
`-1.6` to `-1.74 rad` instead of the `-0.3 rad` target. Artifacts are under
`logs/evaluations/host29_posture_v26_final_256_seed42/`; numerical posture
diagnostics for V25 and V26 are under the corresponding
`host29_posture_*_final_128_seed42/posture_diagnostics.json` paths.

V26 did not fail because the lower-body controller had to move. The gradient
mask preserved every leg and waist-yaw output row bit-for-bit, but the trained
output layer moved only 0.19% relative to its V25 value and the total actor
moved only 0.022%. During training, the upper-body costs stayed effectively
flat: torso tilt about `-30.7`, hands about `-105.5`, left/right arm pose about
`-30/-31`, and waist pose about `-28.7` per episode. Exploration standard
deviation `0.005` was too small to discover a different upper-body attractor.
At the same time, PPO advantages were dominated by large terms produced by
the frozen lower body, including yaw around `-117`, feet alignment around
`-59`, and left-hip-yaw velocity around `-29` per episode. Those terms cannot
be improved directly by the selected waist/arm action rows and obscure the
posture signal.

V27 therefore restarts from the cleaner V25 final checkpoint rather than
stacking V26's ineffective drift. It keeps the same strict output-row mask
(waist roll/pitch and both arms only), but changes one hypothesis: isolate the
controllable upper-body objective. Frozen lower-body target terms receive zero
weight during this stage, while stand-up, base stability, regularization, and
the bounded torso/arm objectives remain active. Exploration is per action:
`0.005` on frozen lower-body outputs and `0.08` only on trainable upper-body
outputs. V27 uses learning rate `1e-4`, four PPO epochs, 1000 updates, a fresh
critic/optimizer, and the frozen V25 actor normalizer/action scale. The
preflight must prove nonzero upper-body rewards and gradients, exact zero
gradients on all frozen rows, finite rollouts, and the expected per-action
noise vector before formal training starts.

## V27 result and V28 decision

V27 completed 1000 updates. Its final checkpoint is
`logs/host29_posture_v27_4096_u1000_20261011/model_1000_final.pt`, SHA256
`7c8608b1f580076b3a851cc70edbc6a74ccd8a591c050f2f7be235e21ed30d26`.
The formal evaluation keeps 100% held/final standing and zero terminations,
but final joint-speed RMS rises to `1.713 / 1.553 / 1.706 / 1.645`, yaw-rate
RMS to `1.379 / 1.156 / 1.363 / 1.247`, and horizontal-speed RMS to
`0.366 / 0.348 / 0.363 / 0.357`. Torso tilt improves from about 0.52 rad to
`0.184 / 0.230 / 0.186 / 0.213 rad`, but the videos show renewed continuous
turning, stepping, crouching, and bent asymmetric arms. V27 final therefore
fails the combined gate.

A checkpoint sweep identifies a clear trade-off rather than a missing amount
of PPO training. Update 100 is quieter than V25 (mean yaw RMS `0.508 rad/s`,
mean horizontal RMS `0.094 m/s`) but leaves torso tilt and waist error almost
unchanged. Later checkpoints progressively straighten the torso only by using
the arms and waist as moving counterweights; motion then deteriorates. A
row-wise decomposition confirms the coupling. Combining V27-final waist rows
with V27-update-100 arm rows gives excellent numerical motion (mean yaw
`0.468 rad/s`, mean horizontal speed `0.094 m/s`, final joint-speed RMS
`0.424 / 0.530 / 0.433 / 0.502`), yet the torso still tilts about 25 degrees
and the right arm remains raised.

The next hypothesis is therefore not another reward-weight increase. A
deterministic post-standing posture controller was tested on that decomposed
checkpoint. A small arm correction preserves the quiet lower-body solution
but cannot remove the visual bias. A torso-frame gravity feedback controller
for waist roll/pitch reduces torso tilt to roughly 7-10 degrees, while a full
neutral-arm correction makes both arms substantially more symmetric. The
combined correction still provokes corrective stepping because the inherited
lower-body policy never learned to balance the shifted upper-body mass.

V28 trains only lower-body and waist-yaw actor output rows while applying the
validated upper-body controller inside the action term after standing. The
upper controller is deterministic and frozen: neutral arms use blend 1.0 and
gain 0.5; torso-frame waist balance uses blend 1.0 and gain 2.0 after a
25-step delay and 25-step ramp. PPO is used only to adapt the lower-body
balance policy to this changed mass distribution. This separates posture
enforcement from locomotion stabilization, which V27's additive reward could
not do. The source checkpoint and all prior checkpoints remain read-only; V28
uses a new script, log directory, fresh critic/optimizer, and an explicit
runtime-controller record. If the composite passes, a later distillation pass
can remove the deterministic action post-processor without changing the
validated target behavior.

## V28 result and V29 decision

V28 completed 800 updates. Its midpoint and final checkpoints are
`logs/host29_posture_v28_4096_u800_20261011/model_400.pt` and
`logs/host29_posture_v28_4096_u800_20261011/model_800_final.pt`, with SHA256
`0f8b670656e5336d6a048268547ae9af13f3fba6cf5c4046f5cb613faa6956cc` and
`d4a38327004d3236168c9562b27465b0509c7540e4b9968156582dca3163f386`.
The final 256-environment composite evaluation retains held/final standing of
`100/100/100/99.61%`, with zero terminations. Final joint-speed RMS is
`0.001 / 0.001 / 0.001 / 0.303`; standing yaw-rate RMS is
`0.161 / 0.378 / 0.196 / 0.387`; standing horizontal-speed RMS is
`0.050 / 0.091 / 0.051 / 0.101`. These are all substantially better than V25
and well inside the numerical motion gate.

The upper-body hypothesis is also validated. The 128-environment diagnosis
measures torso tilt at `0.124-0.126 rad`, torso-to-support error at
`0.045-0.047 m`, and left/right arm target MAE near `0.11 rad`. The four
0/4/8/12/16-second video rows remain in the same world orientation and stance
after standing: there is no continuous turn, periodic reversal, repeated
corrective step, or raised/folded arm. V28 therefore solves the turning,
horizontal drift, torso-bias, and arm-pose failures together.

V28 does not yet pass the complete visual gate because it converges to a
pronounced split stance. The feet remain `0.296-0.316 m` apart in the root
forward direction, although their height difference is only `0.012 m` and
lateral separation is a reasonable `0.284-0.287 m`. The geometry is explained
by a repeatable sagittal joint asymmetry: left/right hip pitch is about
`-0.25/-0.80 rad`, knee `1.39/1.16 rad`, and ankle pitch `-0.93/-0.26 rad`.
This is visible as one foot roughly 0.30 m ahead of the other in every posture.
The existing combined smooth-L1 feet term has weight `-0.40`; it is outweighed
by the stable support solution and does not isolate the sagittal error.

V29 therefore changes only the unresolved degree of freedom. It starts from
V28 final with the same deterministic upper controller and freezes every
upper-body, hip-roll/yaw, ankle-roll, and waist-yaw output. Only the six paired
hip-pitch, knee, and ankle-pitch output rows are trainable. Two controller-gated
linear-tailed objectives replace the combined feet term: direct root-frame
fore-aft foot alignment and paired sagittal-joint symmetry. Their losses are
bounded before weighting so they cannot reproduce V24's critic-scale collapse.
The lateral/yaw rows that produced V28's quiet stance remain bitwise frozen;
the expected change is a gradual sagittal stance closure without reopening
turning or torso compensation.

## V29 result and V30 decision

V29 completed 800 updates. Its final checkpoint is
`logs/host29_posture_v29_4096_u800_20261011/model_800_final.pt`, SHA256
`f1b026d62c37d5ec8b97cc8d0d3a09687003504165ef15ff9465b3957d4a3d17`.
The final policy does make the feet nearly level in the root-forward direction
(`0.029-0.032 m`), but it destroys the standing attractor: held fraction is
zero in all four postures, final standing is only `3.1-5.9%`, final
joint-speed RMS is `3.227-3.316`, yaw-rate RMS is `1.087-1.160`, and
horizontal-speed RMS is `0.964-0.990`. The final videos show continuous
crouching, sweeping steps, large torso excursions, and arm compensation, so
V29 final fails both the numeric and visual gates.

The checkpoint sweep exposes a monotonic geometry-motion trade-off. At update
100, held/final standing remains 100%, joint-speed RMS is `0.090-0.135`, and
yaw/horizontal RMS stays near V28, but the feet are still `0.262-0.284 m`
apart fore-aft. At update 200 the offset improves to `0.221-0.239 m`, while
joint-speed rises to `0.455-0.511` and horizontal speed to `0.093-0.150`.
By update 400, joint-speed is `1.584-1.665` and horizontal speed
`0.364-0.405`, before the final collapse. Contact sheets confirm that updates
100 and 200 remain visually quiet but visibly split; no checkpoint satisfies
both sides of the gate.

Parameter decomposition identifies the mechanism. Relative to V28, the six
trainable final-layer weight rows move by 2.37%, 4.38%, 9.94%, and 15.33% at
updates 100, 200, 400, and 800. Geometry improves only as the inherited
feature-dependent state feedback is progressively rewritten. The bias change
is much smaller at the early stable checkpoints. A separate deterministic
foot controller does not solve the trade-off: additive hip feedback up to its
action limit only reduces the offset to about `0.20 m` while raising motion,
and suppressing the learned left/right action difference makes the offset
worse (`0.35-0.39 m`). This rules out simply increasing the same reward or
blindly adding a stronger post-processor.

V30 returns to the stable V28 final checkpoint and changes the optimization
parameterization instead of the target. Every actor weight is frozen, as are
all ankle, lateral, yaw, waist, and upper-body outputs. Only the four final
output biases for bilateral hip pitch and knee are trainable. Their gradients
are projected into two antisymmetric pairs, preserving each pair's common
action exactly, and each bias delta is clamped to `0.05`. The ankle rows are
excluded because V29 updates 100-200 improved geometry without moving ankle
pitch, whereas later ankle changes coincided with instability. V30 keeps only
the bounded direct foot-forward objective and disables the redundant joint
symmetry loss. It uses standard deviation `0.04` on the four trainable biases,
`0.005` elsewhere, learning rate `2e-4`, four PPO epochs, 400 updates, and
checkpoints every 50 updates.

The 64-environment, 100-step V30 preflight passed. The only nonzero actor
gradients are the four selected bias entries; each left/right pair is exactly
antisymmetric, all actor weights remain frozen, the upper controller reaches
mix `0.76`, and the direct alignment term has mean absolute contribution
`0.099` per step. Every target term remains below `0.41`, and the rollout is
finite with no NaN or Inf. This directly tests whether a bounded static
differential correction can close the stance while preserving V28's validated
closed-loop feedback.

## V30 result and V31 decision

V30 completed 400 updates. Its final checkpoint is
`logs/host29_posture_v30_4096_u400_20261011/model_400_final.pt`, SHA256
`a248e83bdc594ece0eb52c6798f366d87bf87448f23ef0fae54beb06053a083c`.
The formal 256-environment composite evaluation retains 100% held/final
standing and zero terminations in all four postures. Final joint-speed RMS is
`0.000 / 0.001 / 0.003 / 0.002`; yaw-rate RMS is
`0.161 / 0.380 / 0.196 / 0.388`; horizontal-speed RMS is
`0.050 / 0.093 / 0.052 / 0.099`. These are effectively the V28 values, but
the forward foot offset also remains effectively unchanged at
`0.307 / 0.296 / 0.308 / 0.293 m`. All four selected biases reach their
allowed differential limit without moving the planted contacts. V30 is
therefore numerically stable but fails the same split-stance visual gate.

Additional diagnostics change the root-cause hypothesis. Static left-leg
targets do not move the loaded forward foot. Finite right-leg commands do lift
the rear foot by about `0.08 m` and transiently reduce the forward offset to
`0.02-0.03 m`, proving that the geometry is reachable only through a dynamic
repositioning step. Three two-phase pulse searches, including a full 13-output
lower-body search with varied timing, reduce the settled offset only to about
`0.22-0.25 m`: after the pulse, the unchanged V28 actor returns toward its
split-stance attractor. This rules out another static bias, stronger copy of
the same alignment reward, or one-shot open-loop pulse.

A continuous right sagittal target was then tested with the validated V28
upper controller. The target `(0.30, 2.10, -1.00 rad)` gives the best measured
geometry among the stable scans: the final forward offset falls to
`0.169-0.188 m`, with a transient minimum near `0.10 m`. The unadapted policy
still moves too much: final joint-speed RMS is about `1.99-2.41`, yaw-rate RMS
`0.84-0.93`, horizontal-speed RMS `0.23-0.26`, and final standing is
`95.3-96.9%`. Weaker or stronger targets settle near `0.22-0.26 m` and do not
offer a better geometry/stability trade-off.

V31 therefore fixes that right hip-pitch/knee/ankle-pitch target after a
50-step delay and 100-step ramp, then learns the complementary lower-body
stabilizer. The controller-owned rows `6/9/10` and all upper rows remain
bitwise frozen; only the other ten lower rows can train. V31 restarts from the
stable V28 final actor with a fresh critic/optimizer, learning rate `3e-5`,
four PPO epochs, standard deviation `0.04` on trainable rows and `0.005`
elsewhere, 800 updates, and checkpoints every 100 updates. This changes the
strategy after the failed static/open-loop family: the step geometry is now a
fixed constraint while PPO learns closed-loop support around it.

## V31 fixed-right-target lower-body adaptation (final, failed)

V31 final (`model_800_final.pt`) completed without runtime errors, but failed both numeric and visual gates. Formal 256-env evaluation produced held/final standing rates of 70.31/69.53%, 79.69/78.91%, 95.70/93.36%, and 50.39/50.00% for prone/supine/left/right; final joint-speed RMS was 2.494/2.267/2.033/2.728, yaw RMS 1.287/1.235/1.251/1.326, and horizontal RMS 0.567/0.511/0.543/0.571. The 128-env diagnosis found final feet-forward offsets of 0.276/0.279/0.263/0.271 m and torso tilt around 0.30-0.32 rad. Video contact sheets show persistent stepping, crouching, trunk lean, and arm excursions.

The checkpoint sweep demonstrates monotonic degradation from u200 to u800. At u200 the policy was already below the standing gate and final feet offsets remained about 0.19-0.23 m; later training further damaged support without retaining the transient geometric gain. The most likely cause is that the fixed right sagittal target creates a transient aligned configuration, while PPO adaptation of the remaining lower-body feedback learns to retreat toward the original split-stance attractor. A continuous fixed target and a freely adapting contralateral sagittal chain therefore conflict.

Next hypothesis: replace open-loop release and continuous forcing with an event-driven capture controller. Apply the known short right-leg pulse, detect the minimum feet-forward offset, capture the right sagittal joint state once offset falls below the alignment threshold, and hold that captured target with a smooth ramp. First test this read-only on V28. If it preserves <=0.08 m final offset and stability, train only non-sagittal compensation rows while freezing both sagittal chains and all upper-body rows.

## V32 restart after crouch/twist regression

A direct visual comparison of V25, V26, V27-u100, V27-final, and V28 confirms that the persistent crouching/knee-walking and twisting regression begins in V27, not V25/V26. V27-u100 is still close to the stable V25 gait, while V27-final shows repeated bent-knee stepping and torso rotation. V28 stabilizes that later actor with an analytic upper-body controller, so it removes continuous turning but inherits a split, flexed lower-body attractor. V29 and V31 then directly manipulate sagittal hip/knee/ankle behavior and amplify the same failure. Therefore V28-V31 are excluded as restart sources.

A read-only V25 test with the full V28 upper override preserved 100% held/final standing, but increased yaw RMS to 0.98-1.33 and horizontal RMS to 0.24-0.34, showing that a strong instantaneous upper correction itself forces compensatory lower-body motion. V32 consequently restarts from the stable V25 final actor and uses no analytic leg or upper override. All lower-body and waist-yaw outputs (rows 0-12) remain bitwise frozen. Only waist roll/pitch and arm rows 13-28 train. Exploration is reduced to 0.03 on trainable rows and 0.001 elsewhere, learning rate to 3e-5, PPO epochs to 2, and output-layer drift is hard-clamped to 0.0025 per weight and 0.02 per bias. The run is intentionally short (150 updates) with checkpoints every 25 updates; selection will sweep 25/50/75/100/125/150 and will reject the first appearance of crouching, twisting, or repeated corrective stepping.

The 64-env preflight passed: lower-body gradients are exactly zero, upper gradients are nonzero, reward terms are finite and bounded, and no NaN/Inf occurred. Formal V32 training started on GPU4 from `logs/host29_posture_v25_4096_u1000_20261010/model_1000_final.pt` in `logs/host29_posture_v32_4096_u150_20261012`.

## V32 final and V33 decision

All six V32 checkpoints pass the numeric stability gate. Across updates 25-150, held/final standing is 100% in every posture, joint-speed RMS stays 0.676-0.959, yaw RMS 0.509-0.797, and horizontal RMS 0.082-0.117. This confirms that restarting from V25 and freezing rows 0-12 removes the V27-V31 knee-walking regression. However, posture diagnosis is effectively unchanged from update 25 to 150: torso tilt remains 0.51-0.53 rad, waist target MAE about 0.529 rad, and right-hand-to-hip error about 0.75-0.77 m. V32-final video is stable with no knee walking or repeated stepping, but retains the original bent-knee/leaning V25 stance and raised arm. Upper-only output-row PPO therefore cannot move the V25 balance attractor.

V33 changes the controllability assumption without reopening the sagittal failure path. It starts again from V25, applies a slow weak upper controller (arm blend/gain 0.5/0.3; balance blend/gain 0.5/1.0; delay/ramp 50/100), freezes both hip-pitch/knee/ankle-pitch chains and every upper output, and trains only rows 1/2/5/7/8/11/12 (bilateral hip roll/yaw, ankle roll, and waist yaw) to learn lateral compensation. The 64-env 200-step preflight passed: controller mix reached 0.95, controller affects only upper outputs, sagittal rows have zero gradient, target terms are finite, and no NaN/Inf occurred. Formal V33 training started on GPU4 in `logs/host29_posture_v33_4096_u300_20261012`.
## V34 result and V35 stillness hypothesis (2026-10-12)

- V34 completed all eight 256-env sweeps. Update 400 was the quietest: held/final standing were 100% with zero termination; final joint-speed RMS was 0.535/0.139/0.574/0.426, yaw-rate RMS 0.412/0.440/0.487/0.492, and horizontal-speed RMS 0.131/0.100/0.144/0.132.
- Despite the lower motion metrics, V34 does not pass visually. The four videos settle into a persistent crouched, forward/side-leaning split stance. Diagnosis reports torso tilt 0.511-0.519 rad, waist target MAE 0.519-0.522 rad, and final forward foot offset 0.326-0.328 m. The moderate upper controller therefore failed to change the V25 posture attractor and increased translation relative to the V25 baseline in several postures.
- V35 changes the objective hierarchy to post-standing stillness first. It restarts from V25 with no analytic override, freezes all sagittal leg and upper-body outputs, and trains only hip roll/yaw, ankle roll, and waist-yaw rows. Rewards are limited to actor-observable angular velocity, joint velocity/stillness, and action magnitude; world translation, foot anchors, and other unobservable world-frame objectives are excluded. Output-row deltas are tightly bounded to prevent a return to the V27-V31 crouch attractor.

## V35 acceptance and V36 straightening plan (2026-10-12)

V35 update 400 is accepted as the practical A6 initialization under the relaxed
criterion that standing still matters more than a perfect pose. Formal
256-environment evaluation retains 100% held/final standing and zero
terminations in all four postures. Final joint-speed RMS is
`0.887 / 0.688 / 0.933 / 0.787`, yaw-rate RMS is
`0.486 / 0.742 / 0.595 / 0.699`, and horizontal-speed RMS is
`0.081 / 0.111 / 0.102 / 0.111`. The four videos show no knee walking,
sustained rotation, or repeated corrective stepping after recovery. The
remaining visible defects are the inherited torso lean and raised right arm.

V36 starts from the immutable V35-final actor while A6 training continues on
separate GPUs. Every lower-body and waist-yaw output remains bitwise frozen;
only waist roll/pitch and both arms can change. Unlike V32's ineffective
`0.0025/0.02` output clamp and V27's destabilizing unconstrained upper-body
adaptation, V36 uses an intermediate `0.006/0.05` weight/bias budget, standard
deviation `0.02`, learning rate `2e-5`, two PPO epochs, and 300 updates with
checkpoints every 25 updates. It keeps V35's yaw and lateral-stillness terms
active while strengthening bounded torso, waist, support, and arm-pose terms.
Checkpoint selection prioritizes unchanged standing stability and motion first,
then torso and arm improvement; any return of knee walking, stepping, or yaw
drift rejects that checkpoint.

## V36 result (2026-10-12)

V36 completed all 300 updates and all twelve 256-environment checkpoint
evaluations without runtime errors. Every checkpoint retained 100% held/final
standing and zero terminations. Update 50 is the best motion-preserving
candidate: final joint-speed RMS is `0.902 / 0.657 / 0.898 / 0.785`, yaw-rate
RMS is `0.492 / 0.734 / 0.573 / 0.700`, and horizontal-speed RMS is
`0.082 / 0.110 / 0.100 / 0.112`. These values are effectively unchanged from
V35 rather than a meaningful improvement. Later training progressively harms
stillness; at update 300 yaw-rate RMS reaches
`0.520 / 0.955 / 0.697 / 0.817` and horizontal-speed RMS reaches
`0.091 / 0.126 / 0.149 / 0.121`.

The posture diagnosis rules out a useful straightening gain. Mean torso tilt
is `0.5188 rad` for V35, `0.5187 rad` at V36 update 50, and `0.5239 rad` at
update 300. Mean waist target error remains `0.529 rad`; mean right-hand target
error changes from `0.758 m` to `0.759 m` and then `0.766 m`. The update-50
videos remain quiet after recovery with no knee walking, sustained turn, or
repeated corrective step, but the torso lean, split crouched stance, and raised
right arm are visually indistinguishable from V35. V36 therefore provides no
usable posture improvement. V35 final remains the selected A6 source, and no
V37 posture run is started.

## V37 post-standing damping fine-tune (2026-10-14)

User review judged the accepted V35 stance still visibly under-damped. V37
therefore returns to the immutable V35 final actor and optimizes stability
rather than straightening. Unlike V35's seven-row lateral-only update, V37
permits the entire final actor layer to learn, but applies row-specific trust
regions: the seven lateral/yaw outputs receive at most `0.002/0.010`
weight/bias change, while the six sagittal and sixteen upper-body outputs
receive only `0.00075/0.004`. Exploration standard deviation is `0.008` on
lateral rows and `0.004` elsewhere; the learning rate is `8e-6`, with two PPO
epochs and dense 25-update checkpoints.

The objective remains actor-observable. Existing yaw, selected joint-velocity,
stillness, and action penalties are joined by gated all-joint stillness,
all-joint velocity, all-action magnitude, action-rate, and body-frame root
angular-speed terms. They activate only above root height `0.66 m`, just below
V35's measured standing root height of about `0.68 m`, so the main get-up phase
is not forced rigid. A 64-environment, 400-step preflight passed: every new term
activated, the largest mean absolute single-step target contribution was
`2.907`, actor gradients and row-wise standard deviations were correct, and all
actions, rewards, and observations remained finite. Formal training started on
GPU4 from `logs/host29_posture_v35_4096_u400_20261012/model_400_final.pt` in
`logs/host29_posture_v37_4096_u300_20261014`. Selection will prioritize lower
joint/yaw/horizontal motion with unchanged four-posture recovery and will reject
any return of knee walking, repeated stepping, or sustained turning.

## V37 result (2026-10-15)

V37 completed all 300 updates without OOM, NaN/Inf, abnormal termination, or
training interruption. All twelve 256-environment checkpoint evaluations kept
100% held/final standing and zero terminations in every posture. Update 100 is
the selected stability checkpoint:
`logs/host29_posture_v37_4096_u300_20261014/model_100.pt`, SHA256
`6317a1990393bcb603cec449042f30302b71a70a25e50a894e7af57e41945e09`.

At update 100, final joint-speed RMS is
`0.264 / 0.199 / 0.173 / 0.215`, standing yaw-rate RMS is
`0.436 / 0.378 / 0.360 / 0.385`, and standing horizontal-speed RMS is
`0.117 / 0.103 / 0.094 / 0.093` for prone/supine/left/right. Compared with
V35, the four-posture mean final joint-speed RMS falls from `0.824` to `0.213`
(`-74%`), mean standing yaw-rate RMS from `0.631` to `0.390` (`-38%`), and
mean final root linear speed from `0.0493` to `0.0060 m/s` (`-88%`). Mean
standing horizontal-speed RMS remains effectively unchanged at `0.1014 m/s`
because that statistic includes the recovery transient; the final-speed and
joint-speed measurements capture the large post-settle damping improvement.

The 128-environment diagnosis shows no meaningful posture drift relative to
V35: torso tilt remains `0.51-0.53 rad`, torso/support error `0.07-0.08 m`,
feet-forward offset `0.32-0.34 m`, and arm/waist errors are unchanged. The four
20-second videos confirm that after recovery the robot holds the same inherited
leaning, split stance without knee walking, sustained turn, periodic direction
change, repeated corrective stepping, or visible leg jitter. Frames at
4/8/12/16 seconds are visually stationary. V37 therefore succeeds at the
requested stability objective while intentionally not claiming a straight-pose
improvement; update 100 is preferred over the final checkpoint and becomes the
current 29-DoF post-standing stability result.
