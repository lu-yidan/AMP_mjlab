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
