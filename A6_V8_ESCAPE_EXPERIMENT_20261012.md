# A6 29-DoF guided-board escape: v8 experiment (2026-10-12)

## Completion criterion

The objective is an unassisted G1 policy that escapes the fixed 6 kg guided
plate from both prone and supine starts, without triggering the unchanged
`>5000 N` force or `>0.05 m` guided-plate lift termination, and then stands.
Training rewards, assisted rollouts, or a single stochastic escape do not by
themselves meet this criterion. The independent first-episode replay uses no
horizontal or vertical assistance and reports each posture separately.

## Observed baseline and diagnosis

- The frozen 29-DoF flat teacher is
  `logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt`.
  A6 continues the same PPO actor and four critics; a separate copy of this
  checkpoint is fixed for flat and post-escape imitation.
- V6, 100 updates from the flat teacher: last 20 updates had prone
  `[0 escaped, 94 invalid, 270 other]` and supine `[0, 49, 277]`. The 0.25 m
  board offset did not produce a real easy start in yawed poses: the plate is
  world-axis-aligned, and the pelvis was about 0.20 m from the nearest edge in
  seed 42.
- `aligned_edge_v1` places the board along the dominant world lateral axis,
  compensating for the board's unequal half-sizes. At offset 0.25 m, a smoke
  assertion verifies approximately 0.07 m pelvis-to-edge margin. V7, 100
  updates from the flat teacher, still had no guided escapes. V7, 100 updates
  transferred from V6, had only one supine escape and none in its last 20
  updates. Neither is a successful policy.
- A zero-force parameter search over 128 simple joint-action biases, then five
  rounds of local search, found at most about 0.14 m prone / 0.16 m supine
  torso progress, with no complete escape. It shows some autonomous lateral
  control but not a full crawling sequence.
- A diagnostic 300 N horizontal external force produced real guided escape in
  seed 42 at prone step 109 and supine step 134, without lift invalidity. This
  is **a feasibility probe, not an unassisted policy result**. The flat teacher
  also failed to stand from the resulting high-velocity post-escape states.
- The original `safe_actions_v1` multiplies shoulder and hip commands by 0.25
  while the inherited HoST beta is 0.25, reducing their effective command
  magnitude to 0.0625 of the raw policy output. Releasing lateral shoulder and
  hip axes to full scale gave about 0.12 m prone progress but crossed the
  0.05 m plate-lift reset. With a substep motor guard fading commands between
  0.025 and 0.045 m lift, seed 42/43 plate-lift peaks were 0.0355/0.0386 m,
  with no invalid termination. This is a controller experiment, not proof of
  successful escape.

## V8 controlled comparison

Both runs start from the same V7 transfer checkpoint
`logs/host_a6_v7_aligned_fromv6_1099_256_seed42_probe100/model_01199.pt`.
Both use 256 environments, 96 steps per update, 200 additional PPO updates,
focused guided resets, aligned board edges, full lateral shoulder/hip commands,
and the substep lift guard. GPU3 adds `knee_straight_prior_v1`: the trapped
knee-action target is 0.35 rad for both postures and the full pre-escape knee
penalty remains active. GPU4 retains the earlier staged knee prior. The two
variants are recorded distinctly in each checkpoint's `reward_variant`.

| Run | GPU | Server output | Difference |
| --- | ---: | --- | --- |
| strict | 3 | `logs/host_a6_v8_guard_strict_fromv7_1199_256_seed42_probe200` | 0.35 rad knee target and full knee penalty |
| control | 4 | `logs/host_a6_v8_guard_control_fromv7_1199_256_seed42_probe200` | earlier knee prior |

Source snapshot before these runs:
`logs/host_a6_v8_guard_strict_source_20261012.tar.gz`. The active source files
are `src/g1recovery_host/a6_reset.py` (board placement), `a6_task.py` (scene
configuration), `action.py` (joint scaling and lift guard), `a6_reward.py`
(escape and knee rewards), and `scripts/train_host.py` (PPO and checkpoint
metadata). `scripts/diagnose_a6_escape_sequence.py` supplies the independent
first-episode, no-auto-reset evaluation.

## Selection gate

1. Compare last-20-update prone/supine `[escaped, invalid, other]` counts;
   reject a run that converts knee straightness into inactivity or invalidity.
2. Replay saved checkpoints at offset 0.25 m and centered offset 0.0 m with
   seeds 42–49, zero assistance, and four-view prone/supine/flat/free worlds.
   Report success per posture, maximum board lift, board force, and posture.
3. Advance the board inward only after both postures show repeatable safe
   success. Do not report the guided task as reproduced while centered-board
   unassisted success and post-escape standing remain unverified.
