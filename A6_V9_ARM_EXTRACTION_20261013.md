# A6 v9: arm extraction and real reset age

The 29-DoF G1 guided-board experiment continues the same HoST PPO actor and
four critics from the v8 strict-knee `model_01399.pt`. A separate flat
`model_00999.pt` policy remains frozen for flat and post-escape imitation.
This run targets 2,000 additional PPO updates (1400–3399), 4,096 worlds,
96 control steps per update, on server GPU3. The training entry point is
`scripts/train_a6_arm_footprint_2000_server.sh`.

## Why v9 changes the task

At a 3 cm edge-overlap start, independent seed-42 replay of the v8 strict
policy showed that the prone robot briefly cleared the board at control step
183, then re-entered with its left hand. The old episode state changed back
to trapped. Supine never cleared: the left hand remained 16.7 cm inside the
board footprint at step 250. The remaining blocker was an arm, not a knee.

`a6_reward.py` now measures the deepest XY overlap among eight arm collision
geometries. Before escape it rewards a decrease in this overlap, charges for
re-entry, and provides a bounded near-clear signal for the last centimetre.
After 25 consecutive full-body, no-contact clearance steps, the escape state
latches so the PPO can learn the stand-up phase. The controller's 5 cm guided
plate lift guard and the >5,000 N plate-force invalid rule remain active.

The reset grace period now uses `HostAction.real_steps`, which counts actual
control steps since reset. PPO randomizes `episode_length_buf` to stagger
timeouts; using that randomized counter had falsely invalidated fresh
shallow-overlap episodes before they had 25 real steps to contact the board.
The action smoothness first-step mask uses the same real-step counter.

## Full-size preflight

All one-update preflights started from the same v8 `model_01399.pt`, with
4,096 worlds and 96 steps per world. These are training-batch signals, not
independent escape success rates.

| Configuration | Guided invalid count | Mean PPO KL | Final adaptive LR | Mean target reward |
| --- | ---: | ---: | ---: | ---: |
| Old randomized-age rule, LR 3e-4 | 2577 | 0.318 | 1e-5 | -0.0966 |
| Real-step age, LR 3e-4 | 142 | 0.232 | 1e-5 | -0.0020 |
| Real-step age, LR 5e-5 | 144 | 0.0185 | 1e-5 | -0.0020 |

The invalid-count reduction confirms the reset-age bug. A mean KL of 0.0185
still exceeds the configured 0.01 target, so the first 20 updates are a
bounded stability check before committing the remaining 1,980 updates.
Advancing the board requires at least 35% resolved escape success from **each**
guided posture over the success-gated curriculum's 20-update window.

## Checkpoints and continuation

Server root: `/home/czr/code/G1Recovery_HoST`.

- Transfer parent: `logs/host_a6_v8_guard_strict_fromv7_1199_256_seed42_probe200/model_01399.pt`.
- Frozen flat teacher: `logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt`.
- V9 output: `logs/host_a6_v9_arm_footprint_from1399_4096_seed42_2000`.
- Expected final checkpoint if all 2,000 updates finish: `model_03399.pt`.

The first segment runs through iteration 1419 and saves `model_01419.pt`.
To continue without resetting the PPO optimizer or curriculum, set
`A6_RESUME` to that checkpoint and leave the same output directory. The
script's default total `A6_ITERATIONS=3400` then runs iterations 1420–3399.
The pre-v9 server files were saved to
`logs/host_a6_pre_arm_extraction_source_20261012.tar.gz` and
`logs/host_a6_pre_real_age_source_20261012.tar.gz` before edits.

Do not treat curriculum training fractions as proof of success. Independently
replay prone and supine guided scenes with zero assistance, require escape
without force/lift invalidation, then verify standing from the resulting pose.

## First 20-update gate and independent replay

The initial continuation finished at `model_01419.pt`. Its 20-update training
window recorded 1,156 prone escapes among 3,773 resolved prone attempts and
1,331 supine escapes among 3,889 resolved supine attempts, at the easy edge
placement. This is sampled training data, not a deployable-policy success rate.

Independent 10-second, no-assist replay at edge offset 0.25 m over seeds
42–49 found **0/8 deterministic escapes** from either guided posture. With
actions sampled from the checkpoint's PPO distribution instead, the same seeds
gave **4/8 prone** and **3/8 supine** escapes; two supine episodes also stood.
The flat robot stood in all eight seeds. This points to a significant gap
between stochastic exploration and the actor's mean action. Several escaped
prone episodes stayed low or curled their knees to the limits after the
25-control-step escape latch. Directly applying the frozen flat teacher after
escape could stand some supine examples, but could also send a prone robot back
toward the board and trigger a >5,000 N collision.

The PPO implementation had another distribution issue: many raw `log_std`
parameters had drifted just above their clamped maximum, where the clamp's
gradient is zero. `src/g1recovery_host/ppo.py` now projects the trainable
parameter into its bounds before and after PPO updates. A local PyTorch check
measured a zero gradient before projection and a nonzero gradient at the bound.
The running 1420–1599 server process imported the old PPO module on startup;
the projection fix applies only after a restart from a checkpoint.

## 1499–1549 checkpoint observations

At `model_01499.pt`, eight independent 10-second replays at offset 0.25 m
found deterministic escape in 0/8 prone and 1/8 supine episodes. Sampled
actions escaped in 3/8 prone and 5/8 supine episodes; 4/8 sampled supine
episodes ended standing, while prone ended standing in 0/8. The curriculum
advanced from 0.225 to 0.200 m at training iteration 1540 when both sampled
training postures met its 35% gate.

`scripts/eval_a6_batch.py` evaluates 8 low-reset states per scene in a single
32-world, no-assist simulation. At `model_01549.pt` and the current 0.200 m
offset, its baseline found prone escape 0/8 for both mean and sampled actions;
supine escaped 5/8 with mean actions and 3/8 sampled. At the easier 0.250 m
offset, supine mean actions escaped 8/8, while prone mean remained 0/8.

A playback-only arm-actuation sweep at the 0.200 m offset isolated a plausible
prone bottleneck. The trained guided arm scale is 0.25 (shoulder pitch, elbow,
and wrist axes; lateral shoulder axes are already at full scale). Replaying
the same `model_01549.pt` with scale 1.0 raised prone escape from 0/8 to
2/8 mean and 4/8 sampled, with one sampled prone episode standing. One mean
prone and one sampled supine episode hit the safety invalid condition. Scales
0.50 and 0.75 gave smaller gains. This is a feasibility probe, not a trained
policy result; the next stage must learn to use the larger action safely.
