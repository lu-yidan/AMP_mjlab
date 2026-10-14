# HoST flat29 arm, wrist and waist diagnosis

> This is the diagnosis of the original checkpoint. The trained fix and
> independent validation are in [HOST29_ARM_WRIST_FIX.md](HOST29_ARM_WRIST_FIX.md).

Checkpoint: `logs/downloaded/host_ground29_standref_4096_seed42_3000/model_02999.pt`
(SHA-256 `cd5392783c8e1e52b989d9a57d690d7d2efaf3a0824008d77f2eeefb5c96e999`).
The checkpoint is `flat29`, iteration 2999, `init_mode=fresh`, with final
`beta=0.25` and assistance force 0 N.

`scripts/diagnose_host29_arms.py` replays the deterministic policy in the
training MuJoCo model, with no assistance. Two runs used seeds 42 and 50,
eight environments each, 500 control steps per environment. Measurements
below cover steps 350-499 with pelvis height >0.7 m: all 2400 sampled
frames passed that height criterion. Raw per-joint measurements are in
`logs/downloaded/host_ground29_standref_4096_seed42_3000/diagnose_arms_8env_seed{42,50}.json`.

| Joint | Target (rad) | Seed-50 mean (rad) | MJCF range (rad) | Observation |
| --- | ---: | ---: | ---: | --- |
| waist pitch | 0 | -0.532 | ±0.520 | At lower stop in 100% of late frames; action -1.000 |
| left wrist pitch | 0 | -1.623 | ±1.614 | At lower stop in 100%; action -0.987 |
| right wrist pitch | 0 | -1.618 | ±1.614 | At lower stop in 100%; action -0.579 |
| right wrist yaw | 0 | -1.579 | ±1.614 | Near lower stop in 84% |
| left elbow | +0.821 | -1.064 | [-1.047, +2.094] | At lower stop in 100%; action -0.930 |
| left shoulder yaw | +0.160 | -2.005 | ±2.618 | Wrong direction by 2.165 rad |
| right shoulder pitch | +0.242 | +1.183 | [-3.089, +2.670] | Too high by 0.941 rad |

Seed 42 gave the same waist/wrist-pitch limit pattern. The recorded
`waist_pitch_joint` torque is -7.125 Nm and left wrist-pitch torque is
-4.142 Nm at seed 50: they actively push toward the negative stops.
Numerical constraint compliance allows a few hundredths of a radian beyond
the nominal MJCF range; this is not a different configured range.

## Cause

The six added joints **are configured and actuated**. They use the 29-DoF
Unitree MJCF ranges/inertias. `action.py` assigns waist roll/pitch
`kp=28.50`, `kd=1.814`, effort 50 Nm, and wrist pitch/yaw `kp=16.78`,
`kd=1.068`, effort 5 Nm, from mjlab's 5020/4010 motor constants. At the
checkpoint's `beta=0.25`, the tanh actor can request at most 7.125 Nm on
waist roll/pitch and 4.195 Nm on wrist pitch/yaw at zero velocity. Static
gravity torque at the intended upright pose is only about 0.303 Nm for
waist pitch and 0.121 Nm for wrist pitch; an insufficient static holding
torque cannot explain the upright limit-seeking behavior. The learned
negative commands do explain it. Simply increasing these gains would push
harder into the stops with this policy.

The posture reward is the main training-design failure. `task.py` computes
one `exp(-0.75 * sum(square(14 arm-joint errors)))` score. After the policy
has moved several shoulder/elbow/wrist joints far from the target, this
single product collapses: the measured late-standing arm-pose score is
1.34e-9 for seed 42 and 7.76e-10 for seed 50. Its reward and gradient are
effectively zero. Yet the logged `target` reward remains positive because
its other terms reward pelvis height/orientation and velocity. `style_reward`
only checks the shoulder-roll sign, not arm alignment or wrist posture.
The main HoST task rewards getting upright and raising the head, which the
policy achieves while leaving arms and waist at bad angles.
The orientation component reads projected gravity at the root/pelvis, so it
does not directly measure the torso's angle across the waist joint. At the
observed waist-pitch stop, the waist-pose score is still about 0.426 rather
than zero. `regularization_reward` does penalize crossing soft joint limits,
but `ppo.py` weights its normalized advantage by 0.1 versus 2.5 for the
upright task; this provides a much weaker policy-update signal than the
primary objective. That explains why a nonzero `target` training metric and
an upright pelvis can coexist with twisted arms and a bent torso.

The control law adds to this failure. It commands torque from the policy's
relative action (`kp * beta * action - kd * velocity`), without an absolute
position spring toward the reference pose. During the first 30 control
steps, all joints are unactuated. By step 30 the wrists have begun to bend;
by step 100, after standing, both wrist-pitch joints are close to their
negative stops and stay there. The reference recovery project uses a
different absolute-position action controller and different engineering
gains (waist 40/5 and wrist 40/1); direct equality of PD numbers between
the projects is therefore not expected. Its 5 Nm wrist limit matches ours.

The chosen reference's final wrist-roll targets are ±0.690 rad (about 40°),
so that reference itself should be visually reviewed if a more neutral
hand orientation is desired. It does not explain the current ±1.6 rad
wrist-pitch error or the reversed elbows.

## Recommended next experiment

Keep this checkpoint and its physics configuration immutable as the
baseline. Create a new `flat29` reward variant with **separate, non-vanishing
per-joint arm and waist errors** (for example, gated Huber penalties or a
sum of per-joint scores), plus explicit torso-upright and joint-stop
penalties. Give arms an attainable path out of their early contact poses;
test a small fine-tune or ablation first. Evaluate upright success and
per-joint errors together. Do not change PD gains while replaying this
checkpoint and interpret the result as the original policy: that would be
a different controller and would not isolate the training-reward problem.
