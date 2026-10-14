"""Optional standing-only posture controller for 29-DoF replay diagnostics."""

import torch

from .model import HOST29_JOINT_NAMES, HOST29_TARGET_JOINT_POS


def stand_pose_ready(env, min_steps=90):
    """Require real post-reset steps, an upright pelvis, and standing height."""
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    return (
        (term.real_steps >= min_steps)
        & (robot.data.root_link_pos_w[:, 2] > 0.7)
        & (robot.data.projected_gravity_b[:, 2] < -0.8)
    )


def apply_stand_pose_guard(env, action, min_steps=90):
    """Replace upper-body actions once the pelvis is upright and stably high.

    This is an explicit replay controller, not part of the trained policy.
    Its low wrist gains respect the 5 Nm wrist motors while the policy keeps
    controlling the legs throughout recovery and standing.
    """
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    if not hasattr(env, "_host29_guard_cache"):
        names = tuple(robot.joint_names)
        if names != HOST29_JOINT_NAMES or tuple(term._target_names) != names:
            raise ValueError("stand guard requires the native 29-DoF joint order")
        ids = [
            i
            for i, name in enumerate(names)
            if any(part in name for part in ("waist", "shoulder", "elbow", "wrist"))
        ]
        env._host29_guard_cache = (
            ids,
            torch.tensor(
                [HOST29_TARGET_JOINT_POS[name] for name in names], device=env.device
            )[ids],
            torch.tensor(
                [
                    12.0
                    if "wrist_pitch" in names[i] or "wrist_yaw" in names[i]
                    else 40.0
                    for i in ids
                ],
                device=env.device,
            ),
            torch.tensor(
                [
                    2.0 if "wrist_pitch" in names[i] or "wrist_yaw" in names[i] else 4.0
                    for i in ids
                ],
                device=env.device,
            ),
        )
    ids, target, kp, kd = env._host29_guard_cache
    q = robot.data.joint_pos[:, ids]
    v = robot.data.joint_vel[:, ids]
    desired_torque = kp * (target - q) - kd * v
    replacement = (desired_torque + term._kd[ids] * v) / (term._kp[ids] * term.beta)
    ready = stand_pose_ready(env, min_steps)
    guarded = action.clone()
    guarded[:, ids] = torch.where(
        ready[:, None], replacement.clamp(-4.0, 4.0), guarded[:, ids]
    )
    return guarded
