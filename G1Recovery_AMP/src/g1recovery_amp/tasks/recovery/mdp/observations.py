"""Recovery actor and critic observations translated from the source task.

The deployable NoKeyBody actor receives one 96-value proprioceptive frame:

``[base_ang_vel(3), root_orientation(6), joint_pos(29),
joint_vel(29), last_action(29)]``.

The five terms keep separate histories because Isaac Lab 2.3.1 flattened group
history in term-major order.  Matching that ordering is necessary for source
checkpoint compatibility even though frame-major history can look more natural.

The privileged critic receives 118 values per frame.  In addition to the actor
state, it sees root linear velocity, root height, and six key-body positions in
the root frame.  Its five-term history therefore contains 590 values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


RECOVERY_POLICY_FRAME_DIM = 3 + 6 + 29 + 29 + 29
RECOVERY_POLICY_HISTORY_LENGTH = 5
RECOVERY_POLICY_OBS_DIM = RECOVERY_POLICY_FRAME_DIM * RECOVERY_POLICY_HISTORY_LENGTH

RECOVERY_CRITIC_FRAME_DIM = 3 + 3 + 6 + 1 + 29 + 29 + 29 + 18
RECOVERY_CRITIC_HISTORY_LENGTH = 5
RECOVERY_CRITIC_OBS_DIM = RECOVERY_CRITIC_FRAME_DIM * RECOVERY_CRITIC_HISTORY_LENGTH

AMP_DISCRIMINATOR_FRAME_DIM = 6 + 3 + 29 + 29 + 18
AMP_DISCRIMINATOR_HISTORY_LENGTH = 10
AMP_DISCRIMINATOR_OBS_DIM = (
    AMP_DISCRIMINATOR_FRAME_DIM * AMP_DISCRIMINATOR_HISTORY_LENGTH
)

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def root_local_rot_tan_norm_from_quat(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Encode roll and pitch as two columns of a yaw-free rotation matrix.

    ``root_quat_w`` uses the ``(w, x, y, z)`` convention.  Removing yaw makes
    the encoding independent of the compass direction in which the robot lies.
    The first and third matrix columns provide a continuous six-value encoding.
    """

    root_yaw_quat = yaw_quat(root_quat_w)
    root_quat_local = quat_mul(quat_conjugate(root_yaw_quat), root_quat_w)
    root_rotm_local = matrix_from_quat(root_quat_local)
    tangent = root_rotm_local[..., :, 0]
    normal = root_rotm_local[..., :, 2]
    return torch.cat((tangent, normal), dim=-1)


def root_local_rot_tan_norm(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Read the robot root orientation and return its six-value local encoding."""

    robot: Entity = env.scene[asset_cfg.name]
    return root_local_rot_tan_norm_from_quat(robot.data.root_link_quat_w)


def base_pos_z(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return root height in world coordinates as a one-value observation."""

    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.root_link_pos_w[:, 2:3]


def key_body_pos_b_from_world(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    key_body_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Express world-frame key-body positions relative to the robot root frame.

    Args:
        root_pos_w: Root positions with shape ``(num_envs, 3)``.
        root_quat_w: Root quaternions in ``(w, x, y, z)`` order.
        key_body_pos_w: Key-body positions with shape ``(num_envs, bodies, 3)``.

    Returns:
        Root-frame positions flattened to ``(num_envs, bodies * 3)``.
    """

    relative_pos_w = key_body_pos_w - root_pos_w.unsqueeze(1)
    root_quat_for_each_body = root_quat_w.unsqueeze(1).expand(
        -1, key_body_pos_w.shape[1], -1
    )
    key_body_pos_b = quat_apply_inverse(root_quat_for_each_body, relative_pos_w)
    return key_body_pos_b.reshape(root_pos_w.shape[0], -1)


def amp_discriminator_frame_from_state(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    joint_pos_value: torch.Tensor,
    joint_vel_value: torch.Tensor,
    key_body_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Build matching live/demo AMP features while preserving time dimensions."""

    root_orientation = root_local_rot_tan_norm_from_quat(root_quat_w)
    root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
    relative_pos_w = key_body_pos_w - root_pos_w.unsqueeze(-2)
    root_quat_per_body = root_quat_w.unsqueeze(-2).expand(
        *root_quat_w.shape[:-1], key_body_pos_w.shape[-2], 4
    )
    key_body_pos_b_value = quat_apply_inverse(
        root_quat_per_body, relative_pos_w
    ).flatten(start_dim=-2)
    return torch.cat(
        (
            root_orientation,
            root_ang_vel_b,
            joint_pos_value,
            joint_vel_value,
            key_body_pos_b_value,
        ),
        dim=-1,
    )


def amp_demo_observation(
    env: ManagerBasedRlEnv,
    command_name: str,
    key_body_names: tuple[str, ...],
    history_length: int = AMP_DISCRIMINATOR_HISTORY_LENGTH,
) -> torch.Tensor:
    """Sample chronological AMP demonstration windows from the loaded NPZ motion."""

    from g1recovery_amp.tasks.recovery.mdp.commands import RecoveryMotionCommand

    command = cast(
        RecoveryMotionCommand,
        env.command_manager.get_term(command_name),
    )
    motion_ids, frame_ids = command.sample_amp_demo_indices(history_length)
    key_body_ids = torch.tensor(
        [command.cfg.body_names.index(name) for name in key_body_names],
        dtype=torch.long,
        device=env.device,
    )
    root_body_id = command.cfg.body_names.index("pelvis")
    body_pos_w = command.gather_motion_field("body_pos_w", motion_ids, frame_ids)
    return amp_discriminator_frame_from_state(
        body_pos_w[:, :, root_body_id],
        command.gather_motion_field("body_quat_w", motion_ids, frame_ids)[
            :, :, root_body_id
        ],
        command.gather_motion_field("body_ang_vel_w", motion_ids, frame_ids)[
            :, :, root_body_id
        ],
        command.gather_motion_field("joint_pos", motion_ids, frame_ids),
        command.gather_motion_field("joint_vel", motion_ids, frame_ids),
        body_pos_w[:, :, key_body_ids],
    )


def key_body_pos_b(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return selected body positions relative to the robot root frame."""

    robot: Entity = env.scene[asset_cfg.name]
    return key_body_pos_b_from_world(
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        robot.data.body_link_pos_w[:, asset_cfg.body_ids],
    )


def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return absolute joint angles, matching the source recovery task."""

    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return joint angular velocities in the validated 29-joint order."""

    robot: Entity = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, asset_cfg.joint_ids]
