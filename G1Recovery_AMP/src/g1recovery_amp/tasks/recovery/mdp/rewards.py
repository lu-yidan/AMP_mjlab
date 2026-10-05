"""Task rewards for teaching the G1 robot to finish standing up."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


RECOVERY_STANDUP_HEIGHT = 0.65
RECOVERY_TARGET_BASE_HEIGHT = 0.75
RECOVERY_LOW_HEIGHT_FLOOR = 0.10

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def _standing_gate(root_height: torch.Tensor, threshold: float) -> torch.Tensor:
    """Enable final-standing rewards only after the root clears a height threshold."""

    return (root_height > threshold).to(dtype=root_height.dtype)


def ang_vel_xy_from_state(
    root_ang_vel_b: torch.Tensor,
    root_height: torch.Tensor,
    target_base_height_phase3: float,
) -> torch.Tensor:
    """Reward small roll/pitch angular velocity after the robot stands up."""

    stability = torch.exp(-2.0 * torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1))
    return stability * _standing_gate(root_height, target_base_height_phase3)


def lin_vel_xy_from_state(
    root_lin_vel_b: torch.Tensor,
    root_height: torch.Tensor,
    target_base_height_phase3: float,
) -> torch.Tensor:
    """Reward small horizontal root velocity after the robot stands up."""

    stability = torch.exp(-5.0 * torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1))
    return stability * _standing_gate(root_height, target_base_height_phase3)


def target_orientation_from_state(
    projected_gravity_b: torch.Tensor,
    root_height: torch.Tensor,
    target_base_height_phase3: float,
) -> torch.Tensor:
    """Reward an upright root orientation after the robot stands up."""

    upright = torch.exp(
        -5.0 * torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    )
    return upright * _standing_gate(root_height, target_base_height_phase3)


def target_base_height_from_state(
    root_height: torch.Tensor,
    base_height_target: float,
    target_base_height_phase3: float,
) -> torch.Tensor:
    """Reward root height close to the final standing target."""

    height_match = torch.exp(-20.0 * torch.abs(root_height - base_height_target))
    return height_match * _standing_gate(root_height, target_base_height_phase3)


def low_height_progress_from_state(
    root_height: torch.Tensor,
    floor_height: float,
    standup_height: float,
) -> torch.Tensor:
    """Give a small, increasing height signal before final-standing rewards begin.

    This does not prescribe which side the robot should roll toward.  The
    configured weight stays below the standing-height reward at the gate, so
    hovering just under the gate is not more rewarding than crossing it.
    """

    if floor_height >= standup_height:
        raise ValueError("floor_height must be below standup_height")
    progress = ((root_height - floor_height) / (standup_height - floor_height)).clamp(
        0.0, 1.0
    )
    return torch.where(root_height <= standup_height, progress, 0.0)


def target_joint_deviation_l2_from_state(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    root_height: torch.Tensor,
    target_base_height_phase3: float,
) -> torch.Tensor:
    """Measure squared joint deviation from the default standing pose."""

    deviation = torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)
    return deviation * _standing_gate(root_height, target_base_height_phase3)


def ang_vel_xy(
    env: ManagerBasedRlEnv,
    target_base_height_phase3: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Read state and reward small horizontal-axis angular velocity."""

    robot: Entity = env.scene[asset_cfg.name]
    return ang_vel_xy_from_state(
        robot.data.root_link_ang_vel_b,
        robot.data.root_link_pos_w[:, 2],
        target_base_height_phase3,
    )


def lin_vel_xy(
    env: ManagerBasedRlEnv,
    target_base_height_phase3: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Read state and reward small horizontal root velocity."""

    robot: Entity = env.scene[asset_cfg.name]
    return lin_vel_xy_from_state(
        robot.data.root_link_lin_vel_b,
        robot.data.root_link_pos_w[:, 2],
        target_base_height_phase3,
    )


def target_orientation(
    env: ManagerBasedRlEnv,
    target_base_height_phase3: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Read state and reward an upright root orientation."""

    robot: Entity = env.scene[asset_cfg.name]
    return target_orientation_from_state(
        robot.data.projected_gravity_b,
        robot.data.root_link_pos_w[:, 2],
        target_base_height_phase3,
    )


def target_base_height(
    env: ManagerBasedRlEnv,
    base_height_target: float,
    target_base_height_phase3: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Read state and reward the target standing height."""

    robot: Entity = env.scene[asset_cfg.name]
    return target_base_height_from_state(
        robot.data.root_link_pos_w[:, 2],
        base_height_target,
        target_base_height_phase3,
    )


def low_height_progress(
    env: ManagerBasedRlEnv,
    floor_height: float,
    standup_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Read torso height for the early-stage recovery reward."""

    robot: Entity = env.scene[asset_cfg.name]
    return low_height_progress_from_state(
        robot.data.root_link_pos_w[:, 2], floor_height, standup_height
    )


def target_joint_deviation_l2(
    env: ManagerBasedRlEnv,
    target_base_height_phase3: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Read state and measure deviation from the default standing joint pose."""

    robot: Entity = env.scene[asset_cfg.name]
    default_joint_pos = robot.data.default_joint_pos
    assert default_joint_pos is not None
    return target_joint_deviation_l2_from_state(
        robot.data.joint_pos[:, asset_cfg.joint_ids],
        default_joint_pos[:, asset_cfg.joint_ids],
        robot.data.root_link_pos_w[:, 2],
        target_base_height_phase3,
    )
