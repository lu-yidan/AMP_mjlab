"""Read-only physics health metrics for recovery training."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def max_root_angular_speed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return each environment's root angular-speed magnitude in rad/s."""

    robot = cast("Entity", env.scene[asset_cfg.name])
    return torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=-1)


def max_joint_speed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return the largest absolute joint speed in each environment."""

    robot = cast("Entity", env.scene[asset_cfg.name])
    return torch.amax(torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def max_joint_acceleration(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return the largest absolute joint acceleration in each environment."""

    robot = cast("Entity", env.scene[asset_cfg.name])
    return torch.amax(torch.abs(robot.data.joint_acc[:, asset_cfg.joint_ids]), dim=-1)


def max_contact_penetration(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the deepest active MuJoCo contact in each environment, in metres."""

    contact = env.sim.data.contact
    valid = contact.dim > 0
    penetration = torch.where(
        valid,
        torch.clamp(-contact.dist, min=0.0),
        torch.zeros_like(contact.dist),
    )
    world_ids = contact.worldid.to(dtype=torch.long).clamp_(0, env.num_envs - 1)
    maximum = torch.zeros(env.num_envs, device=env.device)
    maximum.scatter_reduce_(
        0,
        world_ids,
        penetration,
        reduce="amax",
        include_self=True,
    )
    return maximum


def batch_max_joint_acceleration(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Repeat the current batch-wide joint-acceleration maximum for every env."""

    batch_maximum = torch.amax(max_joint_acceleration(env, asset_cfg))
    return batch_maximum.expand(env.num_envs)


def batch_max_contact_penetration(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Repeat the current batch-wide contact-penetration maximum for every env."""

    batch_maximum = torch.amax(max_contact_penetration(env))
    return batch_maximum.expand(env.num_envs)


def unsafe_root_angular_speed(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return 1 where root angular speed crosses its warning threshold."""

    return (max_root_angular_speed(env, asset_cfg) > threshold).to(torch.float)


def unsafe_joint_speed(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return 1 where any joint speed crosses its warning threshold."""

    return (max_joint_speed(env, asset_cfg) > threshold).to(torch.float)


def unsafe_joint_acceleration(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return 1 where any joint acceleration crosses its warning threshold."""

    return (max_joint_acceleration(env, asset_cfg) > threshold).to(torch.float)


def unsafe_contact_penetration(
    env: ManagerBasedRlEnv,
    threshold: float,
) -> torch.Tensor:
    """Return 1 where contact penetration crosses its warning threshold."""

    return (max_contact_penetration(env) > threshold).to(torch.float)


def unsafe_physics_state(
    env: ManagerBasedRlEnv,
    root_angular_speed_threshold: float,
    joint_speed_threshold: float,
    joint_acceleration_threshold: float,
    contact_penetration_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Return 1 for environments crossing any warning threshold, else 0."""

    unsafe = (
        unsafe_root_angular_speed(
            env, root_angular_speed_threshold, asset_cfg
        ).bool()
        | unsafe_joint_speed(env, joint_speed_threshold, asset_cfg).bool()
        | unsafe_joint_acceleration(
            env, joint_acceleration_threshold, asset_cfg
        ).bool()
        | unsafe_contact_penetration(env, contact_penetration_threshold).bool()
    )
    return unsafe.to(dtype=torch.float)
