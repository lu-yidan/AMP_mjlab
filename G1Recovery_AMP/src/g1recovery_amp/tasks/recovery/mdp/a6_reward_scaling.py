"""Symmetric task/style/cost scaling for obstructed A6 recovery scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from mjlab.envs import mdp as common_mdp

from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import update_a6_plate_state
from g1recovery_amp.tasks.recovery.mdp.a6_rewards import plate_force
from g1recovery_amp.tasks.recovery.mdp.a6_shared_costs import (
    a6_action_acceleration,
    a6_action_rate,
    a6_foot_motion,
    a6_joint_stall,
    a6_physics_cost,
)
from g1recovery_amp.tasks.recovery.mdp.a6_shared_tasks import (
    a6_task_multiplier_from_state,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


A6ScalableCost = Literal[
    "action_rate",
    "action_acceleration",
    "joint_limits",
    "physics",
    "foot_motion",
    "joint_stall",
    "plate_force",
]


def a6_obstructed_multiplier(
    env: ManagerBasedRlEnv, obstructed_scale: float
) -> torch.Tensor:
    """Return ``obstructed_scale`` until an active plate is escaped, else one."""

    # ObservationManager probes each function once before the first reset to
    # infer its shape.  The reset-bank event creates _a6_reset_scene later.
    if not hasattr(env, "_a6_reset_scene"):
        return torch.ones(env.num_envs, device=env.device)
    update_a6_plate_state(env)
    return a6_task_multiplier_from_state(
        env._a6_reset_scene > 0,
        env._a6_plate_escaped,
        obstructed_scale,
    )


def a6_amp_reward_scale(
    env: ManagerBasedRlEnv, obstructed_scale: float
) -> torch.Tensor:
    """Expose a one-column scale for AMP without changing actor/critic inputs."""

    return a6_obstructed_multiplier(env, obstructed_scale).unsqueeze(-1)


def a6_scaled_cost(
    env: ManagerBasedRlEnv,
    cost_name: A6ScalableCost,
    obstructed_scale: float,
    index: int | None = None,
) -> torch.Tensor:
    """Scale one historical A6 cost only while the robot remains obstructed."""

    if cost_name == "action_rate":
        value = a6_action_rate(env)
    elif cost_name == "action_acceleration":
        value = a6_action_acceleration(env)
    elif cost_name == "joint_limits":
        value = common_mdp.joint_pos_limits(env)
    elif cost_name == "physics":
        if index is None:
            raise ValueError("physics cost requires an index")
        value = a6_physics_cost(env, index)
    elif cost_name == "foot_motion":
        value = a6_foot_motion(env)
    elif cost_name == "joint_stall":
        value = a6_joint_stall(env)
    elif cost_name == "plate_force":
        value = plate_force(env)
    else:
        raise ValueError(f"unknown scalable A6 cost: {cost_name!r}")
    return value * a6_obstructed_multiplier(env, obstructed_scale)


__all__ = [
    "A6ScalableCost",
    "a6_amp_reward_scale",
    "a6_obstructed_multiplier",
    "a6_scaled_cost",
]
