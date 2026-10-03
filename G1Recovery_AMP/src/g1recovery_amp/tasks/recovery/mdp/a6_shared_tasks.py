"""Nine frozen A6 positive recovery and quiet-motion task components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import (
    update_a6_plate_state,
)
from g1recovery_amp.tasks.recovery.mdp.a6_shared_costs import (
    a6_head_state,
    update_a6_shared_cost_state,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

A6_TASK_NAMES = (
    "stage_pose",
    "head_velocity",
    "head_height",
    "upright",
    "quiet_feet",
    "quiet_base",
    "angular_quiet",
    "joint_quiet",
    "action_smoothness",
)
A6_TASK_WEIGHTS = (0.22, 0.18, 0.10, 0.15, 0.08, 0.07, 0.07, 0.06, 0.07)
A6_PHASE_HEIGHT_TARGETS = (0.62, 0.86, 1.15, 1.15)
A6_PHASE_UPRIGHT_TARGETS = (0.60, 0.76, 0.93, 0.93)
A6_PHASE_KNEE_LOW = (0.80, 0.60, 0.00, 0.00)
A6_PHASE_KNEE_HIGH = (1.80, 1.60, 0.65, 0.65)
A6_PHASE_VERTICAL_TARGETS = (0.10, 0.15, 0.20, 0.00)
A6_PHASE_VERTICAL_UPPER_LIMITS = (0.25, 0.30, 0.30, 0.12)
A6_PHASE_VERTICAL_LOWER_LIMITS = (0.16, 0.18, 0.18, 0.12)


def a6_task_multiplier_from_state(
    active_plate: torch.Tensor,
    escaped: torch.Tensor,
    obstructed_scale: float,
) -> torch.Tensor:
    """Scale positive recovery tasks only while a plate still obstructs motion."""

    if not 0.0 <= obstructed_scale <= 1.0:
        raise ValueError("obstructed task scale must be in [0, 1]")
    obstructed = active_plate & ~escaped
    return torch.where(
        obstructed,
        torch.full_like(active_plate, obstructed_scale, dtype=torch.float32),
        torch.ones_like(active_plate, dtype=torch.float32),
    )


def smooth_gate(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Cubic zero-to-one gate used by the frozen A6 quiet-motion tasks."""

    if upper <= lower:
        raise ValueError("upper gate threshold must be greater than lower")
    fraction = ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
    return fraction.square() * (3.0 - 2.0 * fraction)


def a6_task_components_from_state(
    *,
    phase: torch.Tensor,
    head_height: torch.Tensor,
    head_vertical_velocity: torch.Tensor,
    upright: torch.Tensor,
    knee_position: torch.Tensor,
    foot_linear_velocity: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    joint_velocity: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    fresh: torch.Tensor,
) -> torch.Tensor:
    """Return the nine frozen A6 positive task values for each environment."""

    height_target = head_height.new_tensor(A6_PHASE_HEIGHT_TARGETS)[phase]
    upright_target = upright.new_tensor(A6_PHASE_UPRIGHT_TARGETS)[phase]
    knee_low = knee_position.new_tensor(A6_PHASE_KNEE_LOW)[phase, None]
    knee_high = knee_position.new_tensor(A6_PHASE_KNEE_HIGH)[phase, None]
    knee_error = (
        (knee_low - knee_position).clamp_min(0.0).square()
        + (knee_position - knee_high).clamp_min(0.0).square()
    ).mean(dim=-1)
    stage_pose = torch.exp(
        -8.0 * (height_target - head_height).clamp_min(0.0).square()
        - 6.0 * (upright_target - upright).clamp_min(0.0).square()
        - 5.0 * knee_error
    )

    vertical_target = head_height.new_tensor(A6_PHASE_VERTICAL_TARGETS)[phase]
    vertical_upper = head_height.new_tensor(A6_PHASE_VERTICAL_UPPER_LIMITS)[phase]
    vertical_lower = head_height.new_tensor(A6_PHASE_VERTICAL_LOWER_LIMITS)[phase]
    vertical_target = vertical_target * ((height_target - head_height) / 0.2).clamp(
        0.0, 1.0
    )
    vertical_excess = (head_vertical_velocity - vertical_upper).clamp_min(
        0.0
    ).square() + (-head_vertical_velocity - vertical_lower).clamp_min(0.0).square()
    head_velocity = torch.exp(
        -45.0 * (head_vertical_velocity - vertical_target).square()
        - 140.0 * vertical_excess
    )

    near_standing = smooth_gate(head_height, 0.85, 1.15) * smooth_gate(
        upright, 0.70, 0.93
    )
    foot_speed_square = foot_linear_velocity.square().sum(dim=-1).mean(dim=-1)
    action_delta = action - previous_action
    action_delta = torch.where(fresh[:, None], 0.0, action_delta)
    return torch.stack(
        (
            stage_pose,
            head_velocity,
            torch.exp(-2.0 * (1.15 - head_height).clamp_min(0.0).square()),
            upright.square(),
            1.0 - near_standing + near_standing * torch.exp(-20.0 * foot_speed_square),
            1.0
            - near_standing
            + near_standing
            * torch.exp(-8.0 * base_linear_velocity.square().sum(dim=-1)),
            torch.exp(-0.8 * base_angular_velocity.square().sum(dim=-1)),
            torch.exp(-0.04 * joint_velocity.square().mean(dim=-1)),
            torch.exp(-4.0 * action_delta.square().mean(dim=-1)),
        ),
        dim=-1,
    )


def compute_a6_task_components(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Compute and cache all nine components once per control step."""

    update_a6_shared_cost_state(env)
    if getattr(env, "_a6_task_tick", -1) == env.common_step_counter:
        return env._a6_task_components
    robot = env.scene["robot"]
    head_height, head_vertical_velocity = a6_head_state(env)
    values = a6_task_components_from_state(
        phase=env._a6_cost_phase,
        head_height=head_height,
        head_vertical_velocity=head_vertical_velocity,
        upright=(-robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0),
        knee_position=robot.data.joint_pos[:, env._a6_cost_knee_ids],
        foot_linear_velocity=robot.data.body_link_lin_vel_w[:, env._a6_cost_foot_ids],
        base_linear_velocity=robot.data.root_link_lin_vel_w,
        base_angular_velocity=robot.data.root_link_ang_vel_w,
        joint_velocity=robot.data.joint_vel,
        action=env.action_manager.action,
        previous_action=env.action_manager.prev_action,
        fresh=env.episode_length_buf <= 1,
    )
    env._a6_task_components = values
    env._a6_task_tick = env.common_step_counter
    return values


def a6_task_component(
    env: ManagerBasedRlEnv, index: int, obstructed_scale: float = 1.0
) -> torch.Tensor:
    """Return one of the nine cached A6 positive task components."""

    if not 0 <= index < len(A6_TASK_NAMES):
        raise ValueError(
            f"A6 task component index must be in [0, {len(A6_TASK_NAMES) - 1}], "
            f"got {index}"
        )
    value = compute_a6_task_components(env)[:, index]
    if obstructed_scale == 1.0:
        return value
    update_a6_plate_state(env)
    multiplier = a6_task_multiplier_from_state(
        env._a6_reset_scene > 0,
        env._a6_plate_escaped,
        obstructed_scale,
    )
    return value * multiplier


__all__ = [
    "A6_PHASE_HEIGHT_TARGETS",
    "A6_PHASE_KNEE_HIGH",
    "A6_PHASE_KNEE_LOW",
    "A6_PHASE_UPRIGHT_TARGETS",
    "A6_PHASE_VERTICAL_LOWER_LIMITS",
    "A6_PHASE_VERTICAL_TARGETS",
    "A6_PHASE_VERTICAL_UPPER_LIMITS",
    "A6_TASK_NAMES",
    "A6_TASK_WEIGHTS",
    "a6_task_component",
    "a6_task_components_from_state",
    "a6_task_multiplier_from_state",
    "compute_a6_task_components",
    "smooth_gate",
]
