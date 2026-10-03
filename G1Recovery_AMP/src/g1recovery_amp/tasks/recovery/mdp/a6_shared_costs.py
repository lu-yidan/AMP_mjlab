"""Control-step costs and ordered recovery phase state from frozen A6.

This module intentionally contains only terms that are evaluated once per
policy control step.  Costs that must average ten MuJoCo physics substeps live
in a later migration stage so a convenient control-rate approximation cannot
silently replace the historical definition.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import (
    A6_HEAD_OFFSET_IN_TORSO,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

A6_ACTION_RATE_PHASE_FACTORS = (0.5, 0.75, 1.0, 1.0)
A6_ACTION_ACCEL_PHASE_FACTORS = (0.3, 0.6, 1.0, 1.0)
A6_PHASE_HOLD_STEPS = (10, 10, 25, 0)
A6_JOINT_ACC_PHASE_FACTORS = (0.35, 0.65, 1.0, 1.0)
A6_TORQUE_PHASE_FACTORS = (0.5, 0.75, 1.0, 1.0)
A6_JOINT_SPEED_LIMITS = (6.0, 5.0, 4.0, 3.5)
A6_JOINT_POWER_LIMITS = (140.0, 110.0, 90.0, 75.0)
A6_HEAD_UPWARD_SPEED_LIMITS = (0.30, 0.30, 0.30, 0.20)
A6_HEAD_DOWNWARD_SPEED_LIMIT = 0.20
A6_EFFORT_MEMORY_TIME_CONSTANT = 0.5
A6_SUSTAINED_EFFORT_THRESHOLD = 0.45
A6_STALL_HISTORY_STEPS = 25
A6_STALL_LOAD_START = 0.70
A6_STALL_LOAD_FULL = 0.95
A6_STALL_TIME_START = 0.30
A6_STALL_TIME_FULL = 1.0
A6_STALL_SPEED_MAX = 0.30
A6_STALL_POSITION_SPAN = 0.10


def linear_gate(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Map values linearly from zero to one between two thresholds."""

    if upper <= lower:
        raise ValueError("upper gate threshold must be greater than lower")
    return ((value - lower) / (upper - lower)).clamp(0.0, 1.0)


def quiet_foot_motion_cost(
    head_height: torch.Tensor,
    upright: torch.Tensor,
    foot_linear_velocity: torch.Tensor,
) -> torch.Tensor:
    """Frozen A6 Q: penalize foot motion only near an upright stance."""

    foot_speed_square = foot_linear_velocity.square().sum(dim=-1).mean(dim=-1)
    return (
        linear_gate(head_height, 0.85, 1.15)
        * linear_gate(upright, 0.70, 0.93)
        * (foot_speed_square / 0.1**2).clamp_max(10.0)
    )


def update_joint_stall_substep(
    timer: torch.Tensor,
    torque: torch.Tensor,
    effort_limit: torch.Tensor,
    joint_velocity: torch.Tensor,
    physics_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance each joint's high-load timer and return its stall evidence."""

    if physics_dt <= 0.0:
        raise ValueError("physics_dt must be positive")
    if torch.any(effort_limit <= 0.0):
        raise ValueError("effort limits must be positive")
    load_ratio = (torque / effort_limit).abs()
    next_timer = torch.where(
        load_ratio > A6_STALL_LOAD_START,
        timer + physics_dt,
        torch.zeros_like(timer),
    )
    evidence = (
        linear_gate(load_ratio, A6_STALL_LOAD_START, A6_STALL_LOAD_FULL).square()
        * linear_gate(next_timer, A6_STALL_TIME_START, A6_STALL_TIME_FULL)
        * linear_gate(
            A6_STALL_SPEED_MAX - joint_velocity.abs(),
            0.0,
            A6_STALL_SPEED_MAX,
        )
    )
    return next_timer, evidence


def joint_stall_cost(
    averaged_evidence: torch.Tensor, joint_position_span: torch.Tensor
) -> torch.Tensor:
    """Frozen A6 L: take the worst persistently loaded, unmoving joint."""

    low_excursion = (1.0 - joint_position_span / A6_STALL_POSITION_SPAN).clamp(0.0, 1.0)
    return (averaged_evidence * low_excursion).amax(dim=-1)


def initial_phase_from_state(
    head_height: torch.Tensor, upright: torch.Tensor
) -> torch.Tensor:
    """Choose A6 phase 0..3 from the robot pose at episode reset."""

    phase = torch.zeros_like(head_height, dtype=torch.long)
    phase = torch.where((head_height >= 0.55) & (upright >= 0.55), 1, phase)
    phase = torch.where((head_height >= 0.78) & (upright >= 0.72), 2, phase)
    phase = torch.where((head_height >= 1.08) & (upright >= 0.85), 3, phase)
    return phase


def advance_phase_from_state(
    *,
    phase: torch.Tensor,
    hold: torch.Tensor,
    head_height: torch.Tensor,
    upright: torch.Tensor,
    knee_pos: torch.Tensor,
    head_vertical_velocity: torch.Tensor,
    minimum_foot_load: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen A6 hysteretic phase transition rules.

    ``hold`` is the number of consecutive control steps for which the next
    stage's conditions have remained true.  This prevents a single noisy
    contact sample from declaring progress.
    """

    next_phase = phase.clone()
    next_hold = hold.clone()
    fallen = (head_height < 0.65) & (upright < 0.45)
    next_phase = torch.where(fallen, 0, next_phase)
    next_hold = torch.where(fallen, 0, next_hold)

    velocity = head_vertical_velocity.abs()
    knee_min = knee_pos.amin(dim=-1)
    knee_abs_max = knee_pos.abs().amax(dim=-1)
    ready = (
        (
            (next_phase == 0)
            & (head_height >= 0.55)
            & (upright >= 0.55)
            & (knee_min >= 0.8)
            & (velocity <= 0.16)
        )
        | (
            (next_phase == 1)
            & (head_height >= 0.78)
            & (upright >= 0.72)
            & (knee_min >= 0.6)
            & (velocity <= 0.18)
        )
        | (
            (next_phase == 2)
            & (head_height >= 1.08)
            & (upright >= 0.85)
            & (knee_abs_max < 0.8)
            & (minimum_foot_load > 20.0)
            & (velocity <= 0.12)
        )
    )
    next_hold = torch.where(ready, next_hold + 1, 0)
    required = torch.where(
        next_phase == 2,
        torch.full_like(next_hold, A6_PHASE_HOLD_STEPS[2]),
        torch.full_like(next_hold, A6_PHASE_HOLD_STEPS[0]),
    )
    advance = (next_phase < 3) & (next_hold >= required)
    next_phase = torch.where(advance, next_phase + 1, next_phase)
    next_hold = torch.where(advance, 0, next_hold)
    return next_phase, next_hold


def phased_action_costs(
    action: torch.Tensor,
    previous_action: torch.Tensor,
    previous_previous_action: torch.Tensor,
    phase: torch.Tensor,
    fresh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return A6 action-rate and action-acceleration raw cost values."""

    delta = action - previous_action
    acceleration = action - 2.0 * previous_action + previous_previous_action
    delta = torch.where(fresh[:, None], 0.0, delta)
    acceleration = torch.where(fresh[:, None], 0.0, acceleration)
    rate_factors = action.new_tensor(A6_ACTION_RATE_PHASE_FACTORS)[phase]
    acceleration_factors = action.new_tensor(A6_ACTION_ACCEL_PHASE_FACTORS)[phase]
    return (
        delta.square().sum(dim=-1) * rate_factors,
        acceleration.square().sum(dim=-1) * acceleration_factors,
    )


def update_sustained_effort(
    previous_mean_square: torch.Tensor,
    torque: torch.Tensor,
    effort_limit: torch.Tensor,
    physics_dt: float,
) -> torch.Tensor:
    """Update the 0.5-second moving mean of normalized squared torque."""

    if physics_dt <= 0.0:
        raise ValueError("physics_dt must be positive")
    if torch.any(effort_limit <= 0.0):
        raise ValueError("effort limits must be positive")
    alpha = 1.0 - math.exp(-physics_dt / A6_EFFORT_MEMORY_TIME_CONSTANT)
    normalized_square = (torque / effort_limit).square()
    return previous_mean_square + alpha * (normalized_square - previous_mean_square)


def sustained_effort_cost(mean_square: torch.Tensor) -> torch.Tensor:
    """Penalize the mean of the three most persistently loaded joints."""

    normalized_excess = (
        (mean_square.clamp_min(0.0).sqrt() - A6_SUSTAINED_EFFORT_THRESHOLD).clamp_min(
            0.0
        )
        / (1.0 - A6_SUSTAINED_EFFORT_THRESHOLD)
    ).square()
    return normalized_excess.topk(3, dim=-1).values.mean(dim=-1)


def physics_substep_cost_components(
    *,
    joint_acceleration: torch.Tensor,
    torque: torch.Tensor,
    joint_velocity: torch.Tensor,
    head_vertical_velocity: torch.Tensor,
    phase: torch.Tensor,
    effort_mean_square: torch.Tensor,
) -> torch.Tensor:
    """Return the six frozen A6 raw costs for one physics substep."""

    acceleration_factor = joint_acceleration.new_tensor(A6_JOINT_ACC_PHASE_FACTORS)[
        phase
    ]
    torque_factor = torque.new_tensor(A6_TORQUE_PHASE_FACTORS)[phase]
    speed_limit = joint_velocity.new_tensor(A6_JOINT_SPEED_LIMITS)[phase, None]
    power_limit = joint_velocity.new_tensor(A6_JOINT_POWER_LIMITS)[phase, None]
    head_up_limit = head_vertical_velocity.new_tensor(A6_HEAD_UPWARD_SPEED_LIMITS)[
        phase
    ]
    joint_power = (torque * joint_velocity).abs()
    head_overspeed = (head_vertical_velocity - head_up_limit).clamp_min(
        0.0
    ).square() + (-head_vertical_velocity - A6_HEAD_DOWNWARD_SPEED_LIMIT).clamp_min(
        0.0
    ).square()
    return torch.stack(
        (
            joint_acceleration.square().sum(dim=-1) * acceleration_factor,
            torque.square().sum(dim=-1) * torque_factor,
            (joint_velocity.abs() - speed_limit).clamp_min(0.0).square().sum(dim=-1),
            (joint_power - power_limit).clamp_min(0.0).square().mean(dim=-1),
            head_overspeed,
            sustained_effort_cost(effort_mean_square),
        ),
        dim=-1,
    )


def a6_head_state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return A6 head height and vertical velocity from the torso pose."""

    robot = env.scene["robot"]
    offset = robot.data.root_link_pos_w.new_tensor(A6_HEAD_OFFSET_IN_TORSO)
    offset = offset.expand(env.num_envs, -1)
    offset_w = quat_apply(robot.data.root_link_quat_w, offset)
    head_pos = robot.data.root_link_pos_w + offset_w
    head_velocity = robot.data.root_link_lin_vel_w + torch.linalg.cross(
        robot.data.root_link_ang_vel_w, offset_w, dim=-1
    )
    return head_pos[:, 2] - env.scene.env_origins[:, 2], head_velocity[:, 2]


def initialize_a6_shared_cost_state(env: ManagerBasedRlEnv) -> None:
    """Allocate A6 phase, substep, and short-history state buffers."""

    if hasattr(env, "_a6_cost_phase"):
        return
    robot = env.scene["robot"]
    knee_ids, _ = robot.find_joints(
        ("left_knee_joint", "right_knee_joint"), preserve_order=True
    )
    foot_ids, _ = robot.find_bodies(
        ("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True
    )
    env._a6_cost_knee_ids = knee_ids
    env._a6_cost_foot_ids = foot_ids
    env._a6_cost_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._a6_cost_phase_hold = torch.zeros_like(env._a6_cost_phase)
    env._a6_cost_phase_tick = -1
    action_term = env.action_manager.get_term("joint_pos")
    ctrl_ids = getattr(action_term, "_ctrl_ids_for_targets", None)
    if ctrl_ids is None or len(ctrl_ids) != robot.data.joint_pos.shape[1]:
        raise RuntimeError("A6 costs require one mapped control input per robot joint")
    env._a6_cost_ctrl_ids = ctrl_ids
    env._a6_cost_effort_mean_square = torch.zeros_like(robot.data.joint_pos)
    env._a6_cost_substep_accum = torch.zeros(
        (env.num_envs, 6), dtype=torch.float32, device=env.device
    )
    env._a6_cost_stall_timer = torch.zeros_like(robot.data.joint_pos)
    env._a6_cost_stall_substep_accum = torch.zeros_like(robot.data.joint_pos)
    env._a6_cost_joint_position_history = robot.data.joint_pos.new_zeros(
        (A6_STALL_HISTORY_STEPS, *robot.data.joint_pos.shape)
    )
    env._a6_cost_joint_position_cursor = 0
    env._a6_cost_substep_tick = 0


def reset_a6_shared_cost_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Initialize phase from the actual bank pose of each reset environment."""

    initialize_a6_shared_cost_state(env)
    robot = env.scene["robot"]
    head_height, _ = a6_head_state(env)
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    env._a6_cost_phase[env_ids] = initial_phase_from_state(
        head_height[env_ids], upright[env_ids]
    )
    env._a6_cost_phase_hold[env_ids] = 0
    env._a6_cost_effort_mean_square[env_ids] = 0.0
    env._a6_cost_substep_accum[env_ids] = 0.0
    env._a6_cost_stall_timer[env_ids] = 0.0
    env._a6_cost_stall_substep_accum[env_ids] = 0.0
    env._a6_cost_joint_position_history[:, env_ids] = robot.data.joint_pos[
        env_ids
    ].unsqueeze(0)


def update_a6_shared_cost_state(env: ManagerBasedRlEnv) -> None:
    """Advance A6 recovery phase at most once per policy control step."""

    initialize_a6_shared_cost_state(env)
    if env._a6_cost_phase_tick == env.common_step_counter:
        return
    env._a6_cost_phase_tick = env.common_step_counter
    robot = env.scene["robot"]
    head_height, head_velocity = a6_head_state(env)
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    force = env.scene["quality_feet"].data.force
    if force is None:
        raise RuntimeError("A6 quality_feet sensor did not provide force")
    minimum_foot_load = force[..., 2].abs().amin(dim=-1)
    phase, hold = advance_phase_from_state(
        phase=env._a6_cost_phase,
        hold=env._a6_cost_phase_hold,
        head_height=head_height,
        upright=upright,
        knee_pos=robot.data.joint_pos[:, env._a6_cost_knee_ids],
        head_vertical_velocity=head_velocity,
        minimum_foot_load=minimum_foot_load,
    )
    env._a6_cost_phase.copy_(phase)
    env._a6_cost_phase_hold.copy_(hold)
    env._a6_cost_joint_position_history[env._a6_cost_joint_position_cursor].copy_(
        robot.data.joint_pos
    )
    env._a6_cost_joint_position_cursor = (
        env._a6_cost_joint_position_cursor + 1
    ) % A6_STALL_HISTORY_STEPS


def sample_a6_cost_substep(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Accumulate six A6 costs once after every MuJoCo physics substep.

    The return value is also exposed as a diagnostic metric.  Rewards read the
    six-column accumulator after all ``decimation`` substeps are complete.
    """

    initialize_a6_shared_cost_state(env)
    if env._a6_cost_substep_tick % env.cfg.decimation == 0:
        env._a6_cost_substep_accum.zero_()
        env._a6_cost_stall_substep_accum.zero_()
    robot = env.scene["robot"]
    effort_limit = (
        env.sim.model.actuator_forcerange[:, env._a6_cost_ctrl_ids].abs().amax(dim=-1)
    )
    effort_mean_square = update_sustained_effort(
        env._a6_cost_effort_mean_square,
        robot.data.qfrc_actuator,
        effort_limit,
        env.physics_dt,
    )
    env._a6_cost_effort_mean_square.copy_(effort_mean_square)
    stall_timer, stall_evidence = update_joint_stall_substep(
        env._a6_cost_stall_timer,
        robot.data.qfrc_actuator,
        effort_limit,
        robot.data.joint_vel,
        env.physics_dt,
    )
    env._a6_cost_stall_timer.copy_(stall_timer)
    env._a6_cost_stall_substep_accum += stall_evidence
    _, head_vertical_velocity = a6_head_state(env)
    components = physics_substep_cost_components(
        joint_acceleration=robot.data.joint_acc,
        torque=robot.data.qfrc_actuator,
        joint_velocity=robot.data.joint_vel,
        head_vertical_velocity=head_vertical_velocity,
        phase=env._a6_cost_phase,
        effort_mean_square=effort_mean_square,
    )
    env._a6_cost_substep_accum += components
    env._a6_cost_substep_tick += 1
    return components[:, 0]


def _action_costs(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    update_a6_shared_cost_state(env)
    manager = env.action_manager
    return phased_action_costs(
        manager.action,
        manager.prev_action,
        manager.prev_prev_action,
        env._a6_cost_phase,
        env.episode_length_buf <= 1,
    )


def a6_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """A6 phase-scaled squared first difference of raw policy actions."""

    return _action_costs(env)[0]


def a6_action_acceleration(env: ManagerBasedRlEnv) -> torch.Tensor:
    """A6 phase-scaled squared second difference of raw policy actions."""

    return _action_costs(env)[1]


def a6_physics_cost(env: ManagerBasedRlEnv, index: int) -> torch.Tensor:
    """Return one ten-substep-averaged A6 physics cost component."""

    if not 0 <= index < 6:
        raise ValueError(f"A6 physics cost index must be in [0, 5], got {index}")
    update_a6_shared_cost_state(env)
    return env._a6_cost_substep_accum[:, index] / env.cfg.decimation


def a6_foot_motion(env: ManagerBasedRlEnv) -> torch.Tensor:
    """A6 Q: near-standing foot-motion cost using both ankle links."""

    update_a6_shared_cost_state(env)
    robot = env.scene["robot"]
    head_height, _ = a6_head_state(env)
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    return quiet_foot_motion_cost(
        head_height,
        upright,
        robot.data.body_link_lin_vel_w[:, env._a6_cost_foot_ids],
    )


def a6_joint_stall(env: ManagerBasedRlEnv) -> torch.Tensor:
    """A6 L: persistent high-load, low-motion joint stall cost."""

    update_a6_shared_cost_state(env)
    history = env._a6_cost_joint_position_history
    position_span = history.amax(dim=0) - history.amin(dim=0)
    averaged_evidence = env._a6_cost_stall_substep_accum / env.cfg.decimation
    return joint_stall_cost(averaged_evidence, position_span)


__all__ = [
    "A6_ACTION_ACCEL_PHASE_FACTORS",
    "A6_ACTION_RATE_PHASE_FACTORS",
    "A6_EFFORT_MEMORY_TIME_CONSTANT",
    "A6_HEAD_DOWNWARD_SPEED_LIMIT",
    "A6_HEAD_UPWARD_SPEED_LIMITS",
    "A6_JOINT_ACC_PHASE_FACTORS",
    "A6_JOINT_POWER_LIMITS",
    "A6_JOINT_SPEED_LIMITS",
    "A6_PHASE_HOLD_STEPS",
    "A6_STALL_HISTORY_STEPS",
    "A6_STALL_LOAD_FULL",
    "A6_STALL_LOAD_START",
    "A6_STALL_POSITION_SPAN",
    "A6_STALL_SPEED_MAX",
    "A6_STALL_TIME_FULL",
    "A6_STALL_TIME_START",
    "A6_SUSTAINED_EFFORT_THRESHOLD",
    "A6_TORQUE_PHASE_FACTORS",
    "a6_action_acceleration",
    "a6_action_rate",
    "a6_foot_motion",
    "a6_head_state",
    "a6_joint_stall",
    "a6_physics_cost",
    "advance_phase_from_state",
    "initial_phase_from_state",
    "initialize_a6_shared_cost_state",
    "joint_stall_cost",
    "linear_gate",
    "phased_action_costs",
    "physics_substep_cost_components",
    "quiet_foot_motion_cost",
    "reset_a6_shared_cost_state",
    "sample_a6_cost_substep",
    "sustained_effort_cost",
    "update_a6_shared_cost_state",
    "update_joint_stall_substep",
    "update_sustained_effort",
]
