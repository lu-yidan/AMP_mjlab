"""Frozen A6 robot-dynamics randomization for the AMP adaptation task.

The original AMP baseline randomizes one torso mass and samples every actuator
gain independently.  Historical A6 instead uses coherent body and motor groups:
both legs share one factor, the upper body shares another, the pelvis/waist
share a third, and six named motor groups share paired Kp/Kd factors.  A quarter
of resets keep these additions nominal and the remaining range grows from
+/-10% to +/-20% over the first 2000 PPO updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.actuator import BuiltinPositionActuator, IdealPdActuator
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

A6_DYNAMICS_NOMINAL_FRACTION = 0.25
A6_DYNAMICS_INITIAL_WIDTH = 0.10
A6_DYNAMICS_FINAL_WIDTH = 0.20
A6_DYNAMICS_RAMP_UPDATES = 2_000
A6_CONTROL_STEPS_PER_UPDATE = 24
A6_DYNAMICS_RAMP_CONTROL_STEPS = (
    A6_DYNAMICS_RAMP_UPDATES * A6_CONTROL_STEPS_PER_UPDATE
)
A6_MAX_COMMAND_DELAY_STEPS = 5
A6_BODY_GROUP_NAMES = ("legs", "upper", "pelvis_waist")
A6_ACTUATOR_GROUP_NAMES = ("waist", "hip", "knee", "ankle", "arm", "wrist")


def a6_dynamics_width(
    common_step_counter: int,
    curriculum_start_counter: int,
    ramp_control_steps: int = A6_DYNAMICS_RAMP_CONTROL_STEPS,
) -> float:
    """Return the current symmetric randomization half-width."""

    if ramp_control_steps <= 0:
        raise ValueError("ramp_control_steps must be positive")
    age = max(common_step_counter - curriculum_start_counter, 0)
    progress = min(float(age) / ramp_control_steps, 1.0)
    return A6_DYNAMICS_INITIAL_WIDTH + (
        A6_DYNAMICS_FINAL_WIDTH - A6_DYNAMICS_INITIAL_WIDTH
    ) * progress


def sample_a6_dynamics(
    count: int,
    *,
    width: float,
    generator: torch.Generator,
    device: str,
    enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample body factors, gain factors, command lag and nominal mask."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if not 0.0 <= width < 1.0:
        raise ValueError("width must be in [0, 1)")
    body = torch.ones((count, len(A6_BODY_GROUP_NAMES)), device=device)
    gains = torch.ones((count, len(A6_ACTUATOR_GROUP_NAMES)), device=device)
    lag = torch.zeros(count, dtype=torch.long, device=device)
    nominal = torch.ones(count, dtype=torch.bool, device=device)
    if not enabled or count == 0:
        return body, gains, lag, nominal

    nominal = torch.rand(count, generator=generator, device=device) < (
        A6_DYNAMICS_NOMINAL_FRACTION
    )
    body = 1.0 + (
        2.0
        * torch.rand(
            body.shape, generator=generator, device=device, dtype=body.dtype
        )
        - 1.0
    ) * width
    gains = 1.0 + (
        2.0
        * torch.rand(
            gains.shape, generator=generator, device=device, dtype=gains.dtype
        )
        - 1.0
    ) * width
    lag = torch.randint(
        0,
        A6_MAX_COMMAND_DELAY_STEPS + 1,
        (count,),
        generator=generator,
        device=device,
    )
    body[nominal] = 1.0
    gains[nominal] = 1.0
    lag[nominal] = 0
    return body, gains, lag, nominal


def _tensor_env_ids(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[
            env_ids
        ]
    return env_ids.to(device=env.device, dtype=torch.long)


def _actuator_group(name: str) -> int:
    if "waist" in name:
        return 0
    if "hip" in name:
        return 1
    if "knee" in name:
        return 2
    if "ankle" in name:
        return 3
    if "shoulder" in name or "elbow" in name:
        return 4
    return 5


def _initialize_a6_dynamics(env: ManagerBasedRlEnv) -> None:
    if hasattr(env, "_a6_dynamics_rng"):
        return

    robot = env.scene["robot"]
    model = env.sim.mj_model
    body_ids = robot.indexing.body_ids.long()

    def descendants(root_name: str) -> set[int]:
        root_id = model.body(f"robot/{root_name}").id
        selected: set[int] = set()
        for local_index, body_id in enumerate(body_ids.tolist()):
            ancestor = int(body_id)
            while ancestor > 0 and ancestor != root_id:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root_id:
                selected.add(local_index)
        return selected

    legs = descendants("left_hip_pitch_link") | descendants(
        "right_hip_pitch_link"
    )
    upper = descendants("torso_link")
    groups = [
        0 if index in legs else 1 if index in upper else 2
        for index in range(len(body_ids))
    ]

    actuator_groups: list[int] = []
    for actuator in robot.actuators:
        if not actuator.target_names:
            raise ValueError("A6 dynamics requires named actuator targets")
        # Historical A6 classifies an actuator object by its first target.  In
        # this G1 model the shoulder/elbow actuator also owns wrist-roll, so it
        # intentionally receives the shared arm factor as one physical group.
        actuator_groups.append(_actuator_group(actuator.target_names[0]))

    env._a6_dynamics_rng = torch.Generator(device=env.device).manual_seed(
        int(env.cfg.seed or 0) + 131_071
    )
    env._a6_dynamics_body_ids = body_ids
    env._a6_dynamics_body_groups = torch.tensor(
        groups, dtype=torch.long, device=env.device
    )
    env._a6_dynamics_actuator_groups = tuple(actuator_groups)
    env._a6_body_factors = torch.ones(
        (env.num_envs, len(A6_BODY_GROUP_NAMES)), device=env.device
    )
    env._a6_gain_factors = torch.ones(
        (env.num_envs, len(A6_ACTUATOR_GROUP_NAMES)), device=env.device
    )
    env._a6_command_lag = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._a6_dynamics_nominal = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    if not hasattr(env, "_a6_curriculum_start_counter"):
        env._a6_curriculum_start_counter = int(env.common_step_counter)


@requires_model_fields(
    "body_mass",
    "body_inertia",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    recompute=RecomputeLevel.set_const,
)
def reset_a6_dynamics(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    *,
    enabled: bool = True,
) -> None:
    """Apply the frozen grouped A6 randomization to reset environments."""

    _initialize_a6_dynamics(env)
    resolved_ids = _tensor_env_ids(env, env_ids)
    width = a6_dynamics_width(
        env.common_step_counter, env._a6_curriculum_start_counter
    )
    body, gains, lag, nominal = sample_a6_dynamics(
        len(resolved_ids),
        width=width,
        generator=env._a6_dynamics_rng,
        device=env.device,
        enabled=enabled,
    )
    env._a6_body_factors[resolved_ids] = body
    env._a6_gain_factors[resolved_ids] = gains
    env._a6_command_lag[resolved_ids] = lag
    env._a6_dynamics_nominal[resolved_ids] = nominal

    robot = env.scene["robot"]
    body_ids = env._a6_dynamics_body_ids
    body_scale = body[:, env._a6_dynamics_body_groups]
    default_mass = env.sim.get_default_field("body_mass")[body_ids]
    default_inertia = env.sim.get_default_field("body_inertia")[body_ids]
    env.sim.model.body_mass[resolved_ids[:, None], body_ids[None, :]] = (
        default_mass[None, :] * body_scale
    )
    env.sim.model.body_inertia[resolved_ids[:, None], body_ids[None, :]] = (
        default_inertia[None, :, :] * body_scale[:, :, None]
    )

    for actuator, group in zip(
        robot.actuators, env._a6_dynamics_actuator_groups, strict=True
    ):
        factor = gains[:, group, None]
        if isinstance(actuator, BuiltinPositionActuator):
            ctrl_ids = actuator.global_ctrl_ids
            default_gain = env.sim.get_default_field("actuator_gainprm")
            default_bias = env.sim.get_default_field("actuator_biasprm")
            env.sim.model.actuator_gainprm[
                resolved_ids[:, None], ctrl_ids, 0
            ] = default_gain[ctrl_ids, 0] * factor
            env.sim.model.actuator_biasprm[
                resolved_ids[:, None], ctrl_ids, 1
            ] = default_bias[ctrl_ids, 1] * factor
            env.sim.model.actuator_biasprm[
                resolved_ids[:, None], ctrl_ids, 2
            ] = default_bias[ctrl_ids, 2] * factor
        elif isinstance(actuator, IdealPdActuator):
            actuator.set_gains(
                resolved_ids,
                kp=actuator.default_stiffness[resolved_ids] * factor,
                kd=actuator.default_damping[resolved_ids] * factor,
            )
        else:
            raise TypeError(
                "A6 dynamics does not support actuator type "
                f"{type(actuator).__name__}"
            )


def audit_a6_dynamics(env: ManagerBasedRlEnv) -> dict[str, bool]:
    """Assert that sampled factors reached the model and actuator tensors."""

    _initialize_a6_dynamics(env)
    body_ids = env._a6_dynamics_body_ids
    expected_mass = env.sim.get_default_field("body_mass")[body_ids][None, :] * (
        env._a6_body_factors[:, env._a6_dynamics_body_groups]
    )
    expected_inertia = env.sim.get_default_field("body_inertia")[body_ids][
        None, :, :
    ] * env._a6_body_factors[:, env._a6_dynamics_body_groups, None]
    if not torch.allclose(env.sim.model.body_mass[:, body_ids], expected_mass):
        raise AssertionError("A6 body-mass factors were not applied")
    if not torch.allclose(env.sim.model.body_inertia[:, body_ids], expected_inertia):
        raise AssertionError("A6 body-inertia factors were not applied")
    for actuator, group in zip(
        env.scene["robot"].actuators,
        env._a6_dynamics_actuator_groups,
        strict=True,
    ):
        factor = env._a6_gain_factors[:, group, None]
        if isinstance(actuator, BuiltinPositionActuator):
            ctrl_ids = actuator.global_ctrl_ids
            default_gain = env.sim.get_default_field("actuator_gainprm")
            default_bias = env.sim.get_default_field("actuator_biasprm")
            if not torch.allclose(
                env.sim.model.actuator_gainprm[:, ctrl_ids, 0],
                default_gain[ctrl_ids, 0] * factor,
            ):
                raise AssertionError("A6 Kp gain factors were not applied")
            if not torch.allclose(
                env.sim.model.actuator_biasprm[:, ctrl_ids, 1],
                default_bias[ctrl_ids, 1] * factor,
            ):
                raise AssertionError("A6 Kp bias factors were not applied")
            if not torch.allclose(
                env.sim.model.actuator_biasprm[:, ctrl_ids, 2],
                default_bias[ctrl_ids, 2] * factor,
            ):
                raise AssertionError("A6 Kd factors were not applied")
            actual_limits = env.sim.model.actuator_forcerange[:, ctrl_ids]
            expected_limits = env.sim.get_default_field("actuator_forcerange")[
                ctrl_ids
            ][None].expand_as(actual_limits)
            if not torch.equal(actual_limits, expected_limits):
                raise AssertionError("A6 randomization changed effort limits")
        elif isinstance(actuator, IdealPdActuator):
            if not torch.allclose(
                actuator.stiffness, actuator.default_stiffness * factor
            ):
                raise AssertionError("A6 Kp factors were not applied")
            if not torch.allclose(
                actuator.damping, actuator.default_damping * factor
            ):
                raise AssertionError("A6 Kd factors were not applied")
            if not torch.equal(actuator.force_limit, actuator.default_force_limit):
                raise AssertionError("A6 randomization changed effort limits")
        else:
            raise TypeError(
                "A6 dynamics does not support actuator type "
                f"{type(actuator).__name__}"
            )
    return {
        "mass_inertia": True,
        "paired_kp_kd": True,
        "effort_limits_unchanged": True,
        "command_delay_range": bool(
            torch.all(
                (env._a6_command_lag >= 0)
                & (env._a6_command_lag <= A6_MAX_COMMAND_DELAY_STEPS)
            )
        ),
    }


__all__ = [
    "A6_ACTUATOR_GROUP_NAMES",
    "A6_BODY_GROUP_NAMES",
    "A6_CONTROL_STEPS_PER_UPDATE",
    "A6_DYNAMICS_FINAL_WIDTH",
    "A6_DYNAMICS_INITIAL_WIDTH",
    "A6_DYNAMICS_NOMINAL_FRACTION",
    "A6_DYNAMICS_RAMP_CONTROL_STEPS",
    "A6_DYNAMICS_RAMP_UPDATES",
    "A6_MAX_COMMAND_DELAY_STEPS",
    "a6_dynamics_width",
    "audit_a6_dynamics",
    "reset_a6_dynamics",
    "sample_a6_dynamics",
]
