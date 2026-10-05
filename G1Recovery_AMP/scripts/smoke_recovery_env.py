"""Run a short zero-action smoke test of the recovery development environment."""

from __future__ import annotations

import argparse
import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.types import VecEnvObs
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1recovery_amp.tasks.recovery import DEV_TASK_ID
from g1recovery_amp.tasks.recovery.env_cfg import (
    MUJOCO_SAFE_GROUND_FRICTION,
    RecoveryDisturbanceMode,
    g1_recovery_dev_env_cfg,
)
from g1recovery_amp.tasks.recovery.mdp import (
    AMP_DISCRIMINATOR_FRAME_DIM,
    AMP_DISCRIMINATOR_HISTORY_LENGTH,
    AMP_TERMINAL_DISC_OBSERVATION_KEY,
    AMP_TERMINAL_ENV_IDS_KEY,
    RECOVERY_CRITIC_OBS_DIM,
    RECOVERY_POLICY_OBS_DIM,
    GetUpAssistForceCommand,
    MixedRecoveryDisturbance,
    RecoveryMotionCommand,
    SustainedBodyWrench,
)
from g1recovery_amp.tasks.recovery.rl import (
    AmpDiscriminator,
    AmpDiscriminatorOptimizer,
    AmpReplayBuffer,
)


def _assert_finite_observations(observations: VecEnvObs) -> None:
    for group_name, values in observations.items():
        tensors = values.values() if isinstance(values, dict) else (values,)
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise RuntimeError(f"observation group {group_name!r} contains NaN or Inf")


def _check_startup_randomization(
    env: ManagerBasedRlEnv,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Check that startup friction and consistent torso inertia reached the model."""

    material_term = env.event_manager.get_term_cfg("physics_material")
    material_cfg = material_term.params["asset_cfg"]
    if not isinstance(material_cfg, SceneEntityCfg):
        raise TypeError("physics_material asset_cfg was not resolved")
    robot = env.scene[material_cfg.name]
    global_geom_ids = robot.indexing.geom_ids[material_cfg.geom_ids]
    friction = env.sim.model.geom_friction[:, global_geom_ids, 0]
    friction_bounds = tuple(float(value) for value in material_term.params["ranges"])
    friction_range = (float(friction.min()), float(friction.max()))
    if friction_range[0] < friction_bounds[0] or friction_range[1] > friction_bounds[1]:
        raise RuntimeError(
            f"randomized friction {friction_range} is outside {friction_bounds}"
        )
    terrain = env.scene["terrain"]
    ground_friction = env.sim.model.geom_friction[
        :, terrain.indexing.geom_ids, 0
    ]
    ground_friction_range = (
        float(ground_friction.min()),
        float(ground_friction.max()),
    )
    expected_ground_range = (
        MUJOCO_SAFE_GROUND_FRICTION,
        MUJOCO_SAFE_GROUND_FRICTION,
    )
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-8)
        for actual, expected in zip(
            ground_friction_range, expected_ground_range, strict=True
        )
    ):
        raise RuntimeError(
            "unexpected ground friction baseline: "
            f"{ground_friction_range}"
        )

    mass_term = env.event_manager.get_term_cfg("add_base_mass")
    mass_cfg = mass_term.params["asset_cfg"]
    if not isinstance(mass_cfg, SceneEntityCfg):
        raise TypeError("add_base_mass asset_cfg was not resolved")
    global_body_ids = robot.indexing.body_ids[mass_cfg.body_ids]
    randomized_mass = env.sim.model.body_mass[:, global_body_ids]
    default_mass = env.sim.get_default_field("body_mass")[global_body_ids].unsqueeze(0)
    mass_delta = randomized_mass - default_mass
    mass_bounds = tuple(
        float(value) for value in mass_term.params["mass_delta_range"]
    )
    mass_delta_range = (float(mass_delta.min()), float(mass_delta.max()))
    if mass_delta_range[0] < mass_bounds[0] or mass_delta_range[1] > mass_bounds[1]:
        raise RuntimeError(
            f"randomized mass delta {mass_delta_range} is outside {mass_bounds}"
        )

    randomized_inertia = env.sim.model.body_inertia[:, global_body_ids]
    default_inertia = env.sim.get_default_field("body_inertia")[
        global_body_ids
    ].unsqueeze(0)
    expected_scale = (randomized_mass / default_mass).unsqueeze(-1)
    inertia_scale = randomized_inertia / default_inertia
    torch.testing.assert_close(inertia_scale, expected_scale.expand_as(inertia_scale))
    inertia_scale_range = (float(inertia_scale.min()), float(inertia_scale.max()))

    return (
        friction_range,
        ground_friction_range,
        mass_delta_range,
        inertia_scale_range,
    )


def _check_reset_pd_randomization(
    env: ManagerBasedRlEnv,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Check training-only Kp/Kd scale samples after an environment reset."""

    if "actuator_gains" not in env.event_manager.active_terms.get("reset", ()):
        return None

    gains_term = env.event_manager.get_term_cfg("actuator_gains")
    robot = env.scene["robot"]
    ctrl_ids = torch.cat([actuator.global_ctrl_ids for actuator in robot.actuators])
    default_gain = env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0]
    default_damping = -env.sim.get_default_field("actuator_biasprm")[ctrl_ids, 2]
    if torch.any(default_gain <= 0.0) or torch.any(default_damping <= 0.0):
        raise RuntimeError("expected positive default Kp/Kd for every G1 actuator")

    kp_ratio = env.sim.model.actuator_gainprm[:, ctrl_ids, 0] / default_gain
    kd_ratio = -env.sim.model.actuator_biasprm[:, ctrl_ids, 2] / default_damping
    kp_range = (float(kp_ratio.min()), float(kp_ratio.max()))
    kd_range = (float(kd_ratio.min()), float(kd_ratio.max()))
    kp_bounds = tuple(float(value) for value in gains_term.params["kp_range"])
    kd_bounds = tuple(float(value) for value in gains_term.params["kd_range"])
    if kp_range[0] < kp_bounds[0] or kp_range[1] > kp_bounds[1]:
        raise RuntimeError(f"randomized Kp ratios {kp_range} are outside {kp_bounds}")
    if kd_range[0] < kd_bounds[0] or kd_range[1] > kd_bounds[1]:
        raise RuntimeError(f"randomized Kd ratios {kd_range} are outside {kd_bounds}")
    return kp_range, kd_range


def _check_get_up_assist_force(
    env: ManagerBasedRlEnv,
) -> tuple[float, float]:
    """Verify that reset wrote each stored upward force to the torso only."""

    event = env.event_manager.get_term_cfg("get_up_assist_force")
    asset_cfg = event.params["asset_cfg"]
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("get_up_assist_force asset_cfg was not resolved")
    if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 1:
        raise RuntimeError("get-up assistance must resolve exactly torso_link")
    command = env.command_manager.get_term("get_up_assist_force")
    if not isinstance(command, GetUpAssistForceCommand):
        raise TypeError("expected the get-up assist force command")

    robot = env.scene[asset_cfg.name]
    wrench = robot.data.body_external_wrench[:, asset_cfg.body_ids]
    if torch.any(wrench[..., 0:2] != 0.0) or torch.any(wrench[..., 3:6] != 0.0):
        raise RuntimeError("get-up assistance contains a non-vertical force or torque")
    torch.testing.assert_close(wrench[:, 0, 2], command.command[:, 0])
    return float(wrench[:, 0, 2].min()), float(wrench[:, 0, 2].max())


def _check_base_velocity_push(
    env: ManagerBasedRlEnv,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Trigger, inspect, and undo the training-only base-velocity push."""

    if "base_velocity_push" not in env.event_manager.active_terms.get("interval", ()):
        return None

    push_term = env.event_manager.get_term_cfg("base_velocity_push")
    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    velocity_before = robot.data.root_link_vel_w[env_ids].clone()
    push_term.func(env, env_ids, **push_term.params)
    env.sim.forward()
    velocity_after = robot.data.root_link_vel_w[env_ids].clone()

    linear_bounds = tuple(
        float(value) for value in push_term.params["linear_velocity_range"]
    )
    yaw_bounds = tuple(float(value) for value in push_term.params["yaw_velocity_range"])
    horizontal_velocity = velocity_after[:, 0:2]
    yaw_velocity = velocity_after[:, 5]
    if torch.any(horizontal_velocity < linear_bounds[0]) or torch.any(
        horizontal_velocity > linear_bounds[1]
    ):
        raise RuntimeError(
            f"base-push x/y velocity is outside {linear_bounds}: {horizontal_velocity}"
        )
    if torch.any(yaw_velocity < yaw_bounds[0]) or torch.any(
        yaw_velocity > yaw_bounds[1]
    ):
        raise RuntimeError(
            f"base-push yaw velocity is outside {yaw_bounds}: {yaw_velocity}"
        )
    torch.testing.assert_close(
        velocity_after[:, 2:5], velocity_before[:, 2:5], atol=1.0e-5, rtol=0.0
    )

    robot.write_root_link_velocity_to_sim(velocity_before, env_ids=env_ids)
    env.sim.forward()
    horizontal_range = (
        float(horizontal_velocity.min()),
        float(horizontal_velocity.max()),
    )
    yaw_range = (float(yaw_velocity.min()), float(yaw_velocity.max()))
    return horizontal_range, yaw_range


def _check_body_impulse(
    env: ManagerBasedRlEnv,
) -> tuple[tuple[float, float], tuple[float, float], int] | None:
    """Wait for one configured impulse, inspect its wrench, and verify clearing."""

    if "body_impulse" not in env.event_manager.active_terms.get("step", ()):
        return None

    impulse_term = env.event_manager.get_term_cfg("body_impulse")
    impulse_asset_cfg = impulse_term.params["asset_cfg"]
    if not isinstance(impulse_asset_cfg, SceneEntityCfg):
        raise TypeError("body_impulse asset_cfg was not resolved")
    if not isinstance(impulse_asset_cfg.body_ids, list):
        raise TypeError("body_impulse must resolve to an explicit body-id list")

    robot = env.scene[impulse_asset_cfg.name]
    selected_body_ids = impulse_asset_cfg.body_ids
    if len(selected_body_ids) != 7:
        raise RuntimeError(
            f"body impulse resolved {len(selected_body_ids)} bodies instead of 7"
        )
    force_bounds = tuple(float(value) for value in impulse_term.params["force_range"])
    torque_bounds = tuple(float(value) for value in impulse_term.params["torque_range"])
    maximum_cooldown_steps = int(impulse_term.params["cooldown_s"][1] / env.step_dt) + 2

    trigger_step = 0
    selected_wrench = robot.data.body_external_wrench[:, selected_body_ids]
    for trigger_step in range(1, maximum_cooldown_steps + 1):
        impulse_term.func(env, None, **impulse_term.params)
        selected_wrench = robot.data.body_external_wrench[:, selected_body_ids]
        if torch.any(selected_wrench != 0.0):
            break
    else:
        raise RuntimeError("body impulse did not trigger within its maximum cooldown")

    force = selected_wrench[..., 0:3]
    torque = selected_wrench[..., 3:6]
    if torch.any(force < force_bounds[0]) or torch.any(force > force_bounds[1]):
        raise RuntimeError(f"impulse force is outside {force_bounds}")
    if torch.any(torque < torque_bounds[0]) or torch.any(torque > torque_bounds[1]):
        raise RuntimeError(f"impulse torque is outside {torque_bounds}")
    force_range = (float(force.min()), float(force.max()))
    torque_range = (float(torque.min()), float(torque.max()))

    maximum_duration_steps = int(impulse_term.params["duration_s"][1] / env.step_dt) + 2
    for _ in range(maximum_duration_steps):
        impulse_term.func(env, None, **impulse_term.params)
        selected_wrench = robot.data.body_external_wrench[:, selected_body_ids]
        if torch.all(selected_wrench == 0.0):
            break
    else:
        raise RuntimeError("body impulse was not cleared after its maximum duration")

    return force_range, torque_range, trigger_step


def _check_sustained_body_wrench(
    env: ManagerBasedRlEnv,
) -> tuple[tuple[float, float], tuple[float, float], int] | None:
    """Verify mass scaling, smooth ramping, holding, and reset clearing."""

    term_name = "sustained_body_wrench"
    if term_name not in env.event_manager.active_terms.get("step", ()):
        return None

    sustained_term = env.event_manager.get_term_cfg(term_name)
    asset_cfg = sustained_term.params["asset_cfg"]
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError("sustained_body_wrench asset_cfg was not resolved")
    if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 7:
        raise RuntimeError("sustained_body_wrench must resolve exactly seven bodies")

    robot = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    acceleration_bounds = tuple(
        float(value)
        for value in sustained_term.params["linear_acceleration_range"]
    )
    ramp_s = float(sustained_term.params["ramp_s"])
    maximum_period_steps = int(sustained_term.params["period_s"][1] / env.step_dt) + 2

    trigger_step = 0
    selected_wrench = robot.data.body_external_wrench[:, body_ids]
    for trigger_step in range(1, maximum_period_steps + 1):
        sustained_term.func(env, None, **sustained_term.params)
        selected_wrench = robot.data.body_external_wrench[:, body_ids]
        if torch.any(selected_wrench != 0.0):
            break
    else:
        raise RuntimeError("sustained wrench did not trigger within its period")

    if not isinstance(sustained_term.func, SustainedBodyWrench):
        raise TypeError("sustained_body_wrench class was not instantiated")
    global_body_ids = robot.indexing.body_ids[body_ids]
    selected_mass = env.sim.model.body_mass[:, global_body_ids]
    sampled_acceleration = sustained_term.func.force / selected_mass.unsqueeze(-1)
    if torch.any(sampled_acceleration < acceleration_bounds[0]) or torch.any(
        sampled_acceleration > acceleration_bounds[1]
    ):
        raise RuntimeError(
            f"sustained acceleration is outside {acceleration_bounds}"
        )

    ramp_steps = math.ceil(ramp_s / env.step_dt) + 1
    for _ in range(ramp_steps):
        sustained_term.func(env, None, **sustained_term.params)
    held_wrench = robot.data.body_external_wrench[:, body_ids].clone()
    force = held_wrench[..., 0:3]
    torque = held_wrench[..., 3:6]
    if torch.any(torque != 0.0):
        raise RuntimeError("sustained mode unexpectedly applied a direct torque")
    force_range = (float(force.min()), float(force.max()))
    torque_range = (float(torque.min()), float(torque.max()))

    sustained_term.func(env, None, **sustained_term.params)
    torch.testing.assert_close(
        robot.data.body_external_wrench[:, body_ids], held_wrench
    )

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    sustained_term.func.reset(env_ids=env_ids)
    if torch.any(robot.data.body_external_wrench[:, body_ids] != 0.0):
        raise RuntimeError("sustained wrench leaked across reset")
    return force_range, torque_range, trigger_step


def run_smoke_test(
    *,
    steps: int = 100,
    num_envs: int = 1,
    device: str = "cuda:0",
    action_mode: str = "zero",
    config_mode: str = "play",
    disturbance_mode: RecoveryDisturbanceMode = "mixed",
) -> None:
    """Reset one environment and advance it with zero or random actions."""

    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if action_mode not in {"zero", "random"}:
        raise ValueError(f"unknown action mode: {action_mode}")
    if config_mode not in {"play", "train"}:
        raise ValueError(f"unknown config mode: {config_mode}")
    if disturbance_mode not in {
        "mixed",
        "base_push",
        "impulse",
        "sustained",
        "none",
    }:
        raise ValueError(f"unknown disturbance mode: {disturbance_mode}")

    cfg = g1_recovery_dev_env_cfg(
        play=config_mode == "play",
        disturbance_mode=disturbance_mode,
    )
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        observations, _ = env.reset(seed=0)
        _assert_finite_observations(observations)
        friction_range, ground_friction_range, mass_delta_range, inertia_scale_range = (
            _check_startup_randomization(env)
        )
        assist_force_range = _check_get_up_assist_force(env)
        pd_gain_ratio_ranges = _check_reset_pd_randomization(env)
        if "amp_terminal_observation" not in env.recorder_manager:
            raise RuntimeError("AMP terminal observation recorder is not active")

        motion = env.command_manager.get_term("motion")
        if not isinstance(motion, RecoveryMotionCommand):
            raise TypeError("expected the recovery reference-state command")
        robot = env.scene["robot"]
        reset_motion_id = int(motion.reset_motion_ids[0])
        reset_frame = int(motion.reset_time_steps[0])
        reset_motion_counts = torch.bincount(
            motion.reset_motion_ids,
            minlength=len(motion.motions),
        ).tolist()
        expected_body_pos = motion.gather_motion_field(
            "body_pos_w",
            motion.reset_motion_ids,
            motion.reset_time_steps,
        )
        expected_body_quat = motion.gather_motion_field(
            "body_quat_w",
            motion.reset_motion_ids,
            motion.reset_time_steps,
        )
        expected_body_lin_vel = motion.gather_motion_field(
            "body_lin_vel_w",
            motion.reset_motion_ids,
            motion.reset_time_steps,
        )
        expected_body_ang_vel = motion.gather_motion_field(
            "body_ang_vel_w",
            motion.reset_motion_ids,
            motion.reset_time_steps,
        )
        expected_joint_pos = motion.gather_motion_field(
            "joint_pos",
            motion.reset_motion_ids,
            motion.reset_time_steps,
        )
        expected_joint_vel = motion.gather_motion_field(
            "joint_vel",
            motion.reset_motion_ids,
            motion.reset_time_steps,
        )
        expected_root_pos = expected_body_pos[:, 0] + env.scene.env_origins
        expected_root_pos[:, 2] += motion.recovery_cfg.root_height_offset
        reset_joint_pos_error = float(
            torch.max(torch.abs(robot.data.joint_pos - expected_joint_pos))
        )
        reset_joint_vel_error = float(
            torch.max(torch.abs(robot.data.joint_vel - expected_joint_vel))
        )
        reset_root_pos_error = float(
            torch.max(torch.abs(robot.data.root_link_pos_w - expected_root_pos))
        )
        reset_root_quat_error = float(
            torch.max(
                torch.abs(robot.data.root_link_quat_w - expected_body_quat[:, 0])
            )
        )
        reset_root_lin_vel_error = float(
            torch.max(
                torch.abs(
                    robot.data.root_link_lin_vel_w - expected_body_lin_vel[:, 0]
                )
            )
        )
        reset_root_ang_vel_error = float(
            torch.max(
                torch.abs(
                    robot.data.root_link_ang_vel_w - expected_body_ang_vel[:, 0]
                )
            )
        )
        max_reset_error = max(
            reset_joint_pos_error,
            reset_joint_vel_error,
            reset_root_pos_error,
            reset_root_quat_error,
            reset_root_lin_vel_error,
            reset_root_ang_vel_error,
        )
        if max_reset_error > 1.0e-5:
            raise RuntimeError(
                f"reference reset mismatch: maximum error is {max_reset_error}"
            )
        base_push_velocity_ranges = _check_base_velocity_push(env)
        body_impulse_result = _check_body_impulse(env)
        sustained_wrench_result = _check_sustained_body_wrench(env)
        mixed_disturbance: MixedRecoveryDisturbance | None = None
        mixed_mode_counts: dict[str, int] | None = None
        if "mixed_disturbance" in env.event_manager.active_terms.get("step", ()):
            mixed_term = env.event_manager.get_term_cfg("mixed_disturbance")
            if not isinstance(mixed_term.func, MixedRecoveryDisturbance):
                raise TypeError("mixed_disturbance class was not instantiated")
            mixed_disturbance = mixed_term.func
            mixed_mode_counts = mixed_disturbance.mode_counts()

        action_dim = env.action_manager.total_action_dim
        actions = torch.zeros((env.num_envs, action_dim), device=env.device)
        reward_min = float("inf")
        reward_max = float("-inf")
        termination_count = 0
        timeout_count = 0
        observed_natural_body_impulse = False
        observed_natural_sustained_wrench = False
        terminal_amp_observation_count = 0
        terminal_vs_reset_max_difference = 0.0
        replay_length = min(steps, 4)
        disc_replay = AmpReplayBuffer(
            max_length=replay_length,
            batch_size=env.num_envs,
            device=env.device,
        )
        disc_demo_replay = AmpReplayBuffer(
            max_length=replay_length,
            batch_size=env.num_envs,
            device=env.device,
        )

        for _ in range(steps):
            if action_mode == "random":
                actions.uniform_(-1.0, 1.0)
            observations, rewards, terminated, truncated, extras = env.step(actions)
            _assert_finite_observations(observations)
            step_disc = observations["disc"]
            step_disc_demo = observations["disc_demo"]
            if not isinstance(step_disc, torch.Tensor) or not isinstance(
                step_disc_demo, torch.Tensor
            ):
                raise TypeError("expected tensor AMP observations during replay check")
            disc_replay.append(step_disc)
            disc_demo_replay.append(step_disc_demo)
            done_mask = terminated | truncated
            if torch.any(done_mask):
                terminal_env_ids = extras.get(AMP_TERMINAL_ENV_IDS_KEY)
                terminal_disc = extras.get(AMP_TERMINAL_DISC_OBSERVATION_KEY)
                if not isinstance(terminal_env_ids, torch.Tensor) or not isinstance(
                    terminal_disc, torch.Tensor
                ):
                    raise RuntimeError("done step is missing terminal AMP observations")
                expected_env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
                if not torch.equal(terminal_env_ids, expected_env_ids):
                    raise RuntimeError(
                        f"terminal AMP env ids {terminal_env_ids.tolist()} do not "
                        f"match dones {expected_env_ids.tolist()}"
                    )
                expected_terminal_shape = (
                    len(expected_env_ids),
                    AMP_DISCRIMINATOR_HISTORY_LENGTH,
                    AMP_DISCRIMINATOR_FRAME_DIM,
                )
                if tuple(terminal_disc.shape) != expected_terminal_shape:
                    raise RuntimeError(
                        f"terminal AMP shape {tuple(terminal_disc.shape)} != "
                        f"{expected_terminal_shape}"
                    )
                if not torch.isfinite(terminal_disc).all():
                    raise RuntimeError("terminal AMP observation contains NaN or Inf")
                difference = float(
                    (terminal_disc - step_disc[terminal_env_ids]).abs().max()
                )
                if difference <= 0.0:
                    raise RuntimeError(
                        "terminal AMP observation equals the post-reset observation"
                    )
                terminal_amp_observation_count += len(terminal_env_ids)
                terminal_vs_reset_max_difference = max(
                    terminal_vs_reset_max_difference,
                    difference,
                )
            elif (
                AMP_TERMINAL_ENV_IDS_KEY in extras
                or AMP_TERMINAL_DISC_OBSERVATION_KEY in extras
            ):
                raise RuntimeError("stale terminal AMP observation leaked to next step")
            if not torch.isfinite(rewards).all():
                raise RuntimeError("reward contains NaN or Inf")
            reward_min = min(reward_min, float(rewards.min()))
            reward_max = max(reward_max, float(rewards.max()))
            termination_count += int(terminated.sum())
            timeout_count += int(truncated.sum())
            if "body_impulse" in env.event_manager.active_terms.get("step", ()):
                observed_natural_body_impulse |= bool(
                    torch.any(robot.data.body_external_wrench != 0.0)
                )
            if "sustained_body_wrench" in env.event_manager.active_terms.get(
                "step", ()
            ):
                observed_natural_sustained_wrench |= bool(
                    torch.any(robot.data.body_external_wrench != 0.0)
                )

        if (
            disturbance_mode == "impulse"
            and steps * env.step_dt >= 3.0
            and not observed_natural_body_impulse
        ):
            raise RuntimeError(
                "body impulse did not fire naturally during the environment loop"
            )
        if (
            disturbance_mode == "sustained"
            and steps * env.step_dt >= 3.0
            and not observed_natural_sustained_wrench
        ):
            raise RuntimeError(
                "sustained wrench did not fire naturally during the environment loop"
            )

        mixed_trigger_ranges: dict[str, tuple[int, int]] | None = None
        if mixed_disturbance is not None:
            mixed_trigger_ranges = {}
            for mode_index, name in enumerate(mixed_disturbance.mode_names):
                selected_counts = mixed_disturbance.trigger_counts[
                    mixed_disturbance.env_mode == mode_index
                ]
                if len(selected_counts) == 0:
                    mixed_trigger_ranges[name] = (0, 0)
                    continue
                mixed_trigger_ranges[name] = (
                    int(selected_counts.min()),
                    int(selected_counts.max()),
                )
                if steps * env.step_dt >= 4.0:
                    if name == "none" and torch.any(selected_counts != 0):
                        raise RuntimeError(
                            "none-mode environments triggered a disturbance"
                        )
                    if name != "none" and torch.any(selected_counts == 0):
                        raise RuntimeError(
                            f"some {name} environments never triggered a disturbance"
                        )

        root_height = float(robot.data.root_link_pos_w[0, 2])
        health_metrics = {
            name: float(values[0])
            for name, values in env.metrics_manager.get_active_iterable_terms(0)
        }
        expected_health_metrics = (
            "max_root_angular_speed",
            "max_joint_speed",
            "max_joint_acceleration",
            "max_contact_penetration",
            "batch_max_joint_acceleration",
            "batch_max_contact_penetration",
            "unsafe_root_angular_speed",
            "unsafe_joint_speed",
            "unsafe_joint_acceleration",
            "unsafe_contact_penetration",
            "unsafe_physics_state",
        )
        if tuple(health_metrics) != expected_health_metrics:
            raise RuntimeError(
                f"physics health metrics {tuple(health_metrics)} != "
                f"{expected_health_metrics}"
            )
        if not all(math.isfinite(value) for value in health_metrics.values()):
            raise RuntimeError(f"physics health metric is not finite: {health_metrics}")
        actor_observations = observations["actor"]
        if not isinstance(actor_observations, torch.Tensor):
            raise TypeError("expected concatenated actor observations")
        actor_shape = tuple(actor_observations.shape)
        expected_actor_shape = (env.num_envs, RECOVERY_POLICY_OBS_DIM)
        if actor_shape != expected_actor_shape:
            raise RuntimeError(
                f"actor observation shape {actor_shape} != {expected_actor_shape}"
            )
        critic_observations = observations["critic"]
        if not isinstance(critic_observations, torch.Tensor):
            raise TypeError("expected concatenated critic observations")
        critic_shape = tuple(critic_observations.shape)
        expected_critic_shape = (env.num_envs, RECOVERY_CRITIC_OBS_DIM)
        if critic_shape != expected_critic_shape:
            raise RuntimeError(
                f"critic observation shape {critic_shape} != {expected_critic_shape}"
            )
        disc_observations = observations["disc"]
        disc_demo_observations = observations["disc_demo"]
        if not isinstance(disc_observations, torch.Tensor) or not isinstance(
            disc_demo_observations, torch.Tensor
        ):
            raise TypeError("expected concatenated AMP discriminator observations")
        expected_disc_shape = (
            env.num_envs,
            AMP_DISCRIMINATOR_HISTORY_LENGTH,
            AMP_DISCRIMINATOR_FRAME_DIM,
        )
        disc_shape = tuple(disc_observations.shape)
        disc_demo_shape = tuple(disc_demo_observations.shape)
        if disc_shape != expected_disc_shape or disc_demo_shape != expected_disc_shape:
            raise RuntimeError(
                "AMP observation shape mismatch: "
                f"disc={disc_shape}, demo={disc_demo_shape}, "
                f"expected={expected_disc_shape}"
            )
        discriminator = AmpDiscriminator().to(env.device)
        replay_disc_observations = disc_replay.sample(env.num_envs)
        replay_demo_observations = disc_demo_replay.sample(env.num_envs)
        with torch.inference_mode():
            disc_score = discriminator.score(disc_observations)
            disc_demo_score = discriminator.score(disc_demo_observations)
            style_reward, _ = discriminator.style_reward(disc_observations, env.step_dt)
            replay_disc_score = discriminator.score(replay_disc_observations)
            replay_demo_score = discriminator.score(replay_demo_observations)
        output_weight_before_update = discriminator.disc_linear.weight.detach().clone()
        discriminator_optimizer = AmpDiscriminatorOptimizer(discriminator)
        discriminator_update_metrics = discriminator_optimizer.update(
            replay_disc_observations,
            replay_demo_observations,
        )
        output_weight_update = float(
            (discriminator.disc_linear.weight - output_weight_before_update)
            .detach()
            .abs()
            .max()
        )
        if output_weight_update <= 0.0:
            raise RuntimeError(
                "AMP discriminator optimizer did not change output weights"
            )
        expected_normalization_count = (
            2 * env.num_envs * AMP_DISCRIMINATOR_HISTORY_LENGTH
        )
        normalization_count = int(discriminator.disc_obs_normalizer.count)
        if normalization_count != expected_normalization_count:
            raise RuntimeError(
                "AMP normalization count mismatch: "
                f"{normalization_count} != {expected_normalization_count}"
            )
        if disc_score.shape != (env.num_envs,) or disc_demo_score.shape != (
            env.num_envs,
        ):
            raise RuntimeError("AMP discriminator did not return one score per env")
        if not all(
            torch.isfinite(value).all()
            for value in (
                disc_score,
                disc_demo_score,
                style_reward,
                replay_disc_score,
                replay_demo_score,
            )
        ):
            raise RuntimeError("AMP discriminator score or style reward is not finite")
        discriminator_parameter_count = sum(
            parameter.numel() for parameter in discriminator.parameters()
        )
        print(f"task={DEV_TASK_ID}")
        print(f"device={env.device}, num_envs={env.num_envs}")
        print(f"physics_dt={env.physics_dt}, step_dt={env.step_dt}")
        print(
            f"startup_friction_range={friction_range}, "
            f"ground_friction_range={ground_friction_range}, "
            f"startup_torso_mass_delta_range={mass_delta_range}kg, "
            f"startup_torso_inertia_scale_range={inertia_scale_range}"
        )
        print(f"reset_pd_gain_ratio_ranges={pd_gain_ratio_ranges}")
        print(f"reset_get_up_assist_force_range={assist_force_range}N")
        print(f"manual_base_push_velocity_ranges={base_push_velocity_ranges}")
        print(f"manual_body_impulse_result={body_impulse_result}")
        print(f"manual_sustained_wrench_result={sustained_wrench_result}")
        print(f"mixed_mode_counts={mixed_mode_counts}")
        print(f"mixed_trigger_ranges={mixed_trigger_ranges}")
        print(f"final_health_metrics_env0={health_metrics}")
        print(f"observed_natural_body_impulse={observed_natural_body_impulse}")
        print(f"observed_natural_sustained_wrench={observed_natural_sustained_wrench}")
        print(
            f"action_dim={action_dim}, actor_observation_shape={actor_shape}, "
            f"critic_observation_shape={critic_shape}"
        )
        print(
            f"disc_observation_shape={disc_shape}, "
            f"disc_demo_observation_shape={disc_demo_shape}"
        )
        print(
            f"amp_discriminator_parameters={discriminator_parameter_count}, "
            f"policy_score_range=({float(disc_score.min()):.6f}, "
            f"{float(disc_score.max()):.6f}), "
            f"demo_score_range=({float(disc_demo_score.min()):.6f}, "
            f"{float(disc_demo_score.max()):.6f})"
        )
        print(
            f"amp_replay_length={disc_replay.current_length}, "
            f"single_buffer_bytes={disc_replay.required_storage_bytes}, "
            f"sample_shape={tuple(replay_disc_observations.shape)}"
        )
        print(
            f"amp_update_metrics={discriminator_update_metrics}, "
            f"output_weight_max_update={output_weight_update:.6e}, "
            f"normalization_count={normalization_count}"
        )
        print(f"action_mode={action_mode}")
        print(
            f"config_mode={config_mode}, disturbance_mode={disturbance_mode}, "
            f"reset_motion_id={reset_motion_id}, reset_reference_frame={reset_frame}"
        )
        print(f"reset_motion_counts={reset_motion_counts}")
        print(
            "reset_max_errors="
            f"(root_pos={reset_root_pos_error:.3e}, "
            f"root_quat={reset_root_quat_error:.3e}, "
            f"root_lin_vel={reset_root_lin_vel_error:.3e}, "
            f"root_ang_vel={reset_root_ang_vel_error:.3e}, "
            f"joint_pos={reset_joint_pos_error:.3e}, "
            f"joint_vel={reset_joint_vel_error:.3e})"
        )
        print(f"steps={steps}, simulated_time={steps * env.step_dt:.3f}s")
        print(
            f"terminal_amp_observations={terminal_amp_observation_count}, "
            "terminal_vs_reset_max_difference="
            f"{terminal_vs_reset_max_difference:.6e}"
        )
        print(f"reward_range=({reward_min:.6f}, {reward_max:.6f})")
        reward_terms = {
            name: values[0]
            for name, values in env.reward_manager.get_active_iterable_terms(0)
        }
        print(f"final_weighted_reward_rates={reward_terms}")
        print(f"terminations={termination_count}, timeouts={timeout_count}")
        print(f"final_root_height={root_height:.6f}m")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-mode", choices=("zero", "random"), default="zero")
    parser.add_argument("--config-mode", choices=("play", "train"), default="play")
    parser.add_argument(
        "--disturbance-mode",
        choices=("mixed", "base_push", "impulse", "sustained", "none"),
        default="mixed",
    )
    args = parser.parse_args()
    run_smoke_test(
        steps=args.steps,
        num_envs=args.num_envs,
        device=args.device,
        action_mode=args.action_mode,
        config_mode=args.config_mode,
        disturbance_mode=args.disturbance_mode,
    )


if __name__ == "__main__":
    main()
