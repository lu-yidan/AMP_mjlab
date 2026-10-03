"""Disturbance events implemented specifically for G1 recovery training."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.event_manager import (
    EventTermCfg,
    RecomputeLevel,
    requires_model_fields,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_MIXED_MODE_ORDER = ("sustained", "impulse", "base_push", "none")


def _default_body_field(
    env: ManagerBasedRlEnv,
    field: str,
    env_ids: torch.Tensor,
    body_ids: torch.Tensor,
) -> torch.Tensor:
    """Select a model default as an ``(environment, body, ...)`` tensor."""

    default = env.sim.get_default_field(field)
    if field in env.sim.per_world_default_fields:
        env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
        return default[env_grid, body_grid]
    selected = default[body_ids].unsqueeze(0)
    return selected.expand((len(env_ids),) + selected.shape[1:])


@requires_model_fields(
    "body_mass",
    "body_inertia",
    recompute=RecomputeLevel.set_const,
)
def randomize_body_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    mass_delta_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
    """Add uniformly sampled mass and scale inertia by the same ratio.

    Isaac Lab's mass randomizer recomputes inertia by default.  Mjlab's
    ``dr.body_mass`` deliberately changes only mass, so this recovery-specific
    event preserves the source task's uniform additive mass distribution while
    keeping the body's density-style mass/inertia relationship consistent.
    """

    robot: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)

    body_ids = robot.indexing.body_ids[asset_cfg.body_ids].to(
        env.device, dtype=torch.long
    )
    default_mass = _default_body_field(env, "body_mass", env_ids, body_ids)
    default_inertia = _default_body_field(env, "body_inertia", env_ids, body_ids)

    delta = torch.empty_like(default_mass).uniform_(*mass_delta_range)
    new_mass = default_mass + delta
    if torch.any(new_mass <= 0.0):
        raise ValueError(
            "mass_delta_range produced a non-positive body mass; narrow the range"
        )
    inertia_scale = (new_mass / default_mass).unsqueeze(-1)

    env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    env.sim.model.body_mass[env_grid, body_grid] = new_mass
    env.sim.model.body_inertia[env_grid, body_grid] = (
        default_inertia * inertia_scale
    )


def _local_wrench_to_world(
    robot: Entity,
    forces_b: torch.Tensor,
    torques_b: torch.Tensor,
    body_ids: list[int] | slice,
    env_ids: torch.Tensor | slice,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate body-frame wrenches into MuJoCo's required world frame.

    Isaac Lab's ``set_external_force_and_torque`` uses each body's local frame
    unless ``is_global=True`` is requested.  MuJoCo's ``xfrc_applied`` instead
    expects world-frame values, so preserving the source behavior requires this
    conversion immediately before writing the wrench to the simulator.
    """

    body_quat_w = robot.data.body_link_quat_w[env_ids][:, body_ids]
    return quat_apply(body_quat_w, forces_b), quat_apply(body_quat_w, torques_b)


def _sample_mass_scaled_body_force(
    env: ManagerBasedRlEnv,
    global_body_ids: torch.Tensor,
    env_ids: torch.Tensor,
    acceleration_range: tuple[float, float],
) -> torch.Tensor:
    """Sample body-frame accelerations and convert them to forces with ``F=m*a``."""

    shape = (len(env_ids), len(global_body_ids), 3)
    acceleration = torch.empty(shape, device=env.device).uniform_(
        *acceleration_range
    )
    body_mass = env.sim.model.body_mass
    if body_mass.ndim == 1:
        selected_mass = body_mass[global_body_ids].unsqueeze(0).expand(len(env_ids), -1)
    else:
        env_grid, body_grid = torch.meshgrid(env_ids, global_body_ids, indexing="ij")
        selected_mass = body_mass[env_grid, body_grid]
    return acceleration * selected_mass.unsqueeze(-1)


def apply_get_up_assist_force(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Apply each environment's stored upward assistance to the torso."""

    from g1recovery_amp.tasks.recovery.mdp.commands import GetUpAssistForceCommand

    robot: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    force_command = cast(
        GetUpAssistForceCommand,
        env.command_manager.get_term("get_up_assist_force"),
    )
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else robot.num_bodies
    )
    forces = torch.zeros((len(env_ids), num_bodies, 3), device=env.device)
    torques = torch.zeros_like(forces)
    forces[:, 0, 2] = force_command.command[env_ids, 0]
    robot.write_external_wrench_to_sim(
        forces,
        torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


class MixedRecoveryDisturbance:
    """Assign one source-compatible disturbance mode to each environment."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params["asset_cfg"]
        modes = cfg.params["modes"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("mixed_disturbance requires a SceneEntityCfg")
        if not isinstance(modes, dict):
            raise TypeError("mixed_disturbance modes must be a dictionary")

        self._robot: Entity = env.scene[asset_cfg.name]
        self._env = env
        self._body_ids = asset_cfg.body_ids
        self._global_body_ids = self._robot.indexing.body_ids[self._body_ids].to(
            env.device, dtype=torch.long
        )
        self._num_bodies = (
            len(self._body_ids)
            if isinstance(self._body_ids, list)
            else self._robot.num_bodies
        )
        self._num_envs = env.num_envs
        self._device = env.device
        self._step_dt = env.step_dt
        self._modes = cast(dict[str, dict[str, object]], modes)
        self.mode_names = tuple(
            name
            for name in _MIXED_MODE_ORDER
            if self._enabled(name) and self._ratio(name) > 0.0
        )
        if not self.mode_names:
            self.mode_names = ("none",)

        ratios = torch.tensor(
            [self._ratio(name) for name in self.mode_names],
            device=self._device,
        )
        ratios /= ratios.sum()
        bounds = (torch.cumsum(ratios, dim=0) * self._num_envs).long()
        self.env_mode = torch.empty(
            self._num_envs, dtype=torch.long, device=self._device
        )
        start = 0
        for mode_index in range(len(self.mode_names)):
            end = (
                self._num_envs
                if mode_index == len(self.mode_names) - 1
                else int(bounds[mode_index])
            )
            self.env_mode[start:end] = mode_index
            start = end

        self.time_left = torch.zeros(self._num_envs, device=self._device)
        self.impulse_left = torch.zeros(self._num_envs, device=self._device)
        self.force = torch.zeros(
            self._num_envs, self._num_bodies, 3, device=self._device
        )
        self.force_start = torch.zeros_like(self.force)
        self.torque = torch.zeros_like(self.force)
        self.ramp_elapsed = torch.zeros(self._num_envs, device=self._device)
        self.trigger_counts = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )
        self._resample_time(torch.arange(self._num_envs, device=self._device))

    def _mode_cfg(self, name: str) -> dict[str, object]:
        return self._modes.get(name, {})

    def _enabled(self, name: str) -> bool:
        return bool(self._mode_cfg(name).get("enable", True))

    def _scalar(self, name: str, key: str, default: float) -> float:
        value = self._mode_cfg(name).get(key, default)
        if not isinstance(value, int | float):
            raise TypeError(f"mixed mode {name!r} {key!r} must be numeric")
        return float(value)

    def _ratio(self, name: str) -> float:
        return self._scalar(name, "ratio", 0.0)

    def _range(
        self,
        name: str,
        key: str,
        default: tuple[float, float],
    ) -> tuple[float, float]:
        value = self._mode_cfg(name).get(key, default)
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"mixed mode {name!r} {key!r} must be a two-value tuple")
        return float(value[0]), float(value[1])

    def _period(self, name: str) -> tuple[float, float]:
        return self._range(name, "period_s", (1.0, 3.0))

    def mode_counts(self) -> dict[str, int]:
        """Return the current deterministic environment partition sizes."""

        return {
            name: int((self.env_mode == index).sum())
            for index, name in enumerate(self.mode_names)
        }

    def _resample_time(self, env_ids: torch.Tensor) -> None:
        for mode_index, name in enumerate(self.mode_names):
            selected = env_ids[self.env_mode[env_ids] == mode_index]
            if len(selected) == 0:
                continue
            if name == "none":
                self.time_left[selected] = torch.inf
                continue
            lower, upper = self._period(name)
            self.time_left[selected] = torch.empty(
                len(selected), device=self._device
            ).uniform_(lower, upper)

    def _sample_wrench(
        self,
        count: int,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (count, self._num_bodies, 3)
        force = torch.empty(shape, device=self._device).uniform_(*force_range)
        torque = torch.empty(shape, device=self._device).uniform_(*torque_range)
        return force, torque

    def _sustained_force_at(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Interpolate from the previous force to the new mass-scaled target."""

        ramp_s = self._scalar("sustained", "ramp_s", 0.1)
        if ramp_s <= 0.0:
            return self.force[env_ids]
        fraction = (self.ramp_elapsed[env_ids] / ramp_s).clamp_(0.0, 1.0)
        return self.force_start[env_ids] + fraction[:, None, None] * (
            self.force[env_ids] - self.force_start[env_ids]
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        modes: dict[str, dict[str, object]],
    ) -> None:
        """Advance all timers and apply each environment's assigned disturbance."""

        del env_ids, asset_cfg, modes
        all_env_ids = torch.arange(self._num_envs, device=self._device)
        self.time_left -= self._step_dt
        self.impulse_left -= self._step_dt
        self.ramp_elapsed += self._step_dt

        if "impulse" in self.mode_names:
            impulse_index = self.mode_names.index("impulse")
            expired = (self.impulse_left <= 0.0) & (self.env_mode == impulse_index)
            self.force[expired] = 0.0
            self.torque[expired] = 0.0

        firing = self.time_left <= 0.0
        for mode_index, name in enumerate(self.mode_names):
            selected = all_env_ids[firing & (self.env_mode == mode_index)]
            if len(selected) == 0:
                continue
            count = len(selected)
            if name == "sustained":
                self.force_start[selected] = self._sustained_force_at(selected)
                self.force[selected] = _sample_mass_scaled_body_force(
                    self._env,
                    self._global_body_ids,
                    selected,
                    self._range(
                        name, "linear_acceleration", (-10.5, 10.5)
                    ),
                )
                self.torque[selected] = 0.0
                self.ramp_elapsed[selected] = 0.0
            elif name == "impulse":
                self.force[selected], self.torque[selected] = self._sample_wrench(
                    count,
                    self._range(name, "force", (-90.0, 90.0)),
                    self._range(name, "torque", (-18.0, 18.0)),
                )
                duration = self._scalar(name, "duration_s", 0.1)
                self.impulse_left[selected] = duration
            elif name == "base_push":
                pushed_velocity = sample_base_push_velocity(
                    self._robot.data.root_link_vel_w[selected],
                    self._range(name, "vel", (-0.8, 0.8)),
                    self._range(name, "ang_vel", (-1.0, 1.0)),
                )
                self._robot.write_root_link_velocity_to_sim(
                    pushed_velocity, env_ids=selected
                )
            self.trigger_counts[selected] += 1
            self._resample_time(selected)

        applied_force = self.force.clone()
        if "sustained" in self.mode_names:
            sustained_index = self.mode_names.index("sustained")
            sustained_ids = all_env_ids[self.env_mode == sustained_index]
            applied_force[sustained_ids] = self._sustained_force_at(sustained_ids)

        force_w, torque_w = _local_wrench_to_world(
            self._robot,
            applied_force,
            self.torque,
            self._body_ids,
            all_env_ids,
        )
        self._robot.write_external_wrench_to_sim(
            force_w,
            torque_w,
            body_ids=self._body_ids,
            env_ids=all_env_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear forces, transient timers, and verification counters on reset."""

        if env_ids is None or isinstance(env_ids, slice):
            selected = torch.arange(self._num_envs, device=self._device)
        else:
            selected = env_ids
        self.force[selected] = 0.0
        self.force_start[selected] = 0.0
        self.torque[selected] = 0.0
        self.ramp_elapsed[selected] = 0.0
        self.impulse_left[selected] = 0.0
        self.trigger_counts[selected] = 0
        self._robot.write_external_wrench_to_sim(
            self.force[selected],
            self.torque[selected],
            body_ids=self._body_ids,
            env_ids=selected,
        )
        self._resample_time(selected)


class SustainedBodyWrench:
    """Hold mass-scaled body forces and smoothly resample them every few seconds."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("sustained_body_wrench requires a SceneEntityCfg")
        self._robot: Entity = env.scene[asset_cfg.name]
        self._env = env
        self._body_ids = asset_cfg.body_ids
        self._global_body_ids = self._robot.indexing.body_ids[self._body_ids].to(
            env.device, dtype=torch.long
        )
        self._num_bodies = (
            len(self._body_ids)
            if isinstance(self._body_ids, list)
            else self._robot.num_bodies
        )
        self._num_envs = env.num_envs
        self._device = env.device
        self._step_dt = env.step_dt
        self._period_s: tuple[float, float] = cfg.params["period_s"]
        self._time_left = self._sample_period(self._num_envs)
        self.force = torch.zeros(
            self._num_envs, self._num_bodies, 3, device=self._device
        )
        self.force_start = torch.zeros_like(self.force)
        self.ramp_elapsed = torch.zeros(self._num_envs, device=self._device)

    def _sample_period(self, count: int) -> torch.Tensor:
        lower, upper = self._period_s
        return torch.empty(count, device=self._device).uniform_(lower, upper)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        linear_acceleration_range: tuple[float, float],
        ramp_s: float,
        period_s: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Advance timers and smoothly approach each new mass-scaled force."""

        del env, env_ids, period_s, asset_cfg
        self._time_left -= self._step_dt
        self.ramp_elapsed += self._step_dt
        firing_ids = (self._time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(firing_ids) > 0:
            current_force = self._interpolated_force(firing_ids, ramp_s)
            self.force_start[firing_ids] = current_force
            self.force[firing_ids] = _sample_mass_scaled_body_force(
                self._env,
                self._global_body_ids,
                firing_ids,
                linear_acceleration_range,
            )
            self.ramp_elapsed[firing_ids] = 0.0
            self._time_left[firing_ids] = self._sample_period(len(firing_ids))

        all_env_ids = torch.arange(self._num_envs, device=self._device)
        forces = self._interpolated_force(all_env_ids, ramp_s)
        torques = torch.zeros_like(forces)
        force_w, torque_w = _local_wrench_to_world(
            self._robot,
            forces,
            torques,
            self._body_ids,
            all_env_ids,
        )
        self._robot.write_external_wrench_to_sim(
            force_w,
            torque_w,
            body_ids=self._body_ids,
            env_ids=all_env_ids,
        )

    def _interpolated_force(
        self, env_ids: torch.Tensor, ramp_s: float
    ) -> torch.Tensor:
        if ramp_s <= 0.0:
            return self.force[env_ids]
        fraction = (self.ramp_elapsed[env_ids] / ramp_s).clamp_(0.0, 1.0)
        return self.force_start[env_ids] + fraction[:, None, None] * (
            self.force[env_ids] - self.force_start[env_ids]
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear prior-episode wrenches and restart selected environment timers."""

        if env_ids is None:
            env_ids = slice(None)
        count = self._num_envs if isinstance(env_ids, slice) else len(env_ids)
        zeros = torch.zeros((count, self._num_bodies, 3), device=self._device)
        self.force[env_ids] = 0.0
        self.force_start[env_ids] = 0.0
        self.ramp_elapsed[env_ids] = 0.0
        self._robot.write_external_wrench_to_sim(
            zeros,
            zeros,
            body_ids=self._body_ids,
            env_ids=env_ids,
        )
        self._time_left[env_ids] = self._sample_period(count)


def sample_base_push_velocity(
    root_velocity_w: torch.Tensor,
    linear_velocity_range: tuple[float, float],
    yaw_velocity_range: tuple[float, float],
) -> torch.Tensor:
    """Replace world-frame x/y and yaw velocity with uniform push samples.

    The source recovery task leaves vertical, roll, and pitch velocity unchanged.
    Keeping this tensor operation separate makes that behavioral contract easy to
    test without constructing a simulator.
    """

    pushed_velocity = root_velocity_w.clone()
    pushed_velocity[:, 0:2].uniform_(*linear_velocity_range)
    pushed_velocity[:, 5].uniform_(*yaw_velocity_range)
    return pushed_velocity


def base_velocity_push(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    linear_velocity_range: tuple[float, float],
    yaw_velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
    """Give selected robots the source task's instantaneous base-velocity push.

    This is a deliberately cheap disturbance rather than a physical contact force:
    it overwrites world-frame x/y linear velocity and yaw angular velocity.  The
    other three root-velocity components keep their pre-event values.
    """

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    robot: Entity = env.scene[asset_cfg.name]
    pushed_velocity = sample_base_push_velocity(
        robot.data.root_link_vel_w[env_ids],
        linear_velocity_range,
        yaw_velocity_range,
    )
    robot.write_root_link_velocity_to_sim(pushed_velocity, env_ids=env_ids)
