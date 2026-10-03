"""A6 reset-entry protection for the AMP joint-position action.

The A6 reset bank can place the robot far from its default standing pose. A
plain position action would ask the PD controller to move toward the policy
target immediately. This term starts from the bank pose, blends toward the
policy target for a short time, and projects the target into the range allowed
by the current (possibly randomized) PD gains and effort limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


def write_and_select_delayed_targets(
    buffer: torch.Tensor,
    cursor: int,
    current_target: torch.Tensor,
    lag: torch.Tensor,
) -> torch.Tensor:
    """Write one physics-substep target and select each environment's lag."""

    if buffer.ndim != 3 or current_target.shape != buffer.shape[1:]:
        raise ValueError("delay buffer and current target shapes do not match")
    if lag.shape != (buffer.shape[1],):
        raise ValueError("lag must contain one value per environment")
    if torch.any((lag < 0) | (lag >= len(buffer))):
        raise ValueError("lag is outside the delay buffer")
    buffer[cursor].copy_(current_target)
    rows = torch.arange(buffer.shape[1], device=buffer.device)
    return buffer[(cursor - lag) % len(buffer), rows]


class A6EntryProtectedJointPositionAction(JointPositionAction):
    """Position action with entry blending, projection, and A6 command lag."""

    cfg: A6EntryProtectedJointPositionActionCfg

    def __init__(self, cfg: A6EntryProtectedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if cfg.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if cfg.raw_action_clip <= 0.0:
            raise ValueError("raw_action_clip must be positive")
        if cfg.max_delay_steps < 0:
            raise ValueError("max_delay_steps must be non-negative")

        self._entry_joint_pos = self._processed_actions.clone()
        self._steps_since_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._delay_buffer = self._processed_actions.unsqueeze(0).repeat(
            cfg.max_delay_steps + 1, 1, 1
        )
        self._delay_cursor = 0

        ctrl_id_by_joint: dict[str, int] = {}
        for actuator in self._entity.actuators:
            ctrl_ids = actuator.global_ctrl_ids.tolist()
            if len(ctrl_ids) != len(actuator.target_names):
                raise ValueError(
                    "A6 entry protection requires one position control input "
                    "per target joint"
                )
            ctrl_id_by_joint.update(
                dict(zip(actuator.target_names, ctrl_ids, strict=True))
            )
        missing = [name for name in self._target_names if name not in ctrl_id_by_joint]
        if missing:
            raise ValueError(f"A6 action target joints have no control input: {missing}")
        self._ctrl_ids_for_targets = torch.tensor(
            [ctrl_id_by_joint[name] for name in self._target_names],
            dtype=torch.long,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        self._steps_since_reset[ids] = 0
        self._entry_joint_pos[ids] = self._entity.data.joint_pos[ids][
            :, self._target_ids
        ]
        self._delay_buffer[:, ids] = self._entry_joint_pos[ids].unsqueeze(0)

    def process_actions(self, actions: torch.Tensor) -> None:
        clipped_actions = actions.clamp(
            -self.cfg.raw_action_clip, self.cfg.raw_action_clip
        )
        super().process_actions(clipped_actions)

        blend = (
            (self._steps_since_reset + 1).to(dtype=torch.float32)
            / self.cfg.warmup_steps
        ).clamp(max=1.0)[:, None]
        requested_target = (
            (1.0 - blend) * self._entry_joint_pos
            + blend * self._processed_actions
        )

        q = self._entity.data.joint_pos[:, self._target_ids]
        dq = self._entity.data.joint_vel[:, self._target_ids]
        model = self._env.sim.model
        ctrl_ids = self._ctrl_ids_for_targets
        kp = model.actuator_gainprm[:, ctrl_ids, 0]
        kd = -model.actuator_biasprm[:, ctrl_ids, 2]
        effort_limit = model.actuator_forcerange[:, ctrl_ids].abs().amax(dim=-1)
        if torch.any(kp <= 0.0) or not torch.isfinite(kp).all():
            raise ValueError("A6 entry protection requires finite positive PD gains")
        if not torch.isfinite(kd).all() or not torch.isfinite(effort_limit).all():
            raise ValueError("A6 entry protection found invalid controller parameters")

        lower_target = q + (kd * dq - effort_limit) / kp
        upper_target = q + (kd * dq + effort_limit) / kp
        self._processed_actions = torch.clamp(
            requested_target, min=lower_target, max=upper_target
        )
        self._steps_since_reset += 1

    def apply_actions(self) -> None:
        """Apply the target selected by each environment's 2 ms delay."""

        lag = getattr(
            self._env,
            "_a6_command_lag",
            torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
        )
        lag = lag.clamp(0, self.cfg.max_delay_steps)
        delayed_target = write_and_select_delayed_targets(
            self._delay_buffer,
            self._delay_cursor,
            self._processed_actions,
            lag,
        )
        self._delay_cursor = (self._delay_cursor + 1) % len(self._delay_buffer)
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(
            delayed_target - encoder_bias, joint_ids=self._target_ids
        )


@dataclass(kw_only=True)
class A6EntryProtectedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for A6 reset-entry action protection."""

    warmup_steps: int = 10
    raw_action_clip: float = 10.0
    max_delay_steps: int = 5

    def build(self, env):
        return A6EntryProtectedJointPositionAction(self, env)


__all__ = [
    "A6EntryProtectedJointPositionAction",
    "A6EntryProtectedJointPositionActionCfg",
    "write_and_select_delayed_targets",
]
