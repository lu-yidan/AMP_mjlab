"""Frozen HoST base policy with a scene-isolated recovery residual."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.host_recovery.mdp.observations import UNACTUATED_STEPS


class TeacherResidualVecEnvWrapper(RslRlVecEnvWrapper):
  """Apply a residual only in obstacle scenes after motor activation.

  Flat-scene actions are exactly the frozen teacher output. Escaped-scene
  actions can optionally keep the residual so post-escape standing is learned.
  """

  def __init__(
    self,
    env,
    base_policy: Callable[[torch.Tensor], torch.Tensor],
    residual_scale: float,
    clip_actions: float | None,
    apply_after_escape: bool = False,
  ) -> None:
    if residual_scale <= 0.0:
      raise ValueError(residual_scale)
    self.base_policy = base_policy
    self.residual_scale = residual_scale
    self.apply_after_escape = apply_after_escape
    self._latest_actor_observations: torch.Tensor | None = None
    super().__init__(env, clip_actions=clip_actions)

  def get_observations(self):
    observations = super().get_observations()
    self._latest_actor_observations = observations["actor"]
    return observations

  def reset(self):
    observations, extras = super().reset()
    self._latest_actor_observations = observations["actor"]
    return observations, extras

  def step(self, residual_actions: torch.Tensor):
    if self._latest_actor_observations is None:
      self.get_observations()
    assert self._latest_actor_observations is not None
    with torch.inference_mode():
      base_actions = self.base_policy(self._latest_actor_observations)
      motor_on = self.unwrapped.episode_length_buf > UNACTUATED_STEPS
      gate = obstacle_residual_gate(
        self.unwrapped._a6_scene,
        self.unwrapped._a6_escaped,
        motor_on,
        apply_after_escape=self.apply_after_escape,
      ).float().unsqueeze(-1)
      gated_residual = gate * residual_actions * self.residual_scale
      combined_actions = (base_actions + gated_residual).clamp(-1.0, 1.0)
    self.unwrapped._a6_base_action = base_actions
    self.unwrapped._a6_residual_action = residual_actions
    self.unwrapped._a6_residual_gate = gate
    self.unwrapped._a6_gated_residual_action = gated_residual
    self.unwrapped._a6_combined_action = combined_actions
    result = super().step(combined_actions)
    self._latest_actor_observations = result[0]["actor"]
    return result


class SymmetricSagittalCorrectionProjection(nn.Module):
  """Restrict policy means to mirrored sagittal correction pairs."""

  def __init__(self, pair_action_ids: torch.Tensor) -> None:
    super().__init__()
    self.pair_action_ids = tuple(
      (int(left_id), int(right_id))
      for left_id, right_id in pair_action_ids.tolist()
    )

  def forward(self, actions: torch.Tensor) -> torch.Tensor:
    projected = torch.zeros_like(actions)
    for left_id, right_id in self.pair_action_ids:
      pair_mean = 0.5 * (actions[:, left_id] + actions[:, right_id])
      projected[:, left_id] = pair_mean
      projected[:, right_id] = pair_mean
    return projected


class FrozenRecoveryPostEscapeCorrectionVecEnvWrapper(RslRlVecEnvWrapper):
  """Keep a frozen recovery residual and learn a symmetric posture correction."""

  _SAGITTAL_PAIRS = (
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
  )

  def __init__(
    self,
    env,
    base_policy: Callable[[torch.Tensor], torch.Tensor],
    recovery_policy: Callable[[torch.Tensor], torch.Tensor],
    recovery_scale: float,
    correction_scale: float,
    clip_actions: float | None,
  ) -> None:
    if recovery_scale <= 0.0 or correction_scale <= 0.0:
      raise ValueError((recovery_scale, correction_scale))
    self.base_policy = base_policy
    self.recovery_policy = recovery_policy
    self.recovery_scale = recovery_scale
    self.correction_scale = correction_scale
    self._latest_actor_observations: torch.Tensor | None = None

    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    target_ids = [int(index) for index in action._target_ids]
    pair_action_ids = []
    for pair in self._SAGITTAL_PAIRS:
      joint_ids = robot.find_joints(pair, preserve_order=True)[0]
      pair_action_ids.append(
        [target_ids.index(int(joint_id)) for joint_id in joint_ids]
      )
    self._pair_action_ids = torch.tensor(
      pair_action_ids, dtype=torch.long, device=env.device
    )
    self.correction_projection = SymmetricSagittalCorrectionProjection(
      self._pair_action_ids
    )
    super().__init__(env, clip_actions=clip_actions)

  def get_observations(self):
    observations = super().get_observations()
    self._latest_actor_observations = observations["actor"]
    return observations

  def reset(self):
    observations, extras = super().reset()
    self._latest_actor_observations = observations["actor"]
    return observations, extras

  def project_correction(self, actions: torch.Tensor) -> torch.Tensor:
    """Project 29-D actions to three mirrored sagittal joint pairs."""
    return self.correction_projection(actions)

  def step(self, correction_actions: torch.Tensor):
    if self._latest_actor_observations is None:
      self.get_observations()
    assert self._latest_actor_observations is not None
    with torch.inference_mode():
      base_actions = self.base_policy(self._latest_actor_observations)
      recovery_actions = self.recovery_policy(self._latest_actor_observations)
      motor_on = self.unwrapped.episode_length_buf > UNACTUATED_STEPS
      recovery_gate = obstacle_residual_gate(
        self.unwrapped._a6_scene,
        self.unwrapped._a6_escaped,
        motor_on,
        apply_after_escape=True,
      ).float().unsqueeze(-1)
      correction_gate = post_escape_correction_gate(
        self.unwrapped._a6_scene,
        self.unwrapped._a6_escaped,
        motor_on,
      ).float().unsqueeze(-1)
      projected_correction = self.project_correction(correction_actions)
      gated_recovery = recovery_gate * recovery_actions * self.recovery_scale
      gated_correction = (
        correction_gate * projected_correction * self.correction_scale
      )
      combined_actions = (
        base_actions + gated_recovery + gated_correction
      ).clamp(-1.0, 1.0)

    self.unwrapped._a6_base_action = base_actions
    self.unwrapped._a6_recovery_residual_action = recovery_actions
    self.unwrapped._a6_recovery_residual_gate = recovery_gate
    self.unwrapped._a6_gated_recovery_residual_action = gated_recovery
    self.unwrapped._a6_residual_action = correction_actions
    self.unwrapped._a6_projected_residual_action = projected_correction
    self.unwrapped._a6_residual_gate = correction_gate
    self.unwrapped._a6_gated_residual_action = gated_correction
    self.unwrapped._a6_combined_action = combined_actions
    result = super().step(combined_actions)
    self._latest_actor_observations = result[0]["actor"]
    return result


def obstacle_residual_gate(
  scene: torch.Tensor,
  escaped: torch.Tensor,
  motor_on: torch.Tensor,
  *,
  apply_after_escape: bool,
) -> torch.Tensor:
  """Return the scene-isolated residual gate."""
  active = scene > 0
  if not apply_after_escape:
    active = active & ~escaped
  return active & motor_on


def post_escape_correction_gate(
  scene: torch.Tensor,
  escaped: torch.Tensor,
  motor_on: torch.Tensor,
) -> torch.Tensor:
  """Activate posture correction only after obstacle escape and motor-on."""
  return (scene > 0) & escaped & motor_on


def residual_effort(env) -> torch.Tensor:
  """Bounded residual magnitude penalty used by V19."""
  value = getattr(env, "_a6_gated_residual_action", None)
  if value is None:
    return torch.zeros(env.num_envs, device=env.device)
  return value.square().mean(dim=-1)
