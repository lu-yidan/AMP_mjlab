"""Frozen HoST base policy with a scene-isolated recovery residual."""

from __future__ import annotations

from collections.abc import Callable

import torch

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


def residual_effort(env) -> torch.Tensor:
  """Bounded residual magnitude penalty used by V19."""
  value = getattr(env, "_a6_gated_residual_action", None)
  if value is None:
    return torch.zeros(env.num_envs, device=env.device)
  return value.square().mean(dim=-1)
