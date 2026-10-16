"""Frozen HoST base policy with a scene-isolated recovery residual."""

from __future__ import annotations

from collections.abc import Callable

import torch

from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.host_recovery.mdp.observations import UNACTUATED_STEPS


class TeacherResidualVecEnvWrapper(RslRlVecEnvWrapper):
  """Apply a residual only while an obstacle remains uncleared.

  Flat-scene and escaped-scene actions are exactly the frozen teacher output,
  so recovery learning cannot alter the base standing controller.
  """

  def __init__(
    self,
    env,
    base_policy: Callable[[torch.Tensor], torch.Tensor],
    residual_scale: float,
    clip_actions: float | None,
  ) -> None:
    if residual_scale <= 0.0:
      raise ValueError(residual_scale)
    self.base_policy = base_policy
    self.residual_scale = residual_scale
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
      gate = (
        (self.unwrapped._a6_scene > 0)
        & ~self.unwrapped._a6_escaped
        & motor_on
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


def residual_effort(env) -> torch.Tensor:
  """Bounded residual magnitude penalty used by V19."""
  value = getattr(env, "_a6_gated_residual_action", None)
  if value is None:
    return torch.zeros(env.num_envs, device=env.device)
  return value.square().mean(dim=-1)
