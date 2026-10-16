"""Preserve a frozen HoST actor's physical joint increments across scales."""

from __future__ import annotations

from collections.abc import Callable

import torch


def host_history_indices(
  width: int,
  num_joints: int,
  num_actions: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return flattened action-history and action-scale indices."""
  single_step_dim = 7 + 2 * num_joints + num_actions
  if width % single_step_dim:
    raise ValueError((width, single_step_dim))
  action_start = 6 + 2 * num_joints
  action_indices = torch.cat(
    [
      torch.arange(
        offset + action_start,
        offset + action_start + num_actions,
        device=device,
      )
      for offset in range(0, width, single_step_dim)
    ]
  )
  scale_indices = torch.arange(
    single_step_dim - 1,
    width,
    single_step_dim,
    device=device,
  )
  return action_indices, scale_indices


def teacher_observations(
  observations: torch.Tensor,
  source_scale: float,
  target_scale: float,
  num_joints: int,
  num_actions: int,
) -> torch.Tensor:
  """Map target-scale history to the frozen teacher's source-scale contract."""
  if source_scale <= 0.0 or target_scale <= 0.0:
    raise ValueError((source_scale, target_scale))
  mapped = observations.clone()
  action_indices, scale_indices = host_history_indices(
    mapped.shape[-1], num_joints, num_actions, mapped.device
  )
  mapped[..., action_indices] *= target_scale / source_scale
  mapped[..., scale_indices] = source_scale
  return mapped


class ScaleCompensatedPolicy:
  """Keep ``source_scale * teacher_action`` invariant at ``target_scale``."""

  def __init__(
    self,
    teacher: Callable[[torch.Tensor], torch.Tensor],
    source_scale: float,
    target_scale: float,
    num_joints: int,
    num_actions: int,
  ) -> None:
    if source_scale <= 0.0 or target_scale <= 0.0:
      raise ValueError((source_scale, target_scale))
    self.teacher = teacher
    self.source_scale = source_scale
    self.target_scale = target_scale
    self.num_joints = num_joints
    self.num_actions = num_actions
    self.last_unclipped_actions: torch.Tensor | None = None

  def __call__(self, observations: torch.Tensor) -> torch.Tensor:
    mapped = teacher_observations(
      observations,
      self.source_scale,
      self.target_scale,
      self.num_joints,
      self.num_actions,
    )
    teacher_actions = self.teacher(mapped)
    uncompensated = teacher_actions * (self.source_scale / self.target_scale)
    self.last_unclipped_actions = uncompensated
    return uncompensated.clamp(-1.0, 1.0)

  def saturation_fraction(self) -> float:
    if self.last_unclipped_actions is None:
      return 0.0
    return float((self.last_unclipped_actions.abs() > 1.0).float().mean())
