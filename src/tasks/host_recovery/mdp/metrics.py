"""HoST metrics and the shared stand-up-progress state.

HoST drives both training wheels from ``self.old_headheight``: the head height
above the feet, tracked per environment and reset at every episode start
(``reset_idx`` sets ``old_headheight`` and ``max_headheight`` back to zero).

mjlab has no per-environment scratch space, so the running maximum lives in a
small singleton (:class:`HoSTMetrics`) that mirrors the trick already used by
``AMP_mjlab``'s ``MotionResetManager``. The metric term below keeps it up to
date every step; :func:`reset_host_metrics` clears it on episode reset; and the
curriculum terms in :mod:`..mdp.curriculums` read it when deciding whether to
take away a training wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class HoSTMetrics:
  """Per-environment stand-up progress, mirroring HoST's buffers.

  ``max_head_height`` is the running maximum inside an episode and is cleared
  on reset. ``current_head_height`` is HoST's ``old_headheight`` (the latest
  step height), and ``last_episode_head_height`` keeps a copy of the final
  height so curriculum terms can still see it after the reset event runs.
  """

  _instance: "HoSTMetrics | None" = None

  def __init__(self) -> None:
    self.max_head_height: torch.Tensor | None = None
    self.current_head_height: torch.Tensor | None = None
    self.last_episode_head_height: torch.Tensor | None = None
    self.initialized = False

  @classmethod
  def get(cls) -> "HoSTMetrics":
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def init(self, num_envs: int, device: torch.device | str) -> None:
    if self.initialized:
      return
    self.max_head_height = torch.zeros(num_envs, device=device)
    self.current_head_height = torch.zeros(num_envs, device=device)
    self.last_episode_head_height = torch.zeros(num_envs, device=device)
    self.initialized = True

  def update(self, head_height: torch.Tensor) -> None:
    self.current_head_height = head_height
    assert self.max_head_height is not None
    self.max_head_height = torch.maximum(self.max_head_height, head_height)

  def reset(self, env_ids: torch.Tensor) -> None:
    if self.max_head_height is None:
      return
    if self.current_head_height is not None:
      assert self.last_episode_head_height is not None
      self.last_episode_head_height[env_ids] = self.current_head_height[env_ids]
    self.max_head_height[env_ids] = 0.0
    if self.current_head_height is not None:
      self.current_head_height[env_ids] = 0.0


def head_height(
  env: ManagerBasedRlEnv,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=()),
) -> torch.Tensor:
  """Head (torso) height above the feet, HoST's ``head_height``.

  NOTE: as in :func:`..mdp.rewards.standup`, mjlab's G1 has no separate head
  body, so ``body_cfg`` points at ``torso_link``.
  """
  asset: Entity = env.scene[body_cfg.name]
  body_z = asset.data.body_link_pos_w[:, body_cfg.body_ids[0], 2]
  feet_z = asset.data.site_pos_w[:, foot_cfg.site_ids, 2].mean(dim=1)
  return body_z - feet_z


def host_standup_progress(
  env: ManagerBasedRlEnv,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=()),
) -> torch.Tensor:
  """Metric term: log the current head height and update the running maximum."""
  state = HoSTMetrics.get()
  state.init(env.num_envs, env.device)
  current = head_height(env, body_cfg, foot_cfg)
  state.update(current)
  return current


def reset_host_metrics(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  """Reset event: snapshot episode-final progress, then clear the buffers."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  HoSTMetrics.get().reset(env_ids)