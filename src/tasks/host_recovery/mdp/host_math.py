"""HoST adapter layer: constants and helpers shared by the ported MDP terms.

HoST keeps these in two places:

- ``legged_gym/envs/g1/g1_utils.py`` -- the :func:`tolerance` / :func:`sigmoid`
  kernels that almost every reward term calls.
- ``legged_gym/envs/base/host_ground.py`` -- ``HOST_CONSTRAINT_DT``,
  ``reward_group_weights`` and the small tensor accessors.

Concentrating them in one module means the rest of this package never has to
know how HoST spelled things, and a reviewer can diff HoST's own config against
``config/g1/env_cfgs.py`` term by term without chasing magic numbers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "HOST_CONSTRAINT_DT",
  "REGU_GROUP_WEIGHT",
  "STYLE_GROUP_WEIGHT",
  "TARGET_GROUP_WEIGHT",
  "TASK_GROUP_WEIGHT",
  "tolerance",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: HoST's control dt (``control.decimation`` = 4 x ``sim.dt`` = 0.005 s).
HOST_CONSTRAINT_DT = 0.02

#: Reward-group weights from HoST's ``G1Cfg.rewards.reward_group_weights``.
TASK_GROUP_WEIGHT = 2.5
REGU_GROUP_WEIGHT = 0.1
STYLE_GROUP_WEIGHT = 1.0
TARGET_GROUP_WEIGHT = 1.0


##
# Helpers ported from HoST (``legged_gym/envs/g1/g1_utils.py``).
##


def _sigmoid(x: torch.Tensor, value_at_1: float) -> torch.Tensor:
  """HoST's ``sigmoid``: a Gaussian bump that is ``value_at_1`` at ``x == 0``."""
  scale = math.sqrt(-2.0 * math.log(value_at_1))
  return torch.exp(-0.5 * torch.square(x * scale))


def tolerance(
  x: torch.Tensor,
  bounds: tuple[float, float] | list[float] = (0.0, 0.0),
  margin: float = 0.0,
  value_at_margin: float = 0.1,
) -> torch.Tensor:
  """HoST's ``tolerance`` kernel: 1.0 inside ``bounds``, decaying outside.

  NOTE: HoST evaluates the falloff in float64; this port stays in float32 so
  the result can be mixed with the rest of the reward terms.
  """
  lower, upper = bounds
  assert lower < upper
  assert margin >= 0
  in_bounds = torch.logical_and(lower <= x, x <= upper)
  if margin == 0:
    return torch.where(in_bounds, 1.0, 0.0)
  outside = torch.where(x < lower, lower - x, x - upper)
  return torch.where(
    in_bounds, torch.ones_like(outside), _sigmoid(outside / margin, value_at_margin)
  )


def _joint_pos(asset: Entity, joint_ids: list[int] | torch.Tensor) -> torch.Tensor:
  return asset.data.joint_pos[:, joint_ids]


def _root_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Base height above the world origin, HoST's ``root_states[:, 2]``."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2]
