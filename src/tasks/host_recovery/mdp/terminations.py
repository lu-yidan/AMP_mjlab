"""HoST termination terms.

HoST terminates an episode only on

- the time limit (``time_out``, mjlab's built-in), and
- the two velocity guards in ``check_termination``: ``dof_vel_limit`` (300) and
  ``base_vel_limit`` (20), both very loose and both ignored during the
  unactuated window at the start of the episode.

HoST's ground task sets ``terminate_after_contacts_on = []``, i.e. it does
not terminate on contacts. That is essential: a robot standing up from the
ground must be allowed to drag its torso, arms and shins on the floor. The port
keeps that property -- there is no ``illegal_contact`` term here on purpose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import UNACTUATED_STEPS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: HoST ``curriculum.dof_vel_limit`` / ``base_vel_limit``.
DOF_VEL_LIMIT = 300.0
BASE_VEL_LIMIT = 20.0

def dof_velocity_out_of_bounds(
  env: ManagerBasedRlEnv,
  limit: float = DOF_VEL_LIMIT,
  unactuated_steps: int = UNACTUATED_STEPS,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate on joint velocities above ``limit`` (HoST ``dof_vel_out``)."""
  asset: Entity = env.scene[asset_cfg.name]
  max_velocity = torch.abs(asset.data.joint_vel).max(dim=1).values
  return (max_velocity > limit) & (env.episode_length_buf > unactuated_steps)


def base_velocity_out_of_bounds(
  env: ManagerBasedRlEnv,
  limit: float = BASE_VEL_LIMIT,
  unactuated_steps: int = UNACTUATED_STEPS,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate on base linear velocity above ``limit`` (HoST ``base_vel_out``).

  NOTE: HoST tests ``torch.norm(base_lin_vel[:, :3])``, i.e. the world frame
  linear velocity, so ``root_link_lin_vel_w`` is used here rather than the body
  frame one.
  """
  asset: Entity = env.scene[asset_cfg.name]
  velocity = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
  return (velocity > limit) & (env.episode_length_buf > unactuated_steps)
