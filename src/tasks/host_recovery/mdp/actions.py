"""HoST incremental joint-position action.

HoST commands position targets incrementally every control step::

    target = current_dof_pos + action * action_rescale

mjlab's :class:`~mjlab.envs.mdp.actions.JointPositionAction` with
``use_default_offset=True`` instead anchors the target to
``default_joint_pos``. The two conventions are not equivalent:

- Incremental: the PD error is always ``action * action_rescale``, and the
  limb excursion can accumulate over consecutive steps.
- Absolute: the target is ``default_joint_pos + action * action_rescale``, so
  the total excursion from the default pose is capped by the rescaler.

Once the ``action_scale`` curriculum decays the rescaler to 0.25, the absolute
convention leaves the robot with at most +/- 0.25 rad of authority around the
standing pose -- not enough to push itself up from the floor. This module
restores HoST's incremental convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .observations import UNACTUATED_STEPS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class IncrementalJointPositionActionCfg(JointPositionActionCfg):
  """Position action whose target is ``current + scale * action``."""

  use_default_offset: bool = False

  def build(self, env: "ManagerBasedRlEnv") -> "IncrementalJointPositionAction":
    return IncrementalJointPositionAction(self, env)


class IncrementalJointPositionAction(JointPositionAction):
  """Joint position action re-based on the measured position every step.

  Also reproduces HoST's ``unactuated_timesteps``: while the episode is still
  in the motors-off window, the target is the *current* joint position, so the
  PD spring term contributes no active force and the robot settles under
  gravity with only its passive damping.
  """

  def __init__(self, cfg: IncrementalJointPositionActionCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg=cfg, env=env)
    self._offset = torch.zeros(self.num_envs, self.action_dim, device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._offset[:] = self._entity.data.joint_pos[:, self._target_ids]
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    unactuated = (self._env.episode_length_buf <= UNACTUATED_STEPS).unsqueeze(-1)
    if torch.any(unactuated):
      current = self._entity.data.joint_pos[:, self._target_ids]
      target = torch.where(unactuated, current, target)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
