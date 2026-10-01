"""HoST observation terms.

HoST builds the actor input as a single vector per control step and then stacks
``num_actor_history`` (= 6) copies of it, giving 6 x 76 = 456 inputs:

======  ======  =============================================================
size    scale   content
======  ======  =============================================================
3       0.25    base angular velocity (body frame)
3       1.0     projected gravity
23      1.0     joint positions (absolute, HoST convention)
23      0.05    joint velocities
23      1.0     current action (HoST's ``self.actions`` at observation time)
1       1.0     action rescaler (the training-wheel state from the curriculum)
======  ======  =============================================================

The port keeps HoST's "one tensor" design on purpose: HoST applies a
per-channel uniform noise vector (see ``host_noise_vector``) and a global
"motors are off" mask to the whole vector, neither of which maps onto mjlab's
per-term noise/scale mechanism. The actor group in
``host_recovery_env_cfg.py`` therefore declares a single term and sets
``enable_corruption=False`` so mjlab does not add its own noise on top.

NOTE: HoST feeds raw absolute joint positions (``dof_pos``). This port follows
that convention with ``asset.data.joint_pos`` rather than mjlab's
default-relative ``joint_pos_rel``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: HoST ``normalization.obs_scales``.
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05

#: HoST ``noise.noise_scales`` (with ``noise_level`` = 1.0).
NOISE_ANG_VEL = 0.2
NOISE_GRAVITY = 0.05
NOISE_DOF_POS = 0.01
NOISE_DOF_VEL = 1.5

#: HoST ``env.num_one_step_observations`` for the 23-DoF G1.
NUM_ONE_STEP_OBS = 76

#: HoST increments ``real_episode_length_buf`` once per policy step and computes
#: ``30 * 0.02 / control_dt``.  At a 0.02 s control period this remains 30
#: policy steps (0.6 s); using the 0.005 s physics step here incorrectly made
#: the robot passive for 120 policy steps.
UNACTUATED_STEPS = 30


def _get_action_rescale(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-env action rescaler, HoST's ``self.action_rescale``.

  Falls back to the action term's configured scale (a scalar or a per-joint
  dict) when the curriculum has not attached a per-env tensor yet.
  """
  rescale = getattr(env, "_host_action_rescale", None)
  if rescale is not None:
    return rescale
  term = env.action_manager.get_term("joint_pos")
  scale = term.cfg.scale
  if isinstance(scale, dict):
    value = float(sum(scale.values()) / len(scale))
  else:
    value = float(scale)
  return torch.full((env.num_envs, 1), value, device=env.device)


def host_observation(
  env: ManagerBasedRlEnv,
  add_noise: bool = True,
  unactuated_steps: int = UNACTUATED_STEPS,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """HoST's full single-step observation vector, 76 dims on the 23-DoF G1.

  Reproduces ``LeggedRobot.compute_observations`` from HoST, including

  - the per-channel uniform noise of ``_get_noise_scale_vec`` (noise is drawn
    from ``[-1, 1]`` and multiplied by the scale vector, as in HoST), and
  - the mask ``current_obs *= real_episode_length_buf > unactuated_time`` that
    zeroes the whole observation while the motors are switched off.
  """
  asset: Entity = env.scene[asset_cfg.name]

  joint_pos = asset.data.joint_pos * DOF_POS_SCALE
  obs = torch.cat(
    (
      asset.data.root_link_ang_vel_b * ANG_VEL_SCALE,
      asset.data.projected_gravity_b,
      joint_pos,
      asset.data.joint_vel * DOF_VEL_SCALE,
      env.action_manager.get_term("joint_pos").raw_action,
      _get_action_rescale(env),
    ),
    dim=-1,
  )

  if add_noise:
    obs = obs + (2.0 * torch.rand_like(obs) - 1.0) * host_noise_vector(
      env, asset_cfg
    )

  actuated = (env.episode_length_buf > unactuated_steps).unsqueeze(-1)
  return obs * actuated


def host_noise_vector(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """HoST's per-channel noise scales, ``_get_noise_scale_vec``.

  ``[ang_vel, gravity, dof_pos, dof_vel, actions, action_rescale]``; HoST leaves
  the action and action-rescaler channels noise-free.
  """
  asset: Entity = env.scene[asset_cfg.name]
  num_joints = asset.data.joint_pos.shape[-1]
  num_actions = env.action_manager.get_term("joint_pos").raw_action.shape[-1]
  vector = torch.zeros(7 + 2 * num_joints + num_actions, device=env.device)
  vector[0:3] = NOISE_ANG_VEL * ANG_VEL_SCALE
  vector[3:6] = NOISE_GRAVITY
  start = 6
  vector[start : start + num_joints] = NOISE_DOF_POS * DOF_POS_SCALE
  vector[start + num_joints : start + 2 * num_joints] = NOISE_DOF_VEL * DOF_VEL_SCALE
  return vector
