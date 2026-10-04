"""HoST reward terms, ported to mjlab.

Ported from ``legged_gym/legged_gym/envs/base/host_ground.py`` (OpenRobotLab/HoST,
RSS 2025 -- "Learning Humanoid Standing-up Control across Diverse Postures").

Reward groups
-------------
HoST weights four reward groups separately:

======  ======  ==============================================================
group   weight  content
======  ======  ==============================================================
task    2.5     multiplicative stand-up progress, see :func:`standup`
regu    0.1     regularisation: acceleration, torque, power, joint limits
style   1.0     motion style while the robot is still on the ground
target  1.0     post-stand-up pose / velocity convergence
======  ======  ==============================================================

mjlab's ``RewardManager`` sums a scalar, so the group weights are folded into
the individual term weights. For the three additive groups that is exactly
equivalent to HoST::

    sum_g w_g * sum_{t in g} term_t  ==  sum_t (w_g * term_t)

The multiplicative ``task`` group cannot be decomposed that way, so
:func:`standup` computes the whole product itself and already applies the 2.5
group weight.

NOTE: HoST multiplies every constraint scale (regu / style / target) by the
control dt (0.02 s) in ``_prepare_reward_function`` while leaving the ``task``
scales unscaled. :data:`HOST_CONSTRAINT_DT` reproduces that factor so the
numbers in ``config/g1/env_cfgs.py`` can stay identical to HoST's own config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .host_math import (
  HOST_CONSTRAINT_DT,
  TASK_GROUP_WEIGHT,
  _DEFAULT_ASSET_CFG,
  _joint_pos,
  _root_height,
  tolerance,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


##
# task group -- multiplicative stand-up reward (HoST group weight 2.5).
##


def standup(
  env: ManagerBasedRlEnv,
  group_weight: float = TASK_GROUP_WEIGHT,
  orientation_threshold: float = 0.99,
  target_head_height: float = 1.0,
  target_head_margin: float = 1.0,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=()),
) -> torch.Tensor:
  """HoST's multiplicative stand-up reward: ``task_orientation * head_height``.

  Following "Learning to Get Up", the ``task`` group is a product rather
  than a sum, so every condition has to hold at the same time; if any factor is
  zero the whole stand-up reward collapses to zero.

  - ``task_orientation``: ``tolerance(-gz, (0.99, inf), 1.0, 0.05)`` -- the
    base must be upright.
  - ``task_head_height``: ``tolerance(h - feet_z, (1.0, inf), 1.0, 0.1)`` --
    head above the feet, i.e. actual stand-up progress.

  NOTE: HoST reads the head from a dedicated ``keyframe_head`` link that only
  exists in its own URDF. mjlab's G1 MJCF has no head body (the head mesh is
  welded to ``torso_link``), so ``body_cfg`` should point at ``torso_link``.
  That measures torso-above-feet instead of head-above-feet; because the torso
  sits roughly 0.2 m lower than HoST's head keyframe, ``target_head_height``
  may need to be lowered for the reward to become reachable.
  """
  asset: Entity = env.scene[body_cfg.name]

  orientation = tolerance(
    -asset.data.projected_gravity_b[:, 2],
    [orientation_threshold, float("inf")],
    1.0,
    0.05,
  )

  torso_z = asset.data.body_link_pos_w[:, body_cfg.body_ids[0], 2]
  feet_z = asset.data.site_pos_w[:, foot_cfg.site_ids, 2].mean(dim=1)
  head_height = torso_z - feet_z

  height = tolerance(
    head_height,
    [target_head_height, float("inf")],
    target_head_margin,
    0.1,
  )
  return orientation * height * group_weight


##
# regu group -- regularisation (HoST group weight 0.1).
#
# HoST's ``regu_dof_acc``, ``regu_action_rate`` and ``regu_dof_pos_limits`` map
# 1:1 onto mjlab's built-in ``joint_acc_l2`` / ``action_rate_l2`` /
# ``joint_pos_limits``, which are reused directly in the env config rather than
# duplicated here.
##


def regu_dof_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise joint velocities (HoST ``regu_dof_vel``)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def regu_upper_dof_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  """Penalise velocities of the selected upper-body joints (anti-tremor)."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  if len(ids) == 0:
    return torch.zeros_like(asset.data.joint_vel[:, 0])
  return torch.sum(torch.square(asset.data.joint_vel[:, ids]), dim=1)


##
# style group -- motion style on the ground (HoST group weight 1.0).
#
# Every term is a penalty indicator (1.0 when the joint is outside the range
# HoST considers natural), which is why their scales in HoST are negative.
##


def style_waist_deviation(
  env: ManagerBasedRlEnv,
  limit: float = 1.4,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  """Penalise large waist yaw (``style_waist_deviation``).

  NOTE: HoST reads ``waist_joint_indices`` = ``['waist_yaw_joint']`` and the
  term is therefore a no-op on the roll/pitch waist joints as well.
  """
  asset: Entity = env.scene[asset_cfg.name]
  waist = _joint_pos(asset, asset_cfg.joint_ids)
  return (torch.abs(waist) > limit).float().squeeze(1)


def style_hip_yaw_deviation(
  env: ManagerBasedRlEnv,
  upper_limit: float = 1.4,
  lower_limit: float = 0.9,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  """Penalise hip yaw outside ``[-upper_limit, upper_limit]`` (HoST ``style_hip_yaw_deviation``).

  NOTE: reproduced verbatim, including HoST's slightly odd combination of
  ``max(|q|) > upper`` and ``min(|q|) > lower`` rather than a plain range check.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = _joint_pos(asset, asset_cfg.joint_ids)
  return (torch.max(torch.abs(q), dim=-1)[0] > upper_limit) | (
    torch.min(torch.abs(q), dim=-1)[0] > lower_limit
  )


def style_hip_roll_deviation(
  env: ManagerBasedRlEnv,
  upper_limit: float = 1.4,
  lower_limit: float = 0.9,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  """Penalise hip roll outside ``[-upper_limit, upper_limit]`` (HoST ``style_hip_roll_deviation``)."""
  asset: Entity = env.scene[asset_cfg.name]
  q = _joint_pos(asset, asset_cfg.joint_ids)
  return (torch.max(torch.abs(q), dim=-1)[0] > upper_limit) | (
    torch.min(torch.abs(q), dim=-1)[0] > lower_limit
  )


def style_shoulder_roll_deviation(
  env: ManagerBasedRlEnv,
  left_threshold: float = -0.02,
  right_threshold: float = 0.02,
  left_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
  right_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  """Penalise shoulders rolled the wrong way while on the ground.

  Keeps the arms out of the way of the torso when the robot is lying down; the
  left/right asymmetry follows HoST exactly.
  """
  asset: Entity = env.scene[left_cfg.name]
  left = _joint_pos(asset, left_cfg.joint_ids)[:, 0]
  right = _joint_pos(asset, right_cfg.joint_ids)[:, 0]
  return (left < left_threshold) | (right > right_threshold)


def _foot_displacement(
  env: ManagerBasedRlEnv,
  sigma: float,
  z_limit: float,
  min_distance: float,
  phase3_height: float,
  foot_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[foot_cfg.name]
  base_xy = asset.data.root_link_pos_w[:, :2]
  foot = asset.data.body_link_pos_w[:, foot_cfg.body_ids[0]]
  foot_xy, foot_z = foot[:, :2], foot[:, 2]
  error = (
    torch.sum(torch.square(base_xy - foot_xy), dim=-1).clamp(min_distance, float("inf"))
  )
  reward = torch.exp(error * sigma) * (foot_z < z_limit)
  standup = asset.data.root_link_pos_w[:, 2] > phase3_height
  return reward * standup


def style_left_foot_displacement(
  env: ManagerBasedRlEnv,
  sigma: float = -2.0,
  z_limit: float = 0.3,
  min_distance: float = 0.3,
  phase3_height: float = 0.65,
  foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Keep the left foot close to the base once the robot is standing (HoST ``style_left_foot_displacement``)."""
  return _foot_displacement(
    env, sigma, z_limit, min_distance, phase3_height, foot_cfg
  )


def style_right_foot_displacement(
  env: ManagerBasedRlEnv,
  sigma: float = -2.0,
  z_limit: float = 0.3,
  min_distance: float = 0.3,
  phase3_height: float = 0.65,
  foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Keep the right foot close to the base once the robot is standing (HoST ``style_right_foot_displacement``)."""
  return _foot_displacement(
    env, sigma, z_limit, min_distance, phase3_height, foot_cfg
  )


def style_knee_deviation(
  env: ManagerBasedRlEnv,
  upper_limit: float = 2.85,
  lower_limit: float = -0.06,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  """Keep knees inside the range a G1 can actually reach (HoST ``style_knee_deviation``)."""
  asset: Entity = env.scene[asset_cfg.name]
  q = _joint_pos(asset, asset_cfg.joint_ids)
  return (torch.max(torch.abs(q), dim=-1)[0] > upper_limit) | (
    torch.min(q, dim=-1)[0] < lower_limit
  )


def style_elbow_deviation(
  env: ManagerBasedRlEnv,
  upper_limit: float = 2.0,
  lower_limit: float = -0.05,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  """Keep elbows in a human-like range while standing up.

  The G1 elbow range is ``[-1.0472, 2.0944]`` with flexion positive and
  negative angles meaning hyperextension (bending the wrong way). Penalising
  ``q < lower_limit`` stops the elbow from being forced backward, while
  ``|q| > upper_limit`` keeps the final pose from locking into an unnaturally
  folded arm. Mirrors ``style_knee_deviation`` (HoST-style indicator).
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = _joint_pos(asset, asset_cfg.joint_ids)
  return (torch.max(torch.abs(q), dim=-1)[0] > upper_limit) | (
    torch.min(q, dim=-1)[0] < lower_limit
  )


def style_shank_orientation(
  env: ManagerBasedRlEnv,
  phase1_height: float = 0.45,
  threshold: float = 0.8,
  left_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  left_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  right_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  right_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  root_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Keep the shank roughly vertical while on the ground (HoST ``style_shank_orientation``).

  Rewards the mean ``(knee - foot)_z / |knee - foot|`` being close to 1, i.e.
  the shank pointing up, which is what a human does when pushing off the
  ground.
  """
  asset: Entity = env.scene[left_knee_cfg.name]
  left_knee = asset.data.body_link_pos_w[:, left_knee_cfg.body_ids[0], :]
  left_foot = asset.data.body_link_pos_w[:, left_foot_cfg.body_ids[0], :]
  right_knee = asset.data.body_link_pos_w[:, right_knee_cfg.body_ids[0], :]
  right_foot = asset.data.body_link_pos_w[:, right_foot_cfg.body_ids[0], :]

  left_dir = left_knee - left_foot
  right_dir = right_knee - right_foot
  left_ori = left_dir[:, 2] / torch.norm(left_dir, dim=-1)
  right_ori = right_dir[:, 2] / torch.norm(right_dir, dim=-1)
  feet_orientation = 0.5 * (left_ori + right_ori)

  on_ground = _root_height(env, root_cfg) > phase1_height
  return tolerance(feet_orientation, [threshold, float("inf")], 1.0, 0.1) * on_ground


def style_feet_distance(
  env: ManagerBasedRlEnv,
  threshold: float = 0.9,
  left_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  right_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Penalise feet that drift far apart (HoST ``style_feet_distance``).

  NOTE: HoST evaluates a ``tolerance(...)`` kernel here but then discards it and
  returns the raw ``feet_distances > 0.9`` boolean. That dead code is kept out
  of this port; the returned quantity is identical.
  """
  asset: Entity = env.scene[left_foot_cfg.name]
  left = asset.data.body_link_pos_w[:, left_foot_cfg.body_ids[0], :3]
  right = asset.data.body_link_pos_w[:, right_foot_cfg.body_ids[0], :3]
  return (torch.norm(left - right, dim=-1) > threshold).squeeze(-1)


def style_ang_vel_xy(
  env: ManagerBasedRlEnv,
  phase1_height: float = 0.45,
  sigma: float = -2.0,
  root_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Damp base roll/pitch rate while on the ground (HoST ``style_style_ang_vel_xy``)."""
  asset: Entity = env.scene[root_cfg.name]
  ang_xy_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  on_ground = _root_height(env, root_cfg) > phase1_height
  return torch.exp(ang_xy_sq * sigma) * on_ground


def style_ground_parallel(
  env: ManagerBasedRlEnv,
  threshold: float = 0.05,
  left_ankle_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  right_ankle_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Keep the ankles level on the ground (HoST ``style_ground_parallel``).

  NOTE: ported verbatim, including the fact that it evaluates to a constant
  zero. HoST takes the variance of ``rigid_body_states[:, [one ankle], 2]``,
  i.e. the variance of a single element along dim 1, which is NaN for the
  default ``correction=1``; the following ``var < 0.05`` comparison is then
  False, so the term contributes ``0 * scale``. It is kept here so the reward
  table still matches HoST's config 1:1. The ``[body_id]`` brackets keep the
  singleton dimension so ``var(dim=1)`` is well-defined.
  """
  asset: Entity = env.scene[left_ankle_cfg.name]
  left_ankle_z = asset.data.body_link_pos_w[:, [left_ankle_cfg.body_ids[0]], 2] * 10.0
  right_ankle_z = (
    asset.data.body_link_pos_w[:, [right_ankle_cfg.body_ids[0]], 2] * 10.0
  )
  variance = 0.5 * (left_ankle_z.var(dim=1) + right_ankle_z.var(dim=1))
  return variance < threshold


##
# target group -- post stand-up convergence (HoST group weight 1.0).
#
# Every term is gated by ``standup``: the robot only counts as standing once the
# base is above ``target_base_height_phase3``.
##


def _standup_gate(
  env: ManagerBasedRlEnv, phase3_height: float, root_cfg: SceneEntityCfg
) -> torch.Tensor:
  return _root_height(env, root_cfg) > phase3_height


def target_ang_vel_xy(
  env: ManagerBasedRlEnv,
  phase3_height: float = 0.65,
  sigma: float = -2.0,
  root_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Stop rotating once standing (HoST ``target_ang_vel_xy``)."""
  asset: Entity = env.scene[root_cfg.name]
  ang_xy_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  return torch.exp(ang_xy_sq * sigma) * _standup_gate(env, phase3_height, root_cfg)


def target_lin_vel_xy(
  env: ManagerBasedRlEnv,
  phase3_height: float = 0.65,
  sigma: float = -5.0,
  root_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Stop translating once standing (HoST ``target_lin_vel_xy``)."""
  asset: Entity = env.scene[root_cfg.name]
  lin_xy_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(lin_xy_sq * sigma) * _standup_gate(env, phase3_height, root_cfg)


def target_feet_height_var(
  env: ManagerBasedRlEnv,
  phase3_height: float = 0.65,
  sigma: float = -2.0,
  min_difference: float = 0.2,
  left_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  right_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  root_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Keep both feet at the same height once standing (HoST ``target_feet_height_var``)."""
  asset: Entity = env.scene[left_foot_cfg.name]
  left_z = asset.data.body_link_pos_w[:, left_foot_cfg.body_ids[0], 2] * 10.0
  right_z = asset.data.body_link_pos_w[:, right_foot_cfg.body_ids[0], 2] * 10.0
  difference = torch.abs(left_z - right_z).clamp(min_difference, float("inf"))
  return torch.exp(difference * sigma) * _standup_gate(env, phase3_height, root_cfg)


def target_target_upper_dof_pos(
  env: ManagerBasedRlEnv,
  phase3_height: float = 0.65,
  sigma: float = -0.1,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
  target_upper_dof_pos: tuple[float, ...] | None = None,
) -> torch.Tensor:
  """Drive the upper body to HoST's target standing pose (``target_target_upper_dof_pos``).

  HoST compares against ``init_state.target_joint_angles`` for the waist and arm
  joints only, which is what ``asset_cfg`` should select.

  NOTE: ``target_upper_dof_pos`` is an ordered tuple matching ``asset_cfg``'s
  joint order; when omitted the robot's default joint positions are used.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  if target_upper_dof_pos is None:
    target = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  else:
    target = torch.tensor(
      target_upper_dof_pos, device=q.device, dtype=q.dtype
    ).expand_as(q)
  mse = torch.sum(torch.square(q - target), dim=-1)
  return torch.exp(mse * sigma) * _standup_gate(env, phase3_height, asset_cfg)


def target_target_orientation(
  env: ManagerBasedRlEnv,
  phase3_height: float = 0.65,
  sigma: float = -5.0,
  root_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Hold the base upright once standing (HoST ``target_target_orientation``)."""
  asset: Entity = env.scene[root_cfg.name]
  xy_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(xy_sq * sigma) * _standup_gate(env, phase3_height, root_cfg)


def target_target_base_height(
  env: ManagerBasedRlEnv,
  base_height_target: float = 0.75,
  phase3_height: float = 0.65,
  sigma: float = -20.0,
  root_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Drive the base to the standing height (HoST ``target_target_base_height``)."""
  error = torch.abs(_root_height(env, root_cfg) - base_height_target)
  return torch.exp(error * sigma) * _standup_gate(env, phase3_height, root_cfg)
