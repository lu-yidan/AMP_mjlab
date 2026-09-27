"""HoST reset events: the "diverse postures" part of the paper.

HoST starts every episode with the robot lying on the floor and -- crucially --
with its motors switched off for the first ``unactuated_timesteps`` control
steps, so the policy has to deal with a robot that has already settled on the
ground instead of a hand-placed pose.

``G1Cfg.init_state`` in HoST:

- ``pos = [0.0, 0.0, 0.5]`` -- 0.5 m above the ground (a prone G1's pelvis
  height), then gravity settles it,
- ``rot = [0.0, -1, 0, 1.0]`` in ``(x, y, z, w)`` -- 180 deg about ``y``, i.e.
  face down (prone),
- ``default_joint_angles`` -- the loose "limbs slightly bent" pose the reset
  scatters around.

The paper's "across diverse postures" contribution is resetting from prone,
supine and both side postures; that is reproduced here through the ``posture``
argument of :func:`reset_root_state_posture`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: HoST ``init_state.pos[2]``.
DEFAULT_HEIGHT = 0.5


#: ``cos(45 deg) = sin(45 deg) = sqrt(2)/2``; a quarter turn about a unit
#: axis reads ``(cos45, sin45 * axis)`` in mjlab's ``(w, x, y, z)`` order.
_S = 0.7071067811865476

# The G1 ``pelvis`` body has identity orientation in the MJCF, so while the
# robot stands upright its body frame (``x`` forward, ``y`` left, ``z`` up)
# coincides with the world frame. Every posture below is a single quarter
# turn off upright, which is what lays the robot flat on the floor:
#   prone      -- +90 deg about ``y``: body ``+x`` (chest) points down.
#   supine     -- -90 deg about ``y``: body ``+x`` (chest) points up.
#   left_side  -- -90 deg about ``x``: body ``+y`` (left) points down.
#   right_side -- +90 deg about ``x``: body ``+y`` (left) points up.
#
# HoST writes ``init_state.rot = [0.0, -1, 0, 1.0]`` in ``(x, y, z, w)``,
# which normalises to a *quarter* turn, not the half turn the previous values
# here encoded: ``(0, 0, 1, 0)`` is 180 deg about ``y``, i.e. head straight
# down, so every episode used to start upside down.
PRONE_QUAT = (_S, 0.0, _S, 0.0)
SUPINE_QUAT = (_S, 0.0, -_S, 0.0)
LEFT_SIDE_QUAT = (_S, -_S, 0.0, 0.0)
RIGHT_SIDE_QUAT = (_S, _S, 0.0, 0.0)

POSTURE_QUATS: dict[str, tuple[float, float, float, float]] = {
  "prone": PRONE_QUAT,
  "supine": SUPINE_QUAT,
  "left_side": LEFT_SIDE_QUAT,
  "right_side": RIGHT_SIDE_QUAT,
}


#: Settled pelvis (root) heights in metres for each posture, measured with
#: ``probe_settle_natural.py`` (gravity only, no pull force). Setting the reset
#: height to these values starts every
#: episode already resting on the floor instead of dropping from 0.5 m.
POSTURE_SETTLED_HEIGHTS: dict[str, float] = {
  "prone": 0.093,
  "supine": 0.075,
  "left_side": 0.094,
  "right_side": 0.099,
}


def reset_root_state_posture(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  posture: Literal["prone", "supine", "left_side", "right_side"] | None = "prone",
  height: float | None = DEFAULT_HEIGHT,
  xy_range: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Place the robot flat on the ground in one of HoST's start postures.

  ``posture=None`` samples uniformly from all four postures, which is the
  "diverse postures" setting of the paper. ``xy_range`` mirrors HoST's
  ``custom_origins``, which scatters the base by +/- 1 m around the terrain
  origin.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]

  positions = env.scene.env_origins[env_ids].clone()
  positions[:, :2] += torch.empty(len(env_ids), 2, device=env.device).uniform_(
    -xy_range, xy_range
  )

  num_reset = len(env_ids)
  if posture is None:
    names = list(POSTURE_QUATS)
    choice = torch.randint(len(names), (num_reset,), device=env.device)
    quats = torch.tensor(
      [POSTURE_QUATS[names[i]] for i in range(len(names))],
      device=env.device,
      dtype=positions.dtype,
    )[choice]
    if height is None:
      heights = torch.tensor(
        [POSTURE_SETTLED_HEIGHTS[n] for n in names],
        device=env.device,
        dtype=positions.dtype,
      )[choice]
    else:
      heights = torch.full((num_reset,), height, device=env.device, dtype=positions.dtype)
  else:
    quats = torch.tensor(
      POSTURE_QUATS[posture], device=env.device, dtype=positions.dtype
    ).expand(num_reset, 4)
    if height is None:
      heights = torch.full(
        (num_reset,), POSTURE_SETTLED_HEIGHTS[posture], device=env.device, dtype=positions.dtype
      )
    else:
      heights = torch.full((num_reset,), height, device=env.device, dtype=positions.dtype)

  positions[:, 2] += heights

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, quats], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(
    torch.zeros(num_reset, 6, device=env.device), env_ids=env_ids
  )


def reset_joints_scaled(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_scale: tuple[float, float] = (0.9, 1.1),
  position_offset: tuple[float, float] = (-0.1, 0.1),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scatter the joints around their default pose (HoST ``_reset_dofs``).

  HoST computes ``default_dof_pos * U(0.9, 1.1) + U(-0.1, 0.1)`` and clips to
  the joint limits; the same expression is used here, only with mjlab's
  ``default_joint_pos`` as the source of ``default_dof_pos``.

  NOTE: HoST's ``default_dof_pos`` comes from ``init_state.default_joint_angles``
  while ``joint_pos_target`` rewards come from ``init_state.target_joint_angles``
  -- two different poses. In mjlab both map onto the robot config's
  ``init_state.joint_pos`` (see ``G1_23DOF_KEYFRAME``), so the two poses are
  identical here. Retarget the arm/waist entries of that keyframe if you want
  to reproduce HoST's split.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
  num_reset, num_joints = default.shape

  scale = torch.empty(num_reset, num_joints, device=env.device).uniform_(*position_scale)
  offset = torch.empty(num_reset, num_joints, device=env.device).uniform_(
    *position_offset
  )
  joint_pos = default * scale + offset

  limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
  joint_vel = torch.zeros_like(joint_pos)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)
  asset.write_joint_state_to_sim(
    joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids
  )
