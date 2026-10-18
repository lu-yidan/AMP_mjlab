"""A6-compatible standing-height contract for the stable V38 policy."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from src.tasks.host_recovery.posture29_v38 import posture29_v38_env_cfg


A6_HEAD_OFFSET = 0.43
A6_HEIGHT_LOW = 1.05
A6_HEIGHT_TARGET = 1.17
A6_UPRIGHT_LOW = 0.80
A6_UPRIGHT_TARGET = 0.93
POST_STAND_ROOT_HEIGHT = 0.66


def _torso() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=("torso_link",), preserve_order=True)


def a6_virtual_head_height(
  env: ManagerBasedRlEnv,
  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Height of SMP's massless head site reconstructed on ``torso_link``."""
  robot = env.scene[torso_cfg.name]
  torso_id = torso_cfg.body_ids[0]
  local = torch.zeros((env.num_envs, 3), device=env.device)
  local[:, 2] = A6_HEAD_OFFSET
  return (
    robot.data.body_link_pos_w[:, torso_id, 2]
    + quat_apply(robot.data.body_link_quat_w[:, torso_id], local)[:, 2]
    - env.scene.env_origins[:, 2]
  )


def _post_stand_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return (
    (robot.data.root_link_pos_w[:, 2] >= POST_STAND_ROOT_HEIGHT)
    & (-robot.data.projected_gravity_b[:, 2] >= A6_UPRIGHT_LOW)
  ).float()


def _smooth_ramp(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
  fraction = ((value - low) / (high - low)).clamp(0.0, 1.0)
  return fraction.square() * (3.0 - 2.0 * fraction)


def a6_height_progress(
  env: ManagerBasedRlEnv,
  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  height = a6_virtual_head_height(env, torso_cfg)
  upright = -env.scene["robot"].data.projected_gravity_b[:, 2]
  return (
    _smooth_ramp(height, A6_HEIGHT_LOW, A6_HEIGHT_TARGET)
    * _smooth_ramp(upright, A6_UPRIGHT_LOW, A6_UPRIGHT_TARGET)
    * _post_stand_gate(env)
  )


def a6_height_shortfall(
  env: ManagerBasedRlEnv,
  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  height = a6_virtual_head_height(env, torso_cfg)
  shortfall = ((A6_HEIGHT_TARGET - height) / (A6_HEIGHT_TARGET - A6_HEIGHT_LOW))
  return shortfall.clamp(0.0, 2.0).square() * _post_stand_gate(env)


def a6_formal_contract(
  env: ManagerBasedRlEnv,
  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  height = a6_virtual_head_height(env, torso_cfg)
  upright = -env.scene["robot"].data.projected_gravity_b[:, 2]
  return (
    (height >= 1.15)
    & (upright >= 0.90)
    & (_post_stand_gate(env) > 0.0)
  ).float()


def posture29_v39_env_cfg():
  cfg = posture29_v38_env_cfg()
  cfg.rewards["target_a6_height_progress"] = RewardTermCfg(
    func=a6_height_progress,
    weight=3.0,
    params={"torso_cfg": _torso()},
  )
  cfg.rewards["target_a6_height_shortfall"] = RewardTermCfg(
    func=a6_height_shortfall,
    weight=-1.0,
    params={"torso_cfg": _torso()},
  )
  cfg.rewards["target_a6_formal_contract"] = RewardTermCfg(
    func=a6_formal_contract,
    weight=2.0,
    params={"torso_cfg": _torso()},
  )
  return cfg
