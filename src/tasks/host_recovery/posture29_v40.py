"""Phase-isolated A6 height objective for full-actor HoST repair."""

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery.posture29_v38 import (
  _settle_gate,
  posture29_v38_env_cfg,
)
from src.tasks.host_recovery.posture29_v39 import (
  a6_formal_contract,
  a6_height_progress,
  a6_height_shortfall,
)


def _torso() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=("torso_link",), preserve_order=True)


def settled_a6_height_progress(
  env: ManagerBasedRlEnv,
  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
):
  return a6_height_progress(env, torso_cfg) * _settle_gate(env)


def settled_a6_height_shortfall(
  env: ManagerBasedRlEnv,
  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
):
  return a6_height_shortfall(env, torso_cfg) * _settle_gate(env)


def settled_a6_formal_contract(
  env: ManagerBasedRlEnv,
  torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
):
  return a6_formal_contract(env, torso_cfg) * _settle_gate(env)


def posture29_v40_env_cfg():
  cfg = posture29_v38_env_cfg()
  cfg.episode_length_s = 23.0
  cfg.rewards["target_a6_height_progress"] = RewardTermCfg(
    func=settled_a6_height_progress,
    weight=4.0,
    params={"torso_cfg": _torso()},
  )
  cfg.rewards["target_a6_height_shortfall"] = RewardTermCfg(
    func=settled_a6_height_shortfall,
    weight=-2.0,
    params={"torso_cfg": _torso()},
  )
  cfg.rewards["target_a6_formal_contract"] = RewardTermCfg(
    func=settled_a6_formal_contract,
    weight=3.0,
    params={"torso_cfg": _torso()},
  )
  return cfg
