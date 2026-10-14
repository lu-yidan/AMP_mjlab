"""Conservative observable damping objective from the stable V35 policy."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v35 import posture29_v35_env_cfg


STABILITY_HEIGHT = 0.66
ALL_ACTIONS = tuple(range(29))


def _all_joints() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(".*",), preserve_order=True)


def posture29_v37_env_cfg():
  cfg = posture29_v35_env_cfg()

  cfg.rewards["target_yaw_rate"].weight = -2.00
  cfg.rewards["target_leg_dof_vel"].weight = -0.0060
  cfg.rewards["target_hip_yaw_dof_vel"].weight = -0.14
  cfg.rewards["target_lateral_balance_dof_vel"].weight = -0.035
  cfg.rewards["target_left_hip_yaw_velocity"].weight = -0.14
  cfg.rewards["target_lateral_joint_stillness"].weight = 0.90
  cfg.rewards["target_lateral_action"].weight = -0.24
  cfg.rewards["target_left_hip_yaw_action"].weight = -0.18

  for name in (
    "target_yaw_rate",
    "target_leg_dof_vel",
    "target_hip_yaw_dof_vel",
    "target_lateral_balance_dof_vel",
    "target_left_hip_yaw_velocity",
    "target_lateral_joint_stillness",
    "target_lateral_action",
    "target_left_hip_yaw_action",
  ):
    cfg.rewards[name].params["phase3_height"] = STABILITY_HEIGHT

  cfg.rewards["target_all_joint_stillness"] = RewardTermCfg(
    func=mdp.target_dof_stillness,
    weight=0.60,
    params={
      "phase3_height": STABILITY_HEIGHT,
      "velocity_scale": 0.75,
      "asset_cfg": _all_joints(),
    },
  )
  cfg.rewards["target_all_joint_velocity"] = RewardTermCfg(
    func=mdp.target_dof_vel,
    weight=-0.0040,
    params={
      "phase3_height": STABILITY_HEIGHT,
      "asset_cfg": _all_joints(),
    },
  )
  cfg.rewards["target_all_action"] = RewardTermCfg(
    func=mdp.target_action_l2,
    weight=-0.05,
    params={
      "phase3_height": STABILITY_HEIGHT,
      "action_indices": ALL_ACTIONS,
    },
  )
  cfg.rewards["target_standing_action_rate"] = RewardTermCfg(
    func=mdp.target_action_rate_l2,
    weight=-0.20,
    params={
      "phase3_height": STABILITY_HEIGHT,
      "action_indices": ALL_ACTIONS,
    },
  )
  cfg.rewards["target_root_angular_speed"] = RewardTermCfg(
    func=mdp.target_root_angular_speed,
    weight=-0.60,
    params={"phase3_height": STABILITY_HEIGHT},
  )

  return cfg
