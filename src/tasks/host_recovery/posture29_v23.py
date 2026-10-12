"""V23 continuation rewards for foot alignment and torso stabilization."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v16 import POST_MOTION_HEIGHT
from src.tasks.host_recovery.posture29_v21 import posture29_v21_env_cfg


def posture29_v23_env_cfg():
  cfg = posture29_v21_env_cfg()

  cfg.rewards["target_yaw_rate"].weight = -2.00
  cfg.rewards["target_root_horizontal_speed"].weight = -0.80
  cfg.rewards["target_foot_horizontal_speed"].weight = -0.55
  cfg.rewards["target_leg_dof_vel"].weight = -0.0045
  cfg.rewards["target_hip_yaw_dof_vel"].weight = -0.18
  cfg.rewards["target_lateral_balance_dof_vel"].weight = -0.020

  cfg.rewards["target_torso_vertical"].weight = 0.75
  cfg.rewards["target_torso_tilt"].weight = -1.50
  cfg.rewards["target_torso_support_error"].weight = -2.50
  for name in (
    "target_torso_vertical",
    "target_torso_tilt",
    "target_torso_support_error",
  ):
    cfg.rewards[name].params["phase3_height"] = POST_MOTION_HEIGHT

  cfg.rewards["target_feet_height_var"].weight = 0.40
  cfg.rewards["target_feet_height_var"].params.update(
    phase3_height=POST_MOTION_HEIGHT,
    min_difference=0.0,
  )
  cfg.rewards["target_feet_side_by_side_error"] = RewardTermCfg(
    func=mdp.target_feet_side_by_side_error,
    weight=-0.60,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "target_lateral_separation": 0.24,
      "forward_scale": 0.08,
      "height_scale": 0.03,
      "lateral_scale": 0.08,
      "foot_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
    },
  )
  return cfg
