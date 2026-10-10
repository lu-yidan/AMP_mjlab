"""V14 continuation rewards with dense torso and arm geometry costs."""

from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v8 import POST_STAND_HEIGHT, _bodies
from src.tasks.host_recovery.posture29_v13 import posture29_v13_env_cfg


def posture29_v14_env_cfg():
  cfg = posture29_v13_env_cfg()
  cfg.rewards.pop("target_hands_beside_hips", None)
  cfg.rewards.pop("target_torso_over_support", None)

  cfg.rewards["target_torso_vertical"].weight = 0.75
  cfg.rewards["target_torso_tilt"].weight = -1.20
  cfg.rewards["target_upper_pose_linear"].weight = -0.80
  cfg.rewards["target_foot_horizontal_speed"].weight = -0.12
  cfg.rewards["target_leg_dof_vel"].weight = -0.0020
  cfg.rewards["target_upper_dof_vel"].weight = -0.0012
  cfg.rewards["target_elbow_flexion"].weight = -0.08

  cfg.rewards["target_hands_beside_hips_error"] = RewardTermCfg(
    func=mdp.target_hands_beside_hips_error,
    weight=-1.50,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "left_target": (0.0, 0.28, -0.30),
      "right_target": (0.0, -0.28, -0.30),
      "torso_cfg": _bodies("torso_link"),
      "hand_cfg": _bodies("left_wrist_yaw_link", "right_wrist_yaw_link"),
    },
  )
  cfg.rewards["target_torso_support_error"] = RewardTermCfg(
    func=mdp.target_upper_center_support_error,
    weight=-2.00,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "upper_cfg": _bodies("torso_link"),
      "ankle_cfg": _bodies(
        "left_ankle_roll_link", "right_ankle_roll_link"
      ),
    },
  )
  return cfg
