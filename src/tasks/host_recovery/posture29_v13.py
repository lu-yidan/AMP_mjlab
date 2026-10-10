"""V13 rewards targeting V11 torso asymmetry and residual joint motion."""

from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v8 import POST_STAND_HEIGHT, _bodies
from src.tasks.host_recovery.posture29_v12 import posture29_v12_env_cfg


def posture29_v13_env_cfg():
  cfg = posture29_v12_env_cfg()

  cfg.rewards["target_hands_beside_hips"].weight = 0.55
  cfg.rewards["target_torso_tilt"].weight = -0.55
  cfg.rewards["target_upper_pose_linear"].weight = -0.30
  cfg.rewards["target_foot_horizontal_speed"].weight = -0.08
  cfg.rewards["target_torso_vertical"].weight = 0.35
  cfg.rewards["target_leg_dof_vel"].weight = -0.0015
  cfg.rewards["target_upper_dof_vel"].weight = -0.0008

  cfg.rewards["target_torso_over_support"] = RewardTermCfg(
    func=mdp.target_upper_center_over_support,
    weight=0.30,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "error_scale": 0.07,
      "upper_cfg": _bodies("torso_link"),
      "ankle_cfg": _bodies(
        "left_ankle_roll_link", "right_ankle_roll_link"
      ),
    },
  )
  return cfg
