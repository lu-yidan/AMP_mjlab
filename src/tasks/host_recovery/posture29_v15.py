"""V15 from-scratch posture objective with intentionally extreme constraints."""

from src.tasks.host_recovery.flat29 import flat29_env_cfg
from src.tasks.host_recovery.posture29_v14 import posture29_v14_env_cfg


def posture29_v15_env_cfg():
  cfg = posture29_v14_env_cfg()
  curriculum_cfg = flat29_env_cfg(play=False)
  cfg.events["init_pull_force"] = curriculum_cfg.events["init_pull_force"]
  cfg.curriculum = curriculum_cfg.curriculum

  cfg.rewards["target_torso_vertical"].weight = 1.50
  cfg.rewards["target_torso_tilt"].weight = -4.00
  cfg.rewards["target_upper_pose_linear"].weight = -2.50
  cfg.rewards["target_foot_horizontal_speed"].weight = -0.20
  cfg.rewards["target_leg_dof_vel"].weight = -0.0025
  cfg.rewards["target_upper_dof_vel"].weight = -0.0020
  cfg.rewards["target_elbow_up"].weight = -0.10
  cfg.rewards["target_arm_twist"].weight = -0.08
  cfg.rewards["target_wrist_flip"].weight = -0.05
  cfg.rewards["target_elbow_flexion"].weight = -0.50
  cfg.rewards["target_hands_beside_hips_error"].weight = -5.00
  cfg.rewards["target_torso_support_error"].weight = -8.00
  return cfg
