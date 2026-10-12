"""Bounded upper-body straightening while preserving V35 stillness."""

from src.tasks.host_recovery.posture29_v26 import posture29_v26_env_cfg


def posture29_v36_env_cfg():
  cfg = posture29_v26_env_cfg()

  cfg.rewards["target_root_horizontal_speed"].weight = 0.0
  cfg.rewards["target_yaw_rate"].weight = -2.00
  cfg.rewards["target_foot_horizontal_speed"].weight = 0.0
  cfg.rewards["target_leg_dof_vel"].weight = -0.0050
  cfg.rewards["target_hip_yaw_dof_vel"].weight = -0.12
  cfg.rewards["target_lateral_balance_dof_vel"].weight = -0.025
  cfg.rewards["target_left_hip_yaw_velocity"].weight = -0.12
  cfg.rewards["target_lateral_joint_stillness"].weight = 0.75
  cfg.rewards["target_lateral_action"].weight = -0.20
  cfg.rewards["target_left_hip_yaw_action"].weight = -0.15

  cfg.rewards["target_torso_vertical"].weight = 1.50
  cfg.rewards["target_torso_tilt"].weight = -4.00
  cfg.rewards["target_torso_over_support"].weight = 1.00
  cfg.rewards["target_waist_roll_pitch_error"].weight = -1.50
  cfg.rewards["target_left_arm_pose"].weight = -1.50
  cfg.rewards["target_right_arm_pose"].weight = -2.00
  cfg.rewards["target_waist_pose"].weight = -2.00
  cfg.rewards["target_arm_stillness"].weight = 0.30

  return cfg
