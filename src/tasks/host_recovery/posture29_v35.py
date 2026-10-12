"""Observable post-standing stillness tuning from the stable V25 policy."""

from src.tasks.host_recovery.posture29_v25 import posture29_v25_env_cfg


def posture29_v35_env_cfg():
  cfg = posture29_v25_env_cfg()

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

  return cfg
