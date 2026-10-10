"""V17 continuation rewards for stronger post-standing anti-spin control."""

from src.tasks.host_recovery.posture29_v16 import posture29_v16_env_cfg


def posture29_v17_env_cfg():
  cfg = posture29_v16_env_cfg()
  cfg.rewards["target_yaw_rate"].weight = -1.50
  cfg.rewards["target_root_horizontal_speed"].weight = -0.60
  cfg.rewards["target_foot_horizontal_speed"].weight = -0.40
  cfg.rewards["target_leg_dof_vel"].weight = -0.0035
  return cfg
