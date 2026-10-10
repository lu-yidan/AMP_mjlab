"""V19 continuation with strong post-standing hip-yaw damping."""

from src.tasks.host_recovery.posture29_v18 import posture29_v18_env_cfg


def posture29_v19_env_cfg():
  cfg = posture29_v18_env_cfg()
  cfg.rewards["target_hip_yaw_dof_vel"].weight = -0.25
  return cfg
