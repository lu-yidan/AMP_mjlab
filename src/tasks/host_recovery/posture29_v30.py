"""V30 paired-bias sagittal alignment from the stable V28 controller."""

from src.tasks.host_recovery.posture29_v29 import posture29_v29_env_cfg


def posture29_v30_env_cfg():
  cfg = posture29_v29_env_cfg()
  cfg.rewards["target_sagittal_leg_symmetry_error"].weight = 0.0
  return cfg
