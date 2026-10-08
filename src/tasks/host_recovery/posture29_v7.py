"""Targeted V7 continuation that preserves V4 recovery features."""

from src.tasks.host_recovery.flat29 import flat29_env_cfg


POST_STAND_HEIGHT = 0.72


def posture29_v7_env_cfg():
  cfg = flat29_env_cfg(play=False)
  cfg.events["init_pull_force"].params["force"] = 0.0
  cfg.curriculum.pop("pull_force", None)
  cfg.curriculum.pop("action_scale", None)

  for reward in cfg.rewards.values():
    if "phase3_height" in reward.params:
      reward.params["phase3_height"] = POST_STAND_HEIGHT

  cfg.rewards["target_target_upper_dof_pos"].weight *= 1.50
  cfg.rewards["target_elbow_up"].weight = -0.010
  cfg.rewards["target_arm_twist"].weight = -0.006
  cfg.rewards["target_wrist_flip"].weight = -0.004
  cfg.rewards["target_leg_dof_vel"].weight = -0.001
  cfg.rewards["target_upper_dof_vel"].weight = -0.0005
  return cfg
