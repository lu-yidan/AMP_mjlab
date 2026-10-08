"""Conservative V6 continuation toward the stable 23-DoF standing pose."""

from src.tasks.host_recovery.flat29 import flat29_env_cfg


POST_STAND_HEIGHT = 0.70


def posture29_v6_env_cfg():
  cfg = flat29_env_cfg(play=False)
  cfg.events["init_pull_force"].params["force"] = 0.0
  cfg.curriculum.pop("pull_force", None)
  cfg.curriculum.pop("action_scale", None)

  for reward in cfg.rewards.values():
    if "phase3_height" in reward.params:
      reward.params["phase3_height"] = POST_STAND_HEIGHT

  cfg.rewards["target_target_upper_dof_pos"].weight *= 1.10
  cfg.rewards["target_target_orientation"].weight *= 1.25
  cfg.rewards["target_target_base_height"].weight *= 1.10
  cfg.rewards["target_ang_vel_xy"].weight *= 1.10
  cfg.rewards["target_lin_vel_xy"].weight *= 1.10
  cfg.rewards["target_leg_dof_vel"].weight = -0.0008
  cfg.rewards["target_upper_dof_vel"].weight = -0.0004
  return cfg
