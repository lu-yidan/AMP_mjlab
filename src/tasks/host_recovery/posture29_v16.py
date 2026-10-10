"""V16 continuation rewards for V15 yaw spin and stance drift."""

from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v8 import POST_STAND_HEIGHT
from src.tasks.host_recovery.posture29_v15 import posture29_v15_env_cfg


POST_MOTION_HEIGHT = 0.65


def posture29_v16_env_cfg():
  cfg = posture29_v15_env_cfg()
  cfg.events["init_pull_force"].params["force"] = 0.0
  cfg.curriculum.pop("pull_force", None)
  cfg.curriculum.pop("action_scale", None)
  cfg.rewards["target_yaw_rate"] = RewardTermCfg(
    func=mdp.target_yaw_rate,
    weight=-0.75,
    params={"phase3_height": POST_MOTION_HEIGHT},
  )
  cfg.rewards["target_root_horizontal_speed"] = RewardTermCfg(
    func=mdp.target_root_horizontal_speed,
    weight=-0.40,
    params={"phase3_height": POST_MOTION_HEIGHT},
  )
  cfg.rewards["target_foot_horizontal_speed"].weight = -0.30
  cfg.rewards["target_foot_horizontal_speed"].params[
    "phase3_height"
  ] = POST_MOTION_HEIGHT
  cfg.rewards["target_leg_dof_vel"].weight = -0.0030
  cfg.rewards["target_leg_dof_vel"].params["phase3_height"] = POST_MOTION_HEIGHT
  return cfg
