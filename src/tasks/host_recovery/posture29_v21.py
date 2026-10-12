"""V21 continuation rewards for coupled lateral and yaw stabilization."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v16 import POST_MOTION_HEIGHT
from src.tasks.host_recovery.posture29_v19 import posture29_v19_env_cfg


def posture29_v21_env_cfg():
  cfg = posture29_v19_env_cfg()
  cfg.rewards["target_yaw_rate"].weight = -1.75
  cfg.rewards["target_root_horizontal_speed"].weight = -0.70
  cfg.rewards["target_foot_horizontal_speed"].weight = -0.45
  cfg.rewards["target_leg_dof_vel"].weight = -0.0040
  cfg.rewards["target_hip_yaw_dof_vel"].weight = -0.15
  cfg.rewards["target_lateral_balance_dof_vel"] = RewardTermCfg(
    func=mdp.target_dof_vel,
    weight=-0.015,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_hip_roll_joint",
          ".*_ankle_roll_joint",
          "waist_yaw_joint",
        ),
        preserve_order=True,
      ),
    },
  )
  return cfg
