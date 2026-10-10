"""V18 continuation with a targeted post-standing hip-yaw penalty."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v16 import POST_MOTION_HEIGHT
from src.tasks.host_recovery.posture29_v17 import posture29_v17_env_cfg


def posture29_v18_env_cfg():
  cfg = posture29_v17_env_cfg()
  cfg.rewards["target_hip_yaw_dof_vel"] = RewardTermCfg(
    func=mdp.target_dof_vel,
    weight=-0.04,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("left_hip_yaw_joint", "right_hip_yaw_joint"),
        preserve_order=True,
      ),
    },
  )
  return cfg
