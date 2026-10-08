"""V12 dense post-stand posture rewards targeting the V10 failure mode."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.config.g1.env_cfgs import (
  HOST_TARGET_UPPER_DOF_POS,
  UPPER_BODY_JOINTS,
)
from src.tasks.host_recovery.posture29_v8 import POST_STAND_HEIGHT, _bodies
from src.tasks.host_recovery.posture29_v9 import posture29_v9_env_cfg


def posture29_v12_env_cfg():
  cfg = posture29_v9_env_cfg()
  cfg.rewards.pop("target_hands_away_from_torso", None)
  cfg.rewards.pop("target_upper_center_over_support", None)

  upper_targets = dict(
    zip(UPPER_BODY_JOINTS, HOST_TARGET_UPPER_DOF_POS, strict=True)
  )
  for name in (
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  ):
    upper_targets[name] = 0.0

  cfg.rewards["target_hands_beside_hips"] = RewardTermCfg(
    func=mdp.target_hands_beside_hips,
    weight=0.40,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "left_target": (0.0, 0.28, -0.30),
      "right_target": (0.0, -0.28, -0.30),
      "error_scale": 0.14,
      "torso_cfg": _bodies("torso_link"),
      "hand_cfg": _bodies("left_wrist_yaw_link", "right_wrist_yaw_link"),
    },
  )
  cfg.rewards["target_torso_tilt"] = RewardTermCfg(
    func=mdp.target_torso_tilt,
    weight=-0.30,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "torso_cfg": _bodies("torso_link"),
    },
  )
  cfg.rewards["target_upper_pose_linear"] = RewardTermCfg(
    func=mdp.target_dof_pos_deviation,
    weight=-0.20,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=tuple(upper_targets), preserve_order=True
      ),
      "target_pos": tuple(upper_targets.values()),
    },
  )
  cfg.rewards["target_foot_horizontal_speed"] = RewardTermCfg(
    func=mdp.target_site_horizontal_speed,
    weight=-0.05,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "site_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
    },
  )
  return cfg
