"""V26 upper-body-only correction on top of V25's stable stance."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v16 import POST_MOTION_HEIGHT
from src.tasks.host_recovery.posture29_v25 import posture29_v25_env_cfg


LEFT_ARM_JOINTS = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = (
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
LEFT_ARM_TARGET = (0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0)
RIGHT_ARM_TARGET = (0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0)


def posture29_v26_env_cfg():
  cfg = posture29_v25_env_cfg()

  cfg.rewards.pop("target_upper_pose_linear", None)
  cfg.rewards["target_torso_vertical"].weight = 1.25
  cfg.rewards["target_torso_tilt"].weight = -3.00
  cfg.rewards["target_torso_over_support"].weight = 0.75
  cfg.rewards["target_waist_roll_pitch_error"].weight = -1.00
  cfg.rewards["target_hands_beside_hips_error"].weight = -5.00
  cfg.rewards["target_elbows_outside_torso"].weight = 0.40
  cfg.rewards["target_elbow_flexion"].weight = -0.25

  cfg.rewards["target_left_arm_pose"] = RewardTermCfg(
    func=mdp.target_dof_pos_deviation,
    weight=-1.25,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=LEFT_ARM_JOINTS, preserve_order=True
      ),
      "target_pos": LEFT_ARM_TARGET,
    },
  )
  cfg.rewards["target_right_arm_pose"] = RewardTermCfg(
    func=mdp.target_dof_pos_deviation,
    weight=-1.25,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True
      ),
      "target_pos": RIGHT_ARM_TARGET,
    },
  )
  cfg.rewards["target_waist_pose"] = RewardTermCfg(
    func=mdp.target_dof_pos_deviation,
    weight=-1.25,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("waist_roll_joint", "waist_pitch_joint"),
        preserve_order=True,
      ),
      "target_pos": (0.0, 0.0),
    },
  )
  cfg.rewards["target_arm_stillness"] = RewardTermCfg(
    func=mdp.target_dof_stillness,
    weight=0.30,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "velocity_scale": 0.75,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS,
        preserve_order=True,
      ),
    },
  )
  return cfg
