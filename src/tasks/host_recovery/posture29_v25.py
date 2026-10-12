"""V25 observable, bounded correction for V24's rotational gait."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v8 import _bodies
from src.tasks.host_recovery.posture29_v16 import POST_MOTION_HEIGHT
from src.tasks.host_recovery.posture29_v24 import posture29_v24_env_cfg


def posture29_v25_env_cfg():
  cfg = posture29_v24_env_cfg()

  for name in (
    "target_heading_error",
    "target_feet_orientation_error",
    "target_foot_ground_error",
    "target_post_stand_foot_anchor_error",
    "target_upper_chain_support_error",
    "target_pelvis_support_error",
  ):
    cfg.rewards.pop(name, None)
  cfg.events.pop("reset_post_stand_foot_anchor", None)

  cfg.rewards["target_root_horizontal_speed"].weight = 0.0
  cfg.rewards["target_yaw_rate"].weight = -1.50
  cfg.rewards["target_foot_horizontal_speed"].weight = -0.35
  cfg.rewards["target_leg_dof_vel"].weight = -0.0035
  cfg.rewards["target_hip_yaw_dof_vel"].weight = -0.06
  cfg.rewards["target_lateral_balance_dof_vel"].weight = -0.010
  cfg.rewards["target_torso_vertical"].weight = 0.75
  cfg.rewards["target_torso_tilt"].weight = -2.00
  cfg.rewards["target_torso_support_error"].weight = 0.0
  cfg.rewards["target_feet_side_by_side_error"].weight = 0.0
  cfg.rewards["target_feet_height_var"].weight = 0.40
  cfg.rewards["target_target_orientation"].weight = 0.75
  cfg.rewards["target_waist_roll_pitch_error"].weight = -0.40

  feet = SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot"), preserve_order=True
  )
  lateral_joints = (
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
  )

  cfg.rewards["target_feet_side_by_side_huber_error"] = RewardTermCfg(
    func=mdp.target_feet_side_by_side_huber_error,
    weight=-0.40,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "target_lateral_separation": 0.24,
      "forward_scale": 0.08,
      "height_scale": 0.03,
      "lateral_scale": 0.08,
      "foot_cfg": feet,
    },
  )
  cfg.rewards["target_torso_over_support"] = RewardTermCfg(
    func=mdp.target_upper_center_over_support,
    weight=0.50,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "error_scale": 0.08,
      "upper_cfg": _bodies("torso_link"),
      "ankle_cfg": _bodies("left_ankle_roll_link", "right_ankle_roll_link"),
    },
  )
  cfg.rewards["target_pelvis_over_support"] = RewardTermCfg(
    func=mdp.target_upper_center_over_support,
    weight=0.40,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "error_scale": 0.08,
      "upper_cfg": _bodies("pelvis"),
      "ankle_cfg": _bodies("left_ankle_roll_link", "right_ankle_roll_link"),
    },
  )
  cfg.rewards["target_lateral_joint_pose"] = RewardTermCfg(
    func=mdp.target_dof_pos_deviation,
    weight=-0.30,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=lateral_joints, preserve_order=True
      ),
      "target_pos": (0.0,) * len(lateral_joints),
    },
  )
  cfg.rewards["target_lateral_joint_stillness"] = RewardTermCfg(
    func=mdp.target_dof_stillness,
    weight=0.30,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "velocity_scale": 1.0,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=lateral_joints, preserve_order=True
      ),
    },
  )
  cfg.rewards["target_left_hip_yaw_velocity"] = RewardTermCfg(
    func=mdp.target_dof_vel,
    weight=-0.06,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=("left_hip_yaw_joint",), preserve_order=True
      ),
    },
  )
  cfg.rewards["target_lateral_action"] = RewardTermCfg(
    func=mdp.target_action_l2,
    weight=-0.15,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "action_indices": (1, 2, 5, 7, 8, 11, 12),
    },
  )
  cfg.rewards["target_left_hip_yaw_action"] = RewardTermCfg(
    func=mdp.target_action_l2,
    weight=-0.10,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "action_indices": (2,),
    },
  )
  return cfg
