"""V24 strong post-standing heading, stance, and torso stabilization."""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v8 import _bodies
from src.tasks.host_recovery.posture29_v16 import POST_MOTION_HEIGHT
from src.tasks.host_recovery.posture29_v23 import posture29_v23_env_cfg


def posture29_v24_env_cfg():
  cfg = posture29_v23_env_cfg()

  def feet():
    return SceneEntityCfg(
      "robot", site_names=("left_foot", "right_foot"), preserve_order=True
    )

  def ankles():
    return _bodies("left_ankle_roll_link", "right_ankle_roll_link")

  for reward in cfg.rewards.values():
    if "phase3_height" in reward.params:
      reward.params["phase3_height"] = POST_MOTION_HEIGHT

  cfg.events["reset_post_stand_foot_anchor"] = EventTermCfg(
    func=mdp.reset_post_stand_foot_anchor,
    mode="reset",
    params={},
  )

  cfg.rewards["target_yaw_rate"].weight = -3.00
  cfg.rewards["target_root_horizontal_speed"].weight = -1.50
  cfg.rewards["target_foot_horizontal_speed"].weight = -1.20
  cfg.rewards["target_leg_dof_vel"].weight = -0.0060
  cfg.rewards["target_hip_yaw_dof_vel"].weight = -0.30
  cfg.rewards["target_lateral_balance_dof_vel"].weight = -0.040
  cfg.rewards["target_torso_vertical"].weight = 1.50
  cfg.rewards["target_torso_tilt"].weight = -5.00
  cfg.rewards["target_torso_support_error"].weight = -6.00
  cfg.rewards["target_feet_side_by_side_error"].weight = -1.50
  cfg.rewards["target_feet_height_var"].weight = 0.75
  cfg.rewards["target_target_orientation"].weight = 1.00

  upper_chain = _bodies(
    "torso_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.rewards["target_heading_error"] = RewardTermCfg(
    func=mdp.target_heading_error,
    weight=-4.00,
    params={"phase3_height": POST_MOTION_HEIGHT},
  )
  cfg.rewards["target_feet_orientation_error"] = RewardTermCfg(
    func=mdp.target_feet_orientation_error,
    weight=-2.00,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "tilt_weight": 0.75,
      "symmetry_weight": 0.75,
      "foot_cfg": ankles(),
    },
  )
  cfg.rewards["target_foot_ground_error"] = RewardTermCfg(
    func=mdp.target_foot_ground_error,
    weight=-2.00,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "max_clearance": 0.04,
      "clearance_scale": 0.04,
      "foot_cfg": feet(),
    },
  )
  cfg.rewards["target_post_stand_foot_anchor_error"] = RewardTermCfg(
    func=mdp.target_post_stand_foot_anchor_error,
    weight=-2.00,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "settle_steps": 30,
      "anchor_scale": 0.08,
      "foot_cfg": feet(),
    },
  )
  cfg.rewards["target_upper_chain_support_error"] = RewardTermCfg(
    func=mdp.target_upper_center_support_error,
    weight=-4.00,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "upper_cfg": upper_chain,
      "ankle_cfg": ankles(),
    },
  )
  cfg.rewards["target_pelvis_support_error"] = RewardTermCfg(
    func=mdp.target_upper_center_support_error,
    weight=-4.00,
    params={
      "phase3_height": POST_MOTION_HEIGHT,
      "upper_cfg": _bodies("pelvis"),
      "ankle_cfg": ankles(),
    },
  )
  cfg.rewards["target_waist_roll_pitch_error"] = RewardTermCfg(
    func=mdp.target_dof_pos_deviation,
    weight=-1.50,
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
  return cfg
