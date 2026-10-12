"""V27 upper-body objective isolated from frozen lower-body reward variance."""

from src.tasks.host_recovery.posture29_v26 import posture29_v26_env_cfg


FROZEN_LOWER_BODY_TERMS = (
  "target_yaw_rate",
  "target_root_horizontal_speed",
  "target_foot_horizontal_speed",
  "target_leg_dof_vel",
  "target_hip_yaw_dof_vel",
  "target_lateral_balance_dof_vel",
  "target_feet_height_var",
  "target_feet_side_by_side_error",
  "target_feet_side_by_side_huber_error",
  "target_pelvis_over_support",
  "target_lateral_joint_pose",
  "target_lateral_joint_stillness",
  "target_left_hip_yaw_velocity",
  "target_lateral_action",
  "target_left_hip_yaw_action",
)


def posture29_v27_env_cfg():
  cfg = posture29_v26_env_cfg()
  for name in FROZEN_LOWER_BODY_TERMS:
    if name in cfg.rewards:
      cfg.rewards[name].weight = 0.0

  cfg.rewards["target_left_arm_pose"].weight = -1.50
  cfg.rewards["target_right_arm_pose"].weight = -2.00
  cfg.rewards["target_waist_pose"].weight = -2.00
  cfg.rewards["target_arm_stillness"].weight = 0.20
  return cfg
