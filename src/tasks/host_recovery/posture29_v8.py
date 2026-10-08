"""V8 post-stand rewards for elbow geometry and upper-body centering."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.flat29 import flat29_env_cfg


POST_STAND_HEIGHT = 0.72


def _bodies(*names: str) -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=names, preserve_order=True)


def posture29_v8_env_cfg():
  cfg = flat29_env_cfg(play=False)
  cfg.events["init_pull_force"].params["force"] = 0.0
  cfg.curriculum.pop("pull_force", None)
  cfg.curriculum.pop("action_scale", None)

  for reward in cfg.rewards.values():
    if "phase3_height" in reward.params:
      reward.params["phase3_height"] = POST_STAND_HEIGHT

  cfg.rewards["target_target_upper_dof_pos"].weight *= 1.25
  cfg.rewards["target_elbow_up"].weight = -0.006
  cfg.rewards["target_arm_twist"].weight = -0.004
  cfg.rewards["target_wrist_flip"].weight = -0.003
  cfg.rewards["target_leg_dof_vel"].weight = -0.001
  cfg.rewards["target_upper_dof_vel"].weight = -0.0005
  cfg.rewards["target_elbows_outside_torso"] = RewardTermCfg(
    func=mdp.target_elbows_outside_torso,
    weight=0.20,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "lateral_margin": 0.16,
      "error_scale": 0.08,
      "torso_cfg": _bodies("torso_link"),
      "elbow_cfg": _bodies("left_elbow_link", "right_elbow_link"),
    },
  )
  cfg.rewards["target_upper_center_over_support"] = RewardTermCfg(
    func=mdp.target_upper_center_over_support,
    weight=0.25,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "error_scale": 0.08,
      "upper_cfg": _bodies(
        "torso_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
      ),
      "ankle_cfg": _bodies(
        "left_ankle_roll_link", "right_ankle_roll_link"
      ),
    },
  )
  return cfg
