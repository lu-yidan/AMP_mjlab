"""Conservative 29-DoF continuation with post-stand limb alignment and stability."""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.flat29 import LEG_JOINTS, flat29_env_cfg


def _bodies(*names: str) -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=list(names))


def directional29_env_cfg():
  cfg = flat29_env_cfg(play=False)
  cfg.events["init_pull_force"].params["force"] = 0.0
  cfg.curriculum.pop("pull_force", None)

  shoulders = _bodies("left_shoulder_yaw_link", "right_shoulder_yaw_link")
  elbows = _bodies("left_elbow_link", "right_elbow_link")
  wrists = _bodies("left_wrist_roll_link", "right_wrist_roll_link")
  hands = _bodies("left_wrist_yaw_link", "right_wrist_yaw_link")
  hips = _bodies("left_hip_yaw_link", "right_hip_yaw_link")
  knees = _bodies("left_knee_link", "right_knee_link")
  ankles = _bodies("left_ankle_roll_link", "right_ankle_roll_link")

  cfg.rewards["target_upper_forearm_alignment"] = RewardTermCfg(
    func=mdp.target_chain_alignment,
    weight=0.04,
    params={
      "phase3_height": 0.65,
      "proximal_cfg": shoulders,
      "middle_cfg": elbows,
      "distal_cfg": wrists,
    },
  )
  cfg.rewards["target_forearm_hand_alignment"] = RewardTermCfg(
    func=mdp.target_chain_alignment,
    weight=0.04,
    params={
      "phase3_height": 0.65,
      "proximal_cfg": elbows,
      "middle_cfg": wrists,
      "distal_cfg": hands,
    },
  )
  cfg.rewards["target_leg_vertical"] = RewardTermCfg(
    func=mdp.target_leg_vertical,
    weight=0.08,
    params={
      "phase3_height": 0.65,
      "hip_cfg": hips,
      "knee_cfg": knees,
      "ankle_cfg": ankles,
    },
  )
  cfg.rewards["target_leg_stillness"] = RewardTermCfg(
    func=mdp.target_dof_stillness,
    weight=0.06,
    params={
      "phase3_height": 0.65,
      "velocity_scale": 1.5,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=LEG_JOINTS, preserve_order=True
      ),
    },
  )
  cfg.rewards["target_leg_dof_vel"].weight = -0.0012
  return cfg
