"""Unitree G1 (23-DoF) HoST standing-up environment configuration.

The 23-DoF G1 is used because HoST's own URDF is ``g1_23dof.urdf``: 12 leg
joints, ``waist_yaw_joint`` and 2 x (shoulder pitch/roll/yaw, elbow,
wrist_roll). ``src.assets.robots.G1_23DOF_ACTION_SCALE``, ``get_g1_23dof_robot_cfg`` and the
mjlab MJCF all agree with that joint set and with HoST's joint ordering.
"""

import os

from src.assets.robots import get_g1_23dof_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.host_recovery.host_recovery_env_cfg import make_host_recovery_env_cfg

#: G1 links/sites used by HoST's reward terms.
TORSO_BODY = "torso_link"  # stands in for HoST's ``keyframe_head``
FOOT_SITES = ("left_foot", "right_foot")
LEFT_ANKLE_BODY = "left_ankle_roll_link"
RIGHT_ANKLE_BODY = "right_ankle_roll_link"
LEFT_KNEE_BODY = "left_knee_link"
RIGHT_KNEE_BODY = "right_knee_link"

#: HoST ``init_state.target_joint_angles`` for the upper body, in mjlab's joint
#: order (waist, left arm, right arm). HoST encourages the arms to sit flat at
#: the sides once the robot is up.
UPPER_BODY_JOINTS = (
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
)
HOST_TARGET_UPPER_DOF_POS = (0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0)

#: HoST ``control.action_scale``; the ``action_scale`` curriculum decays it.
HOST_ACTION_SCALE = 1.0
#: Play pins the rescaler at the curriculum floor so evaluation matches the
#: final training regime (HoST's own play value is 0.3; ours decays to 0.25).
HOST_PLAY_ACTION_SCALE = 0.25


def unitree_g1_host_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 HoST standing-up configuration."""
  cfg = make_host_recovery_env_cfg()

  cfg.scene.entities = {"robot": get_g1_23dof_robot_cfg()}
  cfg.viewer.body_name = TORSO_BODY

  foot_geoms = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  ##
  # Rewards: fill in the per-robot body/joint names.
  ##

  cfg.rewards["standup"].params["body_cfg"] = SceneEntityCfg(
    "robot", body_names=(TORSO_BODY,)
  )
  cfg.rewards["standup"].params["foot_cfg"] = SceneEntityCfg(
    "robot", site_names=FOOT_SITES
  )

  cfg.rewards["style_left_foot_displacement"].params["foot_cfg"] = SceneEntityCfg(
    "robot", body_names=(LEFT_ANKLE_BODY,)
  )
  cfg.rewards["style_right_foot_displacement"].params["foot_cfg"] = SceneEntityCfg(
    "robot", body_names=(RIGHT_ANKLE_BODY,)
  )
  cfg.rewards["style_shank_orientation"].params["left_knee_cfg"] = SceneEntityCfg(
    "robot", body_names=(LEFT_KNEE_BODY,)
  )
  cfg.rewards["style_shank_orientation"].params["left_foot_cfg"] = SceneEntityCfg(
    "robot", body_names=(LEFT_ANKLE_BODY,)
  )
  cfg.rewards["style_shank_orientation"].params["right_knee_cfg"] = SceneEntityCfg(
    "robot", body_names=(RIGHT_KNEE_BODY,)
  )
  cfg.rewards["style_shank_orientation"].params["right_foot_cfg"] = SceneEntityCfg(
    "robot", body_names=(RIGHT_ANKLE_BODY,)
  )
  for name in (
    "style_ground_parallel",
    "style_feet_distance",
    "target_feet_height_var",
  ):
    cfg.rewards[name].params["left_ankle_cfg" if name == "style_ground_parallel" else "left_foot_cfg"] = SceneEntityCfg(
      "robot", body_names=(LEFT_ANKLE_BODY,)
    )
    cfg.rewards[name].params["right_ankle_cfg" if name == "style_ground_parallel" else "right_foot_cfg"] = SceneEntityCfg(
      "robot", body_names=(RIGHT_ANKLE_BODY,)
    )

  cfg.rewards["target_target_upper_dof_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=UPPER_BODY_JOINTS
  )
  cfg.rewards["regu_upper_dof_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=UPPER_BODY_JOINTS
  )
  cfg.rewards["target_target_upper_dof_pos"].params["target_upper_dof_pos"] = (
    HOST_TARGET_UPPER_DOF_POS
  )

  ##
  # Metrics follow the same torso/feet convention as the stand-up reward.
  ##

  cfg.metrics["host_head_height"].params["body_cfg"] = SceneEntityCfg(
    "robot", body_names=(TORSO_BODY,)
  )
  cfg.metrics["host_head_height"].params["foot_cfg"] = SceneEntityCfg(
    "robot", site_names=FOOT_SITES
  )

  ##
  # Domain randomisation events.
  ##

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  ##
  # Actions: HoST drives a scalar rescaler, not a per-joint one.
  #
  # ``src.assets.robots.G1_23DOF_ACTION_SCALE`` (the torque-normalised per-joint scale used by the
  # other tasks in this repository) is imported above and left unused on
  # purpose; swap it in here if you prefer that convention.
  ##

  cfg.actions["joint_pos"].scale = HOST_ACTION_SCALE

  ##
  # Start posture.
  #
  # HoST's ground task starts prone. The supine variant is selected with
  # ``HOST_POSTURE=supine`` so that train / play / eval / resume all agree
  # on the same start without anyone editing this file.
  ##

  # ``none`` selects the paper's "diverse postures" setting: every episode is
  # reset from a uniformly random corner, which is what makes the policy able
  # to recover after being knocked over rather than only from one start.
  posture_env = os.environ.get("HOST_POSTURE", "prone")
  cfg.events["reset_base"].params["posture"] = (
    None if posture_env == "none" else posture_env
  )

  # Optional: start the robot already resting on the floor instead of
  # Start the robot already resting on the floor instead of dropping from
  # HoST's 0.5 m pelvis height. ``height=None`` uses the measured settled
  # height for each posture (mdp.POSTURE_SETTLED_HEIGHTS); HOST_RESET_HEIGHT
  # overrides it with one fixed height for every posture.
  height_env = os.environ.get("HOST_RESET_HEIGHT")
  cfg.events["reset_base"].params["height"] = (
    None if height_env is None else float(height_env)
  )

  ##
  # Play mode overrides, mirroring HoST's ``play.py``.
  ##

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.curriculum = {}
    cfg.observations["actor"].terms["host"].params["add_noise"] = False
    cfg.actions["joint_pos"].scale = HOST_PLAY_ACTION_SCALE
    # Evaluation runs without HoST's pull-force training wheel.
    cfg.events.pop("init_pull_force", None)
    cfg.events.pop("apply_pull_force", None)
    # The start posture is set above from ``HOST_POSTURE``, so evaluation uses
    # the same posture the policy was trained on (the paper's "diverse
    # postures" evaluation needs a policy trained on diverse postures).

  return cfg
