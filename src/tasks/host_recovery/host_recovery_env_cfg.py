"""HoST standing-up (recovery) task configuration.

Factory for the task that makes a humanoid get up off the floor. Robot-specific
settings live in ``config/<robot>/env_cfgs.py``.

The reward table below is HoST's ``G1Cfg`` verbatim, with the four group weights
folded in (see ``mdp/rewards.py`` for why that is exact for the additive groups
and why the ``task`` group is computed inside :func:`rewards.standup`)::

    weight = <HoST scale> * <group weight> * HOST_CONSTRAINT_DT

``HOST_CONSTRAINT_DT`` is the 0.02 s control period that HoST multiplies every
constraint scale by in ``_prepare_reward_function``.

Rewards HoST defines but that this port cannot express without an unverified
mjlab field are kept as commented-out entries carrying their exact HoST value,
so the table stays auditable and nothing silently disappears.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

import src.tasks.host_recovery.mdp as mdp

#: HoST's single-step actor observation is 76 dims on the 23-DoF G1, stacked
#: ``num_actor_history`` (= 6) times.
NUM_ONE_STEP_OBS = 76
NUM_ACTOR_HISTORY = 6


def make_host_recovery_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base HoST standing-up task configuration."""

  ##
  # Observations
  #
  # A single composite term: HoST builds one 76-dim vector, applies a
  # per-channel noise vector to it and zeroes the whole thing while the motors
  # are off. mjlab's per-term noise/scale cannot express that, so the term does
  # it internally and ``enable_corruption`` is off.
  #
  # The critic group is declared separately because mjlab needs it for the value
  # function. NOTE: HoST feeds the same noisy vector to actor and critic; here
  # the critic gets the clean vector instead of an independent noise draw.
  ##

  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "host": ObservationTermCfg(
          func=mdp.host_observation,
          params={"add_noise": True},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=NUM_ACTOR_HISTORY,
      history_ordering="time",
    ),
    "critic": ObservationGroupCfg(
      terms={
        "host": ObservationTermCfg(
          func=mdp.host_observation,
          params={"add_noise": False},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=NUM_ACTOR_HISTORY,
      history_ordering="time",
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "host_head_height": MetricsTermCfg(
      func=mdp.host_standup_progress,
      params={
        "body_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "foot_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
  }

  ##
  # Actions
  #
  # HoST's position target is incremental: ``target = dof_pos + action *
  # action_rescale``, re-based on the measured position every step, so the PD
  # error is always ``action * action_rescale`` and limb excursions accumulate
  # across steps. ``mdp.IncrementalJointPositionAction`` restores that
  # convention instead of mjlab's default-relative absolute target.
  #
  # ``scale`` starts at HoST's ``control.action_scale`` (= 1.0) and is decayed
  # to 0.25 by the ``action_scale`` curriculum term.
  ##

  actions = {
    "joint_pos": mdp.IncrementalJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=1.0,
      use_default_offset=False,
    ),
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_posture,
      mode="reset",
      params={
        "posture": "prone",  # HoST's ground task always starts prone.
        "height": mdp.DEFAULT_HEIGHT,
        "xy_range": 1.0,
      },
    ),
    "reset_joints": EventTermCfg(
      func=mdp.reset_joints_scaled,
      mode="reset",
      params={
        "position_scale": (0.9, 1.1),
        "position_offset": (-0.1, 0.1),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_host_metrics": EventTermCfg(
      func=mdp.reset_host_metrics,
      mode="reset",
      params={},
    ),
    "init_pull_force": EventTermCfg(
      func=mdp.init_pull_force,
      mode="startup",
      params={"force": mdp.PULL_FORCE},
    ),
    "apply_pull_force": EventTermCfg(
      func=mdp.apply_pull_force,
      mode="step",
      params={"force_when_down": True},
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.1, 1.0),  # HoST domain_rand.friction_range
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        # HoST domain_rand.com_displacement_range
        "ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)},
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    # ---------------- task (group weight 2.5, no dt factor) ----------------
    "standup": RewardTermCfg(
      func=mdp.standup,
      weight=1.0,  # The 2.5 group weight is applied inside mdp.standup.
      params={
        "group_weight": mdp.TASK_GROUP_WEIGHT,
        "orientation_threshold": 0.99,
        "target_head_height": 0.8,
        "target_head_margin": 1.0,
        "body_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "foot_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # ---------------- regu (group weight 0.1) ----------------
    "regu_dof_acc": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-6 * mdp.REGU_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
    ),
    "regu_action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.05 * mdp.REGU_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
    ),
    "regu_smoothness": RewardTermCfg(
      func=mdp.action_acc_l2,
      weight=-0.08 * mdp.REGU_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
    ),
    "regu_dof_vel": RewardTermCfg(
      func=mdp.regu_dof_vel,
      weight=-2e-3 * mdp.REGU_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "regu_upper_dof_vel": RewardTermCfg(
      func=mdp.regu_upper_dof_vel,
      weight=-2e-3 * mdp.REGU_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=())},  # Set per-robot.
    ),
    "regu_dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-150.0 * mdp.REGU_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
    ),
    # HoST regu_smoothness (-0.01): second-order action smoothness needs
    # ``last_last_actions``; mjlab only exposes ``last_action``.
    # "regu_smoothness": RewardTermCfg(func=..., weight=-0.01 * 0.1 * 0.02),
    # HoST regu_torques (-2.5e-6) and regu_joint_power (-2.5e-5): need the
    # applied actuator torque; verify the mjlab EntityData field name first.
    # "regu_torques": RewardTermCfg(func=..., weight=-2.5e-6 * 0.1 * 0.02),
    # "regu_joint_power": RewardTermCfg(func=..., weight=-2.5e-5 * 0.1 * 0.02),
    # HoST regu_joint_tracking_error (-0.00025): needs the PD target.
    # HoST regu_dof_vel_limits (-1): needs the soft joint velocity limits.
    # ---------------- style (group weight 1.0) ----------------
    "style_waist_deviation": RewardTermCfg(
      func=mdp.style_waist_deviation,
      weight=-10.0 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",))},
    ),
    "style_hip_yaw_deviation": RewardTermCfg(
      func=mdp.style_hip_yaw_deviation,
      weight=-12.0 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_yaw_joint",))},
    ),
    "style_hip_roll_deviation": RewardTermCfg(
      func=mdp.style_hip_roll_deviation,
      weight=-12.0 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_roll_joint",))},
    ),
    "style_shoulder_roll_deviation": RewardTermCfg(
      func=mdp.style_shoulder_roll_deviation,
      weight=-2.5 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={
        "left_cfg": SceneEntityCfg("robot", joint_names=("left_shoulder_roll_joint",)),
        "right_cfg": SceneEntityCfg("robot", joint_names=("right_shoulder_roll_joint",)),
      },
    ),
    "style_left_foot_displacement": RewardTermCfg(
      func=mdp.style_left_foot_displacement,
      weight=2.5 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={
        "phase3_height": 0.65,
        "foot_cfg": SceneEntityCfg("robot", body_names=("left_ankle_roll_link",)),
      },
    ),
    "style_right_foot_displacement": RewardTermCfg(
      func=mdp.style_right_foot_displacement,
      weight=2.5 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={
        "phase3_height": 0.65,
        "foot_cfg": SceneEntityCfg("robot", body_names=("right_ankle_roll_link",)),
      },
    ),
    "style_knee_deviation": RewardTermCfg(
      func=mdp.style_knee_deviation,
      weight=-1.5 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",))},
    ),
    "style_elbow_deviation": RewardTermCfg(
      func=mdp.style_elbow_deviation,
      weight=-2.0 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_elbow_joint",))},
    ),
    "style_shank_orientation": RewardTermCfg(
      func=mdp.style_shank_orientation,
      weight=14.0 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={
        "phase1_height": 0.45,
        "left_knee_cfg": SceneEntityCfg("robot", body_names=("left_knee_link",)),
        "left_foot_cfg": SceneEntityCfg("robot", body_names=("left_ankle_roll_link",)),
        "right_knee_cfg": SceneEntityCfg("robot", body_names=("right_knee_link",)),
        "right_foot_cfg": SceneEntityCfg("robot", body_names=("right_ankle_roll_link",)),
      },
    ),
    "style_ground_parallel": RewardTermCfg(
      func=mdp.style_ground_parallel,
      weight=25.0 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={
        "left_ankle_cfg": SceneEntityCfg("robot", body_names=("left_ankle_roll_link",)),
        "right_ankle_cfg": SceneEntityCfg("robot", body_names=("right_ankle_roll_link",)),
      },
    ),
    "style_feet_distance": RewardTermCfg(
      func=mdp.style_feet_distance,
      weight=-14.0 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={
        "left_foot_cfg": SceneEntityCfg("robot", body_names=("left_ankle_roll_link",)),
        "right_foot_cfg": SceneEntityCfg("robot", body_names=("right_ankle_roll_link",)),
      },
    ),
    "style_style_ang_vel_xy": RewardTermCfg(
      func=mdp.style_ang_vel_xy,
      weight=1.0 * mdp.STYLE_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"phase1_height": 0.45},
    ),
    # ---------------- target (group weight 1.0) ----------------
    "target_ang_vel_xy": RewardTermCfg(
      func=mdp.target_ang_vel_xy,
      weight=16.0 * mdp.TARGET_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"phase3_height": 0.65},
    ),
    "target_lin_vel_xy": RewardTermCfg(
      func=mdp.target_lin_vel_xy,
      weight=16.0 * mdp.TARGET_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"phase3_height": 0.65},
    ),
    "target_feet_height_var": RewardTermCfg(
      func=mdp.target_feet_height_var,
      weight=6.0 * mdp.TARGET_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={
        "phase3_height": 0.65,
        "left_foot_cfg": SceneEntityCfg("robot", body_names=("left_ankle_roll_link",)),
        "right_foot_cfg": SceneEntityCfg("robot", body_names=("right_ankle_roll_link",)),
      },
    ),
    "target_target_upper_dof_pos": RewardTermCfg(
      func=mdp.target_target_upper_dof_pos,
      weight=25.0 * mdp.TARGET_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={
        "phase3_height": 0.65,
        "asset_cfg": SceneEntityCfg("robot", joint_names=()),  # Set per-robot.
        # HoST rewrites this pose in init_state.target_joint_angles; leave None
        # to use the robot's default joint positions.
        "target_upper_dof_pos": None,
      },
    ),
    "target_target_orientation": RewardTermCfg(
      func=mdp.target_target_orientation,
      weight=16.0 * mdp.TARGET_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"phase3_height": 0.65, "sigma": -8.0},
    ),
    "target_target_base_height": RewardTermCfg(
      func=mdp.target_target_base_height,
      weight=16.0 * mdp.TARGET_GROUP_WEIGHT * mdp.HOST_CONSTRAINT_DT,
      params={"base_height_target": 0.75, "phase3_height": 0.65, "sigma": -25.0},
    ),
  }

  ##
  # Terminations
  #
  # No contact termination, on purpose: HoST's ground task sets
  # ``terminate_after_contacts_on = []`` so the robot may drag itself along the
  # floor while getting up.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "dof_vel_out": TerminationTermCfg(
      func=mdp.dof_velocity_out_of_bounds,
      params={"limit": mdp.DOF_VEL_LIMIT},
    ),
    "base_vel_out": TerminationTermCfg(
      func=mdp.base_velocity_out_of_bounds,
      params={"limit": mdp.BASE_VEL_LIMIT},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "action_scale": CurriculumTermCfg(
      func=mdp.action_scale_decay,
      params={"threshold_height": mdp.THRESHOLD_HEIGHT, "decay": 0.005, "min_scale": 0.25},
    ),
    "pull_force": CurriculumTermCfg(
      func=mdp.pull_force_decay,
      params={"threshold_height": mdp.THRESHOLD_HEIGHT, "decay": 20.0},
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=128,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
