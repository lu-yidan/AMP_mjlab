"""Development environment configuration for the G1 recovery migration.

This development task combines the verified robot, action, motion reset,
actor/critic observations, task rewards, timeout, basic randomization, and a
provisional disturbance selector.  Mixed disturbances and AMP remain.
"""

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.noise import UniformNoiseCfg

from g1recovery_amp.g1_model import (
    G1_KEY_BODY_NAMES,
    get_recovery_g1_robot_cfg,
    get_recovery_joint_position_action_cfg,
)
from g1recovery_amp.motion_assets import (
    CONVERTED_GET_UP_MOTIONS,
    GET_UP_DEFAULT_MOTION,
    GET_UP_TRAINING_MOTION_WEIGHTS,
)
from g1recovery_amp.tasks.recovery import mdp

RECOVERY_EPISODE_LENGTH_S = 10.0
MUJOCO_SAFE_GROUND_FRICTION = 1.0e-5
RECOVERY_SUSTAINED_LINEAR_ACCELERATION = 10.5
RECOVERY_SUSTAINED_RAMP_S = 0.1
PHYSICS_WARNING_ROOT_ANGULAR_SPEED = 50.0
PHYSICS_WARNING_JOINT_SPEED = 50.0
PHYSICS_WARNING_JOINT_ACCELERATION = 10_000.0
PHYSICS_WARNING_CONTACT_PENETRATION = 0.05
RecoveryDisturbanceMode = Literal["mixed", "base_push", "impulse", "sustained", "none"]


def g1_recovery_dev_env_cfg(
    *,
    play: bool = False,
    disturbance_mode: RecoveryDisturbanceMode = "mixed",
) -> ManagerBasedRlEnvCfg:
    """Create the first runnable G1 recovery development environment.

    Recovery model, action, observations, rewards, motion asset, physics
    timestep, and decimation are real migration inputs covered by target tests.
    The mixed robust scheduler and source-compatible get-up assistance are active
    for training; standalone disturbance modes remain available for diagnosis.
    """

    cfg = unitree_g1_flat_tracking_env_cfg(play=play)

    cfg.scene.entities = {"robot": get_recovery_g1_robot_cfg()}
    if cfg.scene.terrain is None:
        raise TypeError("expected the tracking scaffold to provide flat terrain")
    # Isaac Lab multiplies each randomized robot-shape friction by the ground's
    # fixed coefficient of 1.0.  MuJoCo combines equal-priority geom friction
    # with an elementwise maximum instead.  A near-zero ground baseline therefore
    # makes every robot-ground contact use the robot geom's sampled value.
    # 1e-5 is MuJoCo's minimum safe coefficient for frictional contacts.
    cfg.scene.terrain.geoms = (
        spec_cfg.GeomCfg(
            geom_names_expr=("terrain",),
            friction=(MUJOCO_SAFE_GROUND_FRICTION,),
        ),
    )
    cfg.actions = {"joint_pos": get_recovery_joint_position_action_cfg()}
    robot_metric_cfg = SceneEntityCfg("robot")
    cfg.metrics = {
        "max_root_angular_speed": MetricsTermCfg(
            func=mdp.max_root_angular_speed,
            params={"asset_cfg": robot_metric_cfg},
            reduce="max",
        ),
        "max_joint_speed": MetricsTermCfg(
            func=mdp.max_joint_speed,
            params={"asset_cfg": robot_metric_cfg},
            reduce="max",
        ),
        "max_joint_acceleration": MetricsTermCfg(
            func=mdp.max_joint_acceleration,
            params={"asset_cfg": robot_metric_cfg},
            reduce="max",
        ),
        "max_contact_penetration": MetricsTermCfg(
            func=mdp.max_contact_penetration,
            reduce="max",
        ),
        "batch_max_joint_acceleration": MetricsTermCfg(
            func=mdp.batch_max_joint_acceleration,
            params={"asset_cfg": robot_metric_cfg},
            reduce="max",
        ),
        "batch_max_contact_penetration": MetricsTermCfg(
            func=mdp.batch_max_contact_penetration,
            reduce="max",
        ),
        "unsafe_root_angular_speed": MetricsTermCfg(
            func=mdp.unsafe_root_angular_speed,
            params={
                "asset_cfg": robot_metric_cfg,
                "threshold": PHYSICS_WARNING_ROOT_ANGULAR_SPEED,
            },
            reduce="max",
        ),
        "unsafe_joint_speed": MetricsTermCfg(
            func=mdp.unsafe_joint_speed,
            params={
                "asset_cfg": robot_metric_cfg,
                "threshold": PHYSICS_WARNING_JOINT_SPEED,
            },
            reduce="max",
        ),
        "unsafe_joint_acceleration": MetricsTermCfg(
            func=mdp.unsafe_joint_acceleration,
            params={
                "asset_cfg": robot_metric_cfg,
                "threshold": PHYSICS_WARNING_JOINT_ACCELERATION,
            },
            reduce="max",
        ),
        "unsafe_contact_penetration": MetricsTermCfg(
            func=mdp.unsafe_contact_penetration,
            params={"threshold": PHYSICS_WARNING_CONTACT_PENETRATION},
            reduce="max",
        ),
        "unsafe_physics_state": MetricsTermCfg(
            func=mdp.unsafe_physics_state,
            params={
                "asset_cfg": robot_metric_cfg,
                "root_angular_speed_threshold": PHYSICS_WARNING_ROOT_ANGULAR_SPEED,
                "joint_speed_threshold": PHYSICS_WARNING_JOINT_SPEED,
                "joint_acceleration_threshold": PHYSICS_WARNING_JOINT_ACCELERATION,
                "contact_penetration_threshold": (
                    PHYSICS_WARNING_CONTACT_PENETRATION
                ),
            },
            reduce="max",
        ),
    }
    cfg.episode_length_s = RECOVERY_EPISODE_LENGTH_S
    # Get-up poses create more ground and self-contact candidates than ordinary
    # walking.  The tracking default of 35 overflowed during the first smoke run.
    cfg.sim.nconmax = 64

    motion_cmd = cfg.commands["motion"]
    if not isinstance(motion_cmd, MotionCommandCfg):
        raise TypeError("expected the tracking scaffold to provide MotionCommandCfg")
    cfg.commands["motion"] = mdp.RecoveryMotionCommandCfg(
        entity_name=motion_cmd.entity_name,
        resampling_time_range=motion_cmd.resampling_time_range,
        debug_vis=motion_cmd.debug_vis,
        motion_file=str(GET_UP_DEFAULT_MOTION),
        anchor_body_name=motion_cmd.anchor_body_name,
        body_names=motion_cmd.body_names,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        adaptive_kernel_size=motion_cmd.adaptive_kernel_size,
        adaptive_lambda=motion_cmd.adaptive_lambda,
        adaptive_uniform_ratio=motion_cmd.adaptive_uniform_ratio,
        adaptive_alpha=motion_cmd.adaptive_alpha,
        sampling_mode="start" if play else "uniform",
        motion_files=tuple(
            str(path) for path in CONVERTED_GET_UP_MOTIONS.values()
        ),
        motion_weights=tuple(
            GET_UP_TRAINING_MOTION_WEIGHTS[name]
            for name in CONVERTED_GET_UP_MOTIONS
        ),
        viz=motion_cmd.viz,
    )
    cfg.commands["get_up_assist_force"] = mdp.GetUpAssistForceCommandCfg(
        force=0.0 if play else mdp.INITIAL_GET_UP_ASSIST_FORCE,
    )

    # Isaac Lab 2.3.1 flattens history term by term.  Keeping these five terms
    # separate preserves the source checkpoint's exact 480-value input order.
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={
            "base_ang_vel": ObservationTermCfg(
                func=common_mdp.base_ang_vel,
                noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            ),
            "root_local_rot_tan_norm": ObservationTermCfg(
                func=mdp.root_local_rot_tan_norm,
                noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            ),
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos,
                noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            ),
            "joint_vel": ObservationTermCfg(
                func=mdp.joint_vel,
                noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
            ),
            "actions": ObservationTermCfg(func=common_mdp.last_action),
        },
        concatenate_terms=True,
        enable_corruption=not play,
        history_length=mdp.RECOVERY_POLICY_HISTORY_LENGTH,
        flatten_history_dim=True,
        nan_policy="error",
    )

    # The critic is used only while training.  It receives simulator-only state
    # that is useful for judging a pose, including six body positions expressed
    # relative to the robot root.  preserve_order keeps checkpoint semantics.
    critic_key_bodies = SceneEntityCfg(
        "robot",
        body_names=G1_KEY_BODY_NAMES,
        preserve_order=True,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(func=common_mdp.base_lin_vel),
            "base_ang_vel": ObservationTermCfg(func=common_mdp.base_ang_vel),
            "root_local_rot_tan_norm": ObservationTermCfg(
                func=mdp.root_local_rot_tan_norm
            ),
            "root_height": ObservationTermCfg(func=mdp.base_pos_z),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel),
            "actions": ObservationTermCfg(func=common_mdp.last_action),
            "key_body_pos_b": ObservationTermCfg(
                func=mdp.key_body_pos_b,
                params={"asset_cfg": critic_key_bodies},
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=mdp.RECOVERY_CRITIC_HISTORY_LENGTH,
        flatten_history_dim=True,
        nan_policy="error",
    )

    # AMP compares a chronological ten-frame window from the simulated robot
    # against an independently sampled ten-frame window from the demonstration.
    discriminator_key_bodies = SceneEntityCfg(
        "robot",
        body_names=G1_KEY_BODY_NAMES,
        preserve_order=True,
    )
    cfg.observations["disc"] = ObservationGroupCfg(
        terms={
            "root_local_rot_tan_norm": ObservationTermCfg(
                func=mdp.root_local_rot_tan_norm
            ),
            "base_ang_vel": ObservationTermCfg(func=common_mdp.base_ang_vel),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel),
            "key_body_pos_b": ObservationTermCfg(
                func=mdp.key_body_pos_b,
                params={"asset_cfg": discriminator_key_bodies},
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=mdp.AMP_DISCRIMINATOR_HISTORY_LENGTH,
        flatten_history_dim=False,
        nan_policy="error",
    )
    cfg.observations["disc_demo"] = ObservationGroupCfg(
        terms={
            "motion_window": ObservationTermCfg(
                func=mdp.amp_demo_observation,
                params={
                    "command_name": "motion",
                    "key_body_names": G1_KEY_BODY_NAMES,
                },
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="error",
    )
    cfg.recorders = {
        "amp_terminal_observation": RecorderTermCfg(
            func=mdp.AmpTerminalObservationRecorder,
            params={"asset_cfg": discriminator_key_bodies},
        )
    }

    # Four generic penalties discourage violent or unsafe control.  The five
    # recovery terms are gated above 0.65 m so lying still is not mistaken for
    # successfully standing still.
    cfg.rewards = {
        "joint_acc_l2": RewardTermCfg(
            func=common_mdp.joint_acc_l2,
            weight=-1.0e-7,
        ),
        "action_rate_l2": RewardTermCfg(
            func=common_mdp.action_rate_l2,
            weight=-0.005,
        ),
        "joint_torques_l2": RewardTermCfg(
            func=common_mdp.joint_torques_l2,
            weight=-2.0e-6,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=common_mdp.joint_pos_limits,
            weight=-10.0,
        ),
        "ang_vel_xy": RewardTermCfg(
            func=mdp.ang_vel_xy,
            weight=2.0,
            params={"target_base_height_phase3": mdp.RECOVERY_STANDUP_HEIGHT},
        ),
        "lin_vel_xy": RewardTermCfg(
            func=mdp.lin_vel_xy,
            weight=2.0,
            params={"target_base_height_phase3": mdp.RECOVERY_STANDUP_HEIGHT},
        ),
        "target_orientation": RewardTermCfg(
            func=mdp.target_orientation,
            weight=2.0,
            params={"target_base_height_phase3": mdp.RECOVERY_STANDUP_HEIGHT},
        ),
        "low_height_progress": RewardTermCfg(
            func=mdp.low_height_progress,
            weight=0.5,
            params={
                "floor_height": mdp.RECOVERY_LOW_HEIGHT_FLOOR,
                "standup_height": mdp.RECOVERY_STANDUP_HEIGHT,
            },
        ),
        "target_base_height": RewardTermCfg(
            func=mdp.target_base_height,
            weight=5.0,
            params={
                "base_height_target": mdp.RECOVERY_TARGET_BASE_HEIGHT,
                "target_base_height_phase3": mdp.RECOVERY_STANDUP_HEIGHT,
            },
        ),
        "target_joint_deviation_l2": RewardTermCfg(
            func=mdp.target_joint_deviation_l2,
            weight=-0.1,
            params={"target_base_height_phase3": mdp.RECOVERY_STANDUP_HEIGHT},
        ),
    }

    # The source get-up task ends only because its 10-second time budget is
    # exhausted.  Tracking-error terminations would cut off valid exploration.
    cfg.terminations = {
        "time_out": TerminationTermCfg(
            func=common_mdp.time_out,
            time_out=True,
        )
    }

    # Replace tracking randomization rather than selectively removing its terms.
    # These two startup events match the source get-up task's basic variations.
    cfg.events = {
        "physics_material": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "operation": "abs",
                "ranges": (0.3, 1.0),
                "shared_random": False,
            },
        ),
        "add_base_mass": EventTermCfg(
            func=mdp.randomize_body_mass_and_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("torso_link",),
                    preserve_order=True,
                ),
                "mass_delta_range": (-1.0, 1.0),
            },
        ),
        "get_up_assist_force": EventTermCfg(
            func=mdp.apply_get_up_assist_force,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("torso_link",),
                    preserve_order=True,
                ),
            },
        ),
    }
    cfg.curriculum = {}
    if not play:
        cfg.curriculum["get_up_assist_force_level"] = CurriculumTermCfg(
            func=mdp.get_up_assist_force_level,
            params={"reward_term_name": "target_base_height"},
        )
        cfg.events["actuator_gains"] = EventTermCfg(
            func=dr.pd_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "kp_range": (0.85, 1.15),
                "kd_range": (0.85, 1.15),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        if disturbance_mode == "mixed":
            cfg.events["mixed_disturbance"] = EventTermCfg(
                func=mdp.MixedRecoveryDisturbance,
                mode="step",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=(
                            "pelvis",
                            ".*_ankle_roll_link",
                            ".*_wrist_yaw_link",
                            ".*_shoulder_roll_link",
                        ),
                        preserve_order=True,
                    ),
                    "modes": {
                        "sustained": {
                            "enable": True,
                            "ratio": 0.30,
                            "linear_acceleration": (
                                -RECOVERY_SUSTAINED_LINEAR_ACCELERATION,
                                RECOVERY_SUSTAINED_LINEAR_ACCELERATION,
                            ),
                            "ramp_s": RECOVERY_SUSTAINED_RAMP_S,
                            "period_s": (1.0, 3.0),
                        },
                        "impulse": {
                            "enable": True,
                            "ratio": 0.30,
                            "force": (-90.0, 90.0),
                            "torque": (-18.0, 18.0),
                            "duration_s": 0.1,
                            "period_s": (1.0, 3.0),
                        },
                        "base_push": {
                            "enable": True,
                            "ratio": 0.25,
                            "vel": (-0.8, 0.8),
                            "ang_vel": (-1.0, 1.0),
                            "period_s": (2.0, 4.0),
                        },
                        "none": {"enable": True, "ratio": 0.15},
                    },
                },
            )
        elif disturbance_mode == "base_push":
            # Source robust training assigns this mode to 25% of environments.
            # It is exercised alone here before adding the mixed scheduler.
            cfg.events["base_velocity_push"] = EventTermCfg(
                func=mdp.base_velocity_push,
                mode="interval",
                interval_range_s=(2.0, 4.0),
                is_global_time=False,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "linear_velocity_range": (-0.8, 0.8),
                    "yaw_velocity_range": (-1.0, 1.0),
                },
            )
        elif disturbance_mode == "impulse":
            cfg.events["body_impulse"] = EventTermCfg(
                func=common_mdp.apply_body_impulse,
                mode="step",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=(
                            "pelvis",
                            ".*_ankle_roll_link",
                            ".*_wrist_yaw_link",
                            ".*_shoulder_roll_link",
                        ),
                        preserve_order=True,
                    ),
                    "force_range": (-90.0, 90.0),
                    "torque_range": (-18.0, 18.0),
                    "duration_s": (0.1, 0.1),
                    "cooldown_s": (1.0, 3.0),
                },
            )
        elif disturbance_mode == "sustained":
            cfg.events["sustained_body_wrench"] = EventTermCfg(
                func=mdp.SustainedBodyWrench,
                mode="step",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=(
                            "pelvis",
                            ".*_ankle_roll_link",
                            ".*_wrist_yaw_link",
                            ".*_shoulder_roll_link",
                        ),
                        preserve_order=True,
                    ),
                    "linear_acceleration_range": (
                        -RECOVERY_SUSTAINED_LINEAR_ACCELERATION,
                        RECOVERY_SUSTAINED_LINEAR_ACCELERATION,
                    ),
                    "ramp_s": RECOVERY_SUSTAINED_RAMP_S,
                    "period_s": (1.0, 3.0),
                },
            )
        elif disturbance_mode != "none":
            raise ValueError(f"unknown recovery disturbance mode: {disturbance_mode}")

    return cfg
