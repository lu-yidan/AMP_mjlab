"""Configuration checks for the provisional recovery development task."""

import math
import unittest

from mjlab.envs import mdp as common_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

from g1recovery_amp.g1_model import G1_KEY_BODY_NAMES, G1_RECOVERY_ACTION_SCALE
from g1recovery_amp.motion_assets import (
    CONVERTED_GET_UP_MOTIONS,
    GET_UP_DEFAULT_MOTION,
    GET_UP_TRAINING_MOTION_WEIGHTS,
)
from g1recovery_amp.tasks.recovery import DEV_TASK_ID
from g1recovery_amp.tasks.recovery.env_cfg import (
    MUJOCO_SAFE_GROUND_FRICTION,
    PHYSICS_WARNING_CONTACT_PENETRATION,
    PHYSICS_WARNING_JOINT_ACCELERATION,
    PHYSICS_WARNING_JOINT_SPEED,
    PHYSICS_WARNING_ROOT_ANGULAR_SPEED,
    RECOVERY_EPISODE_LENGTH_S,
    g1_recovery_dev_env_cfg,
)
from g1recovery_amp.tasks.recovery.mdp import (
    AMP_DISCRIMINATOR_HISTORY_LENGTH,
    AMP_REFERENCE_STEPS,
    INITIAL_GET_UP_ASSIST_FORCE,
    RECOVERY_CRITIC_HISTORY_LENGTH,
    RECOVERY_LOW_HEIGHT_FLOOR,
    RECOVERY_POLICY_HISTORY_LENGTH,
    RECOVERY_ROOT_HEIGHT_OFFSET,
    RECOVERY_STANDUP_HEIGHT,
    RECOVERY_TARGET_BASE_HEIGHT,
    AmpTerminalObservationRecorder,
    GetUpAssistForceCommandCfg,
    MixedRecoveryDisturbance,
    RecoveryMotionCommandCfg,
    SustainedBodyWrench,
    amp_demo_observation,
    apply_get_up_assist_force,
    base_velocity_push,
    batch_max_contact_penetration,
    batch_max_joint_acceleration,
    get_up_assist_force_level,
    max_contact_penetration,
    max_joint_acceleration,
    max_joint_speed,
    max_root_angular_speed,
    randomize_body_mass_and_inertia,
    unsafe_contact_penetration,
    unsafe_joint_acceleration,
    unsafe_joint_speed,
    unsafe_physics_state,
    unsafe_root_angular_speed,
)


class TestRecoveryDevEnvCfg(unittest.TestCase):
    def test_recovery_task_is_registered(self) -> None:
        self.assertIn(DEV_TASK_ID, list_tasks())

    def test_config_connects_verified_robot_action_and_motion(self) -> None:
        cfg = g1_recovery_dev_env_cfg(play=True)

        self.assertEqual(tuple(cfg.scene.entities), ("robot",))
        self.assertEqual(cfg.sim.mujoco.timestep, 0.005)
        self.assertEqual(cfg.decimation, 4)
        self.assertAlmostEqual(cfg.sim.mujoco.timestep * cfg.decimation, 0.02)
        self.assertEqual(cfg.episode_length_s, RECOVERY_EPISODE_LENGTH_S)
        self.assertEqual(
            math.ceil(
                cfg.episode_length_s / (cfg.sim.mujoco.timestep * cfg.decimation)
            ),
            500,
        )
        self.assertEqual(cfg.sim.nconmax, 64)
        assert cfg.scene.terrain is not None
        self.assertEqual(len(cfg.scene.terrain.geoms), 1)
        ground_geom = cfg.scene.terrain.geoms[0]
        self.assertEqual(ground_geom.geom_names_expr, ("terrain",))
        self.assertEqual(
            ground_geom.friction,
            (MUJOCO_SAFE_GROUND_FRICTION,),
        )

        expected_metrics = {
            "max_root_angular_speed": max_root_angular_speed,
            "max_joint_speed": max_joint_speed,
            "max_joint_acceleration": max_joint_acceleration,
            "max_contact_penetration": max_contact_penetration,
            "batch_max_joint_acceleration": batch_max_joint_acceleration,
            "batch_max_contact_penetration": batch_max_contact_penetration,
            "unsafe_root_angular_speed": unsafe_root_angular_speed,
            "unsafe_joint_speed": unsafe_joint_speed,
            "unsafe_joint_acceleration": unsafe_joint_acceleration,
            "unsafe_contact_penetration": unsafe_contact_penetration,
            "unsafe_physics_state": unsafe_physics_state,
        }
        self.assertEqual(tuple(cfg.metrics), tuple(expected_metrics))
        for name, function in expected_metrics.items():
            metric = cfg.metrics[name]
            self.assertIsInstance(metric, MetricsTermCfg)
            self.assertIs(metric.func, function)
            self.assertEqual(metric.reduce, "max")
        warning_params = cfg.metrics["unsafe_physics_state"].params
        self.assertEqual(
            warning_params["root_angular_speed_threshold"],
            PHYSICS_WARNING_ROOT_ANGULAR_SPEED,
        )
        self.assertEqual(
            warning_params["joint_speed_threshold"],
            PHYSICS_WARNING_JOINT_SPEED,
        )
        self.assertEqual(
            warning_params["joint_acceleration_threshold"],
            PHYSICS_WARNING_JOINT_ACCELERATION,
        )
        self.assertEqual(
            warning_params["contact_penetration_threshold"],
            PHYSICS_WARNING_CONTACT_PENETRATION,
        )

        action = cfg.actions["joint_pos"]
        assert isinstance(action, JointPositionActionCfg)
        self.assertEqual(action.scale, G1_RECOVERY_ACTION_SCALE)
        self.assertTrue(action.use_default_offset)

        motion = cfg.commands["motion"]
        assert isinstance(motion, RecoveryMotionCommandCfg)
        self.assertEqual(
            motion.motion_file,
            str(GET_UP_DEFAULT_MOTION),
        )
        self.assertEqual(
            motion.motion_files,
            tuple(str(path) for path in CONVERTED_GET_UP_MOTIONS.values()),
        )
        self.assertEqual(
            motion.motion_weights,
            tuple(
                GET_UP_TRAINING_MOTION_WEIGHTS[name]
                for name in CONVERTED_GET_UP_MOTIONS
            ),
        )
        self.assertEqual(motion.anchor_body_name, "torso_link")
        self.assertEqual(motion.sampling_mode, "start")
        self.assertEqual(motion.pose_range, {})
        self.assertEqual(motion.velocity_range, {})
        self.assertEqual(motion.joint_position_range, (0.0, 0.0))
        self.assertEqual(motion.root_height_offset, RECOVERY_ROOT_HEIGHT_OFFSET)
        self.assertEqual(motion.future_reference_steps, AMP_REFERENCE_STEPS)
        assist_command = cfg.commands["get_up_assist_force"]
        assert isinstance(assist_command, GetUpAssistForceCommandCfg)
        self.assertEqual(assist_command.force, 0.0)

        actor_obs = cfg.observations["actor"]
        self.assertIsInstance(actor_obs, ObservationGroupCfg)
        self.assertEqual(
            tuple(actor_obs.terms),
            (
                "base_ang_vel",
                "root_local_rot_tan_norm",
                "joint_pos",
                "joint_vel",
                "actions",
            ),
        )
        self.assertEqual(actor_obs.history_length, RECOVERY_POLICY_HISTORY_LENGTH)
        self.assertTrue(actor_obs.flatten_history_dim)
        self.assertFalse(actor_obs.enable_corruption)

        critic_obs = cfg.observations["critic"]
        self.assertIsInstance(critic_obs, ObservationGroupCfg)
        self.assertEqual(
            tuple(critic_obs.terms),
            (
                "base_lin_vel",
                "base_ang_vel",
                "root_local_rot_tan_norm",
                "root_height",
                "joint_pos",
                "joint_vel",
                "actions",
                "key_body_pos_b",
            ),
        )
        self.assertEqual(critic_obs.history_length, RECOVERY_CRITIC_HISTORY_LENGTH)
        self.assertTrue(critic_obs.flatten_history_dim)
        self.assertFalse(critic_obs.enable_corruption)
        key_body_cfg = critic_obs.terms["key_body_pos_b"].params["asset_cfg"]
        self.assertEqual(tuple(key_body_cfg.body_names), G1_KEY_BODY_NAMES)
        self.assertTrue(key_body_cfg.preserve_order)

        disc_obs = cfg.observations["disc"]
        self.assertIsInstance(disc_obs, ObservationGroupCfg)
        self.assertEqual(
            tuple(disc_obs.terms),
            (
                "root_local_rot_tan_norm",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
                "key_body_pos_b",
            ),
        )
        self.assertEqual(disc_obs.history_length, AMP_DISCRIMINATOR_HISTORY_LENGTH)
        self.assertFalse(disc_obs.flatten_history_dim)
        self.assertFalse(disc_obs.enable_corruption)
        disc_key_body_cfg = disc_obs.terms["key_body_pos_b"].params["asset_cfg"]
        self.assertEqual(tuple(disc_key_body_cfg.body_names), G1_KEY_BODY_NAMES)

        disc_demo_obs = cfg.observations["disc_demo"]
        self.assertIsInstance(disc_demo_obs, ObservationGroupCfg)
        self.assertEqual(tuple(disc_demo_obs.terms), ("motion_window",))
        self.assertIs(disc_demo_obs.terms["motion_window"].func, amp_demo_observation)
        self.assertEqual(
            disc_demo_obs.terms["motion_window"].params["key_body_names"],
            G1_KEY_BODY_NAMES,
        )
        self.assertEqual(tuple(cfg.recorders), ("amp_terminal_observation",))
        terminal_recorder = cfg.recorders["amp_terminal_observation"]
        self.assertIsInstance(terminal_recorder, RecorderTermCfg)
        self.assertIs(terminal_recorder.func, AmpTerminalObservationRecorder)
        terminal_asset_cfg = terminal_recorder.params["asset_cfg"]
        self.assertEqual(tuple(terminal_asset_cfg.body_names), G1_KEY_BODY_NAMES)
        self.assertTrue(terminal_asset_cfg.preserve_order)

        expected_reward_weights = {
            "joint_acc_l2": -1.0e-7,
            "action_rate_l2": -0.005,
            "joint_torques_l2": -2.0e-6,
            "joint_pos_limits": -10.0,
            "ang_vel_xy": 2.0,
            "lin_vel_xy": 2.0,
            "target_orientation": 2.0,
            "low_height_progress": 0.5,
            "target_base_height": 5.0,
            "target_joint_deviation_l2": -0.1,
        }
        self.assertEqual(tuple(cfg.rewards), tuple(expected_reward_weights))
        for name, expected_weight in expected_reward_weights.items():
            reward = cfg.rewards[name]
            self.assertIsInstance(reward, RewardTermCfg)
            self.assertEqual(reward.weight, expected_weight)
        self.assertEqual(
            cfg.rewards["target_base_height"].params,
            {
                "base_height_target": RECOVERY_TARGET_BASE_HEIGHT,
                "target_base_height_phase3": RECOVERY_STANDUP_HEIGHT,
            },
        )
        self.assertEqual(
            cfg.rewards["low_height_progress"].params,
            {
                "floor_height": RECOVERY_LOW_HEIGHT_FLOOR,
                "standup_height": RECOVERY_STANDUP_HEIGHT,
            },
        )

        self.assertEqual(tuple(cfg.terminations), ("time_out",))
        time_out = cfg.terminations["time_out"]
        self.assertIsInstance(time_out, TerminationTermCfg)
        self.assertIs(time_out.func, common_mdp.time_out)
        self.assertTrue(time_out.time_out)
        self.assertEqual(time_out.params, {})

        self.assertEqual(
            tuple(cfg.events),
            ("physics_material", "add_base_mass", "get_up_assist_force"),
        )
        physics_material = cfg.events["physics_material"]
        self.assertIsInstance(physics_material, EventTermCfg)
        self.assertEqual(physics_material.mode, "startup")
        self.assertIs(physics_material.func, dr.geom_friction)
        self.assertEqual(physics_material.params["ranges"], (0.3, 1.0))
        self.assertEqual(physics_material.params["operation"], "abs")
        self.assertFalse(physics_material.params["shared_random"])
        material_asset_cfg = physics_material.params["asset_cfg"]
        self.assertEqual(material_asset_cfg.geom_names, ".*")

        add_base_mass = cfg.events["add_base_mass"]
        self.assertIsInstance(add_base_mass, EventTermCfg)
        self.assertEqual(add_base_mass.mode, "startup")
        self.assertIs(add_base_mass.func, randomize_body_mass_and_inertia)
        self.assertEqual(add_base_mass.params["mass_delta_range"], (-1.0, 1.0))
        mass_asset_cfg = add_base_mass.params["asset_cfg"]
        self.assertEqual(tuple(mass_asset_cfg.body_names), ("torso_link",))
        self.assertTrue(mass_asset_cfg.preserve_order)

        assist_event = cfg.events["get_up_assist_force"]
        self.assertEqual(assist_event.mode, "reset")
        self.assertIs(assist_event.func, apply_get_up_assist_force)
        assist_asset_cfg = assist_event.params["asset_cfg"]
        self.assertEqual(tuple(assist_asset_cfg.body_names), ("torso_link",))
        self.assertEqual(cfg.curriculum, {})

    def test_training_reset_uniformly_samples_valid_reference_frames(self) -> None:
        cfg = g1_recovery_dev_env_cfg(play=False)
        motion = cfg.commands["motion"]
        assert isinstance(motion, RecoveryMotionCommandCfg)

        self.assertEqual(motion.sampling_mode, "uniform")
        self.assertEqual(motion.pose_range, {})
        self.assertEqual(motion.velocity_range, {})
        self.assertEqual(motion.joint_position_range, (0.0, 0.0))
        self.assertEqual(cfg.episode_length_s, RECOVERY_EPISODE_LENGTH_S)
        assist_command = cfg.commands["get_up_assist_force"]
        assert isinstance(assist_command, GetUpAssistForceCommandCfg)
        self.assertEqual(assist_command.force, INITIAL_GET_UP_ASSIST_FORCE)

        assist_curriculum = cfg.curriculum["get_up_assist_force_level"]
        self.assertIsInstance(assist_curriculum, CurriculumTermCfg)
        self.assertIs(assist_curriculum.func, get_up_assist_force_level)
        self.assertEqual(
            assist_curriculum.params, {"reward_term_name": "target_base_height"}
        )

        self.assertEqual(
            tuple(cfg.events),
            (
                "physics_material",
                "add_base_mass",
                "get_up_assist_force",
                "actuator_gains",
                "mixed_disturbance",
            ),
        )
        actuator_gains = cfg.events["actuator_gains"]
        self.assertIsInstance(actuator_gains, EventTermCfg)
        self.assertEqual(actuator_gains.mode, "reset")
        self.assertIs(actuator_gains.func, dr.pd_gains)
        self.assertEqual(actuator_gains.params["kp_range"], (0.85, 1.15))
        self.assertEqual(actuator_gains.params["kd_range"], (0.85, 1.15))
        self.assertEqual(actuator_gains.params["operation"], "scale")
        self.assertEqual(actuator_gains.params["distribution"], "uniform")
        gains_asset_cfg = actuator_gains.params["asset_cfg"]
        self.assertEqual(gains_asset_cfg.actuator_names, ".*")

        mixed = cfg.events["mixed_disturbance"]
        self.assertIsInstance(mixed, EventTermCfg)
        self.assertEqual(mixed.mode, "step")
        self.assertIs(mixed.func, MixedRecoveryDisturbance)
        modes = mixed.params["modes"]
        self.assertEqual(
            {name: mode["ratio"] for name, mode in modes.items()},
            {"sustained": 0.30, "impulse": 0.30, "base_push": 0.25, "none": 0.15},
        )
        self.assertEqual(modes["sustained"]["period_s"], (1.0, 3.0))
        self.assertEqual(
            modes["sustained"]["linear_acceleration"], (-10.5, 10.5)
        )
        self.assertEqual(modes["sustained"]["ramp_s"], 0.1)
        self.assertEqual(modes["impulse"]["duration_s"], 0.1)
        self.assertEqual(modes["base_push"]["period_s"], (2.0, 4.0))
        mixed_asset_cfg = mixed.params["asset_cfg"]
        self.assertEqual(
            tuple(mixed_asset_cfg.body_names),
            (
                "pelvis",
                ".*_ankle_roll_link",
                ".*_wrist_yaw_link",
                ".*_shoulder_roll_link",
            ),
        )

    def test_base_push_can_be_isolated_during_migration(self) -> None:
        cfg = g1_recovery_dev_env_cfg(play=False, disturbance_mode="base_push")

        base_push = cfg.events["base_velocity_push"]
        self.assertIsInstance(base_push, EventTermCfg)
        self.assertEqual(base_push.mode, "interval")
        self.assertIs(base_push.func, base_velocity_push)
        self.assertEqual(base_push.interval_range_s, (2.0, 4.0))
        self.assertFalse(base_push.is_global_time)
        self.assertEqual(base_push.params["linear_velocity_range"], (-0.8, 0.8))
        self.assertEqual(base_push.params["yaw_velocity_range"], (-1.0, 1.0))

    def test_impulse_can_be_isolated_during_migration(self) -> None:
        cfg = g1_recovery_dev_env_cfg(play=False, disturbance_mode="impulse")

        self.assertEqual(
            tuple(cfg.events),
            (
                "physics_material",
                "add_base_mass",
                "get_up_assist_force",
                "actuator_gains",
                "body_impulse",
            ),
        )
        impulse = cfg.events["body_impulse"]
        self.assertIsInstance(impulse, EventTermCfg)
        self.assertEqual(impulse.mode, "step")
        self.assertIs(impulse.func, common_mdp.apply_body_impulse)
        self.assertEqual(impulse.params["force_range"], (-90.0, 90.0))
        self.assertEqual(impulse.params["torque_range"], (-18.0, 18.0))
        self.assertEqual(impulse.params["duration_s"], (0.1, 0.1))
        self.assertEqual(impulse.params["cooldown_s"], (1.0, 3.0))
        impulse_asset_cfg = impulse.params["asset_cfg"]
        self.assertEqual(
            tuple(impulse_asset_cfg.body_names),
            (
                "pelvis",
                ".*_ankle_roll_link",
                ".*_wrist_yaw_link",
                ".*_shoulder_roll_link",
            ),
        )
        self.assertTrue(impulse_asset_cfg.preserve_order)

    def test_disturbances_can_be_disabled_during_migration(self) -> None:
        cfg = g1_recovery_dev_env_cfg(play=False, disturbance_mode="none")

        self.assertEqual(
            tuple(cfg.events),
            (
                "physics_material",
                "add_base_mass",
                "get_up_assist_force",
                "actuator_gains",
            ),
        )

    def test_sustained_wrench_can_be_isolated_during_migration(self) -> None:
        cfg = g1_recovery_dev_env_cfg(play=False, disturbance_mode="sustained")

        sustained = cfg.events["sustained_body_wrench"]
        self.assertIsInstance(sustained, EventTermCfg)
        self.assertEqual(sustained.mode, "step")
        self.assertIs(sustained.func, SustainedBodyWrench)
        self.assertEqual(
            sustained.params["linear_acceleration_range"], (-10.5, 10.5)
        )
        self.assertEqual(sustained.params["ramp_s"], 0.1)
        self.assertEqual(sustained.params["period_s"], (1.0, 3.0))
        asset_cfg = sustained.params["asset_cfg"]
        self.assertEqual(
            tuple(asset_cfg.body_names),
            (
                "pelvis",
                ".*_ankle_roll_link",
                ".*_wrist_yaw_link",
                ".*_shoulder_roll_link",
            ),
        )

    def test_registry_returns_independent_configs(self) -> None:
        first = load_env_cfg(DEV_TASK_ID, play=True)
        second = load_env_cfg(DEV_TASK_ID, play=True)

        self.assertIsNot(first, second)
        self.assertIsNot(first.scene.entities["robot"], second.scene.entities["robot"])


if __name__ == "__main__":
    unittest.main()
