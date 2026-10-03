"""The A6 physics comparison must not change the policy control period."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from g1recovery_amp.tasks.recovery import (
    A6_G_MINUS_70_30_TASK_ID,
    A6_G_MINUS_SYMMETRIC_005_TASK_ID,
    A6_G_MINUS_TASK_ID,
    A6_G_PLUS_70_30_TASK_ID,
    A6_G_PLUS_CONSTRAINT_AWARE_001_FIXED6_TASK_ID,
    A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID,
    A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID,
    A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID,
    A6_G_PLUS_FORCE_HOLD_INVALID2_FIXED6_TASK_ID,
    A6_G_PLUS_PRONE_LATERAL_020_FIXED6_TASK_ID,
    A6_G_PLUS_STRICT_ESCAPE_EXPLORATION_025_FIXED6_TASK_ID,
    A6_G_PLUS_STRICT_ESCAPE_HEADWARD_FIXED6_TASK_ID,
    A6_G_PLUS_SYMMETRIC_005_TASK_ID,
    A6_G_PLUS_TASK_ID,
    A6_TASK_ID,
)
from g1recovery_amp.tasks.recovery.a6_env_cfg import (
    A6_ACTION_WARMUP_STEPS,
    A6_BALANCED_30_50_20_SCENE_WEIGHTS,
    A6_CONSTRAINED_70_30_SCENE_WEIGHTS,
    A6_CONSTRAINT_AWARE_001_AMP_SCALE,
    A6_CONSTRAINT_AWARE_AMP_SCALE,
    A6_CONTROL_DT,
    A6_DECIMATION,
    A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
    A6_FIXED_PLATE_MASS,
    A6_FORCE_LIMIT_HOLD_S,
    A6_INVALID_PLATE_TERMINATION_WEIGHT,
    A6_OBSTRUCTED_TASK_SCALE,
    A6_PHYSICS_TIMESTEP,
    A6_PLATE_NO_PROGRESS_WEIGHT,
    A6_PRONE_LATERAL_PROGRESS_WEIGHT,
    A6_RAW_ACTION_CLIP,
    A6_STRICT_ESCAPE_CLEAR_HOLD_S,
    A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
    A6_SYMMETRIC_OBSTRUCTED_SCALE,
    A6_TRAPPED_SHOULDERS_HEADWARD_WEIGHT,
    apply_a6_timing,
    g1_recovery_a6_env_cfg,
)
from g1recovery_amp.tasks.recovery.env_cfg import g1_recovery_dev_env_cfg
from g1recovery_amp.tasks.recovery.mdp import (
    A6_TASK_NAMES,
    A6_TASK_WEIGHTS,
    A6EntryProtectedJointPositionActionCfg,
    RecoveryMotionCommandCfg,
    a6_action_acceleration,
    a6_action_rate,
    a6_amp_reward_scale,
    a6_foot_motion,
    a6_invalid_plate,
    a6_joint_stall,
    a6_physics_cost,
    a6_scaled_cost,
    a6_task_component,
    plate_clearance,
    plate_completion,
    plate_force,
    plate_geometry_progress,
    plate_no_progress,
    plate_separation,
    prone_lateral_progress,
    reset_a6_dynamics,
    reset_from_a6_three_scene_curriculum,
    sample_a6_cost_substep,
    trapped_shoulders_headward,
)
from g1recovery_amp.tasks.recovery.rl.rl_cfg import (
    A6_STABLE_ENTROPY_COEF,
    A6_STABLE_LEARNING_RATE,
    A6_STABLE_STD_RANGE,
)


class TestA6Timing(unittest.TestCase):
    def test_invalid_plate_preserves_reason_across_automatic_reset(self) -> None:
        env = SimpleNamespace(
            num_envs=3,
            device=torch.device("cpu"),
            _a6_reset_scene=torch.tensor([1, 1, 2]),
            _a6_plate_invalid=torch.tensor([True, False, True]),
            _a6_plate_last_invalid_force=torch.tensor([True, False, False]),
            _a6_plate_last_invalid_penetration=torch.tensor([False, False, True]),
            _a6_plate_last_invalid_no_contact=torch.tensor([False, False, False]),
            _a6_plate_terminal_invalid_force=torch.zeros(3, dtype=torch.bool),
            _a6_plate_terminal_invalid_penetration=torch.zeros(3, dtype=torch.bool),
            _a6_plate_terminal_invalid_no_contact=torch.zeros(3, dtype=torch.bool),
        )

        with patch(
            "g1recovery_amp.tasks.recovery.mdp.a6_plate_state."
            "update_a6_plate_state"
        ):
            result = a6_invalid_plate(env)  # type: ignore[arg-type]

        self.assertTrue(torch.equal(result, torch.tensor([True, False, True])))
        self.assertTrue(
            torch.equal(
                env._a6_plate_terminal_invalid_force,
                torch.tensor([True, False, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                env._a6_plate_terminal_invalid_penetration,
                torch.tensor([False, False, True]),
            )
        )

        # Simulate reset-time code overwriting the live reason.  The preserved
        # terminal snapshot must remain unchanged for the rollout reporter.
        env._a6_plate_last_invalid_force.zero_()
        self.assertTrue(env._a6_plate_terminal_invalid_force[0])

    def test_a6_bridge_task_is_registered(self) -> None:
        self.assertTrue(
            {
                A6_TASK_ID,
                A6_G_MINUS_TASK_ID,
                A6_G_PLUS_TASK_ID,
                A6_G_MINUS_70_30_TASK_ID,
                A6_G_MINUS_SYMMETRIC_005_TASK_ID,
                A6_G_PLUS_70_30_TASK_ID,
                A6_G_PLUS_CONSTRAINT_AWARE_001_FIXED6_TASK_ID,
                A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID,
                A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID,
                A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID,
                A6_G_PLUS_FORCE_HOLD_INVALID2_FIXED6_TASK_ID,
                A6_G_PLUS_PRONE_LATERAL_020_FIXED6_TASK_ID,
                A6_G_PLUS_STRICT_ESCAPE_HEADWARD_FIXED6_TASK_ID,
                A6_G_PLUS_STRICT_ESCAPE_EXPLORATION_025_FIXED6_TASK_ID,
                A6_G_PLUS_SYMMETRIC_005_TASK_ID,
            }.issubset(list_tasks())
        )
        cfg = load_env_cfg(A6_TASK_ID, play=True)
        self.assertEqual(cfg.sim.mujoco.timestep, A6_PHYSICS_TIMESTEP)
        self.assertEqual(cfg.decimation, A6_DECIMATION)

        constrained_cfg = load_env_cfg(A6_G_PLUS_70_30_TASK_ID, play=False)
        self.assertEqual(
            constrained_cfg.events["a6_three_scene_curriculum_reset"].params[
                "scene_weights"
            ],
            A6_CONSTRAINED_70_30_SCENE_WEIGHTS,
        )

    def test_symmetric_task_scales_all_costs_but_not_escape_guidance(self) -> None:
        candidate = g1_recovery_a6_env_cfg(
            guidance="plus",
            scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
            symmetric_obstructed_scale=A6_SYMMETRIC_OBSTRUCTED_SCALE,
        )

        self.assertEqual(
            candidate.events["a6_three_scene_curriculum_reset"].params["scene_weights"],
            A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        )
        scale_group = candidate.observations["amp_reward_scale"]
        scale_term = scale_group.terms["scale"]
        self.assertIs(scale_term.func, a6_amp_reward_scale)
        self.assertEqual(
            scale_term.params,
            {"obstructed_scale": A6_SYMMETRIC_OBSTRUCTED_SCALE},
        )
        for index, name in enumerate(A6_TASK_NAMES):
            self.assertEqual(
                candidate.rewards[f"a6_{name}"].params,
                {
                    "index": index,
                    "obstructed_scale": A6_SYMMETRIC_OBSTRUCTED_SCALE,
                },
            )
        expected_cost_sources = {
            "a6_action_rate": ("action_rate", None),
            "a6_action_acceleration": ("action_acceleration", None),
            "a6_joint_limits": ("joint_limits", None),
            "a6_joint_acceleration": ("physics", 0),
            "a6_torque": ("physics", 1),
            "a6_joint_overspeed": ("physics", 2),
            "a6_joint_overpower": ("physics", 3),
            "a6_head_overspeed": ("physics", 4),
            "a6_sustained_effort": ("physics", 5),
            "a6_foot_motion": ("foot_motion", None),
            "a6_joint_stall": ("joint_stall", None),
            "plate_force": ("plate_force", None),
        }
        for name, (source, index) in expected_cost_sources.items():
            term = candidate.rewards[name]
            self.assertIs(term.func, a6_scaled_cost)
            self.assertEqual(term.params["cost_name"], source)
            self.assertEqual(
                term.params["obstructed_scale"], A6_SYMMETRIC_OBSTRUCTED_SCALE
            )
            self.assertEqual(term.params.get("index"), index)
        for name in (
            "plate_geometry_progress",
            "plate_clearance",
            "plate_completion",
            "plate_separation",
        ):
            self.assertIsNot(candidate.rewards[name].func, a6_scaled_cost)

        baseline = g1_recovery_a6_env_cfg(guidance="plus")
        self.assertNotIn("amp_reward_scale", baseline.observations)
        self.assertIs(baseline.rewards["a6_action_rate"].func, a6_action_rate)

        for task_id in (
            A6_G_MINUS_SYMMETRIC_005_TASK_ID,
            A6_G_PLUS_SYMMETRIC_005_TASK_ID,
        ):
            agent = load_rl_cfg(task_id)
            self.assertEqual(
                agent.actor.distribution_cfg["std_range"], A6_STABLE_STD_RANGE
            )
            self.assertEqual(agent.algorithm.entropy_coef, A6_STABLE_ENTROPY_COEF)
            self.assertEqual(agent.algorithm.learning_rate, A6_STABLE_LEARNING_RATE)
            self.assertEqual(agent.algorithm.schedule, "fixed")
            self.assertEqual(
                agent.algorithm.amp_cfg.style_reward_scale_obs_group,
                "amp_reward_scale",
            )
            self.assertEqual(
                agent.algorithm.amp_cfg.actor_std_range_on_load,
                A6_STABLE_STD_RANGE,
            )

    def test_constraint_aware_task_scales_amp_to_005_but_keeps_costs_full(self) -> None:
        candidate = load_env_cfg(A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID)

        scale_term = candidate.observations["amp_reward_scale"].terms["scale"]
        self.assertIs(scale_term.func, a6_amp_reward_scale)
        self.assertEqual(
            scale_term.params,
            {"obstructed_scale": A6_CONSTRAINT_AWARE_AMP_SCALE},
        )
        for index, name in enumerate(A6_TASK_NAMES):
            self.assertEqual(
                candidate.rewards[f"a6_{name}"].params,
                {"index": index, "obstructed_scale": A6_OBSTRUCTED_TASK_SCALE},
            )
        self.assertIs(candidate.rewards["a6_action_rate"].func, a6_action_rate)
        self.assertIs(
            candidate.rewards["a6_action_acceleration"].func,
            a6_action_acceleration,
        )
        self.assertIs(
            candidate.rewards["a6_joint_limits"].func, common_mdp.joint_pos_limits
        )
        self.assertIs(candidate.rewards["a6_joint_acceleration"].func, a6_physics_cost)
        self.assertIs(candidate.rewards["a6_foot_motion"].func, a6_foot_motion)
        self.assertIs(candidate.rewards["a6_joint_stall"].func, a6_joint_stall)
        self.assertIs(candidate.rewards["plate_force"].func, plate_force)

        agent = load_rl_cfg(A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID)
        self.assertEqual(
            agent.algorithm.amp_cfg.obstructed_style_reward_scale,
            A6_CONSTRAINT_AWARE_AMP_SCALE,
        )
        self.assertEqual(
            agent.algorithm.amp_cfg.style_reward_scale_obs_group,
            "amp_reward_scale",
        )
        self.assertEqual(
            agent.algorithm.amp_cfg.ppo_learning_rate_on_load,
            A6_STABLE_LEARNING_RATE,
        )

    def test_constraint_aware_fixed6_task_uses_six_kg_in_training(self) -> None:
        fixed = load_env_cfg(A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID)
        scheduled = load_env_cfg(A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID)
        fixed_reset = fixed.events["a6_three_scene_curriculum_reset"]
        scheduled_reset = scheduled.events["a6_three_scene_curriculum_reset"]

        self.assertEqual(fixed_reset.params["plate_mass_override"], A6_FIXED_PLATE_MASS)
        self.assertIsNone(scheduled_reset.params["plate_mass_override"])
        self.assertEqual(
            fixed_reset.params["scene_weights"], A6_BALANCED_30_50_20_SCENE_WEIGHTS
        )

    def test_constraint_aware_001_changes_only_obstructed_amp_scale(self) -> None:
        low_amp = load_env_cfg(A6_G_PLUS_CONSTRAINT_AWARE_001_FIXED6_TASK_ID)
        baseline = load_env_cfg(A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID)

        scale_term = low_amp.observations["amp_reward_scale"].terms["scale"]
        self.assertEqual(
            scale_term.params,
            {"obstructed_scale": A6_CONSTRAINT_AWARE_001_AMP_SCALE},
        )
        self.assertEqual(low_amp.rewards, baseline.rewards)
        self.assertEqual(low_amp.actions, baseline.actions)
        self.assertEqual(tuple(low_amp.scene.entities), tuple(baseline.scene.entities))
        self.assertEqual(low_amp.scene.sensors, baseline.scene.sensors)
        low_reset = low_amp.events["a6_three_scene_curriculum_reset"]
        baseline_reset = baseline.events["a6_three_scene_curriculum_reset"]
        self.assertIs(low_reset.func, baseline_reset.func)
        self.assertEqual(low_reset.params, baseline_reset.params)

        agent = load_rl_cfg(A6_G_PLUS_CONSTRAINT_AWARE_001_FIXED6_TASK_ID)
        self.assertEqual(
            agent.algorithm.amp_cfg.obstructed_style_reward_scale,
            A6_CONSTRAINT_AWARE_001_AMP_SCALE,
        )

    def test_no_progress_task_changes_only_the_reward_term(self) -> None:
        candidate = load_env_cfg(
            A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID
        )
        baseline = load_env_cfg(A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID)

        term = candidate.rewards["plate_no_progress"]
        self.assertIs(term.func, plate_no_progress)
        self.assertEqual(term.weight, A6_PLATE_NO_PROGRESS_WEIGHT)
        candidate_rewards = dict(candidate.rewards)
        candidate_rewards.pop("plate_no_progress")
        self.assertEqual(candidate_rewards, baseline.rewards)
        self.assertEqual(candidate.observations, baseline.observations)
        self.assertEqual(candidate.actions, baseline.actions)
        self.assertEqual(candidate.events, baseline.events)
        self.assertEqual(
            tuple(candidate.scene.entities), tuple(baseline.scene.entities)
        )
        self.assertEqual(candidate.scene.sensors, baseline.scene.sensors)

        agent = load_rl_cfg(A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID)
        self.assertEqual(
            agent.algorithm.amp_cfg.obstructed_style_reward_scale,
            A6_CONSTRAINT_AWARE_AMP_SCALE,
        )

    def test_strict_escape_headward_task_is_checkpoint_compatible(self) -> None:
        candidate = load_env_cfg(A6_G_PLUS_STRICT_ESCAPE_HEADWARD_FIXED6_TASK_ID)
        baseline = load_env_cfg(
            A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID
        )

        reset_params = candidate.events["a6_three_scene_curriculum_reset"].params
        self.assertEqual(
            reset_params["escape_min_planar_clearance"],
            A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        )
        self.assertEqual(
            reset_params["escape_clear_hold_steps"],
            round(A6_STRICT_ESCAPE_CLEAR_HOLD_S / A6_CONTROL_DT),
        )
        term = candidate.rewards["trapped_shoulders_headward"]
        self.assertIs(term.func, trapped_shoulders_headward)
        self.assertEqual(term.weight, A6_TRAPPED_SHOULDERS_HEADWARD_WEIGHT)
        candidate_rewards = dict(candidate.rewards)
        candidate_rewards.pop("trapped_shoulders_headward")
        self.assertEqual(candidate_rewards, baseline.rewards)
        self.assertEqual(candidate.observations, baseline.observations)
        self.assertEqual(candidate.actions, baseline.actions)

        agent = load_rl_cfg(A6_G_PLUS_STRICT_ESCAPE_HEADWARD_FIXED6_TASK_ID)
        self.assertEqual(
            agent.algorithm.amp_cfg.obstructed_style_reward_scale,
            A6_CONSTRAINT_AWARE_AMP_SCALE,
        )

    def test_strict_escape_exploration_task_removes_shoulder_reward_and_scales_only_exploration_costs(
        self,
    ) -> None:
        candidate = load_env_cfg(
            A6_G_PLUS_STRICT_ESCAPE_EXPLORATION_025_FIXED6_TASK_ID
        )

        self.assertNotIn("trapped_shoulders_headward", candidate.rewards)
        scaled_costs = {
            "a6_action_rate": ("action_rate", None),
            "a6_action_acceleration": ("action_acceleration", None),
            "a6_joint_overspeed": ("physics", 2),
            "a6_head_overspeed": ("physics", 4),
            "a6_sustained_effort": ("physics", 5),
            "a6_foot_motion": ("foot_motion", None),
            "a6_joint_stall": ("joint_stall", None),
        }
        for name, (source, index) in scaled_costs.items():
            term = candidate.rewards[name]
            self.assertIs(term.func, a6_scaled_cost)
            self.assertEqual(term.params["cost_name"], source)
            self.assertEqual(
                term.params["obstructed_scale"],
                A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
            )
            self.assertEqual(term.params.get("index"), index)

        full_strength_costs = {
            "a6_joint_limits": common_mdp.joint_pos_limits,
            "a6_joint_acceleration": a6_physics_cost,
            "a6_torque": a6_physics_cost,
            "a6_joint_overpower": a6_physics_cost,
            "plate_force": plate_force,
        }
        for name, function in full_strength_costs.items():
            self.assertIs(candidate.rewards[name].func, function)

        reset_params = candidate.events["a6_three_scene_curriculum_reset"].params
        self.assertEqual(
            reset_params["escape_min_planar_clearance"],
            A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        )
        self.assertEqual(
            reset_params["escape_clear_hold_steps"],
            round(A6_STRICT_ESCAPE_CLEAR_HOLD_S / A6_CONTROL_DT),
        )
        self.assertEqual(
            candidate.observations["amp_reward_scale"].terms["scale"].params,
            {"obstructed_scale": A6_CONSTRAINT_AWARE_AMP_SCALE},
        )

    def test_force_hold_task_changes_only_termination_and_terminal_penalty(
        self,
    ) -> None:
        candidate = load_env_cfg(A6_G_PLUS_FORCE_HOLD_INVALID2_FIXED6_TASK_ID)
        baseline = load_env_cfg(
            A6_G_PLUS_STRICT_ESCAPE_EXPLORATION_025_FIXED6_TASK_ID
        )

        self.assertEqual(
            candidate.terminations["invalid_plate"].params,
            {
                "force_limit": 1500.0,
                "force_hold_steps": round(A6_FORCE_LIMIT_HOLD_S / A6_CONTROL_DT),
                "catastrophic_force": 2500.0,
            },
        )
        self.assertEqual(
            baseline.terminations["invalid_plate"].params,
            {
                "force_limit": 1500.0,
                "force_hold_steps": 1,
                "catastrophic_force": None,
            },
        )
        terminal_penalty = candidate.rewards["invalid_plate_termination"]
        self.assertIs(terminal_penalty.func, common_mdp.is_terminated)
        self.assertEqual(
            terminal_penalty.weight, A6_INVALID_PLATE_TERMINATION_WEIGHT
        )
        candidate_rewards = dict(candidate.rewards)
        candidate_rewards.pop("invalid_plate_termination")
        self.assertEqual(candidate_rewards, baseline.rewards)
        self.assertEqual(candidate.observations, baseline.observations)
        self.assertEqual(candidate.actions, baseline.actions)

    def test_prone_lateral_task_adds_only_the_targeted_reward(self) -> None:
        candidate = load_env_cfg(A6_G_PLUS_PRONE_LATERAL_020_FIXED6_TASK_ID)
        baseline = load_env_cfg(A6_G_PLUS_FORCE_HOLD_INVALID2_FIXED6_TASK_ID)

        lateral = candidate.rewards["prone_lateral_progress"]
        self.assertIs(lateral.func, prone_lateral_progress)
        self.assertEqual(lateral.weight, A6_PRONE_LATERAL_PROGRESS_WEIGHT)
        candidate_rewards = dict(candidate.rewards)
        candidate_rewards.pop("prone_lateral_progress")
        self.assertEqual(candidate_rewards, baseline.rewards)
        self.assertEqual(candidate.terminations, baseline.terminations)
        self.assertEqual(candidate.observations, baseline.observations)
        self.assertEqual(candidate.actions, baseline.actions)

    def test_preserves_control_period_and_default_config(self) -> None:
        original = g1_recovery_dev_env_cfg(play=True)
        candidate = g1_recovery_a6_env_cfg(play=True)

        self.assertEqual(original.sim.mujoco.timestep, 0.005)
        self.assertEqual(original.decimation, 4)
        self.assertEqual(candidate.sim.mujoco.timestep, A6_PHYSICS_TIMESTEP)
        self.assertEqual(candidate.decimation, A6_DECIMATION)
        self.assertAlmostEqual(A6_CONTROL_DT, 0.02)
        self.assertAlmostEqual(
            original.sim.mujoco.timestep * original.decimation,
            candidate.sim.mujoco.timestep * candidate.decimation,
        )

    def test_preserves_amp_observations_action_shape_and_replaces_shared_costs(
        self,
    ) -> None:
        original = g1_recovery_dev_env_cfg(play=False)
        candidate = g1_recovery_a6_env_cfg(play=False)

        self.assertEqual(candidate.observations, original.observations)
        unchanged_rewards = set(original.rewards) - {
            "joint_acc_l2",
            "action_rate_l2",
            "joint_torques_l2",
            "joint_pos_limits",
            "ang_vel_xy",
            "lin_vel_xy",
            "target_orientation",
            "low_height_progress",
            "target_base_height",
            "target_joint_deviation_l2",
        }
        self.assertEqual(
            {name: candidate.rewards[name] for name in unchanged_rewards},
            {name: original.rewards[name] for name in unchanged_rewards},
        )
        self.assertNotIn("action_rate_l2", candidate.rewards)
        self.assertNotIn("joint_acc_l2", candidate.rewards)
        self.assertNotIn("joint_torques_l2", candidate.rewards)
        self.assertNotIn("joint_pos_limits", candidate.rewards)
        for name in (
            "ang_vel_xy",
            "lin_vel_xy",
            "target_orientation",
            "low_height_progress",
            "target_base_height",
            "target_joint_deviation_l2",
        ):
            self.assertNotIn(name, candidate.rewards)
        for index, (name, weight) in enumerate(
            zip(A6_TASK_NAMES, A6_TASK_WEIGHTS, strict=True)
        ):
            reward = candidate.rewards[f"a6_{name}"]
            self.assertIs(reward.func, a6_task_component)
            self.assertEqual(reward.weight, weight)
            self.assertEqual(
                reward.params,
                {"index": index, "obstructed_scale": A6_OBSTRUCTED_TASK_SCALE},
            )
        expected_shared_costs = {
            "a6_action_rate": (a6_action_rate, -0.0015),
            "a6_action_acceleration": (a6_action_acceleration, -0.0012),
            "a6_joint_limits": (common_mdp.joint_pos_limits, -0.1),
            "a6_foot_motion": (a6_foot_motion, -0.03),
            "a6_joint_stall": (a6_joint_stall, -0.20),
        }
        for name, (function, weight) in expected_shared_costs.items():
            self.assertIs(candidate.rewards[name].func, function)
            self.assertEqual(candidate.rewards[name].weight, weight)
        expected_physics_costs = {
            "a6_joint_acceleration": (-5.0e-8, 0),
            "a6_torque": (-1.0e-6, 1),
            "a6_joint_overspeed": (-0.02, 2),
            "a6_joint_overpower": (-2.0e-6, 3),
            "a6_head_overspeed": (-1.0, 4),
            "a6_sustained_effort": (-0.05, 5),
        }
        for name, (weight, index) in expected_physics_costs.items():
            self.assertIs(candidate.rewards[name].func, a6_physics_cost)
            self.assertEqual(candidate.rewards[name].weight, weight)
            self.assertEqual(candidate.rewards[name].params, {"index": index})
        sampler = candidate.metrics["a6_joint_acc_substep_raw"]
        self.assertIs(sampler.func, sample_a6_cost_substep)
        self.assertTrue(sampler.per_substep)
        expected_plate_rewards = {
            "plate_geometry_progress": (plate_geometry_progress, 0.45),
            "plate_clearance": (plate_clearance, 0.08),
            "plate_completion": (plate_completion, 0.60),
            "plate_force": (plate_force, -0.03),
            "plate_separation": (plate_separation, 0.10),
        }
        for name, (function, weight) in expected_plate_rewards.items():
            self.assertIs(candidate.rewards[name].func, function)
            self.assertEqual(candidate.rewards[name].weight, weight)
        original_action = original.actions["joint_pos"]
        candidate_action = candidate.actions["joint_pos"]
        self.assertIsInstance(candidate_action, A6EntryProtectedJointPositionActionCfg)
        self.assertEqual(candidate_action.scale, original_action.scale)
        self.assertEqual(
            candidate_action.use_default_offset, original_action.use_default_offset
        )
        self.assertEqual(candidate_action.warmup_steps, A6_ACTION_WARMUP_STEPS)
        self.assertEqual(candidate_action.raw_action_clip, A6_RAW_ACTION_CLIP)
        self.assertEqual(candidate_action.max_delay_steps, 5)
        candidate_commands = dict(candidate.commands)
        candidate_motion = candidate_commands["motion"]
        self.assertIsInstance(candidate_motion, RecoveryMotionCommandCfg)
        candidate_commands["motion"] = replace(
            candidate_motion, initialize_robot_from_motion=True
        )
        expected_commands = dict(original.commands)
        expected_commands.pop("get_up_assist_force")
        self.assertEqual(candidate_commands, expected_commands)
        self.assertNotIn("get_up_assist_force", candidate.events)
        self.assertNotIn("get_up_assist_force_level", candidate.curriculum)

        self.assertNotIn("add_base_mass", candidate.events)
        self.assertNotIn("actuator_gains", candidate.events)
        self.assertNotIn("mixed_disturbance", candidate.events)
        dynamics = candidate.events["a6_dynamics"]
        self.assertIs(dynamics.func, reset_a6_dynamics)
        self.assertEqual(dynamics.mode, "reset")
        self.assertEqual(dynamics.params, {"enabled": True})
        friction = candidate.events["physics_material"]
        self.assertIs(friction.func, dr.geom_friction)
        self.assertEqual(friction.params["ranges"], (0.3, 1.2))
        self.assertTrue(friction.params["shared_random"])
        self.assertEqual(
            friction.params["asset_cfg"].geom_names,
            r"^(left|right)_foot[1-7]_collision$",
        )
        self.assertIs(candidate.events["encoder_bias"].func, dr.encoder_bias)
        self.assertIs(candidate.events["base_com"].func, dr.body_com_offset)
        push = candidate.events["push_robot"]
        self.assertIs(push.func, common_mdp.push_by_setting_velocity)
        self.assertEqual(push.interval_range_s, (1.0, 3.0))
        self.assertEqual(push.params["velocity_range"]["z"], (-0.4, 0.4))

    def test_g_minus_and_g_plus_are_matched_except_for_guidance(self) -> None:
        minus = g1_recovery_a6_env_cfg(play=False, guidance="minus")
        plus = g1_recovery_a6_env_cfg(play=False, guidance="plus")

        guidance_rewards = {
            "plate_geometry_progress",
            "plate_clearance",
            "plate_separation",
        }
        self.assertTrue(guidance_rewards.isdisjoint(minus.rewards))
        self.assertTrue(guidance_rewards.issubset(plus.rewards))
        for name in (
            "plate_completion",
            "plate_force",
            "a6_foot_motion",
            "a6_joint_stall",
        ):
            self.assertEqual(minus.rewards[name], plus.rewards[name])
        for index, name in enumerate(A6_TASK_NAMES):
            self.assertEqual(
                minus.rewards[f"a6_{name}"].params,
                {"index": index, "obstructed_scale": 1.0},
            )
            self.assertEqual(
                plus.rewards[f"a6_{name}"].params,
                {"index": index, "obstructed_scale": A6_OBSTRUCTED_TASK_SCALE},
            )
        self.assertEqual(minus.observations, plus.observations)
        self.assertEqual(minus.actions, plus.actions)

    def test_rejects_unknown_guidance_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown A6 guidance mode"):
            g1_recovery_a6_env_cfg(guidance="typo")  # type: ignore[arg-type]

    def test_a6_task_uses_bank_without_motion_teleport(self) -> None:
        original = g1_recovery_dev_env_cfg(play=False)
        candidate = g1_recovery_a6_env_cfg(play=False)
        original_motion = original.commands["motion"]
        candidate_motion = candidate.commands["motion"]
        self.assertIsInstance(original_motion, RecoveryMotionCommandCfg)
        self.assertIsInstance(candidate_motion, RecoveryMotionCommandCfg)
        self.assertTrue(original_motion.initialize_robot_from_motion)
        self.assertFalse(candidate_motion.initialize_robot_from_motion)

        self.assertNotIn("a6_three_scene_curriculum_reset", original.events)
        bank_reset = candidate.events["a6_three_scene_curriculum_reset"]
        self.assertIsInstance(bank_reset, EventTermCfg)
        self.assertIs(bank_reset.func, reset_from_a6_three_scene_curriculum)
        self.assertEqual(bank_reset.mode, "reset")
        self.assertIsNone(bank_reset.params["plate_mass_override"])
        self.assertEqual(
            tuple(candidate.scene.entities),
            ("robot", "escape_obstacle", "free_obstacle"),
        )
        self.assertTrue(
            {"guided_contact", "free_contact", "path_hands", "quality_feet"}.issubset(
                {sensor.name for sensor in candidate.scene.sensors}
            )
        )
        self.assertNotIn("invalid_plate", original.terminations)
        self.assertIs(candidate.terminations["invalid_plate"].func, a6_invalid_plate)

        play_candidate = g1_recovery_a6_env_cfg(play=True)
        self.assertNotIn("physics_material", play_candidate.events)
        self.assertNotIn("encoder_bias", play_candidate.events)
        self.assertNotIn("base_com", play_candidate.events)
        self.assertNotIn("push_robot", play_candidate.events)
        self.assertEqual(
            play_candidate.events["a6_dynamics"].params, {"enabled": False}
        )
        self.assertEqual(
            play_candidate.events["a6_three_scene_curriculum_reset"].params[
                "plate_mass_override"
            ],
            6.0,
        )

    def test_rejects_an_unexpected_control_period(self) -> None:
        cfg = g1_recovery_dev_env_cfg(play=True)
        cfg.decimation = 5

        with self.assertRaisesRegex(ValueError, "would change control period"):
            apply_a6_timing(cfg)

        self.assertEqual(cfg.sim.mujoco.timestep, 0.005)
        self.assertEqual(cfg.decimation, 5)


if __name__ == "__main__":
    unittest.main()
