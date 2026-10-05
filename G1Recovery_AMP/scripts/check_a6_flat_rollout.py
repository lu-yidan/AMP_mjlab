"""Run a short non-training rollout from A6 three-scene reset states."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from g1recovery_amp.a6_reset_assets import A6_DIRECTION_NAMES, A6_STAGE_NAMES
from g1recovery_amp.tasks.recovery import (
    A6_G_MINUS_TASK_ID,
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
from g1recovery_amp.tasks.recovery.env_cfg import (
    PHYSICS_WARNING_CONTACT_PENETRATION,
    PHYSICS_WARNING_JOINT_ACCELERATION,
    PHYSICS_WARNING_JOINT_SPEED,
    PHYSICS_WARNING_ROOT_ANGULAR_SPEED,
)
from g1recovery_amp.tasks.recovery.mdp import (
    A6_SCENE_NAMES,
    A6_TASK_NAMES,
    max_contact_penetration,
    max_joint_acceleration,
    max_joint_speed,
    max_root_angular_speed,
)
from g1recovery_amp.tasks.recovery.mdp.a6_shared_costs import a6_head_state
from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import update_a6_plate_state

A6_STABLE_HEAD_HEIGHT = 1.15
A6_STABLE_UPRIGHT = 0.93
A6_STABLE_MAX_KNEE_ANGLE = 0.80
A6_STABLE_MAX_BASE_SPEED = 0.15
A6_STABLE_MAX_ANGULAR_SPEED = 0.30
A6_STABLE_MAX_JOINT_SPEED_RMS = 0.50
A6_STABLE_MAX_FOOT_SPEED = 0.10
A6_STABLE_MIN_FOOT_LOAD = 20.0
A6_STABLE_MIN_STANCE_WIDTH = 0.12
A6_STABLE_MAX_STANCE_WIDTH = 0.45
A6_SUCCESS_HEAD_HEIGHT = 1.15
A6_SUCCESS_UPRIGHT = 0.90
A6_SUCCESS_MIN_FOOT_LOAD = 20.0
A6_STALL_MOVING_JOINT_SPEED_RMS = 0.25
A6_STALL_MOVING_BASE_SPEED = 0.04
A6_STALL_QUIET_JOINT_SPEED_RMS = 0.10
A6_STALL_QUIET_BASE_SPEED = 0.02
A6_STALL_HOLD_S = 0.50
A6_RELAXED_EVAL_HORIZON_S = 10.0
A6_RELAXED_EVAL_MAX_FORCE = 5000.0
A6_RELAXED_EVAL_MAX_PENETRATION = 0.050
A6_RELAXED_EVAL_NO_CONTACT_S = 10.0


def _stable_standing_gates(
    *,
    head_height: torch.Tensor,
    upright: torch.Tensor,
    knee_position: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    joint_velocity: torch.Tensor,
    foot_linear_velocity: torch.Tensor,
    minimum_foot_load: torch.Tensor,
    stance_width: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate the available parts of A6's historical stable-stance test.

    The historical SMP test also required non-foot ground force below 20 N.
    AMP-A6 currently has no ``quality_other`` sensor, so that one gate is
    deliberately omitted and reported as an evaluation limitation.
    """

    gates = {
        "height": head_height >= A6_STABLE_HEAD_HEIGHT,
        "upright": upright >= A6_STABLE_UPRIGHT,
        "knees": knee_position.abs().amax(dim=-1) < A6_STABLE_MAX_KNEE_ANGLE,
        "base_speed": base_linear_velocity.norm(dim=-1) < A6_STABLE_MAX_BASE_SPEED,
        "angular_speed": base_angular_velocity.norm(dim=-1)
        < A6_STABLE_MAX_ANGULAR_SPEED,
        "joint_speed": joint_velocity.square().mean(dim=-1).sqrt()
        < A6_STABLE_MAX_JOINT_SPEED_RMS,
        "foot_speed": foot_linear_velocity.norm(dim=-1).amax(dim=-1)
        < A6_STABLE_MAX_FOOT_SPEED,
        "foot_load": minimum_foot_load > A6_STABLE_MIN_FOOT_LOAD,
        "stance_width": (stance_width >= A6_STABLE_MIN_STANCE_WIDTH)
        & (stance_width <= A6_STABLE_MAX_STANCE_WIDTH),
    }
    stable = torch.ones_like(head_height, dtype=torch.bool)
    for passed in gates.values():
        stable &= passed
    gates["stable"] = stable
    return gates


def _advance_continuous_hold(
    stable: torch.Tensor,
    current_hold: torch.Tensor,
    longest_hold: torch.Tensor,
    step_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance consecutive stable time and retain the longest achieved hold."""

    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    current_hold = torch.where(
        stable, current_hold + step_dt, torch.zeros_like(current_hold)
    )
    return current_hold, torch.maximum(longest_hold, current_hold)


def _basic_stand_success_gate(
    *,
    head_height: torch.Tensor,
    upright: torch.Tensor,
    minimum_foot_load: torch.Tensor,
    done: torch.Tensor,
    active_trial: torch.Tensor,
) -> torch.Tensor:
    """Require head height, upright torso, and support from both feet."""

    return (
        (head_height >= A6_SUCCESS_HEAD_HEIGHT)
        & (upright >= A6_SUCCESS_UPRIGHT)
        & (minimum_foot_load > A6_SUCCESS_MIN_FOOT_LOAD)
        & ~done
        & active_trial
    )


def _advance_post_motion_stall(
    *,
    moving: torch.Tensor,
    quiet: torch.Tensor,
    eligible: torch.Tensor,
    done: torch.Tensor,
    ever_moved: torch.Tensor,
    quiet_steps: torch.Tensor,
    reported: torch.Tensor,
    hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect a new sustained physical stall after an environment has moved.

    ``eligible`` is deliberately supplied by the caller: for the A6 diagnosis
    it selects guided-plate worlds that have not escaped.  ``reported`` makes
    the event fire once per episode, while ``done`` clears all temporal state
    for the next reset.
    """

    if hold_steps <= 0:
        raise ValueError("hold_steps must be positive")
    active = eligible & ~done
    next_ever_moved = ever_moved | (moving & active)
    accumulating = active & next_ever_moved & quiet
    next_quiet_steps = torch.where(
        accumulating, quiet_steps + 1, torch.zeros_like(quiet_steps)
    )
    new_stall = (next_quiet_steps >= hold_steps) & ~reported
    next_reported = reported | new_stall
    next_ever_moved = torch.where(done, False, next_ever_moved)
    next_quiet_steps = torch.where(done, 0, next_quiet_steps)
    next_reported = torch.where(done, False, next_reported)
    return next_ever_moved, next_quiet_steps, next_reported, new_stall


def _finite_observations(observations: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all() for value in observations.values())


def _success_fraction(outcome: torch.Tensor, mask: torch.Tensor) -> str:
    """Format an exact count and rate for one formal evaluation cohort."""

    total = int(mask.sum())
    passed = int((outcome & mask).sum())
    rate = passed / total if total else float("nan")
    return f"{passed}/{total} ({rate:.3f})"


def _relaxed_invalid_reasons_from_state(
    *,
    active: torch.Tensor,
    force: torch.Tensor,
    penetration: torch.Tensor,
    ever_contact: torch.Tensor,
    episode_steps: torch.Tensor,
    max_force: float,
    max_penetration: float,
    max_no_contact_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return evaluator-only invalid masks using deliberately loose limits.

    This helper is separate from the environment's training termination.  It
    lets a capability evaluation wait for slow escape attempts without
    silently changing the dynamics or reward contract used during training.
    """

    if max_force <= 0.0:
        raise ValueError("max_force must be positive")
    if max_penetration <= 0.0:
        raise ValueError("max_penetration must be positive")
    if max_no_contact_steps < 0:
        raise ValueError("max_no_contact_steps must be nonnegative")
    invalid_force = active & (force > max_force)
    invalid_penetration = active & (penetration > max_penetration)
    invalid_no_contact = (
        active & (episode_steps > max_no_contact_steps) & ~ever_contact
    )
    return invalid_force, invalid_penetration, invalid_no_contact


def _relaxed_capability_invalid_plate(
    env: ManagerBasedRlEnv,
    max_force: float = A6_RELAXED_EVAL_MAX_FORCE,
    max_penetration: float = A6_RELAXED_EVAL_MAX_PENETRATION,
    max_no_contact_steps: int = 100,
) -> torch.Tensor:
    """Evaluator-only plate termination that preserves only extreme failures."""

    if not hasattr(env, "_a6_reset_scene"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    update_a6_plate_state(env)
    active = env._a6_reset_scene > 0
    free_scene = env._a6_reset_scene == 2
    guided_data = env.scene["guided_contact"].data
    free_data = env.scene["free_contact"].data
    if (
        guided_data.force is None
        or guided_data.dist is None
        or free_data.force is None
        or free_data.dist is None
    ):
        raise RuntimeError("A6 plate contact sensor data is incomplete")
    guided_force = guided_data.force.norm(dim=-1).amax(dim=-1)
    free_force = free_data.force.norm(dim=-1).amax(dim=-1)
    guided_penetration = (-guided_data.dist).clamp_min(0.0).amax(dim=-1)
    free_penetration = (-free_data.dist).clamp_min(0.0).amax(dim=-1)
    force = torch.where(free_scene, free_force, guided_force)
    penetration = torch.where(
        free_scene, free_penetration, guided_penetration
    )
    reasons = _relaxed_invalid_reasons_from_state(
        active=active,
        force=force,
        penetration=penetration,
        ever_contact=env._a6_plate_ever_contact,
        episode_steps=env.episode_length_buf,
        max_force=max_force,
        max_penetration=max_penetration,
        max_no_contact_steps=max_no_contact_steps,
    )
    for reason, mask in zip(("force", "penetration", "no_contact"), reasons):
        getattr(env, f"_a6_plate_terminal_invalid_{reason}").copy_(mask)
    return reasons[0] | reasons[1] | reasons[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-config",
        action="store_true",
        help="enable A6 training randomization during the rollout",
    )
    parser.add_argument("--action-mode", choices=("zero", "policy"), default="zero")
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument(
        "--formal-success-eval",
        action="store_true",
        help=(
            "run the fixed 2048-trial low-state evaluation with "
            "1024/512/512 flat/guided/free scenes over 10 seconds"
        ),
    )
    parser.add_argument(
        "--relaxed-capability-eval",
        action="store_true",
        help=(
            "with --formal-success-eval, terminate only at 5000 N plate force "
            "or 5 cm penetration; missing initial contact is tolerated for the "
            "full 10-second trial and training/runtime settings are not changed"
        ),
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        help=(
            "evaluation-only timeout override for non-formal rollouts"
        ),
    )
    parser.add_argument(
        "--task-id",
        choices=(
            A6_TASK_ID,
            A6_G_MINUS_TASK_ID,
            A6_G_PLUS_CONSTRAINT_AWARE_001_FIXED6_TASK_ID,
            A6_G_PLUS_TASK_ID,
            A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID,
            A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID,
            A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID,
            A6_G_PLUS_STRICT_ESCAPE_HEADWARD_FIXED6_TASK_ID,
            A6_G_PLUS_STRICT_ESCAPE_EXPLORATION_025_FIXED6_TASK_ID,
            A6_G_PLUS_FORCE_HOLD_INVALID2_FIXED6_TASK_ID,
            A6_G_PLUS_PRONE_LATERAL_020_FIXED6_TASK_ID,
            A6_G_PLUS_SYMMETRIC_005_TASK_ID,
        ),
        default=A6_TASK_ID,
    )
    args = parser.parse_args()
    if args.action_mode == "policy" and args.checkpoint_file is None:
        parser.error("--checkpoint-file is required for --action-mode policy")
    if args.checkpoint_file is not None and not args.checkpoint_file.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint_file}")
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        parser.error("--episode-length-s must be positive")
    if args.relaxed_capability_eval and not args.formal_success_eval:
        parser.error("--relaxed-capability-eval requires --formal-success-eval")
    if args.formal_success_eval:
        if args.training_config:
            parser.error("formal evaluation must use the deterministic play config")
        if args.action_mode != "policy":
            parser.error("formal evaluation requires --action-mode policy")
        args.num_envs = 2048

    cfg = load_env_cfg(args.task_id, play=not args.training_config)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.formal_success_eval:
        step_dt = cfg.sim.mujoco.timestep * cfg.decimation
        horizon_s = (
            A6_RELAXED_EVAL_HORIZON_S
            if args.relaxed_capability_eval
            else 10.0
        )
        args.steps = round(horizon_s / step_dt)
        reset_cfg = cfg.events["a6_three_scene_curriculum_reset"]
        reset_cfg.params["scene_weights"] = (0.50, 0.25, 0.25)
        reset_cfg.params["low_only"] = True
        if args.relaxed_capability_eval:
            cfg.terminations["invalid_plate"].func = (
                _relaxed_capability_invalid_plate
            )
            cfg.terminations["invalid_plate"].params = {
                "max_force": A6_RELAXED_EVAL_MAX_FORCE,
                "max_penetration": A6_RELAXED_EVAL_MAX_PENETRATION,
                "max_no_contact_steps": round(
                    A6_RELAXED_EVAL_NO_CONTACT_S / step_dt
                ),
            }
        # Keep the environment timeout just beyond the selected horizon so
        # every world contributes exactly one trial. Invalid physics still
        # terminates a trial immediately and is counted as failure.
        cfg.episode_length_s = (args.steps + 1) * step_dt
    elif args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    agent_cfg = load_rl_cfg(args.task_id)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        observations = env.get_observations()
        policy = None
        runner = None
        if args.action_mode == "policy":
            runner_cls = load_runner_cls(args.task_id)
            if runner_cls is None:
                raise RuntimeError("AMP-A6 task has no registered runner")
            runner = runner_cls(
                env, asdict(agent_cfg), log_dir=None, device=args.device
            )
            runner.load(
                str(args.checkpoint_file),
                load_cfg={"actor": True, "amp": True},
                strict=True,
                map_location=args.device,
            )
            policy = runner.get_inference_policy(device=args.device)

        maxima = {
            "root_ang_vel": torch.zeros(args.num_envs, device=args.device),
            "joint_speed": torch.zeros(args.num_envs, device=args.device),
            "joint_acceleration": torch.zeros(args.num_envs, device=args.device),
            "contact_penetration": torch.zeros(args.num_envs, device=args.device),
        }
        peak_steps = {
            name: torch.full(
                (args.num_envs,), -1, dtype=torch.int64, device=args.device
            )
            for name in maxima
        }
        metric_functions = {
            "root_ang_vel": max_root_angular_speed,
            "joint_speed": max_joint_speed,
            "joint_acceleration": max_joint_acceleration,
            "contact_penetration": max_contact_penetration,
        }
        thresholds = {
            "root_ang_vel": PHYSICS_WARNING_ROOT_ANGULAR_SPEED,
            "joint_speed": PHYSICS_WARNING_JOINT_SPEED,
            "joint_acceleration": PHYSICS_WARNING_JOINT_ACCELERATION,
            "contact_penetration": PHYSICS_WARNING_CONTACT_PENETRATION,
        }

        if not _finite_observations(observations):
            raise FloatingPointError("initial observation contains NaN or Inf")
        reset_state_metrics = {
            name: function(raw_env) for name, function in metric_functions.items()
        }
        scene = raw_env._a6_reset_scene.clone()
        stage = raw_env._a6_reset_stage.clone()
        direction = raw_env._a6_reset_direction.clone()
        source = raw_env._a6_reset_source.clone()
        rows = torch.arange(args.num_envs, device=args.device)
        active_trial = torch.ones(
            args.num_envs, dtype=torch.bool, device=args.device
        )
        if args.formal_success_eval:
            expected_scene_counts = (1024, 512, 512)
            actual_scene_counts = tuple(
                int((scene == scene_id).sum()) for scene_id in range(3)
            )
            if actual_scene_counts != expected_scene_counts:
                raise RuntimeError(
                    "formal scene allocation mismatch: "
                    f"{actual_scene_counts} != {expected_scene_counts}"
                )
            if torch.any(stage != 0):
                raise RuntimeError("formal evaluation must use only low reset states")
            for scene_id, per_direction in ((0, 256), (1, 128), (2, 128)):
                for direction_id in range(4):
                    cohort = (scene == scene_id) & (direction == direction_id)
                    natural = int((cohort & (source == 0)).sum())
                    procedural = int((cohort & (source == 1)).sum())
                    expected = (3 * per_direction // 4, per_direction // 4)
                    if (natural, procedural) != expected:
                        raise RuntimeError(
                            "formal direction/source allocation mismatch for "
                            f"scene={scene_id}, direction={direction_id}: "
                            f"{(natural, procedural)} != {expected}"
                        )
            plate_worlds = scene > 0
            expected_mass = torch.full_like(
                raw_env._a6_plate_mass[plate_worlds], 6.0
            )
            if not torch.allclose(
                raw_env._a6_plate_mass[plate_worlds], expected_mass
            ):
                raise RuntimeError("formal evaluation plate mass is not exactly 6 kg")
        plate_geom_ids = torch.stack(raw_env._a6_plate_geom_ids)
        active_geom_ids = plate_geom_ids[(scene - 1).clamp(0, 1)]
        initial_plate_pos = raw_env.sim.data.geom_xpos[rows, active_geom_ids].clone()
        max_plate_force = torch.zeros(args.num_envs, device=args.device)
        max_plate_penetration = torch.zeros(args.num_envs, device=args.device)
        invalid_events = torch.zeros(
            args.num_envs, dtype=torch.int64, device=args.device
        )
        timeout_events = torch.zeros(
            args.num_envs, dtype=torch.int64, device=args.device
        )
        invalid_reason_events = {
            reason: torch.zeros(args.num_envs, dtype=torch.int64, device=args.device)
            for reason in ("force", "penetration", "no_contact")
        }
        ever_contact_seen = torch.zeros(
            args.num_envs, dtype=torch.bool, device=args.device
        )
        escaped_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        plate_reward_names = tuple(
            name
            for name in (
                "plate_geometry_progress",
                "plate_clearance",
                "plate_completion",
                "plate_force",
                "plate_separation",
                "plate_no_progress",
                "trapped_shoulders_headward",
                "invalid_plate_termination",
                "prone_lateral_progress",
            )
            if name in raw_env.reward_manager.active_terms
        )
        task_reward_names = tuple(f"a6_{name}" for name in A6_TASK_NAMES)
        shared_cost_names = (
            "a6_action_rate",
            "a6_action_acceleration",
            "a6_joint_limits",
            "a6_joint_acceleration",
            "a6_torque",
            "a6_joint_overspeed",
            "a6_joint_overpower",
            "a6_head_overspeed",
            "a6_sustained_effort",
            "a6_foot_motion",
            "a6_joint_stall",
        )
        reward_indices = {
            name: raw_env.reward_manager.active_terms.index(name)
            for name in (
                *task_reward_names,
                *plate_reward_names,
                *shared_cost_names,
            )
        }
        task_reward_integrals = {
            name: torch.zeros(args.num_envs, device=args.device)
            for name in task_reward_names
        }
        plate_reward_integrals = {
            name: torch.zeros(args.num_envs, device=args.device)
            for name in plate_reward_names
        }
        shared_cost_integrals = {
            name: torch.zeros(args.num_envs, device=args.device)
            for name in shared_cost_names
        }
        phase_seen = torch.zeros(
            (args.num_envs, 4), dtype=torch.bool, device=args.device
        )
        task_total_integral = torch.zeros(args.num_envs, device=args.device)
        style_reward_integral = torch.zeros(args.num_envs, device=args.device)
        blended_reward_integral = torch.zeros(args.num_envs, device=args.device)
        discriminator_score_sum = torch.zeros(args.num_envs, device=args.device)
        task_reward_weight_sum = torch.zeros(args.num_envs, device=args.device)
        stable_gate_pass_counts = {
            name: torch.zeros(args.num_envs, device=args.device)
            for name in (
                "height",
                "upright",
                "knees",
                "base_speed",
                "angular_speed",
                "joint_speed",
                "foot_speed",
                "foot_load",
                "stance_width",
            )
        }
        current_stable_hold = torch.zeros(args.num_envs, device=args.device)
        longest_stable_hold = torch.zeros(args.num_envs, device=args.device)
        current_success_hold = torch.zeros(args.num_envs, device=args.device)
        longest_success_hold = torch.zeros(args.num_envs, device=args.device)
        stall_hold_steps = max(1, round(A6_STALL_HOLD_S / raw_env.step_dt))
        stall_ever_moved = torch.zeros(
            args.num_envs, dtype=torch.bool, device=args.device
        )
        stall_quiet_steps = torch.zeros(
            args.num_envs, dtype=torch.int64, device=args.device
        )
        stall_reported = torch.zeros(
            args.num_envs, dtype=torch.bool, device=args.device
        )
        stall_event_counts = torch.zeros(
            args.num_envs, dtype=torch.int64, device=args.device
        )
        first_stall_step = torch.full(
            (args.num_envs,), -1, dtype=torch.int64, device=args.device
        )
        stall_snapshot_names = (
            "action_rms",
            "action_delta_rms",
            "action_clip_fraction",
            "joint_speed_rms",
            "base_planar_speed",
            "torque_ratio_rms",
            "covered_body_count",
            "initial_covered_body_count",
            "path_clearance",
            "coverage_delta",
            "clearance_delta",
            "hand_support",
            "head_height",
            "plate_force",
        )
        first_stall_snapshot = {
            name: torch.full(
                (args.num_envs,), torch.nan, dtype=torch.float32, device=args.device
            )
            for name in stall_snapshot_names
        }
        first_stall_phase = torch.full(
            (args.num_envs,), -1, dtype=torch.int64, device=args.device
        )
        first_stall_bank_row = torch.full_like(first_stall_phase, -1)
        previous_policy_actions = torch.zeros(
            (args.num_envs, env.num_actions), device=args.device
        )

        with torch.inference_mode():
            for step in range(args.steps):
                if policy is None:
                    actions = torch.zeros(
                        (args.num_envs, env.num_actions), device=args.device
                    )
                else:
                    actions = policy(observations)
                active_before_step = active_trial.clone()
                if args.formal_success_eval:
                    actions = torch.where(
                        active_before_step[:, None], actions, torch.zeros_like(actions)
                    )
                if not torch.isfinite(actions).all():
                    raise FloatingPointError(
                        f"action contains NaN or Inf at step {step}"
                    )
                observations, rewards, dones, extras = env.step(actions)
                if not _finite_observations(observations):
                    raise FloatingPointError(
                        f"observation contains NaN or Inf at step {step}"
                    )
                if not torch.isfinite(rewards).all():
                    raise FloatingPointError(
                        f"reward contains NaN or Inf at step {step}"
                    )
                sample_weight = active_before_step.to(rewards.dtype)
                task_total_integral += rewards * sample_weight
                if runner is not None:
                    disc_observations = (
                        runner.alg._replace_reset_observations_with_terminal(
                            observations["disc"], dones, extras
                        )
                    )
                    style_reward, discriminator_score = (
                        runner.alg.amp_discriminator.style_reward(
                            disc_observations,
                            step_dt=runner.alg.amp_cfg.step_dt,
                        )
                    )
                    style_reward *= runner.alg._style_reward_multiplier(
                        observations, dones, extras
                    )
                    blended_reward = runner.alg.amp_discriminator.blend_reward(
                        rewards,
                        style_reward,
                    )
                    task_reward_weight_sum += (
                        runner.alg.amp_discriminator.cfg.task_style_lerp
                        * sample_weight
                    )
                    style_reward_integral += style_reward * sample_weight
                    blended_reward_integral += blended_reward * sample_weight
                    discriminator_score_sum += discriminator_score * sample_weight

                robot = raw_env.scene["robot"]
                head_height, _ = a6_head_state(raw_env)
                upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
                feet = robot.data.body_link_pos_w[:, raw_env._a6_cost_foot_ids]
                stance_width = (feet[:, 0, :2] - feet[:, 1, :2]).norm(dim=-1)
                foot_force = raw_env.scene["quality_feet"].data.force
                if foot_force is None:
                    raise RuntimeError("quality_feet sensor did not provide force")
                minimum_foot_load = foot_force[..., 2].abs().amin(dim=-1)
                stable_gates = _stable_standing_gates(
                    head_height=head_height,
                    upright=upright,
                    knee_position=robot.data.joint_pos[:, raw_env._a6_cost_knee_ids],
                    base_linear_velocity=robot.data.root_link_lin_vel_w,
                    base_angular_velocity=robot.data.root_link_ang_vel_w,
                    joint_velocity=robot.data.joint_vel,
                    foot_linear_velocity=robot.data.body_link_lin_vel_w[
                        :, raw_env._a6_cost_foot_ids
                    ],
                    minimum_foot_load=minimum_foot_load,
                    stance_width=stance_width,
                )
                for name, counts in stable_gate_pass_counts.items():
                    counts += stable_gates[name].float() * sample_weight
                escaped_or_flat = (scene == 0) | raw_env._a6_plate_escaped
                if args.relaxed_capability_eval:
                    escaped_or_flat = torch.ones_like(escaped_or_flat)
                qualified_stable = (
                    stable_gates["stable"]
                    & escaped_or_flat
                    & ~dones.bool()
                    & active_before_step
                )
                current_stable_hold, longest_stable_hold = _advance_continuous_hold(
                    qualified_stable,
                    current_stable_hold,
                    longest_stable_hold,
                    raw_env.step_dt,
                )
                qualified_success = _basic_stand_success_gate(
                    head_height=head_height,
                    upright=upright,
                    minimum_foot_load=minimum_foot_load,
                    done=dones.bool(),
                    active_trial=active_before_step,
                )
                current_success_hold, longest_success_hold = _advance_continuous_hold(
                    qualified_success,
                    current_success_hold,
                    longest_success_hold,
                    raw_env.step_dt,
                )
                invalid = raw_env.termination_manager.get_term("invalid_plate")
                time_out = raw_env.termination_manager.get_term("time_out")
                unexpected = dones.bool() & ~(invalid | time_out)
                if torch.any(unexpected):
                    ids = unexpected.nonzero(as_tuple=False).squeeze(-1).tolist()
                    raise RuntimeError(
                        f"unexpected non-plate reset at step {step}: envs={ids}"
                    )
                invalid_events += (invalid & active_before_step).long()
                timeout_events += (time_out & active_before_step).long()
                terminal_reason_union = torch.zeros_like(invalid)
                for reason, counts in invalid_reason_events.items():
                    reason_mask = getattr(
                        raw_env, f"_a6_plate_terminal_invalid_{reason}"
                    )
                    terminal_reason_union |= reason_mask
                    counts += (invalid & reason_mask & active_before_step).long()
                unclassified = (
                    invalid
                    & ~terminal_reason_union
                    & active_before_step
                )
                if torch.any(unclassified):
                    ids = unclassified.nonzero(as_tuple=False).squeeze(-1).tolist()
                    raise RuntimeError(
                        "invalid plate termination has no preserved reason: "
                        f"step={step}, envs={ids}"
                    )

                # Detect the failure mode seen in playback: a guided-plate
                # robot moves first, then remains physically still.  Capturing
                # action and load at that instant distinguishes a policy that
                # gives up from one that keeps commanding motion but is stuck.
                applied_actions = raw_env.action_manager.action
                action_rms = applied_actions.square().mean(dim=-1).sqrt()
                action_delta_rms = (
                    (applied_actions - previous_policy_actions)
                    .square()
                    .mean(dim=-1)
                    .sqrt()
                )
                if agent_cfg.clip_actions is None:
                    action_clip_fraction = torch.zeros_like(action_rms)
                else:
                    action_clip_fraction = (
                        (actions.abs() >= agent_cfg.clip_actions).float().mean(dim=-1)
                    )
                joint_speed_rms = robot.data.joint_vel.square().mean(dim=-1).sqrt()
                base_planar_speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
                moving = (joint_speed_rms >= A6_STALL_MOVING_JOINT_SPEED_RMS) | (
                    base_planar_speed >= A6_STALL_MOVING_BASE_SPEED
                )
                quiet = (joint_speed_rms <= A6_STALL_QUIET_JOINT_SPEED_RMS) & (
                    base_planar_speed <= A6_STALL_QUIET_BASE_SPEED
                )
                stall_eligible = (
                    (scene == 1)
                    & ~raw_env._a6_plate_escaped
                    & ~invalid
                    & active_before_step
                )
                (
                    stall_ever_moved,
                    stall_quiet_steps,
                    stall_reported,
                    new_stall,
                ) = _advance_post_motion_stall(
                    moving=moving,
                    quiet=quiet,
                    eligible=stall_eligible,
                    done=dones.bool(),
                    ever_moved=stall_ever_moved,
                    quiet_steps=stall_quiet_steps,
                    reported=stall_reported,
                    hold_steps=stall_hold_steps,
                )
                stall_event_counts += new_stall.long()
                first = new_stall & (first_stall_step < 0)
                if torch.any(first):
                    effort_limit = (
                        raw_env.sim.model.actuator_forcerange[
                            :, raw_env._a6_cost_ctrl_ids
                        ]
                        .abs()
                        .amax(dim=-1)
                    )
                    torque_ratio_rms = (
                        (robot.data.qfrc_actuator / effort_limit)
                        .square()
                        .mean(dim=-1)
                        .sqrt()
                    )
                    snapshot_values = {
                        "action_rms": action_rms,
                        "action_delta_rms": action_delta_rms,
                        "action_clip_fraction": action_clip_fraction,
                        "joint_speed_rms": joint_speed_rms,
                        "base_planar_speed": base_planar_speed,
                        "torque_ratio_rms": torque_ratio_rms,
                        "covered_body_count": raw_env._a6_plate_path_current_count,
                        "initial_covered_body_count": (
                            raw_env._a6_plate_path_initial_count
                        ),
                        "path_clearance": raw_env._a6_plate_path_current_clearance,
                        "coverage_delta": raw_env._a6_plate_path_coverage_delta,
                        "clearance_delta": raw_env._a6_plate_path_clearance_delta,
                        "hand_support": raw_env._a6_plate_path_hand_support,
                        "head_height": raw_env._a6_plate_head_height,
                        "plate_force": raw_env._a6_plate_force,
                    }
                    first_stall_step[first] = step
                    first_stall_phase[first] = raw_env._a6_cost_phase[first]
                    first_stall_bank_row[first] = raw_env._a6_reset_bank_index[first]
                    for name, values in snapshot_values.items():
                        first_stall_snapshot[name][first] = values[first]
                previous_policy_actions.copy_(applied_actions)
                previous_policy_actions[dones.bool()] = 0.0
                for name, term_index in reward_indices.items():
                    if name in task_reward_integrals:
                        destination = task_reward_integrals
                    elif name in plate_reward_integrals:
                        destination = plate_reward_integrals
                    else:
                        destination = shared_cost_integrals
                    destination[name] += (
                        raw_env.reward_manager._step_reward[:, term_index]
                        * raw_env.step_dt
                        * sample_weight
                    )
                phase_seen[
                    rows[active_before_step],
                    raw_env._a6_cost_phase[active_before_step],
                ] = True
                ever_contact_seen |= (
                    raw_env._a6_plate_ever_contact & active_before_step
                )
                escaped_seen |= raw_env._a6_plate_escaped & active_before_step
                guided_data = raw_env.scene["guided_contact"].data
                free_data = raw_env.scene["free_contact"].data
                guided_force = guided_data.force.norm(dim=-1).amax(dim=-1)
                free_force = free_data.force.norm(dim=-1).amax(dim=-1)
                guided_depth = (-guided_data.dist).clamp_min(0).amax(dim=-1)
                free_depth = (-free_data.dist).clamp_min(0).amax(dim=-1)
                active_force = torch.where(scene == 2, free_force, guided_force)
                active_depth = torch.where(scene == 2, free_depth, guided_depth)
                max_plate_force = torch.maximum(
                    max_plate_force,
                    torch.where((scene > 0) & active_before_step, active_force, 0.0),
                )
                max_plate_penetration = torch.maximum(
                    max_plate_penetration,
                    torch.where((scene > 0) & active_before_step, active_depth, 0.0),
                )
                for name, function in metric_functions.items():
                    current = function(raw_env)
                    current = torch.where(active_before_step, current, 0.0)
                    improved = current > maxima[name]
                    maxima[name] = torch.maximum(maxima[name], current)
                    peak_steps[name][improved] = step
                if args.formal_success_eval:
                    active_trial &= ~dones.bool()

        final_plate_pos = raw_env.sim.data.geom_xpos[rows, active_geom_ids]
        expected_substeps = args.steps * raw_env.cfg.decimation
        if raw_env._a6_cost_substep_tick != expected_substeps:
            raise RuntimeError(
                "A6 cost sampler count mismatch: "
                f"{raw_env._a6_cost_substep_tick} != {expected_substeps}"
            )
        plate_delta = final_plate_pos - initial_plate_pos
        plate_planar_displacement = plate_delta[:, :2].norm(dim=-1)
        plate_vertical_displacement = plate_delta[:, 2].abs()
        success_1s = longest_success_hold >= (1.0 - 0.5 * raw_env.step_dt)
        success_10s = longest_success_hold >= (10.0 - 0.5 * raw_env.step_dt)
        strict_stable_1s = longest_stable_hold >= (1.0 - 0.5 * raw_env.step_dt)
        print(
            f"a6_three_scene_rollout_valid=True, mode={args.action_mode}, "
            f"num_envs={args.num_envs}, steps={args.steps}, "
            f"cost_substeps={raw_env._a6_cost_substep_tick}"
        )
        reset_state = ", ".join(
            f"{name}={float(value.max()):.6g}"
            for name, value in reset_state_metrics.items()
        )
        print(f"  reset_state_before_first_action: {reset_state}")
        print(f"  timeout_events={int(timeout_events.sum())}")
        print(
            "  strict_stable_diagnostic_definition=a6_historical_available_gates; "
            "limitation=non_foot_ground_force_gate_not_measured"
        )
        print(
            "  basic_stand_success_definition=head_height_ge_1.15m_"
            "upright_ge_0.90_both_feet_load_gt_20N_hold_1s"
        )
        print(
            "  basic_stand_outcome: "
            f"success_1s={float(success_1s.float().mean()):.3f} "
            f"({int(success_1s.sum())}/{args.num_envs}), "
            f"success_10s={float(success_10s.float().mean()):.3f} "
            f"({int(success_10s.sum())}/{args.num_envs}), "
            f"mean_longest_hold_s={float(longest_success_hold.mean()):.3f}, "
            f"max_longest_hold_s={float(longest_success_hold.max()):.3f}"
        )
        print(
            "  strict_stable_diagnostic: "
            f"hold_1s={float(strict_stable_1s.float().mean()):.3f} "
            f"({int(strict_stable_1s.sum())}/{args.num_envs})"
        )
        if args.formal_success_eval:
            horizon_s = args.steps * raw_env.step_dt
            print(
                f"  formal_eval_profile: trials=2048, horizon_s={horizon_s:.1f}, "
                "scene_counts=flat:1024/guided_plate:512/free_plate:512, "
                "directions_per_scene=balanced, source_ratio=natural:procedural=3:1, "
                "single_episode=True, plate_size_m=0.90x0.64x0.07, "
                "plate_mass_kg=6"
            )
            if args.relaxed_capability_eval:
                print(
                    "  formal_invalid_definition: mode=relaxed_capability, "
                    f"plate_force_gt_N={A6_RELAXED_EVAL_MAX_FORCE:.0f}, "
                    "force_hold_s=0, "
                    "plate_penetration_gt_m="
                    f"{A6_RELAXED_EVAL_MAX_PENETRATION:.3f}, "
                    "initial_no_contact_gt_s="
                    f"{A6_RELAXED_EVAL_NO_CONTACT_S:.1f}"
                )
            else:
                print("  formal_invalid_definition: mode=native_environment")
            print(
                "  formal_success_definition: "
                "all_scenes=head_height_ge_1.15m_upright_ge_0.90_"
                "both_feet_load_gt_20N_hold_1s_within_10s; "
                "plate_escape=diagnostic_only_clearance_4cm_hold_0.5s"
            )
            invalid_failure = invalid_events > 0
            for scene_id, scene_name in enumerate(A6_SCENE_NAMES):
                scene_mask = scene == scene_id
                values = [
                    f"trials={int(scene_mask.sum())}",
                    f"primary_success={_success_fraction(success_1s, scene_mask)}",
                ]
                if scene_id > 0:
                    values.append(
                        "escape_diagnostic="
                        f"{_success_fraction(escaped_seen, scene_mask)}"
                    )
                    values.append(
                        "basic_stand_success="
                        f"{_success_fraction(success_1s, scene_mask)}"
                    )
                values.append(
                    "strict_stable_diagnostic="
                    f"{_success_fraction(strict_stable_1s, scene_mask)}"
                )
                values.append(
                    "invalid_failure="
                    f"{_success_fraction(invalid_failure, scene_mask)}"
                )
                print(f"  formal_scene={scene_name}: " + ", ".join(values))

            flat = scene == 0
            for direction_id, direction_name in enumerate(A6_DIRECTION_NAMES):
                cohort = flat & (direction == direction_id)
                natural = cohort & (source == 0)
                procedural = cohort & (source == 1)
                print(
                    f"  formal_flat_direction={direction_name}: "
                    f"basic_stand_success={_success_fraction(success_1s, cohort)}, "
                    "strict_stable_diagnostic="
                    f"{_success_fraction(strict_stable_1s, cohort)}, "
                    "natural_basic_stand_success="
                    f"{_success_fraction(success_1s, natural)}, "
                    "procedural_basic_stand_success="
                    f"{_success_fraction(success_1s, procedural)}"
                )

            guided = scene == 1
            for direction_id, direction_name in enumerate(A6_DIRECTION_NAMES):
                cohort = guided & (direction == direction_id)
                natural = cohort & (source == 0)
                procedural = cohort & (source == 1)
                print(
                    f"  formal_guided_direction={direction_name}: "
                    f"escape={_success_fraction(escaped_seen, cohort)}, "
                    f"basic_stand_success={_success_fraction(success_1s, cohort)}, "
                    f"natural_escape={_success_fraction(escaped_seen, natural)}, "
                    "natural_basic_stand_success="
                    f"{_success_fraction(success_1s, natural)}, "
                    "procedural_escape="
                    f"{_success_fraction(escaped_seen, procedural)}, "
                    "procedural_basic_stand_success="
                    f"{_success_fraction(success_1s, procedural)}, "
                    f"invalid_failure={_success_fraction(invalid_failure, cohort)}, "
                    "invalid_reasons="
                    + "/".join(
                        f"{reason}:{int(counts[cohort].sum())}"
                        for reason, counts in invalid_reason_events.items()
                    )
                )
        if runner is not None:
            task_style_lerp = runner.alg.amp_discriminator.cfg.task_style_lerp
            print(
                "  mean_amp_reward_integrals: "
                f"task={float(task_total_integral.mean()):.6g}, "
                f"style={float(style_reward_integral.mean()):.6g}, "
                f"blended={float(blended_reward_integral.mean()):.6g}, "
                f"disc_score={float(discriminator_score_sum.mean() / args.steps):.6g}, "
                "mean_task_weight="
                f"{float(task_reward_weight_sum.mean() / args.steps):.3f}, "
                f"default_task_weight={task_style_lerp:.3f}, "
                f"style_scale={runner.alg.amp_discriminator.cfg.style_reward_scale:.3f}"
            )
        print(
            "  a6_phase_seen_envs: "
            + ", ".join(
                f"phase_{phase_id}={int(phase_seen[:, phase_id].sum())}"
                for phase_id in range(4)
            )
        )
        print(
            "  mean_task_reward_integrals: "
            + ", ".join(
                f"{name.removeprefix('a6_')}={float(values.mean()):.6g}"
                for name, values in task_reward_integrals.items()
            )
        )
        print(
            "  mean_shared_cost_integrals: "
            + ", ".join(
                f"{name.removeprefix('a6_')}={float(values.mean()):.6g}"
                for name, values in shared_cost_integrals.items()
            )
        )
        for stage_id, stage_name in enumerate(A6_STAGE_NAMES):
            mask = stage == stage_id
            if not torch.any(mask):
                continue
            values = ", ".join(
                f"{name}={float(maximum[mask].max()):.6g}"
                for name, maximum in maxima.items()
            )
            print(f"  stage={stage_name}: {values}")
        for scene_id, scene_name in enumerate(A6_SCENE_NAMES):
            mask = scene == scene_id
            if not torch.any(mask):
                continue
            values = ", ".join(
                f"{name}={float(maximum[mask].max()):.6g}"
                for name, maximum in maxima.items()
            )
            plate_values = ""
            if scene_id > 0:
                plate_values = (
                    f", plate_force={float(max_plate_force[mask].max()):.6g}, "
                    f"plate_penetration={float(max_plate_penetration[mask].max()):.6g}, "
                    f"invalid_events={int(invalid_events[mask].sum())}, "
                    "invalid_reasons="
                    + "/".join(
                        f"{reason}:{int(counts[mask].sum())}"
                        for reason, counts in invalid_reason_events.items()
                    )
                    + ", "
                    f"ever_contact_envs={int(ever_contact_seen[mask].sum())}, "
                    f"escaped_envs={int(escaped_seen[mask].sum())}, "
                    "mean_reward_integrals="
                    + "/".join(
                        f"{name.removeprefix('plate_')}:"
                        f"{float(values[mask].mean()):.6g}"
                        for name, values in plate_reward_integrals.items()
                    )
                    + ", "
                    "plate_planar_displacement="
                    f"{float(plate_planar_displacement[mask].max()):.6g}, "
                    "plate_vertical_displacement="
                    f"{float(plate_vertical_displacement[mask].max()):.6g}"
                )
            gate_rates = {
                name: float(counts[mask].mean() / args.steps)
                for name, counts in stable_gate_pass_counts.items()
            }
            bottlenecks = sorted(gate_rates.items(), key=lambda item: item[1])[:3]
            outcome_values = (
                ", basic_stand_success_1s="
                f"{float(success_1s[mask].float().mean()):.3f}, "
                "strict_stable_1s="
                f"{float(strict_stable_1s[mask].float().mean()):.3f}, "
                "mean_longest_basic_stand_hold_s="
                f"{float(longest_success_hold[mask].mean()):.3f}, "
                "lowest_gate_pass_rates="
                + "/".join(f"{name}:{rate:.3f}" for name, rate in bottlenecks)
            )
            amp_values = ""
            if runner is not None:
                amp_values = (
                    ", mean_amp_integrals="
                    f"task:{float(task_total_integral[mask].mean()):.6g}/"
                    f"style:{float(style_reward_integral[mask].mean()):.6g}/"
                    f"blend:{float(blended_reward_integral[mask].mean()):.6g}, "
                    "mean_disc_score="
                    f"{float(discriminator_score_sum[mask].mean() / args.steps):.6g}, "
                    "mean_task_weight="
                    f"{float(task_reward_weight_sum[mask].mean() / args.steps):.3f}"
                )
            print(
                f"  scene={scene_name}: {values}{plate_values}"
                f"{outcome_values}{amp_values}"
            )
        if args.num_envs <= 16:
            print("  per_env_outcomes:")
            for env_id in range(args.num_envs):
                print(
                    f"    env={env_id}, scene={A6_SCENE_NAMES[int(scene[env_id])]}, "
                    f"stage={A6_STAGE_NAMES[int(stage[env_id])]}, "
                    f"escaped={bool(escaped_seen[env_id])}, "
                    f"longest_basic_stand_hold_s={float(longest_success_hold[env_id]):.3f}, "
                    f"basic_stand_success_1s={bool(success_1s[env_id])}, "
                    f"strict_stable_1s={bool(strict_stable_1s[env_id])}"
                )
        guided = scene == 1
        print(
            "  guided_post_motion_stalls: "
            f"events={int(stall_event_counts[guided].sum())}, "
            f"affected_envs={int((first_stall_step[guided] >= 0).sum())}/"
            f"{int(guided.sum())}, detection_hold_s={A6_STALL_HOLD_S:.2f}"
        )
        for env_id in (
            (guided & (first_stall_step >= 0))
            .nonzero(as_tuple=False)
            .squeeze(-1)
            .tolist()
        ):
            snapshot = ", ".join(
                f"{name}={float(values[env_id]):.6g}"
                for name, values in first_stall_snapshot.items()
            )
            print(
                f"    first_stall: env={env_id}, "
                f"step={int(first_stall_step[env_id])}, "
                f"phase={int(first_stall_phase[env_id])}, "
                f"direction={A6_DIRECTION_NAMES[int(direction[env_id])]}, "
                f"source={int(raw_env._a6_reset_source[env_id])}, "
                f"bank_row={int(first_stall_bank_row[env_id])}, {snapshot}"
            )
        for direction_id, direction_name in enumerate(A6_DIRECTION_NAMES):
            mask = (stage == 0) & (direction == direction_id)
            if not torch.any(mask):
                continue
            values = ", ".join(
                f"{name}={float(maximum[mask].max()):.6g}"
                for name, maximum in maxima.items()
            )
            print(f"  low_direction={direction_name}: {values}")

        violations = {
            name: int((maximum > thresholds[name]).sum())
            for name, maximum in maxima.items()
        }
        flat = scene == 0
        if torch.any(flat):
            leaked = {
                name: float(values[flat].abs().max())
                for name, values in plate_reward_integrals.items()
                if float(values[flat].abs().max()) > 1.0e-8
            }
            if leaked:
                raise RuntimeError(f"plate rewards leaked into flat scenes: {leaked}")
        overall = ", ".join(
            f"{name}={float(maximum.max()):.6g}" for name, maximum in maxima.items()
        )
        print(f"  overall: {overall}")
        for name, maximum in maxima.items():
            env_id = int(torch.argmax(maximum))
            stage_id = int(stage[env_id])
            if stage_id == 0:
                direction_name = A6_DIRECTION_NAMES[int(direction[env_id])]
                bank_name = "multiterrain_low"
                bank_row = int(raw_env._a6_reset_bank_index[env_id])
            else:
                direction_name = "not_applicable"
                bank_name = "natural_curriculum"
                bank_row = int(raw_env._a6_reset_natural_bank_index[env_id])
            print(
                f"  {name}_peak: step={int(peak_steps[name][env_id])}, "
                f"env={env_id}, stage={A6_STAGE_NAMES[stage_id]}, "
                f"scene={A6_SCENE_NAMES[int(scene[env_id])]}, "
                f"direction={direction_name}, "
                f"source={int(raw_env._a6_reset_source[env_id])}, "
                f"bank={bank_name}, bank_row={bank_row}"
            )
        print(f"  warning_threshold_violations={violations}")
        has_physics_warning = any(violations.values())
        if args.formal_success_eval:
            warning_by_metric = {
                name: maximum > thresholds[name]
                for name, maximum in maxima.items()
            }
            any_warning = torch.zeros(
                args.num_envs, dtype=torch.bool, device=args.device
            )
            for warning_mask in warning_by_metric.values():
                any_warning |= warning_mask
            for scene_id, scene_name in enumerate(A6_SCENE_NAMES):
                scene_mask = scene == scene_id
                values = ", ".join(
                    f"{name}={_success_fraction(mask, scene_mask)}"
                    for name, mask in warning_by_metric.items()
                )
                print(
                    f"  formal_physics_scene={scene_name}: "
                    f"any_warning={_success_fraction(any_warning, scene_mask)}, "
                    + values
                )
            guided = scene == 1
            for direction_id, direction_name in enumerate(A6_DIRECTION_NAMES):
                cohort = guided & (direction == direction_id)
                print(
                    f"  formal_physics_guided_direction={direction_name}: "
                    f"any_warning={_success_fraction(any_warning, cohort)}, "
                    "contact_penetration="
                    f"{_success_fraction(warning_by_metric['contact_penetration'], cohort)}"
                )
            status = (
                "completed_with_physics_warnings"
                if has_physics_warning
                else "completed_clean"
            )
            print(f"  formal_evaluation_complete=True, status={status}")
        elif has_physics_warning:
            raise RuntimeError("physics warning threshold crossed during rollout")
    finally:
        env.close()


if __name__ == "__main__":
    main()
