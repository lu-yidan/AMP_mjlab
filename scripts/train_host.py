"""Train HoST flat or A6 get-up tasks using mjlab and four-critic PPO."""

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host import host_ground29_env_cfg, host_ground_env_cfg
from g1recovery_host.a6_geometry import A6_PHYSICS_SUFFIX
from g1recovery_host.a6_reset import (
    MAX_GUIDED_EDGE_OFFSET_M,
    edge_offset_for_iteration,
)
from g1recovery_host.a6_reward import (
    A6_RESET_VARIANT,
    lateral_travel,
    staged_guided_knee_action,
)
from g1recovery_host.action import PHYSICS_REVISION
from g1recovery_host.model import HOST29_JOINT_NAMES, HOST_JOINT_NAMES
from g1recovery_host.phase_switch import reset_prone_history_after_escape
from g1recovery_host.ppo import (
    GAMMA,
    GROUPS,
    HostPolicy,
    discounted_progress_advantage,
    grouped_gae,
    ppo_update,
)
from g1recovery_host.stand_guard import apply_stand_pose_guard, stand_pose_ready


def train(args):
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device is unavailable; use --device cpu or fix GPU access"
        )
    is_a6 = args.task == "a6"
    staged_escape = (
        args.a6_staged_escape_v1
        or args.a6_staged_escape_v2
        or args.a6_staged_escape_v3
        or args.a6_staged_escape_v4
        or args.a6_staged_escape_v5
    )
    is_29 = args.task in ("flat29", "a6")
    model_source = "unitree_g1_29" if is_29 else args.model_source
    reset_variant = A6_RESET_VARIANT if is_a6 else None
    physics_revision = (
        f"{PHYSICS_REVISION}_{A6_PHYSICS_SUFFIX}"
        if is_a6
        else f"{PHYSICS_REVISION}_flat29_29_v1"
        if is_29
        else PHYSICS_REVISION
    )
    if is_a6:
        reward_variant = (
            "historical_a6_G+_29dof_pose_v6_v8_edge_curriculum_v1"
            if args.a6_edge_curriculum
            else f"historical_a6_{args.arm}_29dof_pose_v6_v{7 if args.arm == 'G+' else 3}"
            if args.a6_pose_v6
            else f"historical_a6_{args.arm}_29dof_v{7 if args.arm == 'G+' else 2}"
        )
        if args.frozen_flat_teacher:
            reward_variant += "_frozen_flat_teacher_v1"
        reward_variant += "_preescape_knee_v2"
        if args.a6_staged_escape_v1:
            reward_variant += "_staged_escape_v1"
        if args.a6_staged_escape_v2:
            reward_variant += "_staged_escape_v2"
        if args.a6_staged_escape_v3:
            reward_variant += "_staged_escape_v3"
        if args.a6_staged_escape_v4:
            reward_variant += "_staged_escape_v4"
        if args.a6_staged_escape_v5:
            reward_variant += "_staged_escape_v5"
        if args.a6_focused_guided:
            reward_variant += "_focused_guided_v1"
        if args.a6_safe_actions_v1:
            reward_variant += "_safe_actions_v1"
        if args.a6_progress_aux_v1:
            reward_variant += "_progress_aux_v1"
        if args.a6_aligned_edge_v1:
            reward_variant += "_aligned_edge_v1"
        if args.a6_lateral_actions_v1:
            reward_variant += "_lateral_actions_v1"
        if args.a6_lift_guard_v1:
            reward_variant += "_lift_guard_v1"
        if args.a6_knee_straight_prior_v1:
            reward_variant += "_knee_straight_prior_v1"
        if args.a6_footprint_edge_v1:
            reward_variant += "_footprint_edge_v1"
        if args.a6_arm_extraction_v1:
            reward_variant += "_arm_extraction_v1"
        if args.a6_prone_full_arms_v1:
            reward_variant += "_prone_full_arms_v1"
        if args.a6_supine_full_arms_v1:
            reward_variant += "_supine_full_arms_v1"
        if args.a6_postescape_prone_replay_v1:
            reward_variant += "_postescape_prone_replay_v1"
        if args.a6_prone_recovery_v1:
            reward_variant += "_prone_recovery_v1"
        if args.a6_bidirectional_recovery_v1:
            reward_variant += "_bidirectional_recovery_v1"
        if args.a6_phase_history_reset_v1:
            reward_variant += "_phase_history_reset_v1"
        if args.a6_postescape_live_obs_v1:
            reward_variant += "_postescape_live_obs_v1"
        if args.frozen_supine_teacher:
            reward_variant += "_frozen_supine_teacher_v1"
        if args.a6_episode_seconds == 15.0:
            reward_variant += "_episode_15s_v1"
        if args.a6_guided_teacher_v1:
            reward_variant += "_guided_teacher_v1"
    elif args.task == "flat29":
        reward_variant = (
            "height_guard_style_v2_29dof_pose_v6"
            if args.flat29_pose_v6
            else "height_guard_style_v2_29dof_pose_v5"
            if args.flat29_pose_v5
            else "height_guard_style_v2_29dof_pose_v4"
            if args.flat29_pose_v4
            else "height_guard_style_v2_29dof_pose_v3"
            if args.flat29_pose_v3
            else "height_guard_style_v2_29dof_standref_v2"
        )
    else:
        reward_variant = (
            ("height_guard_v1" if args.height_guard else "paper")
            if args.legacy_style
            else ("height_guard_style_v2" if args.height_guard else "paper_style_v2")
        )
    ppo_schedule = "adaptive_kl" if args.adaptive_kl else "fixed"
    if args.entropy_coef != 0.01:
        ppo_schedule += f"_entropy_{args.entropy_coef:g}"
    if is_a6:
        from g1recovery_host.a6_task import host_a6_env_cfg

        cfg = host_a6_env_cfg(
            args.num_envs,
            args.seed,
            arm=args.arm,
            preserve_pose=args.a6_pose_v6,
            edge_curriculum=args.a6_edge_curriculum,
            plate_observation=args.a6_staged_escape_v5,
            focused_guided=args.a6_focused_guided,
            postescape_prone=args.a6_postescape_prone_replay_v1,
            safe_actions=args.a6_safe_actions_v1,
            prone_full_arms=args.a6_prone_full_arms_v1,
            supine_full_arms=args.a6_supine_full_arms_v1,
            aligned_edge=args.a6_aligned_edge_v1,
            lateral_actions=args.a6_lateral_actions_v1,
            lift_guard=args.a6_lift_guard_v1,
            footprint_edge=args.a6_footprint_edge_v1,
            footprint_overlap_m=args.a6_footprint_overlap_m,
        )
        cfg.episode_length_s = args.a6_episode_seconds
    elif args.task == "flat29":
        cfg = host_ground29_env_cfg(
            args.num_envs,
            args.seed,
            pose_version="v6"
            if args.flat29_pose_v6
            else "v5"
            if args.flat29_pose_v5
            else "v4"
            if args.flat29_pose_v4
            else "v3"
            if args.flat29_pose_v3
            else "v2",
        )
    else:
        cfg = host_ground_env_cfg(
            args.num_envs,
            args.seed,
            model_source=args.model_source,
            height_guard=args.height_guard,
            legacy_style=args.legacy_style,
        )
    env = ManagerBasedRlEnv(cfg, device=args.device)
    if is_a6:
        env._a6_staged_escape_v1 = args.a6_staged_escape_v1
        env._a6_staged_escape_v2 = args.a6_staged_escape_v2
        env._a6_staged_escape_v3 = args.a6_staged_escape_v3
        env._a6_staged_escape_v4 = args.a6_staged_escape_v4
        env._a6_staged_escape_v5 = args.a6_staged_escape_v5
        env._a6_knee_straight_prior_v1 = args.a6_knee_straight_prior_v1
        env._a6_arm_extraction_v1 = args.a6_arm_extraction_v1
        env._a6_prone_recovery_v1 = args.a6_prone_recovery_v1
        env._a6_bidirectional_recovery_v1 = args.a6_bidirectional_recovery_v1
        env._a6_postescape_live_obs_v1 = args.a6_postescape_live_obs_v1
    pose_teacher_enabled = args.flat29_pose_v6 or (is_a6 and args.a6_pose_v6)
    policy = HostPolicy(
        obs_dim=612 if args.a6_staged_escape_v5 else 564 if is_29 else 456,
        action_dim=29 if is_29 else 23,
        log_std_bounds=(-2.3, math.log(args.action_std_max))
        if args.flat29_pose_v3
        or args.flat29_pose_v4
        or args.flat29_pose_v5
        or pose_teacher_enabled
        else None,
        upper_residual=args.flat29_pose_v4
        or args.flat29_pose_v5
        or pose_teacher_enabled,
        waist_residual_scale=5.0
        if args.flat29_pose_v5 or pose_teacher_enabled
        else 2.0,
    ).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    start_iteration = 0
    state = None
    parent_sha256 = None
    parent_iteration = None
    transfer_source_sha256 = None
    transfer_source_iteration = None
    transfer_source_reward_variant = None
    transfer_source_physics_revision = None
    optimizer_reset_on_transfer = False
    continued_curriculum = None
    init_mode = "fresh"
    if args.init_from:
        from g1recovery_host.a6_transfer import transfer_actor_23_to_29

        with Path(args.init_from).open("rb") as stream:
            parent_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        parent = torch.load(
            args.init_from, map_location=args.device, weights_only=False
        )
        transfer_actor_23_to_29(policy, parent)
        parent_iteration = parent.get("iteration")
        init_mode = "actor_23_to_29"
    if args.continue_from:
        from g1recovery_host.a6_transfer import continue_full_23_to_29

        with Path(args.continue_from).open("rb") as stream:
            parent_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        parent = torch.load(
            args.continue_from, map_location=args.device, weights_only=False
        )
        if parent.get("ppo_schedule") != ppo_schedule and not args.reset_optimizer:
            raise ValueError(
                "continuation PPO schedule differs; use --reset-optimizer "
                "or match the parent's --adaptive-kl setting"
            )
        continue_full_23_to_29(policy, optimizer, parent)
        if args.reset_optimizer:
            optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
            optimizer_reset_on_transfer = True
        continued_curriculum = parent["curriculum"]
        parent_iteration = parent["iteration"]
        start_iteration = parent_iteration + 1
        init_mode = "full_23_to_29"
    if args.continue_from29:
        from g1recovery_host.a6_transfer import continue_full_29_to_a6

        with Path(args.continue_from29).open("rb") as stream:
            parent_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        parent = torch.load(
            args.continue_from29, map_location=args.device, weights_only=False
        )
        if parent.get("ppo_schedule") != ppo_schedule and not args.reset_optimizer:
            raise ValueError(
                "continuation PPO schedule differs; use --reset-optimizer "
                "or match the parent's --adaptive-kl setting"
            )
        continue_full_29_to_a6(
            policy,
            optimizer,
            parent,
            extra_obs_dim=48 if args.a6_staged_escape_v5 else 0,
        )
        if args.reset_optimizer:
            optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
            optimizer_reset_on_transfer = True
        continued_curriculum = parent["curriculum"]
        parent_iteration = parent["iteration"]
        start_iteration = parent_iteration + 1
        init_mode = "full_29_pose_v6_to_a6"
    if args.init_from29:
        with Path(args.init_from29).open("rb") as stream:
            parent_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        parent = torch.load(
            args.init_from29, map_location=args.device, weights_only=False
        )
        if (
            parent.get("task") != "flat29"
            or parent.get("model_source") != "unitree_g1_29"
        ):
            raise ValueError(
                "--init-from29 requires a native 29-DoF flat HoST checkpoint"
            )
        source = parent["policy"]
        if tuple(source["actor.0.weight"].shape) != (512, 564):
            raise ValueError("29-DoF parent has incompatible actor dimensions")
        actor = {
            k.removeprefix("actor."): v
            for k, v in source.items()
            if k.startswith("actor.")
        }
        policy.actor.load_state_dict(actor)
        policy.log_std.data.copy_(source["log_std"])
        parent_iteration = parent.get("iteration")
        init_mode = "actor_29_to_a6"
    if args.warmstart_flat29:
        with Path(args.warmstart_flat29).open("rb") as stream:
            parent_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        parent = torch.load(
            args.warmstart_flat29, map_location=args.device, weights_only=False
        )
        if (
            parent.get("task") != "flat29"
            or parent.get("model_source") != model_source
            or parent.get("physics_revision") != physics_revision
        ):
            raise ValueError("--warmstart-flat29 requires compatible flat29 physics")
        missing, unexpected = policy.load_state_dict(parent["policy"], strict=False)
        if unexpected or any(
            not name.startswith("upper_residual.") for name in missing
        ):
            raise ValueError(
                f"warmstart parameter mismatch: missing={missing}, unexpected={unexpected}"
            )
        policy.log_std.data.clamp_(max=math.log(0.7))
        continued_curriculum = parent["curriculum"]
        parent_iteration = parent["iteration"]
        init_mode = "flat29_policy_new_reward_optimizer"
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        transfer_source_sha256 = state.get("transfer_source_sha256")
        transfer_source_iteration = state.get("transfer_source_iteration")
        transfer_source_reward_variant = state.get("transfer_source_reward_variant")
        transfer_source_physics_revision = state.get("transfer_source_physics_revision")
        optimizer_reset_on_transfer = state.get("optimizer_reset_on_transfer", False)
        checkpoint_source = state.get("model_source", "mjlab")
        checkpoint_revision = state.get("physics_revision", "single_torso_force_v1")
        checkpoint_reward = state.get("reward_variant", "paper")
        checkpoint_reset = state.get("a6_reset_variant")
        checkpoint_schedule = state.get("ppo_schedule", "fixed")
        same_setup = (
            checkpoint_source == model_source
            and checkpoint_revision == physics_revision
            and checkpoint_reward == reward_variant
            and checkpoint_reset == reset_variant
            and state.get("action_std_max", 0.7) == args.action_std_max
            and state.get("entropy_coef", 0.01) == args.entropy_coef
            and (
                not args.a6_footprint_edge_v1
                or state.get("a6_footprint_overlap_m", 0.25)
                == args.a6_footprint_overlap_m
            )
        )
        if not same_setup and not args.allow_model_transfer:
            raise ValueError(
                f"checkpoint model/physics/reward/reset={checkpoint_source!r}/"
                f"{checkpoint_revision!r}/{checkpoint_reward!r}/"
                f"{checkpoint_reset!r}, current={model_source!r}/"
                f"{physics_revision!r}/{reward_variant!r}/{reset_variant!r}; "
                "pass --allow-model-transfer to fine-tune across different "
                "robot dynamics, rewards, or reset rules"
            )
        if args.keep_curriculum_on_transfer and (
            checkpoint_source != model_source or checkpoint_revision != physics_revision
        ):
            raise ValueError(
                "--keep-curriculum-on-transfer requires matching model and "
                "physics revision; only reward or reset rules may change"
            )
        if checkpoint_schedule != ppo_schedule and not args.reset_optimizer:
            raise ValueError(
                f"checkpoint PPO schedule={checkpoint_schedule!r}, requested "
                f"{ppo_schedule!r}; use --reset-optimizer when changing schedule"
            )
        policy.load_state_dict(state["policy"])
        parent_sha256 = state.get("parent_sha256")
        parent_iteration = state.get("parent_iteration")
        init_mode = state.get("init_mode", "resume")
        if same_setup and not args.reset_optimizer:
            optimizer.load_state_dict(state["optimizer"])
        if not same_setup:
            with Path(args.resume).open("rb") as stream:
                transfer_source_sha256 = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            transfer_source_iteration = state["iteration"]
            transfer_source_reward_variant = checkpoint_reward
            transfer_source_physics_revision = checkpoint_revision
            optimizer_reset_on_transfer = True
        start_iteration = state["iteration"] + 1
    frozen_teacher = None
    frozen_teacher_sha256 = None
    if args.frozen_flat_teacher:
        teacher_path = Path(args.frozen_flat_teacher)
        with teacher_path.open("rb") as stream:
            frozen_teacher_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        teacher_checkpoint = torch.load(
            teacher_path, map_location=args.device, weights_only=False
        )
        if (
            teacher_checkpoint.get("task") != "flat29"
            or teacher_checkpoint.get("model_source") != "unitree_g1_29"
            or teacher_checkpoint.get("reward_variant")
            != "height_guard_style_v2_29dof_pose_v6"
            or tuple(teacher_checkpoint["policy"]["actor.6.weight"].shape) != (29, 128)
        ):
            raise ValueError("frozen teacher must be a 29-DoF flat pose-v6 checkpoint")
        if args.continue_from29 and frozen_teacher_sha256 != parent_sha256:
            raise ValueError(
                "frozen teacher must match the continuation parent exactly"
            )
        if args.resume and (
            frozen_teacher_sha256 != state.get("frozen_flat_teacher_sha256")
            or args.flat_teacher_weight != state.get("flat_teacher_weight")
        ):
            raise ValueError("resumed frozen teacher hash or weight differs")
        frozen_teacher = (
            HostPolicy(
                obs_dim=564,
                action_dim=29,
                log_std_bounds=(-2.3, math.log(0.7)),
                upper_residual=True,
                waist_residual_scale=5.0,
            ).to(args.device)
            if args.a6_staged_escape_v5
            else deepcopy(policy)
        )
        frozen_teacher.load_state_dict(teacher_checkpoint["policy"], strict=True)
        frozen_teacher.eval().requires_grad_(False)
        if args.continue_from29:
            init_mode += "_frozen_flat_teacher"
    supine_teacher = None
    supine_teacher_sha256 = None
    if args.frozen_supine_teacher:
        supine_path = Path(args.frozen_supine_teacher)
        with supine_path.open("rb") as stream:
            supine_teacher_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        supine_checkpoint = torch.load(
            supine_path, map_location=args.device, weights_only=False
        )
        if (
            supine_checkpoint.get("task") != "a6"
            or supine_checkpoint.get("model_source") != "unitree_g1_29"
            or "_staged_escape_v5" not in supine_checkpoint.get("reward_variant", "")
            or tuple(supine_checkpoint["policy"]["actor.6.weight"].shape) != (29, 128)
        ):
            raise ValueError("supine teacher must be a 29-DoF staged-v5 A6 checkpoint")
        guided_teacher_transfer = args.a6_guided_teacher_v1 and args.allow_model_transfer
        if args.resume and state.get("frozen_supine_teacher_sha256") is not None and (
            supine_teacher_sha256 != state["frozen_supine_teacher_sha256"]
            or args.supine_teacher_weight != state.get("supine_teacher_weight")
        ) and not guided_teacher_transfer:
            raise ValueError("resumed supine teacher hash or weight differs")
        supine_teacher = HostPolicy(
            obs_dim=612,
            action_dim=29,
            log_std_bounds=(-2.3, math.log(supine_checkpoint.get("action_std_max", 0.7))),
            upper_residual=True,
            waist_residual_scale=5.0,
        ).to(args.device)
        supine_teacher.load_state_dict(supine_checkpoint["policy"], strict=True)
        supine_teacher.eval().requires_grad_(False)
    if start_iteration >= args.iterations:
        raise ValueError(
            f"start iteration {start_iteration} must be less than "
            f"--iterations {args.iterations}; --iterations is the total count"
        )
    edge_curriculum = None
    adaptive_edge = None
    edge_restarted_from_m = None
    if args.a6_edge_curriculum:
        if args.a6_staged_escape_v5:
            from g1recovery_host.a6_curriculum import SuccessGatedEdgeCurriculum

            saved = deepcopy(state.get("edge_curriculum")) if args.resume else None
            if saved is not None and (
                saved.get("mode") != "success_gated"
                or saved.get("start_offset_m") != args.a6_edge_start_offset_m
            ):
                raise ValueError("resumed A6 success-gated curriculum differs")
            if args.a6_edge_restart_offset_m is not None:
                if saved is None:
                    raise ValueError("edge restart needs a saved success-gated curriculum")
                edge_restarted_from_m = float(saved["state"]["offset_m"])
                if args.a6_edge_restart_offset_m < edge_restarted_from_m:
                    raise ValueError("edge restart may only move the plate to an easier offset")
                saved["state"]["offset_m"] = args.a6_edge_restart_offset_m
                saved["state"]["history"] = []
                saved["state"]["advances"] = round(
                    (args.a6_edge_start_offset_m - args.a6_edge_restart_offset_m)
                    / 0.025
                )
                init_mode += "_edge_curriculum_restarted"
                print(
                    f"A6 edge curriculum restarted from {edge_restarted_from_m:.3f} "
                    f"to {args.a6_edge_restart_offset_m:.3f} m",
                    flush=True,
                )
            adaptive_edge = SuccessGatedEdgeCurriculum(
                args.a6_edge_start_offset_m,
                state=saved["state"] if saved else None,
            )
            edge_curriculum = {
                "mode": "success_gated",
                "start_offset_m": args.a6_edge_start_offset_m,
                "state": adaptive_edge.state_dict(),
            }
            env._a6_edge_offset_m = adaptive_edge.offset_m
        else:
            requested = {
                "start_offset_m": args.a6_edge_start_offset_m,
                "hold_updates": args.a6_edge_hold_updates,
                "anneal_updates": args.a6_edge_anneal_updates,
            }
            if args.resume:
                saved = state.get("edge_curriculum")
                if not isinstance(saved, dict) or any(
                    saved.get(key) != value for key, value in requested.items()
                ):
                    raise ValueError("resumed A6 edge-curriculum parameters differ")
                anchor_iteration = saved["anchor_iteration"]
            else:
                anchor_iteration = start_iteration
            edge_curriculum = {**requested, "anchor_iteration": anchor_iteration}
            env._a6_edge_offset_m = edge_offset_for_iteration(
                start_iteration,
                anchor_iteration,
                requested["start_offset_m"],
                requested["hold_updates"],
                requested["anneal_updates"],
            )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    try:
        obs, _ = env.reset()
        obs = obs["actor"]
        action_term = env.action_manager.get_term("joint_pos")
        restore_curriculum = not args.reset_curriculum and (
            continued_curriculum is not None
            or (state is not None and (same_setup or args.keep_curriculum_on_transfer))
        )
        if restore_curriculum:
            saved_curriculum = (
                continued_curriculum
                if continued_curriculum is not None
                else state.get("curriculum")
            )
            if saved_curriculum is None and state is not None:
                metrics_path = Path(args.resume).parent / "metrics.jsonl"
                if metrics_path.exists():
                    for line in reversed(
                        metrics_path.read_text(encoding="utf-8").splitlines()
                    ):
                        metric = json.loads(line)
                        if metric["iteration"] == state["iteration"]:
                            saved_curriculum = {
                                "beta": torch.tensor([metric["beta"]]),
                                "assist_force": torch.tensor(
                                    [metric["assist_force_n"]]
                                ),
                            }
                            break
            if saved_curriculum is not None:
                for name in ("beta", "assist_force"):
                    previous = saved_curriculum[name].to(args.device).reshape(-1)
                    current = getattr(action_term, name)
                    if len(previous) == args.num_envs:
                        current[:, 0] = previous
                    else:
                        current.fill_(float(previous.mean()))
        if not args.no_random_episode_lengths:
            env.episode_length_buf[:] = torch.randint(
                0,
                env.max_episode_length,
                (args.num_envs,),
                device=args.device,
            )
        expected_obs, expected_actions = (
            (612 if args.a6_staged_escape_v5 else 564, 29)
            if is_29
            else (456, 23)
        )
        if (
            obs.shape[1] != expected_obs
            or env.action_manager.total_action_dim != expected_actions
        ):
            raise ValueError(
                f"Expected HoST's {expected_obs} observations and {expected_actions} actions"
            )
        robot = env.scene["robot"]
        joint_names = HOST29_JOINT_NAMES if is_29 else HOST_JOINT_NAMES
        if (
            tuple(robot.joint_names) != joint_names
            or set(robot.actuator_names) != set(joint_names)
            or tuple(action_term._target_names) != joint_names
        ):
            raise ValueError(
                "Model joint/actuator order differs from the selected HoST task"
            )
        if tuple(env.reward_manager.active_terms) != GROUPS:
            raise ValueError("Reward groups do not match four-critic ordering")
        obstacle_worlds = env._a6_scene > 0 if is_a6 else None
        prone_guided_worlds = (
            (env._a6_scene == 1) & (env._a6_direction == 1) if is_a6 else None
        )
        supine_guided_worlds = (
            (env._a6_scene == 1) & (env._a6_direction == 0) if is_a6 else None
        )
        guided_worlds = env._a6_scene == 1 if is_a6 else None
        free_worlds = env._a6_scene == 2 if is_a6 else None
        for iteration in range(start_iteration, args.iterations):
            if edge_curriculum is not None:
                env._a6_edge_offset_m = (
                    adaptive_edge.offset_m
                    if adaptive_edge is not None
                    else edge_offset_for_iteration(
                        iteration,
                        edge_curriculum["anchor_iteration"],
                        edge_curriculum["start_offset_m"],
                        edge_curriculum["hold_updates"],
                        edge_curriculum["anneal_updates"],
                    )
                )
            env_reward_sum = torch.zeros(len(GROUPS), device=args.device)
            a6_obstacle_samples = torch.zeros(27, device=args.device) if is_a6 else None
            a6_arm_samples = torch.zeros(2, device=args.device) if is_a6 else None
            knee_actor_sum = torch.zeros(2, device=args.device)
            knee_guide_sum = torch.zeros(2, device=args.device)
            knee_guide_count = 0
            timeout_count = 0
            a6_attempts = (
                torch.zeros(2, 3, device=args.device, dtype=torch.long)
                if adaptive_edge is not None
                else None
            )
            rollout_keys = (
                "obs",
                "next_obs",
                "actions",
                "logp",
                "old_mu",
                "old_sigma",
                "values",
                "rewards",
                "dones",
                "active",
                "continue_mask",
            )
            if pose_teacher_enabled:
                rollout_keys += ("pose_teacher", "pose_teacher_mask")
            if frozen_teacher is not None:
                rollout_keys += ("frozen_teacher_mask",)
            if supine_teacher is not None:
                rollout_keys += ("supine_teacher_mask",)
            if is_a6 and args.arm == "G+":
                rollout_keys += ("advantage_scales",)
            if staged_escape:
                rollout_keys += ("knee_prior_actions", "knee_prior_mask")
            if args.a6_progress_aux_v1:
                rollout_keys += ("aux_progress_reward",)
            rollout = {key: [] for key in rollout_keys}
            for _ in range(args.steps_per_env):
                with torch.no_grad():
                    distribution = policy.distribution(obs)
                    action = distribution.sample()
                    logp = distribution.log_prob(action).sum(-1)
                    values = policy.values(obs)
                    if pose_teacher_enabled:
                        pose_teacher = apply_stand_pose_guard(
                            env, action, min_steps=90
                        )[:, 12:15]
                        pose_teacher_mask = stand_pose_ready(env, min_steps=90)
                        if is_a6:
                            pose_teacher_mask &= (env._a6_scene == 0) | env._a6_escaped
                active = (
                    env.action_manager.get_term("joint_pos").real_steps > 30
                )
                if is_a6 and hasattr(env, "_a6_postescape_start"):
                    active |= env._a6_postescape_start
                active = active.float()
                if staged_escape:
                    displacement = (
                        env.scene["robot"].data.root_link_pos_w[:, :2]
                        - env._a6_initial_root_xy
                    )
                    near_edge = (
                        0.18 * env._a6_near_edge_phase
                        if args.a6_staged_escape_v5
                        else lateral_travel(
                            displacement,
                            env._a6_initial_lateral_axis_xy,
                            toward_near_edge=True,
                        )
                    )
                    knee_prior_actions = staged_guided_knee_action(
                        env.scene["robot"].data.joint_pos[:, env._a6_knee_ids],
                        env._a6_direction,
                        near_edge,
                        strict=args.a6_knee_straight_prior_v1,
                    )
                    knee_prior_mask = (
                        active
                        * (
                            (env._a6_scene == 1)
                            & ~env._a6_escaped
                            & ((env._a6_direction == 0) | (env._a6_direction == 1))
                        ).float()
                    )
                    selected = knee_prior_mask > 0
                    if selected.any():
                        knee_actor_sum += distribution.mean[selected, :][
                            :, env._a6_knee_ids
                        ].sum(0)
                        knee_guide_sum += knee_prior_actions[selected].sum(0)
                        knee_guide_count += int(selected.sum())
                if frozen_teacher is not None:
                    frozen_teacher_mask = (
                        active * ((env._a6_scene == 0) | env._a6_escaped).float()
                    )
                    if args.a6_prone_recovery_v1:
                        # The flat teacher stands from only a minority of real
                        # prone exit poses. Let PPO learn that transition, then
                        # restore the frozen teacher for the upright pose.
                        recovering_prone = (
                            (env._a6_scene == 1)
                            & (env._a6_direction == 1)
                            & env._a6_escaped
                            & ~stand_pose_ready(env, min_steps=90)
                        )
                        frozen_teacher_mask *= (~recovering_prone).float()
                if supine_teacher is not None:
                    guided_teacher_scene = (
                        (env._a6_scene == 1)
                        if args.a6_guided_teacher_v1
                        else (env._a6_scene == 1) & (env._a6_direction == 0)
                    )
                    supine_teacher_mask = active * guided_teacher_scene.float()
                    if args.a6_guided_teacher_v1:
                        supine_teacher_mask *= (~env._a6_escaped).float()
                    if args.a6_bidirectional_recovery_v1:
                        # The old supine teacher is useful under the board, but
                        # competes with flat recovery imitation after exit.
                        supine_teacher_mask *= (~env._a6_escaped).float()
                if is_a6 and args.arm == "G+":
                    obstructed = (env._a6_scene > 0) & ~env._a6_escaped
                    advantage_scales = torch.ones(
                        args.num_envs, len(GROUPS), device=args.device
                    )
                    advantage_scales[:, (0, 2)] = torch.where(obstructed, 0.05, 1.0)[
                        :, None
                    ]
                pre_escaped = (
                    env._a6_escaped.clone() if adaptive_edge is not None else None
                )
                next_obs, _, terminated, truncated, _ = env.step(action)
                if args.a6_phase_history_reset_v1:
                    next_obs, _ = reset_prone_history_after_escape(
                        env, pre_escaped, terminated | truncated
                    )
                if adaptive_edge is not None:
                    new_escape = ~pre_escaped & env._a6_escaped
                    invalid_term = env.termination_manager.get_term("a6_invalid")
                    unresolved_end = (terminated | truncated) & ~pre_escaped
                    for posture, scene_mask in enumerate(
                        (prone_guided_worlds, supine_guided_worlds)
                    ):
                        a6_attempts[posture, 0] += (scene_mask & new_escape).sum()
                        a6_attempts[posture, 1] += (
                            scene_mask & unresolved_end & invalid_term
                        ).sum()
                        a6_attempts[posture, 2] += (
                            scene_mask & unresolved_end & ~invalid_term
                        ).sum()
                if is_a6:
                    a6_obstacle_samples[0] += env._a6_escaped[obstacle_worlds].sum()
                    a6_obstacle_samples[1] += env._a6_ever_contact[
                        obstacle_worlds
                    ].sum()
                    a6_obstacle_samples[2] += env._a6_escape_values[
                        prone_guided_worlds, 7
                    ].sum()
                    a6_obstacle_samples[3] += env._a6_escape_values[
                        obstacle_worlds, 8
                    ].sum()
                    a6_obstacle_samples[4] += env._a6_escape_values[
                        guided_worlds, 10
                    ].sum()
                    a6_obstacle_samples[5] += env._a6_escape_values[
                        supine_guided_worlds, 11
                    ].sum()
                    for metric_idx, value_idx in (
                        (6, 12),
                        (7, 13),
                        (8, 14),
                        (9, 15),
                        (15, 16),
                        (16, 17),
                        (17, 18),
                        (18, 19),
                        (19, 20),
                        (20, 21),
                        (21, 22),
                        (22, 23),
                        (23, 24),
                        (24, 25),
                    ):
                        a6_obstacle_samples[metric_idx] += env._a6_escape_values[
                            guided_worlds, value_idx
                        ].sum()
                    a6_obstacle_samples[10] += env._a6_escaped[guided_worlds].sum()
                    a6_obstacle_samples[11] += env._a6_escaped[free_worlds].sum()
                    a6_obstacle_samples[12] += env._a6_escaped[
                        prone_guided_worlds
                    ].sum()
                    a6_obstacle_samples[13] += env._a6_escaped[
                        supine_guided_worlds
                    ].sum()
                    # Auto-reset clears env._a6_invalid before step() returns;
                    # the termination manager retains the pre-reset flags.
                    a6_obstacle_samples[14] += env.termination_manager.get_term(
                        "a6_invalid"
                    )[guided_worlds].sum()
                    a6_obstacle_samples[25] += env._a6_escape_values[
                        guided_worlds, 26
                    ].sum()
                    a6_obstacle_samples[26] += env._a6_escape_values[
                        free_worlds, 26
                    ].sum()
                    a6_arm_samples[0] += env._a6_arm_depth[guided_worlds].sum()
                    a6_arm_samples[1] += env._a6_arm_extraction_reward[
                        guided_worlds
                    ].sum()
                done = terminated | truncated
                group_rewards = env.reward_manager._step_reward.clone()
                if cfg.scale_rewards_by_dt:
                    # mjlab's private group tensor is always an unscaled rate.
                    group_rewards *= env.step_dt
                env_reward_sum += group_rewards.sum(0)
                timeout_count += int(truncated.sum())
                # Auto-reset discards terminal observations. HoST's RSL-RL
                # runner bootstraps time-outs with the pre-step value estimate.
                group_rewards += GAMMA * values * truncated.unsqueeze(1)
                rollout["obs"].append(obs.clone())
                rollout["next_obs"].append(next_obs["actor"].clone())
                rollout["actions"].append(action)
                rollout["logp"].append(logp)
                rollout["old_mu"].append(distribution.mean)
                rollout["old_sigma"].append(distribution.stddev)
                rollout["values"].append(values)
                rollout["rewards"].append(group_rewards)
                rollout["dones"].append(done)
                rollout["active"].append(active)
                rollout["continue_mask"].append((~done).float())
                if args.a6_progress_aux_v1:
                    rollout["aux_progress_reward"].append(
                        env._a6_aux_progress_reward.clone()
                    )
                if pose_teacher_enabled:
                    rollout["pose_teacher"].append(pose_teacher)
                    rollout["pose_teacher_mask"].append(pose_teacher_mask.float())
                if frozen_teacher is not None:
                    rollout["frozen_teacher_mask"].append(frozen_teacher_mask)
                if supine_teacher is not None:
                    rollout["supine_teacher_mask"].append(supine_teacher_mask)
                if is_a6 and args.arm == "G+":
                    rollout["advantage_scales"].append(advantage_scales)
                if staged_escape:
                    rollout["knee_prior_actions"].append(knee_prior_actions)
                    rollout["knee_prior_mask"].append(knee_prior_mask)
                obs = next_obs["actor"]
            data = {key: torch.stack(value) for key, value in rollout.items()}
            with torch.no_grad():
                returns, advantages = grouped_gae(
                    data["rewards"],
                    data["values"],
                    data["dones"],
                    policy.values(obs),
                    advantage_scales=(
                        data["advantage_scales"] if is_a6 and args.arm == "G+" else None
                    ),
                )
                if args.a6_progress_aux_v1:
                    progress_advantage = discounted_progress_advantage(
                        data["aux_progress_reward"], data["dones"]
                    )
                    advantages += 0.5 * progress_advantage * data["active"]
            batch_tensors = (
                data["obs"],
                data["next_obs"],
                data["actions"],
                data["logp"],
                data["old_mu"],
                data["old_sigma"],
                data["values"],
                returns,
                advantages,
                data["active"],
                data["continue_mask"],
            )
            if pose_teacher_enabled:
                batch_tensors += (
                    data["pose_teacher"],
                    data["pose_teacher_mask"],
                )
            batch = tuple(
                tensor.reshape((-1,) + tensor.shape[2:]) for tensor in batch_tensors
            )
            (
                policy_loss,
                value_loss,
                actor_grad,
                critic_grad,
                teacher_loss,
                frozen_teacher_loss,
                supine_teacher_loss,
                knee_prior_loss,
                mean_kl,
            ) = ppo_update(
                policy,
                optimizer,
                batch,
                epochs=args.epochs,
                minibatches=args.minibatches,
                adaptive_kl=args.adaptive_kl,
                max_learning_rate=args.max_learning_rate,
                frozen_teacher=frozen_teacher,
                frozen_teacher_mask=(
                    data["frozen_teacher_mask"].reshape(-1)
                    if frozen_teacher is not None
                    else None
                ),
                frozen_teacher_weight=args.flat_teacher_weight,
                supine_teacher=supine_teacher,
                supine_teacher_mask=(
                    data["supine_teacher_mask"].reshape(-1)
                    if supine_teacher is not None else None
                ),
                supine_teacher_weight=(
                    args.supine_teacher_weight if supine_teacher is not None else 0.0
                ),
                knee_prior_actions=(
                    data["knee_prior_actions"].reshape(-1, 2) if staged_escape else None
                ),
                knee_prior_mask=(
                    data["knee_prior_mask"].reshape(-1) if staged_escape else None
                ),
                knee_prior_joint_ids=(
                    tuple(env._a6_knee_ids) if staged_escape else (3, 9)
                ),
                knee_prior_weight=0.02 if staged_escape else 0.0,
                entropy_coef=args.entropy_coef,
            )
            means = (env_reward_sum / (args.steps_per_env * args.num_envs)).tolist()
            metrics = {
                "iteration": iteration,
                "model_source": model_source,
                "physics_revision": physics_revision,
                "reward_variant": reward_variant,
                "a6_reset_variant": reset_variant,
                "task": args.task,
                "arm": args.arm if is_a6 else None,
                "ppo_schedule": ppo_schedule,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "max_learning_rate": args.max_learning_rate,
                "mean_kl": mean_kl,
                "time_outs": timeout_count,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "actor_grad": actor_grad,
                "critic_grad": critic_grad,
                "teacher_loss": teacher_loss,
                "frozen_teacher_loss": frozen_teacher_loss,
                "supine_teacher_loss": supine_teacher_loss,
                "knee_prior_loss": knee_prior_loss,
                "frozen_teacher_fraction": (
                    float(data["frozen_teacher_mask"].mean())
                    if frozen_teacher is not None
                    else 0.0
                ),
                "supine_teacher_fraction": (
                    float(data["supine_teacher_mask"].mean())
                    if supine_teacher is not None else 0.0
                ),
                "beta": float(env.action_manager.get_term("joint_pos").beta.mean()),
                "assist_force_n": float(
                    env.action_manager.get_term("joint_pos").assist_force.mean()
                ),
                "rewards": dict(zip(GROUPS, means)),
            }
            if is_a6 and args.arm == "G+":
                metrics["a6_obstructed_advantage_fraction"] = float(
                    (data["advantage_scales"][:, :, 2] < 1.0).float().mean()
                )
            if staged_escape:
                count = max(knee_guide_count, 1)
                metrics["a6_knee_prior_fraction"] = float(
                    data["knee_prior_mask"].mean()
                )
                metrics["a6_knee_actor_mean"] = (knee_actor_sum / count).tolist()
                metrics["a6_knee_guide_mean"] = (knee_guide_sum / count).tolist()
            if args.a6_progress_aux_v1:
                metrics["a6_aux_progress_reward_mean"] = float(
                    data["aux_progress_reward"].mean()
                )
                metrics["a6_aux_progress_positive_fraction"] = float(
                    (data["aux_progress_reward"] > 0).float().mean()
                )
            if is_a6:
                denominator = args.steps_per_env * int(obstacle_worlds.sum())
                metrics["a6_escaped_fraction"] = float(
                    a6_obstacle_samples[0] / denominator
                )
                metrics["a6_contact_fraction"] = float(
                    a6_obstacle_samples[1] / denominator
                )
                metrics["a6_prone_lateral_gain"] = float(
                    a6_obstacle_samples[2]
                    / (args.steps_per_env * int(prone_guided_worlds.sum()))
                )
                metrics["a6_no_progress"] = float(a6_obstacle_samples[3] / denominator)
                metrics["a6_guided_lift_cost"] = float(
                    a6_obstacle_samples[4]
                    / (args.steps_per_env * int(guided_worlds.sum()))
                )
                guided_denominator = args.steps_per_env * int(guided_worlds.sum())
                metrics["a6_guided_arm_depth_m"] = float(
                    a6_arm_samples[0] / guided_denominator
                )
                metrics["a6_guided_arm_extraction_reward"] = float(
                    a6_arm_samples[1] / guided_denominator
                )
                metrics["a6_supine_lateral_gain"] = float(
                    a6_obstacle_samples[5]
                    / (args.steps_per_env * int(supine_guided_worlds.sum()))
                )
                for name, idx in (
                    ("a6_guided_knee_bend", 6),
                    ("a6_guided_knee_extension", 7),
                    ("a6_guided_leg_progress", 8),
                    ("a6_guided_leg_release", 9),
                    ("a6_guided_leg_straightness", 15),
                    ("a6_guided_hip_bend", 16),
                    ("a6_guided_moving_straight", 17),
                    ("a6_guided_torso_travel", 18),
                    ("a6_guided_headward_arms", 19),
                    ("a6_guided_linear_leg_alignment", 20),
                    ("a6_guided_hand_push", 21),
                    ("a6_guided_hand_drive", 22),
                    ("a6_guided_hand_plant", 23),
                    ("a6_guided_overcurl", 24),
                    ("a6_guided_escaped_fraction", 10),
                    ("a6_guided_invalid_fraction", 14),
                ):
                    metrics[name] = float(a6_obstacle_samples[idx] / guided_denominator)
                metrics["a6_guided_invalid_count"] = int(a6_obstacle_samples[14])
                metrics["a6_guided_pre_escape_knee_curl"] = float(
                    a6_obstacle_samples[25] / guided_denominator
                )
                metrics["a6_guided_pre_escape_knee_penalty_step"] = (
                    -env.step_dt * metrics["a6_guided_pre_escape_knee_curl"]
                )
                metrics["a6_free_pre_escape_knee_curl"] = float(
                    a6_obstacle_samples[26]
                    / (args.steps_per_env * int(free_worlds.sum()))
                )
                metrics["a6_free_pre_escape_knee_penalty_step"] = (
                    -env.step_dt * metrics["a6_free_pre_escape_knee_curl"]
                )
                metrics["a6_free_escaped_fraction"] = float(
                    a6_obstacle_samples[11]
                    / (args.steps_per_env * int(free_worlds.sum()))
                )
                metrics["a6_prone_guided_escaped_fraction"] = float(
                    a6_obstacle_samples[12]
                    / (args.steps_per_env * int(prone_guided_worlds.sum()))
                )
                metrics["a6_supine_guided_escaped_fraction"] = float(
                    a6_obstacle_samples[13]
                    / (args.steps_per_env * int(supine_guided_worlds.sum()))
                )
                if edge_curriculum is not None:
                    metrics["a6_guided_edge_target_m"] = env._a6_edge_offset_m
                    metrics["a6_guided_edge_actual_mean_m"] = float(
                        env._a6_initial_edge_offset_m[guided_worlds].mean()
                    )
                if adaptive_edge is not None:
                    attempts = a6_attempts.tolist()
                    advanced, totals = adaptive_edge.observe(
                        tuple(attempts[0]), tuple(attempts[1])
                    )
                    edge_curriculum["state"] = adaptive_edge.state_dict()
                    metrics["a6_edge_advanced"] = advanced
                    metrics["a6_edge_next_offset_m"] = adaptive_edge.offset_m
                    metrics["a6_prone_attempts"] = attempts[0]
                    metrics["a6_supine_attempts"] = attempts[1]
                    metrics["a6_edge_window_attempts"] = totals
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if iteration == start_iteration or (iteration + 1) % args.log_every == 0:
                print(
                    f"iteration={iteration} policy_loss={policy_loss:.5f} "
                    f"value_loss={value_loss:.5f} "
                    f"actor_grad={actor_grad:.3f} critic_grad={critic_grad:.3f} "
                    f"teacher_loss={teacher_loss:.4f} "
                    f"frozen_teacher_loss={frozen_teacher_loss:.4f} "
                    f"supine_teacher_loss={supine_teacher_loss:.4f} "
                    + " ".join(
                        f"{name}={value:.5f}" for name, value in zip(GROUPS, means)
                    )
                    + f" beta={metrics['beta']:.3f}"
                    + (
                        f" a6_escape={metrics['a6_escaped_fraction']:.4f}"
                        if is_a6
                        else ""
                    ),
                    flush=True,
                )
            if (
                iteration + 1
            ) % args.save_every == 0 or iteration + 1 == args.iterations:
                path = output / f"model_{iteration:05d}.pt"
                torch.save(
                    {
                        "policy": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "curriculum": {
                            "beta": action_term.beta.detach().cpu().clone(),
                            "assist_force": action_term.assist_force.detach()
                            .cpu()
                            .clone(),
                        },
                        "iteration": iteration,
                        "model_source": model_source,
                        "physics_revision": physics_revision,
                        "reward_variant": reward_variant,
                        "a6_reset_variant": reset_variant,
                        "task": args.task,
                        "arm": args.arm if is_a6 else None,
                        "parent_sha256": parent_sha256,
                        "parent_iteration": parent_iteration,
                        "transfer_source_sha256": transfer_source_sha256,
                        "transfer_source_iteration": transfer_source_iteration,
                        "transfer_source_reward_variant": transfer_source_reward_variant,
                        "transfer_source_physics_revision": transfer_source_physics_revision,
                        "optimizer_reset_on_transfer": optimizer_reset_on_transfer,
                        "init_mode": init_mode,
                        "frozen_flat_teacher_sha256": frozen_teacher_sha256,
                        "frozen_supine_teacher_sha256": supine_teacher_sha256,
                        "supine_teacher_weight": (
                            args.supine_teacher_weight if supine_teacher is not None else None
                        ),
                        "flat_teacher_weight": (
                            args.flat_teacher_weight
                            if frozen_teacher is not None
                            else None
                        ),
                        "edge_curriculum": edge_curriculum,
                        "a6_safe_actions_v1": args.a6_safe_actions_v1,
                        "a6_prone_full_arms_v1": args.a6_prone_full_arms_v1,
                        "a6_supine_full_arms_v1": args.a6_supine_full_arms_v1,
                        "a6_postescape_prone_replay_v1": args.a6_postescape_prone_replay_v1,
                        "a6_prone_recovery_v1": args.a6_prone_recovery_v1,
                        "a6_bidirectional_recovery_v1": args.a6_bidirectional_recovery_v1,
                        "a6_guided_teacher_v1": args.a6_guided_teacher_v1,
                        "a6_phase_history_reset_v1": args.a6_phase_history_reset_v1,
                        "a6_postescape_live_obs_v1": args.a6_postescape_live_obs_v1,
                        "a6_progress_aux_v1": args.a6_progress_aux_v1,
                        "a6_footprint_overlap_m": args.a6_footprint_overlap_m,
                        "a6_edge_restart_offset_m": args.a6_edge_restart_offset_m,
                        "a6_edge_restarted_from_m": edge_restarted_from_m,
                        "a6_episode_seconds": args.a6_episode_seconds,
                        "ppo_schedule": ppo_schedule,
                        "max_learning_rate": args.max_learning_rate,
                        "action_std_max": args.action_std_max,
                        "entropy_coef": args.entropy_coef,
                        "num_envs": args.num_envs,
                        "steps_per_env": args.steps_per_env,
                        "seed": args.seed,
                    },
                    path,
                )
                print(f"checkpoint={path}", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("flat", "flat29", "a6"), default="flat")
    parser.add_argument("--arm", choices=("G+", "G-"), default="G+")
    parser.add_argument(
        "--init-from",
        default=None,
        help="23-DoF HoST checkpoint for 29-DoF actor-only transfer",
    )
    parser.add_argument(
        "--init-from29",
        default=None,
        help="29-DoF flat HoST checkpoint for A6 actor-only transfer",
    )
    parser.add_argument(
        "--continue-from",
        default=None,
        help="23-DoF flat checkpoint for full 29-DoF A6 continuation",
    )
    parser.add_argument(
        "--continue-from29",
        default=None,
        help="29-DoF flat pose-v6 checkpoint for full A6 PPO continuation",
    )
    parser.add_argument(
        "--a6-pose-v6",
        action="store_true",
        help="retain pose-v6 residual and standing reward when resuming A6",
    )
    parser.add_argument(
        "--frozen-flat-teacher",
        default=None,
        help="fixed 29-DoF flat pose-v6 checkpoint for flat and post-escape imitation",
    )
    parser.add_argument("--flat-teacher-weight", type=float, default=0.5)
    parser.add_argument(
        "--frozen-supine-teacher",
        default=None,
        help="fixed 29-DoF A6 checkpoint for guided-supine action imitation",
    )
    parser.add_argument("--supine-teacher-weight", type=float, default=0.5)
    parser.add_argument("--a6-edge-curriculum", action="store_true")
    parser.add_argument("--a6-staged-escape-v1", action="store_true")
    parser.add_argument("--a6-staged-escape-v2", action="store_true")
    parser.add_argument("--a6-staged-escape-v3", action="store_true")
    parser.add_argument("--a6-staged-escape-v4", action="store_true")
    parser.add_argument("--a6-staged-escape-v5", action="store_true")
    parser.add_argument("--a6-focused-guided", action="store_true")
    parser.add_argument("--a6-safe-actions-v1", action="store_true")
    parser.add_argument("--a6-progress-aux-v1", action="store_true")
    parser.add_argument("--a6-aligned-edge-v1", action="store_true")
    parser.add_argument("--a6-lateral-actions-v1", action="store_true")
    parser.add_argument("--a6-lift-guard-v1", action="store_true")
    parser.add_argument("--a6-knee-straight-prior-v1", action="store_true")
    parser.add_argument("--a6-footprint-edge-v1", action="store_true")
    parser.add_argument("--a6-footprint-overlap-m", type=float, default=0.25)
    parser.add_argument("--a6-arm-extraction-v1", action="store_true")
    parser.add_argument("--a6-prone-full-arms-v1", action="store_true")
    parser.add_argument("--a6-supine-full-arms-v1", action="store_true")
    parser.add_argument("--a6-postescape-prone-replay-v1", action="store_true")
    parser.add_argument("--a6-prone-recovery-v1", action="store_true")
    parser.add_argument("--a6-bidirectional-recovery-v1", action="store_true")
    parser.add_argument("--a6-guided-teacher-v1", action="store_true")
    parser.add_argument("--a6-phase-history-reset-v1", action="store_true")
    parser.add_argument("--a6-postescape-live-obs-v1", action="store_true")
    parser.add_argument("--a6-edge-start-offset-m", type=float, default=0.25)
    parser.add_argument("--a6-episode-seconds", type=float, choices=(10.0, 15.0), default=10.0)
    parser.add_argument("--a6-edge-restart-offset-m", type=float, default=None)
    parser.add_argument("--a6-edge-hold-updates", type=int, default=100)
    parser.add_argument("--a6-edge-anneal-updates", type=int, default=300)
    parser.add_argument("--flat29-pose-v3", action="store_true")
    parser.add_argument("--flat29-pose-v4", action="store_true")
    parser.add_argument("--flat29-pose-v5", action="store_true")
    parser.add_argument("--flat29-pose-v6", action="store_true")
    parser.add_argument(
        "--warmstart-flat29",
        default=None,
        help="compatible 29-DoF flat checkpoint for a new reward run",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-source",
        choices=("mjlab", "host_urdf", "host_urdf_capsule"),
        default="mjlab",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps-per-env", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-learning-rate", type=float, default=1e-2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--action-std-max", type=float, default=0.7)
    parser.add_argument("--adaptive-kl", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--allow-model-transfer", action="store_true")
    parser.add_argument("--keep-curriculum-on-transfer", action="store_true")
    parser.add_argument("--height-guard", action="store_true")
    parser.add_argument("--legacy-style", action="store_true")
    parser.add_argument("--reset-curriculum", action="store_true")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--no-random-episode-lengths", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        parser.error("--entropy-coef must be finite and nonnegative")
    if not math.isfinite(args.max_learning_rate) or args.max_learning_rate < 1e-5:
        parser.error("--max-learning-rate must be finite and at least 1e-5")
    if not math.isfinite(args.action_std_max) or not 0.1 <= args.action_std_max <= 0.7:
        parser.error("--action-std-max must be in [0.1, 0.7]")
    if (
        sum(
            bool(x)
            for x in (
                args.init_from,
                args.init_from29,
                args.continue_from,
                args.continue_from29,
                args.warmstart_flat29,
                args.resume,
            )
        )
        > 1
    ):
        parser.error("initialization and resume options are mutually exclusive")
    if args.init_from and args.task == "flat":
        parser.error("--init-from is only supported for 29-DoF tasks")
    if args.init_from29 and args.task != "a6":
        parser.error("--init-from29 is only supported for --task a6")
    if args.continue_from and args.task != "a6":
        parser.error("--continue-from is only supported for --task a6")
    if args.continue_from29 and args.task != "a6":
        parser.error("--continue-from29 is only supported for --task a6")
    if args.continue_from29:
        args.a6_pose_v6 = True
    if args.a6_pose_v6 and (
        args.task != "a6" or not (args.continue_from29 or args.resume)
    ):
        parser.error("--a6-pose-v6 requires A6 --continue-from29 or --resume")
    if args.frozen_flat_teacher and (
        args.task != "a6"
        or not args.a6_pose_v6
        or not (args.continue_from29 or args.resume)
    ):
        parser.error(
            "--frozen-flat-teacher requires A6 pose-v6 --continue-from29 or --resume"
        )
    if args.frozen_supine_teacher and not (
        args.task == "a6"
        and args.arm == "G+"
        and args.a6_staged_escape_v5
        and args.frozen_flat_teacher
        and args.resume
    ):
        parser.error(
            "--frozen-supine-teacher requires G+ staged-v5 A6 resume with flat teacher"
        )
    if args.a6_edge_curriculum and (
        args.task != "a6"
        or args.arm != "G+"
        or not args.a6_pose_v6
        or not args.frozen_flat_teacher
        or not (args.continue_from29 or args.resume)
    ):
        parser.error(
            "--a6-edge-curriculum requires G+ A6 pose-v6, a 29-DoF "
            "continuation/resume, and the frozen flat teacher"
        )
    if (
        sum(
            (
                args.a6_staged_escape_v1,
                args.a6_staged_escape_v2,
                args.a6_staged_escape_v3,
                args.a6_staged_escape_v4,
                args.a6_staged_escape_v5,
            )
        )
        > 1
    ):
        parser.error("choose only one A6 staged-escape version")
    if (
        args.a6_staged_escape_v1
        or args.a6_staged_escape_v2
        or args.a6_staged_escape_v3
        or args.a6_staged_escape_v4
        or args.a6_staged_escape_v5
    ) and not args.a6_edge_curriculum:
        parser.error("A6 staged escape requires --a6-edge-curriculum")
    if args.a6_focused_guided and not args.a6_staged_escape_v5:
        parser.error("--a6-focused-guided requires --a6-staged-escape-v5")
    if args.a6_aligned_edge_v1 and not args.a6_edge_curriculum:
        parser.error("--a6-aligned-edge-v1 requires --a6-edge-curriculum")
    if args.a6_lateral_actions_v1 and not args.a6_safe_actions_v1:
        parser.error("--a6-lateral-actions-v1 requires --a6-safe-actions-v1")
    if args.a6_lift_guard_v1 and not args.a6_safe_actions_v1:
        parser.error("--a6-lift-guard-v1 requires --a6-safe-actions-v1")
    if args.a6_knee_straight_prior_v1 and not args.a6_staged_escape_v5:
        parser.error("--a6-knee-straight-prior-v1 requires staged escape v5")
    if args.a6_footprint_edge_v1 and not args.a6_aligned_edge_v1:
        parser.error("--a6-footprint-edge-v1 requires aligned edge v1")
    if args.a6_arm_extraction_v1 and not args.a6_staged_escape_v5:
        parser.error("--a6-arm-extraction-v1 requires staged escape v5")
    if args.a6_prone_full_arms_v1 and not (
        args.a6_safe_actions_v1 and args.a6_staged_escape_v5 and args.arm == "G+"
    ):
        parser.error("--a6-prone-full-arms-v1 requires G+ staged v5 safe actions")
    if args.a6_supine_full_arms_v1 and not (
        args.a6_safe_actions_v1 and args.a6_staged_escape_v5 and args.arm == "G+"
    ):
        parser.error("--a6-supine-full-arms-v1 requires G+ staged v5 safe actions")
    if args.a6_postescape_prone_replay_v1 and not (
        args.a6_focused_guided and args.a6_staged_escape_v5 and args.arm == "G+"
    ):
        parser.error("--a6-postescape-prone-replay-v1 requires G+ focused staged v5")
    if args.a6_postescape_live_obs_v1 and not args.a6_postescape_prone_replay_v1:
        parser.error("--a6-postescape-live-obs-v1 requires postescape prone replay")
    if args.a6_prone_recovery_v1 and not (
        args.a6_postescape_prone_replay_v1 and args.frozen_flat_teacher
    ):
        parser.error("--a6-prone-recovery-v1 requires replay and a flat teacher")
    if args.a6_bidirectional_recovery_v1 and not (
        args.a6_prone_recovery_v1 and args.frozen_supine_teacher
    ):
        parser.error("--a6-bidirectional-recovery-v1 requires prone recovery and a supine teacher")
    if args.a6_guided_teacher_v1 and not (
        args.task == "a6" and args.arm == "G+"
        and args.a6_staged_escape_v5 and args.frozen_supine_teacher
    ):
        parser.error("--a6-guided-teacher-v1 requires G+ staged v5 and a guided teacher")
    if args.a6_phase_history_reset_v1 and not (
        args.a6_prone_recovery_v1 and args.a6_edge_curriculum
    ):
        parser.error("--a6-phase-history-reset-v1 requires prone recovery and edge curriculum")
    if not math.isfinite(args.a6_footprint_overlap_m) or not 0 < args.a6_footprint_overlap_m <= 0.50:
        parser.error("--a6-footprint-overlap-m must be in (0, 0.50]")
    if args.a6_episode_seconds != 10.0 and args.task != "a6":
        parser.error("--a6-episode-seconds is only valid for A6 training")
    if (args.a6_safe_actions_v1 or args.a6_progress_aux_v1) and not (
        args.a6_staged_escape_v5 and args.a6_edge_curriculum
    ):
        parser.error("A6 safe actions and progress auxiliary require staged v5")
    if (
        not math.isfinite(args.a6_edge_start_offset_m)
        or not 0 < args.a6_edge_start_offset_m <= MAX_GUIDED_EDGE_OFFSET_M
        or args.a6_edge_hold_updates < 0
        or args.a6_edge_anneal_updates <= 0
    ):
        parser.error("invalid A6 edge-curriculum offset or update schedule")
    if args.a6_edge_restart_offset_m is not None and not (
        args.task == "a6"
        and args.a6_staged_escape_v5
        and args.a6_edge_curriculum
        and args.resume
        and args.allow_model_transfer
        and args.keep_curriculum_on_transfer
        and math.isfinite(args.a6_edge_restart_offset_m)
        and 0 <= args.a6_edge_restart_offset_m <= args.a6_edge_start_offset_m
    ):
        parser.error("edge restart requires a finite easier staged-v5 transfer offset")
    if not math.isfinite(args.flat_teacher_weight) or args.flat_teacher_weight <= 0:
        parser.error("--flat-teacher-weight must be finite and positive")
    if not math.isfinite(args.supine_teacher_weight) or args.supine_teacher_weight <= 0:
        parser.error("--supine-teacher-weight must be finite and positive")
    if (
        sum(
            (
                args.flat29_pose_v3,
                args.flat29_pose_v4,
                args.flat29_pose_v5,
                args.flat29_pose_v6,
            )
        )
        > 1
    ):
        parser.error("select only one flat29 pose version")
    if (
        args.flat29_pose_v3
        or args.flat29_pose_v4
        or args.flat29_pose_v5
        or args.flat29_pose_v6
    ) and args.task != "flat29":
        parser.error("flat29 pose versions require --task flat29")
    if args.warmstart_flat29 and not (
        args.flat29_pose_v3
        or args.flat29_pose_v4
        or args.flat29_pose_v5
        or args.flat29_pose_v6
    ):
        parser.error("--warmstart-flat29 requires a flat29 pose version")
    if args.task == "a6" and not (
        args.init_from
        or args.init_from29
        or args.continue_from
        or args.continue_from29
        or args.resume
    ):
        parser.error("A6 requires a parent checkpoint or --resume")
    args.num_envs = args.num_envs or (4096 if args.task == "a6" else 64)
    args.steps_per_env = args.steps_per_env or (24 if args.task == "a6" else 50)
    args.iterations = args.iterations or (10000 if args.task == "a6" else 12000)
    args.save_every = args.save_every or (500 if args.task == "a6" else 100)
    args.output = args.output or (
        f"logs/host_a6_29_{args.arm.replace('+', 'plus').replace('-', 'minus')}_seed{args.seed}"
        if args.task == "a6"
        else "logs/host_ground29_mjlab"
        if args.task == "flat29"
        else "logs/host_ground_mjlab"
    )
    if args.keep_curriculum_on_transfer and (
        not args.resume or not args.allow_model_transfer or args.reset_curriculum
    ):
        parser.error(
            "--keep-curriculum-on-transfer requires --resume and "
            "--allow-model-transfer, without --reset-curriculum"
        )
    if (
        min(
            args.num_envs,
            args.steps_per_env,
            args.iterations,
            args.epochs,
            args.minibatches,
        )
        <= 0
    ):
        parser.error("counts must be positive")
    train(args)


if __name__ == "__main__":
    main()
