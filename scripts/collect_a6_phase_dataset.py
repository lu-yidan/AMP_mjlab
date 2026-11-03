"""Collect proprio-only observations with simulator escape labels for phase detection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_distill import (
    effective_guided_teacher_action,
    effective_guided_side_teacher_action,
    legacy_teacher_observation,
    validate_legacy_teacher,
)
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.model import HOST29_JOINT_NAMES
from g1recovery_host.ppo import HostPolicy
from g1recovery_host.proprio_phase import supine_flat_recovery_label
from g1recovery_host.proprio_recovery import InitialSideRouter, load_prone_specialist
from g1recovery_host.proprio_symmetry import (
    mirror_host29_action, mirror_host29_actor_obs,
)


def layout_and_driver(balanced_supine: bool = False, balanced_prone: bool = False,
                      balanced_side: bool = False) -> tuple[np.ndarray, np.ndarray]:
    # The same physical world is represented with both teacher-driven and
    # student-driven exits; flat/free/side worlds probe false positives.
    if balanced_supine:
        rows = np.concatenate([
            np.tile((scene, 0, 0), (256, 1)) for scene in range(3)
        ])
        return rows, np.zeros(len(rows), dtype=np.bool_)
    if balanced_prone:
        rows = np.concatenate([
            np.tile((scene, 1, 0), (256, 1)) for scene in range(3)
        ])
        return rows, np.zeros(len(rows), dtype=np.bool_)
    if balanced_side:
        rows = np.concatenate([
            np.tile((scene, direction, 0), (128, 1))
            for scene in range(3) for direction in (2, 3)
        ])
        return rows, np.zeros(len(rows), dtype=np.bool_)
    groups = (
        (1, 1, 128, True), (1, 1, 128, False),
        (1, 0, 64, True), (1, 0, 64, False),
        (1, 2, 32, False), (1, 3, 32, False),
        (2, 1, 32, False), (2, 0, 32, False),
        (0, 1, 32, False), (0, 0, 32, False),
    )
    rows = np.concatenate(
        [np.tile((scene, direction, 0), (count, 1)) for scene, direction, count, _ in groups]
    )
    teacher = np.concatenate(
        [np.full(count, driven, dtype=np.bool_) for _, _, count, driven in groups]
    )
    return rows, teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--guided-offset-m", type=float, default=0.0)
    parser.add_argument("--pulse-steps", type=int, default=0,
                        help="training-only oracle pulse after a physically confirmed prone exit")
    parser.add_argument("--balanced-supine", action="store_true",
                        help="collect 256 student-driven supine worlds per scene every five steps")
    parser.add_argument("--balanced-prone", action="store_true",
                        help="collect 256 student-driven prone worlds per scene every five steps")
    parser.add_argument("--balanced-side", action="store_true",
                        help="collect 128 left and 128 right student-driven side worlds per scene")
    parser.add_argument("--side-safe-through-step", type=int, default=0,
                        help="apply scene-independent side-safe joint scaling during early side-scene collection")
    parser.add_argument("--mirror-right-initial-side-main", action="store_true",
                        help="collect side scenes under the proprio-only mirrored main actor")
    parser.add_argument("--left-side-specialist", type=Path,
                        help="mirrored 738D side actor for a short initial left-side probe")
    parser.add_argument("--left-side-probe-through-step", type=int, default=0)
    parser.add_argument("--left-side-probe-safe-scale", type=float, default=0.5)
    parser.add_argument("--shuffle-layout", action="store_true",
                        help="scatter scenes across environment indices to avoid index confounding")
    parser.add_argument("--supine-probe-v1", action="store_true",
                        help="small alternating shoulder-roll action at control steps 31-60")
    parser.add_argument("--capture-imu-accel", action="store_true",
                        help="record six recent torso IMU linear-acceleration frames")
    args = parser.parse_args()
    if args.steps < 100 or not 0 <= args.guided_offset_m <= 0.5 or args.pulse_steps < 0:
        parser.error("need at least 100 steps and a guided offset in [0, 0.5]")
    if args.balanced_supine and args.pulse_steps:
        parser.error("balanced supine data have no prone pulse episodes")
    if args.balanced_prone and (args.balanced_supine or args.balanced_side or args.pulse_steps):
        parser.error("balanced prone data are exclusive and have no pulse episodes")
    if args.balanced_side and (args.balanced_supine or args.pulse_steps):
        parser.error("balanced side data are exclusive and have no prone pulse episodes")
    if args.side_safe_through_step and not (
        args.balanced_side and 30 < args.side_safe_through_step <= args.steps
    ):
        parser.error("side-safe collection requires balanced side scenes and a valid horizon")
    if args.mirror_right_initial_side_main and not args.balanced_side:
        parser.error("mirrored right-side collection requires balanced side scenes")
    if (args.left_side_specialist is None) != (args.left_side_probe_through_step == 0):
        parser.error("left-side specialist and probe horizon must be supplied together")
    if args.left_side_specialist is not None and not (
        args.balanced_side and 30 < args.left_side_probe_through_step <= args.steps
        and 0 < args.left_side_probe_safe_scale <= 1
    ):
        parser.error("left-side expert probe requires balanced side scenes and valid controls")
    if args.supine_probe_v1 and not args.balanced_supine:
        parser.error("supine probe requires a balanced supine layout")
    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    source = torch.load(args.teacher, map_location=args.device, weights_only=False)
    if (
        state.get("observation_contract") != "proprio_torque_actor_privileged_critic_v2"
        or state.get("action_contract") != "scene_independent_v1"
        or state.get("model_source") != "unitree_g1_29"
    ):
        raise ValueError("phase data require a 738D proprioceptive A6 checkpoint")
    validate_legacy_teacher(source, state["physics_revision"], state["a6_reset_variant"])
    student = HostPolicy(
        738, 29, critic_obs_dim=786,
        log_std_bounds=(-2.3, math.log(state.get("action_std_max", 0.7))),
        upper_residual=True, waist_residual_scale=5.0,
    ).to(args.device)
    student.load_state_dict(state["policy"], strict=True)
    student.eval()
    left_side_policy = (
        load_prone_specialist(args.left_side_specialist, state, args.device)[0]
        if args.left_side_specialist is not None else None
    )
    teacher = HostPolicy(
        612, 29,
        log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
        upper_residual=True, waist_residual_scale=5.0,
    ).to(args.device)
    teacher.load_state_dict(source["policy"], strict=True)
    teacher.eval()
    layout, driven = layout_and_driver(args.balanced_supine, args.balanced_prone,
                                       args.balanced_side)
    if args.shuffle_layout:
        permutation = np.random.default_rng(args.seed).permutation(len(layout))
        layout, driven = layout[permutation], driven[permutation]
    cfg = host_a6_env_cfg(
        len(layout), args.seed, arm="G+", edge_curriculum=True,
        critic_plate_observation=True, proprio_motor_torque=True,
        aligned_edge=True, footprint_edge=True,
        footprint_overlap_m=state.get("a6_footprint_overlap_m", 0.03),
        privileged_guided_actions=False,
    )
    cfg.events["reset_scene"].params["layout"] = layout
    cfg.auto_reset = False
    cfg.episode_length_s = 15.0
    env = ManagerBasedRlEnv(cfg, device=args.device)
    features: list[np.ndarray] = []
    imu_accel_features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    recovery_labels: list[np.ndarray] = []
    live_frames: list[np.ndarray] = []
    control_steps: list[np.ndarray] = []
    episode_keys: list[np.ndarray] = []
    scenes: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    drivers: list[np.ndarray] = []
    pulse_features: list[np.ndarray] = []
    pulse_keys: list[np.ndarray] = []
    successful_keys: set[int] = set()
    probe_invalid_episodes = {scene: set() for scene in range(3)}
    probe_max_plate_force_n = {scene: 0.0 for scene in (1, 2)}
    probe_max_guided_lift_m = 0.0
    try:
        env._a6_staged_escape_v5 = True
        env._a6_knee_straight_prior_v1 = "_knee_straight_prior_v1" in state["reward_variant"]
        env._a6_arm_extraction_v1 = "_arm_extraction_v1" in state["reward_variant"]
        env._a6_prone_flexible_exit_v1 = "_prone_flexible_exit_v1" in state["reward_variant"]
        env._a6_prone_recovery_v1 = True
        env._a6_bidirectional_recovery_v1 = True
        env._a6_strong_recovery_v1 = "_strong_recovery_v1" in state["reward_variant"]
        env._a6_recovery_bridge_v2 = "_recovery_bridge_v2" in state["reward_variant"]
        env._a6_edge_offset_m = args.guided_offset_m
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(state["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        term.guided_escape_assist_force.zero_()
        term.guided_escape_assist_speed_target.zero_()
        obs, _ = env.reset()
        side_router = (
            InitialSideRouter(env.num_envs, args.device)
            if args.mirror_right_initial_side_main or left_side_policy is not None else None
        )
        if side_router is not None:
            side_router.arm_initial_gravity(
                env.scene["robot"].data.projected_gravity_b
            )
        if args.capture_imu_accel:
            imu_sensor = env.sim.mj_model.sensor("robot/imu_lin_acc")
            if imu_sensor is None or imu_sensor.dim[0] != 3:
                raise RuntimeError("robot/imu_lin_acc is unavailable or not 3D")
            imu_accel_address = int(imu_sensor.adr[0])
            imu_accel_history = torch.zeros(
                (env.num_envs, 6, 3), device=args.device, dtype=torch.float32
            )
        teacher_rows = torch.as_tensor(driven, device=args.device)
        guided_prone = (env._a6_scene == 1) & (env._a6_direction == 1)
        pulse_action = student.deterministic_action(torch.zeros(1, 738, device=args.device))
        escape_age = torch.full((env.num_envs,), -1, device=args.device, dtype=torch.long)
        episode_serial = torch.zeros_like(escape_age)
        had_pulse = torch.zeros(env.num_envs, device=args.device, dtype=torch.bool)
        stand_age = torch.zeros_like(escape_age)
        row_ids = torch.arange(env.num_envs, device=args.device)
        left_roll = HOST29_JOINT_NAMES.index("left_shoulder_roll_joint")
        right_roll = HOST29_JOINT_NAMES.index("right_shoulder_roll_joint")
        for step in range(args.steps):
            with torch.no_grad():
                escaped_before = env._a6_escaped.clone()
                action = student.deterministic_action(obs["actor"])
                if side_router is not None:
                    mirrored_right = (
                        side_router.initial_side & ~side_router.initial_left_side
                    )
                    if mirrored_right.any():
                        mirrored_input = mirror_host29_actor_obs(
                            obs["actor"][mirrored_right]
                        )
                        action[mirrored_right] = mirror_host29_action(
                            student.deterministic_action(mirrored_input)
                        )
                if left_side_policy is not None:
                    probe_left = (
                        side_router.initial_left_side
                        & (term.real_steps <= args.left_side_probe_through_step)
                    )
                    if probe_left.any():
                        mirrored_input = mirror_host29_actor_obs(
                            obs["actor"][probe_left]
                        )
                        expert_action = mirror_host29_action(
                            left_side_policy.deterministic_action(mirrored_input)
                        )
                        action[probe_left] = effective_guided_side_teacher_action(
                            expert_action, scale=args.left_side_probe_safe_scale,
                        )
                teacher_action = effective_guided_teacher_action(
                    teacher.deterministic_action(
                        legacy_teacher_observation(obs["critic"])
                    )
                )
                take_teacher = (
                    teacher_rows & (env._a6_scene == 1)
                    & (env._a6_direction < 2) & ~env._a6_escaped
                )
                action = torch.where(take_teacher[:, None], teacher_action, action)
                if args.side_safe_through_step and step + 1 <= args.side_safe_through_step:
                    action = effective_guided_side_teacher_action(action)
                if args.supine_probe_v1:
                    probing = (term.real_steps >= 31) & (term.real_steps <= 60)
                    if probing.any():
                        phase = ((term.real_steps - 31) // 5).remainder(2)
                        amplitude = torch.where(
                            phase == 0, action.new_full((env.num_envs,), 0.12),
                            action.new_full((env.num_envs,), -0.12),
                        )
                        probe_action = torch.zeros_like(action)
                        probe_action[:, left_roll] = amplitude
                        probe_action[:, right_roll] = -amplitude
                        action = torch.where(probing[:, None], probe_action, action)
                pulse = (
                    guided_prone & (escape_age >= 0)
                    & (escape_age < args.pulse_steps)
                )
                if pulse.any():
                    pulse_features.append(
                        obs["actor"][pulse].cpu().numpy().astype(np.float16)
                    )
                    pulse_keys.append(
                        (episode_serial[pulse] * env.num_envs + row_ids[pulse])
                        .cpu().numpy()
                    )
                    had_pulse[pulse] = True
                    action = torch.where(pulse[:, None], pulse_action, action)
                obs, _, terminated, truncated, _ = env.step(action)
                if args.capture_imu_accel:
                    imu_accel_history = torch.cat(
                        (imu_accel_history[:, 1:],
                         env.sim.data.sensordata[:, imu_accel_address:imu_accel_address + 3][:, None]),
                        dim=1,
                    )
                probe_window = (term.real_steps >= 32) & (term.real_steps <= 61)
                if probe_window.any():
                    for scene in range(3):
                        mask = probe_window & (env._a6_scene == scene)
                        invalid = mask & env._a6_invalid
                        if invalid.any():
                            keys = episode_serial[invalid] * env.num_envs + row_ids[invalid]
                            probe_invalid_episodes[scene].update(keys.cpu().tolist())
                        if scene in (1, 2) and mask.any():
                            sensor = env.scene[
                                "guided_contact" if scene == 1 else "free_contact"
                            ].data
                            force = sensor.force_history.norm(dim=-1).amax(dim=(1, 2))
                            probe_max_plate_force_n[scene] = max(
                                probe_max_plate_force_n[scene], float(force[mask].amax())
                            )
                            if scene == 1:
                                probe_max_guided_lift_m = max(
                                    probe_max_guided_lift_m,
                                    float(env._a6_plate_lift_peak[mask].amax()),
                                )
                new_escape = guided_prone & ~escaped_before & env._a6_escaped
                escape_age = torch.where(
                    new_escape, 0,
                    torch.where(escape_age >= 0, escape_age + 1, escape_age),
                )
                data = env.scene["robot"].data
                stood = (
                    had_pulse & (data.root_link_pos_w[:, 2] > 0.6)
                    & (-data.projected_gravity_b[:, 2] > 0.8)
                )
                stand_age = torch.where(stood, stand_age + 1, 0)
                stable = stand_age >= 25
                if stable.any():
                    keys = episode_serial[stable] * env.num_envs + row_ids[stable]
                    successful_keys.update(keys.cpu().tolist())
                valid = ~(terminated | truncated)
                sample = valid & (
                    (step % 5 == 0) if (args.balanced_supine or args.balanced_prone or args.balanced_side) else (
                        (guided_prone & (step % 5 == 0))
                        | (~guided_prone & (step % 20 == 0))
                    )
                )
                if sample.any():
                    features.append(obs["actor"][sample].cpu().numpy().astype(np.float16))
                    if args.capture_imu_accel:
                        imu_accel_features.append(
                            imu_accel_history[sample].cpu().numpy().astype(np.float16)
                        )
                    labels.append((guided_prone & env._a6_escaped)[sample].cpu().numpy())
                    recovery_labels.append(
                        supine_flat_recovery_label(
                            env._a6_scene, env._a6_direction, env._a6_escaped
                        )[sample].cpu().numpy()
                    )
                    live_frames.append((term.real_steps > 30)[sample].cpu().numpy())
                    control_steps.append(term.real_steps[sample].cpu().numpy())
                    episode_keys.append(
                        (episode_serial[sample] * env.num_envs + row_ids[sample])
                        .cpu().numpy()
                    )
                    scenes.append(env._a6_scene[sample].cpu().numpy())
                    directions.append(env._a6_direction[sample].cpu().numpy())
                    drivers.append(teacher_rows[sample].cpu().numpy())
                finished = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(finished):
                    obs, _ = env.reset(env_ids=finished)
                    if side_router is not None:
                        side_router.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            ids=finished,
                        )
                    if args.capture_imu_accel:
                        imu_accel_history[finished] = 0
                    escape_age[finished] = -1
                    had_pulse[finished] = False
                    stand_age[finished] = 0
                    episode_serial[finished] += 1
        archive = {
            "actor_obs": np.concatenate(features),
            "escaped_prone": np.concatenate(labels),
            "supine_flat_recovery_safe": np.concatenate(recovery_labels),
            "live_frame": np.concatenate(live_frames),
            "control_step": np.concatenate(control_steps),
            "episode_key": np.concatenate(episode_keys),
            "scene": np.concatenate(scenes),
            "direction": np.concatenate(directions),
            "teacher_driven": np.concatenate(drivers),
        }
        if args.capture_imu_accel:
            archive["imu_accel_history"] = np.concatenate(imu_accel_features)
        if args.pulse_steps:
            if not pulse_features:
                raise RuntimeError("no prone escape pulse frames were collected")
            pulse_episode_keys = np.concatenate(pulse_keys)
            archive.update(
                pulse_actor_obs=np.concatenate(pulse_features),
                pulse_success=np.isin(
                    pulse_episode_keys,
                    np.fromiter(successful_keys, dtype=np.int64),
                ),
                pulse_target_action=pulse_action[0].cpu().numpy(),
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **archive)
        result = {
            "checkpoint": str(args.checkpoint),
            "teacher": str(args.teacher),
            "seed": args.seed,
            "guided_offset_m": args.guided_offset_m,
            "steps": args.steps,
            "samples": len(archive["escaped_prone"]),
            "positive": int(archive["escaped_prone"].sum()),
            "supine_flat_recovery_positive_live": int((
                archive["supine_flat_recovery_safe"] & archive["live_frame"]
            ).sum()),
            "teacher_positive": int((archive["escaped_prone"] & archive["teacher_driven"]).sum()),
            "student_positive": int((archive["escaped_prone"] & ~archive["teacher_driven"]).sum()),
            "pulse_steps": args.pulse_steps,
            "pulse_frames": len(archive.get("pulse_actor_obs", [])),
            "pulse_success_frames": int(archive.get("pulse_success", np.zeros(0)).sum()),
            "supine_probe_v1": args.supine_probe_v1,
            "capture_imu_accel": args.capture_imu_accel,
            "balanced_prone": args.balanced_prone,
            "balanced_side": args.balanced_side,
            "side_safe_through_step": args.side_safe_through_step,
            "mirror_right_initial_side_main": args.mirror_right_initial_side_main,
            "left_side_specialist": str(args.left_side_specialist) if args.left_side_specialist else None,
            "left_side_probe_through_step": args.left_side_probe_through_step,
            "left_side_probe_safe_scale": args.left_side_probe_safe_scale,
            "shuffle_layout": args.shuffle_layout,
            "probe_window_invalid_episodes": {
                str(scene): len(keys) for scene, keys in probe_invalid_episodes.items()
            },
            "probe_window_max_plate_force_n": probe_max_plate_force_n,
            "probe_window_max_guided_lift_m": probe_max_guided_lift_m,
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
