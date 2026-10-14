"""Measure first-episode A6 plate force, clearance, and motion in four-view worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_geometry import (
    completion_clearance,
    plate_footprint_inputs,
)
from g1recovery_host.a6_reset import MAX_GUIDED_EDGE_OFFSET_M
from g1recovery_host.a6_reward import (
    A6_RESET_VARIANT,
    EFFORT,
    PLATE_FORCE_INVALID_N,
    guided_exit_margin,
)
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.ppo import HostPolicy

FOUR_VIEW_LAYOUT = ((2, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))
LABELS = ("free_supine", "guided_prone", "guided_supine", "flat_supine")


def load_policy(path: Path, device: str) -> tuple[HostPolicy, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_source") != "unitree_g1_29":
        raise ValueError(f"not a 29-DoF checkpoint: {path}")
    policy = HostPolicy(
        612
        if "_staged_escape_v5" in checkpoint.get("reward_variant", "")
        else 564,
        29,
        log_std_bounds=(-2.3, math.log(checkpoint.get("action_std_max", 0.7))),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    return policy, checkpoint


def summarize(series: list[dict], key: str) -> dict[str, float]:
    values = torch.tensor([row[key] for row in series], dtype=torch.float64)
    return {
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "p95": round(float(torch.quantile(values, 0.95)), 4),
    }


def first_step(series: list[dict], key: str) -> int | None:
    return next((row["step"] for row in series if row[key]), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--sample-action", action="store_true",
        help="Sample the PPO action distribution instead of using its mean.",
    )
    parser.add_argument("--guided-offset-m", type=float, default=0.0)
    parser.add_argument("--aligned-edge-override", action="store_true")
    parser.add_argument("--lateral-actions-override", action="store_true")
    parser.add_argument("--guided-lateral-scale", type=float, default=1.0)
    parser.add_argument("--lift-guard-override", action="store_true")
    parser.add_argument("--footprint-edge-override", action="store_true")
    parser.add_argument("--footprint-overlap-m", type=float, default=None)
    parser.add_argument("--arm-extraction-override", action="store_true")
    parser.add_argument("--guided-horizontal-assist-n", type=float, default=0.0)
    parser.add_argument("--guided-horizontal-assist-until-step", type=int, default=None)
    parser.add_argument("--guided-horizontal-speed-target-mps", type=float, default=0.0)
    parser.add_argument("--teacher-after-escape", action="store_true")
    parser.add_argument("--guided-knee-action", type=float, default=None)
    parser.add_argument("--guided-knee-target-rad", type=float, default=None)
    parser.add_argument(
        "--guided-zero-joints",
        choices=("arms", "waist", "upper", "hip_pitch", "hips", "legs", "all"),
        default=None,
    )
    parser.add_argument("--guided-arm-scale", type=float, default=1.0)
    parser.add_argument("--guided-hip-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.guided_knee_action is not None and args.guided_knee_target_rad is not None:
        parser.error("choose one guided knee override")
    if not 0 <= args.guided_arm_scale <= 1 or not 0 <= args.guided_hip_scale <= 1:
        parser.error("guided arm and hip scales must be in [0, 1]")
    if not 0 < args.guided_lateral_scale <= 1:
        parser.error("guided lateral scale must be in (0, 1]")
    if not math.isfinite(args.guided_horizontal_assist_n) or args.guided_horizontal_assist_n < 0:
        parser.error("guided horizontal assist must be finite and nonnegative")
    if args.guided_horizontal_assist_until_step is not None and not (
        0 <= args.guided_horizontal_assist_until_step <= args.steps
    ):
        parser.error("guided horizontal assist end must be within the episode")
    if not math.isfinite(args.guided_horizontal_speed_target_mps) or args.guided_horizontal_speed_target_mps < 0:
        parser.error("guided horizontal speed target must be finite and nonnegative")

    torch.manual_seed(args.seed)
    student, checkpoint = load_policy(args.checkpoint, args.device)
    teacher, teacher_checkpoint = load_policy(args.teacher_checkpoint, args.device)
    if checkpoint.get("frozen_flat_teacher_sha256") is None:
        raise ValueError("checkpoint has no frozen flat teacher metadata")
    with args.teacher_checkpoint.open("rb") as stream:
        teacher_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if checkpoint["frozen_flat_teacher_sha256"] != teacher_sha256:
        raise ValueError("frozen teacher SHA256 does not match the checkpoint")
    if teacher_checkpoint.get("task") != "flat29":
        raise ValueError("teacher is not a flat29 checkpoint")
    edge_curriculum = "_v8_edge_curriculum_v1" in checkpoint.get("reward_variant", "")
    staged_escape_v5 = "_staged_escape_v5" in checkpoint.get("reward_variant", "")
    safe_actions = "_safe_actions_v1" in checkpoint.get("reward_variant", "")
    aligned_edge = (
        "_aligned_edge_v1" in checkpoint.get("reward_variant", "")
        or args.aligned_edge_override
    )
    lateral_actions = (
        "_lateral_actions_v1" in checkpoint.get("reward_variant", "")
        or args.lateral_actions_override
    )
    lift_guard = (
        "_lift_guard_v1" in checkpoint.get("reward_variant", "")
        or args.lift_guard_override
    )
    knee_straight_prior = "_knee_straight_prior_v1" in checkpoint.get(
        "reward_variant", ""
    )
    arm_extraction = (
        "_arm_extraction_v1" in checkpoint.get("reward_variant", "")
        or args.arm_extraction_override
    )
    prone_full_arms = "_prone_full_arms_v1" in checkpoint.get("reward_variant", "")
    footprint_edge = (
        "_footprint_edge_v1" in checkpoint.get("reward_variant", "")
        or args.footprint_edge_override
    )
    footprint_overlap_m = (
        args.footprint_overlap_m
        if args.footprint_overlap_m is not None
        else checkpoint.get("a6_footprint_overlap_m", 0.25)
    )
    if not math.isfinite(footprint_overlap_m) or not 0 < footprint_overlap_m <= 0.50:
        parser.error("footprint overlap must be in (0, 0.50] metres")
    if not 0 <= args.guided_offset_m <= MAX_GUIDED_EDGE_OFFSET_M or (
        args.guided_offset_m > 0 and not edge_curriculum
    ):
        parser.error(
            "guided offset requires a v8 checkpoint and must be in [0, 0.50] m"
        )
    cfg = host_a6_env_cfg(
        4,
        args.seed,
        arm=checkpoint["arm"],
        edge_curriculum=edge_curriculum,
        plate_observation=staged_escape_v5,
        safe_actions=safe_actions,
        prone_full_arms=prone_full_arms,
        aligned_edge=aligned_edge,
        lateral_actions=lateral_actions,
        lift_guard=lift_guard,
        footprint_edge=footprint_edge,
        footprint_overlap_m=footprint_overlap_m,
    )
    if lateral_actions:
        cfg.actions["joint_pos"].guided_lateral_scale = args.guided_lateral_scale
    cfg.events["reset_scene"].params["layout"] = FOUR_VIEW_LAYOUT
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps * 0.02
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env._a6_staged_escape_v1 = checkpoint.get("reward_variant", "").endswith(
            "_staged_escape_v1"
        )
        env._a6_staged_escape_v2 = checkpoint.get("reward_variant", "").endswith(
            "_staged_escape_v2"
        )
        env._a6_staged_escape_v3 = checkpoint.get("reward_variant", "").endswith(
            "_staged_escape_v3"
        )
        env._a6_staged_escape_v4 = checkpoint.get("reward_variant", "").endswith(
            "_staged_escape_v4"
        )
        env._a6_staged_escape_v5 = staged_escape_v5
        env._a6_knee_straight_prior_v1 = knee_straight_prior
        env._a6_arm_extraction_v1 = arm_extraction
        env._a6_edge_offset_m = args.guided_offset_m
        obs, _ = env.reset()
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(checkpoint["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        term.guided_escape_assist_force.fill_(args.guided_horizontal_assist_n)
        term.guided_escape_assist_speed_target.fill_(
            args.guided_horizontal_speed_target_mps
        )
        rows = torch.arange(4, device=env.device)
        plate_ids = env._a6_plate_geoms[env._a6_scene.clamp_min(1) - 1]
        plate_initial = env.sim.data.geom_xpos[rows, plate_ids].clone()
        collision_names = [
            mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            for geom_id in env._a6_collision_geoms
        ]
        robot = env.scene["robot"]
        effort = EFFORT.to(env.device)
        arm_ids = [
            i
            for i, name in enumerate(robot.joint_names)
            if "shoulder" in name or "elbow" in name or "wrist" in name
        ]
        leg_ids = [
            i
            for i, name in enumerate(robot.joint_names)
            if "hip" in name or "knee" in name or "ankle" in name
        ]
        waist_ids = [i for i, name in enumerate(robot.joint_names) if "waist" in name]
        active_episode = [True] * 4
        series = [[] for _ in range(4)]
        for step in range(args.steps):
            if args.guided_horizontal_assist_until_step is not None:
                term.guided_escape_assist_force.fill_(
                    args.guided_horizontal_assist_n
                    if step < args.guided_horizontal_assist_until_step
                    else 0.0
                )
            with torch.no_grad():
                action = (
                    student.distribution(obs["actor"]).sample()
                    if args.sample_action
                    else student.deterministic_action(obs["actor"])
                )
                policy_action = action
                if (
                    args.guided_knee_action is not None
                    or args.guided_knee_target_rad is not None
                ):
                    action = action.clone()
                    guided_trapped = (env._a6_scene == 1) & ~env._a6_escaped
                    if args.guided_knee_target_rad is None:
                        for knee_id in env._a6_knee_ids:
                            action[guided_trapped, knee_id] = args.guided_knee_action
                    else:
                        knees_before = env.scene["robot"].data.joint_pos[
                            :, env._a6_knee_ids
                        ]
                        desired = (
                            2.0 * (args.guided_knee_target_rad - knees_before)
                        ).clamp(-1.0, 1.0)
                        for j, knee_id in enumerate(env._a6_knee_ids):
                            action[guided_trapped, knee_id] = desired[guided_trapped, j]
                if args.guided_zero_joints is not None:
                    action = action.clone()
                    guided_trapped = (env._a6_scene == 1) & ~env._a6_escaped
                    joint_groups = {
                        "arms": arm_ids,
                        "waist": waist_ids,
                        "upper": arm_ids + waist_ids,
                        "hip_pitch": env._a6_hip_pitch_ids,
                        "hips": [
                            i for i, name in enumerate(robot.joint_names) if "hip" in name
                        ],
                        "legs": leg_ids,
                        "all": list(range(len(robot.joint_names))),
                    }
                    selected = guided_trapped.nonzero(as_tuple=False).squeeze(-1)
                    joint_ids = joint_groups[args.guided_zero_joints]
                    if len(selected):
                        action[selected[:, None], joint_ids] = 0.0
                if args.guided_arm_scale != 1 or args.guided_hip_scale != 1:
                    action = action.clone()
                    selected = (
                        (env._a6_scene == 1) & ~env._a6_escaped
                    ).nonzero(as_tuple=False).squeeze(-1)
                    if len(selected):
                        action[selected[:, None], arm_ids] *= args.guided_arm_scale
                        hip_ids = [
                            i for i, name in enumerate(robot.joint_names) if "hip" in name
                        ]
                        action[selected[:, None], hip_ids] *= args.guided_hip_scale
                teacher_action = teacher.deterministic_action(obs["actor"][:, :564])
                if args.teacher_after_escape:
                    guided_escaped = (env._a6_scene == 1) & env._a6_escaped
                    action = torch.where(
                        guided_escaped[:, None], teacher_action, action
                    )
                teacher_mask = (term.real_steps > 30) & (
                    (env._a6_scene == 0) | env._a6_escaped
                )
                obs, _, terminated, truncated, _ = env.step(action)
                guided = env.scene["guided_contact"].data
                free = env.scene["free_contact"].data
                guided_force = guided.force.norm(dim=-1).amax(-1)
                free_force = free.force.norm(dim=-1).amax(-1)
                guided_substep_force = guided.force_history.norm(dim=-1).amax(
                    dim=(1, 2)
                )
                free_substep_force = free.force_history.norm(dim=-1).amax(dim=(1, 2))
                depth = torch.where(
                    env._a6_scene == 2,
                    free.dist.amin(-1),
                    guided.dist.amin(-1),
                )
                substep_depth = torch.where(
                    env._a6_scene == 2,
                    free.dist_history.amin(dim=(1, 2)),
                    guided.dist_history.amin(dim=(1, 2)),
                )
                guided_contact = (guided.found > 0).any(-1)
                free_contact = (free.found > 0).any(-1)
                force = torch.where(env._a6_scene == 2, free_force, guided_force)
                substep_force = torch.where(
                    env._a6_scene == 2, free_substep_force, guided_substep_force
                )
                contact = torch.where(env._a6_scene == 2, free_contact, guided_contact)
                plate_pos = env.sim.data.geom_xpos[rows, plate_ids]
                geom_pos, geom_ext, footprint_plate_pos, plate_ext = (
                    plate_footprint_inputs(env)
                )
                xy_overlap = (
                    geom_ext[..., :2]
                    + plate_ext[:, None, :2]
                    - (geom_pos[..., :2] - footprint_plate_pos[:, None, :2]).abs()
                )
                covered = (xy_overlap > 0).all(-1)
                xy_clearance = torch.where(
                    covered,
                    0.0,
                    (-xy_overlap).clamp_min(0).norm(dim=-1),
                ).amin(-1)
                sampled_covered = step + 1 in (1, 40, 100, 200, args.steps)
                done = terminated | truncated
                head_z = (
                    robot.data.geom_pos_w[:, env._a6_head_geom_id, 2]
                    - env.scene.env_origins[:, 2]
                )
                reward = env.reward_manager._step_reward * env.step_dt
                completion = completion_clearance(env)
                action_diff = (action - teacher_action).square().sum(-1).sqrt()
                torque_ratio = robot.data.qfrc_actuator.abs() / effort
                max_torque_ratio = torque_ratio.amax(-1)
                arm_torque_ratio = torque_ratio[:, arm_ids].amax(-1)
                leg_torque_ratio = torque_ratio[:, leg_ids].amax(-1)
                waist_torque_ratio = torque_ratio[:, waist_ids].amax(-1)
                escape_values = env._a6_escape_values
                hand_support = (env.scene["a6_hands"].data.found > 0).float().mean(-1)
                hand_load = (
                    (env.scene["a6_hands"].data.force.norm(dim=-1) / 60.0)
                    .clamp(0, 1)
                    .mean(-1)
                )
                knees = robot.data.joint_pos[:, env._a6_knee_ids]
                hips = robot.data.joint_pos[:, env._a6_hip_pitch_ids]
                signed_lateral = (
                    (robot.data.root_link_pos_w[:, :2] - env._a6_initial_root_xy)
                    * env._a6_initial_lateral_axis_xy
                ).sum(-1)
                lateral_displacement = signed_lateral.abs()
                near_edge_displacement = (-signed_lateral).clamp_min(0)
                torso_xy_distance = (
                    robot.data.root_link_pos_w[:, :2] - env._a6_initial_root_xy
                ).norm(dim=-1)
                exit_margin = guided_exit_margin(
                    robot.data.root_link_pos_w[:, :2], plate_pos[:, :2]
                )
                for i in range(4):
                    if not active_episode[i]:
                        continue
                    series[i].append(
                        {
                            "step": step + 1,
                            "time_s": round((step + 1) * 0.02, 3),
                            "initial_edge_offset_m": round(
                                float(env._a6_initial_edge_offset_m[i]), 5
                            ),
                            "contact_force_n": round(float(force[i]), 3),
                            "substep_contact_force_peak_n": round(
                                float(substep_force[i]), 3
                            ),
                            "contact_depth_m": round(float(depth[i]), 5),
                            "substep_contact_depth_min_m": round(
                                float(substep_depth[i]), 5
                            ),
                            "max_torque_ratio": round(float(max_torque_ratio[i]), 4),
                            "arm_torque_ratio": round(float(arm_torque_ratio[i]), 4),
                            "arm_joint_torque_ratios": [
                                round(float(torque_ratio[i, joint]), 4)
                                for joint in arm_ids
                            ],
                            "leg_torque_ratio": round(float(leg_torque_ratio[i]), 4),
                            "waist_torque_ratio": round(
                                float(waist_torque_ratio[i]), 4
                            ),
                            "plate_lift_m": round(
                                float(plate_pos[i, 2] - plate_initial[i, 2]), 5
                            ),
                            "plate_lift_peak_m": round(
                                float(env._a6_plate_lift_peak[i]), 5
                            ),
                            "plate_xy_shift_m": round(
                                float((plate_pos[i, :2] - plate_initial[i, :2]).norm()),
                                5,
                            ),
                            "torso_xy_distance_m": round(
                                float(torso_xy_distance[i]), 5
                            ),
                            "exit_margin_m": round(float(exit_margin[i]), 5),
                            "exit_progress_m": round(
                                float(
                                    (env._a6_initial_exit_margin[i] - exit_margin[i])
                                    .clamp_min(0)
                                ),
                                5,
                            ),
                            "lateral_gain": round(float(escape_values[i, 7]), 5),
                            "hand_support": round(float(hand_support[i]), 5),
                            "hand_load": round(float(hand_load[i]), 5),
                            "lateral_displacement_m": round(
                                float(lateral_displacement[i]), 5
                            ),
                            "near_edge_displacement_m": round(
                                float(near_edge_displacement[i]), 5
                            ),
                            "near_edge_phase": round(
                                float(env._a6_near_edge_phase[i]), 4
                            ),
                            "knee_penalty_weight": round(
                                float(env._a6_knee_penalty_weight[i]), 4
                            ),
                            "plate_path_progress": round(float(escape_values[i, 0]), 5),
                            "supine_lateral_gain": round(
                                float(escape_values[i, 11]), 5
                            ),
                            "left_knee_rad": round(float(knees[i, 0]), 4),
                            "right_knee_rad": round(float(knees[i, 1]), 4),
                            "knee_action": [
                                round(float(v), 4) for v in action[i, env._a6_knee_ids]
                            ],
                            "policy_knee_action": [
                                round(float(v), 4)
                                for v in policy_action[i, env._a6_knee_ids]
                            ],
                            "teacher_knee_action": [
                                round(float(v), 4)
                                for v in teacher_action[i, env._a6_knee_ids]
                            ],
                            "knee_motor_torque_nm": [
                                round(float(v), 3)
                                for v in robot.data.qfrc_actuator[i, env._a6_knee_ids]
                            ],
                            "left_hip_pitch_rad": round(float(hips[i, 0]), 4),
                            "right_hip_pitch_rad": round(float(hips[i, 1]), 4),
                            "knee_bend_cost": round(float(escape_values[i, 12]), 5),
                            "pre_escape_knee_curl": round(
                                float(escape_values[i, 26]), 5
                            ),
                            "knee_extension_gain": round(
                                float(escape_values[i, 13]), 5
                            ),
                            "leg_xy_progress": round(float(escape_values[i, 14]), 5),
                            "leg_release": round(float(escape_values[i, 15]), 5),
                            "leg_straightness": round(float(escape_values[i, 16]), 5),
                            "hip_bend_cost": round(float(escape_values[i, 17]), 5),
                            "moving_straight": round(float(escape_values[i, 18]), 5),
                            "torso_travel": round(float(escape_values[i, 19]), 5),
                            "headward_arms": round(float(escape_values[i, 20]), 5),
                            "linear_leg_alignment": round(
                                float(escape_values[i, 21]), 5
                            ),
                            "hand_push": round(float(escape_values[i, 22]), 5),
                            "hand_drive": round(float(escape_values[i, 23]), 5),
                            "hand_plant": round(float(escape_values[i, 24]), 5),
                            "overcurl": round(float(escape_values[i, 25]), 5),
                            "no_progress": round(float(escape_values[i, 8]), 5),
                            "stalled_no_progress": round(float(escape_values[i, 9]), 5),
                            "plate_lift_cost": round(float(escape_values[i, 10]), 5),
                            "pelvis_height_m": round(
                                float(robot.data.root_link_pos_w[i, 2]), 5
                            ),
                            "head_height_m": round(float(head_z[i]), 5),
                            "upright": round(
                                float(-robot.data.projected_gravity_b[i, 2]), 4
                            ),
                            "xy_clearance_m": round(float(xy_clearance[i]), 5),
                            "covered_geom_count": int(covered[i].sum()),
                            "arm_overlap_depth_m": round(
                                float(env._a6_arm_depth[i]), 5
                            ),
                            "arm_extraction_reward": round(
                                float(env._a6_arm_extraction_reward[i]), 5
                            ),
                            "covered_geoms": [
                                {
                                    "name": collision_names[j],
                                    "xy_overlap_min_m": round(
                                        float(xy_overlap[i, j].min()), 5
                                    ),
                                    "height_below_plate_top_m": round(
                                        float(
                                            footprint_plate_pos[i, 2]
                                            + plate_ext[i, 2]
                                            - geom_pos[i, j, 2]
                                            + geom_ext[i, j, 2]
                                        ),
                                        5,
                                    ),
                                }
                                for j in covered[i].nonzero(as_tuple=False).flatten().tolist()
                            ]
                            if sampled_covered and i in (1, 2)
                            else None,
                            "completion_clearance_m": round(float(completion[i]), 5),
                            "contact": bool(contact[i]),
                            "clear_age": int(env._a6_clear_age[i]),
                            "escaped": bool(env._a6_escaped[i]),
                            "invalid": bool(env._a6_invalid[i]),
                            "stage": int(env._a6_stage[i]),
                            "teacher_mask": bool(teacher_mask[i]),
                            "student_teacher_action_l2": round(
                                float(action_diff[i]), 4
                            ),
                            "task_reward": round(float(reward[i, 0]), 6),
                            "style_reward": round(float(reward[i, 2]), 6),
                            "target_reward": round(float(reward[i, 3]), 6),
                            "done": bool(done[i]),
                        }
                    )
                    if bool(done[i]):
                        active_episode[i] = False
            if not any(active_episode):
                break
            # With auto_reset=False, every finished world must be reset before
            # the next batched step. Keep only its original episode in series.
            finished_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(finished_ids):
                obs, _ = env.reset(env_ids=finished_ids)
        worlds = {}
        for name, records in zip(LABELS, series, strict=True):
            worlds[name] = {
                "steps_recorded": len(records),
                "final_pelvis_height_m": records[-1]["pelvis_height_m"],
                "final_upright": records[-1]["upright"],
                "first_escaped_step": first_step(records, "escaped"),
                "first_invalid_step": first_step(records, "invalid"),
                "first_done_step": first_step(records, "done"),
                "first_force_over_5000_step": next(
                    (
                        row["step"]
                        for row in records
                        if row["substep_contact_force_peak_n"] > PLATE_FORCE_INVALID_N
                    ),
                    None,
                ),
                "first_plate_lift_over_5cm_step": next(
                    (row["step"] for row in records if row["plate_lift_peak_m"] > 0.05),
                    None,
                ),
                "first_depth_below_minus_002_step": next(
                    (
                        row["step"]
                        for row in records
                        if row["substep_contact_depth_min_m"] < -0.02
                    ),
                    None,
                ),
                "max_clear_age": max(row["clear_age"] for row in records),
                "teacher_mask_steps": sum(row["teacher_mask"] for row in records),
                "force_n": summarize(records, "contact_force_n"),
                "max_torque_ratio": summarize(records, "max_torque_ratio"),
                "arm_torque_ratio": summarize(records, "arm_torque_ratio"),
                "arm_joint_torque_ratio_mean": {
                    robot.joint_names[joint]: round(
                        sum(row["arm_joint_torque_ratios"][column] for row in records)
                        / len(records),
                        4,
                    )
                    for column, joint in enumerate(arm_ids)
                },
                "leg_torque_ratio": summarize(records, "leg_torque_ratio"),
                "waist_torque_ratio": summarize(records, "waist_torque_ratio"),
                "plate_lift_m": summarize(records, "plate_lift_m"),
                "plate_lift_peak_m": summarize(records, "plate_lift_peak_m"),
                "plate_xy_shift_m": summarize(records, "plate_xy_shift_m"),
                "pelvis_height_m": summarize(records, "pelvis_height_m"),
                "head_height_m": summarize(records, "head_height_m"),
                "student_teacher_action_l2": summarize(
                    records, "student_teacher_action_l2"
                ),
                "torso_xy_distance_m": summarize(records, "torso_xy_distance_m"),
                "lateral_gain": summarize(records, "lateral_gain"),
                "hand_support": summarize(records, "hand_support"),
                "hand_load": summarize(records, "hand_load"),
                "lateral_displacement_m": summarize(records, "lateral_displacement_m"),
                "near_edge_displacement_m": summarize(
                    records, "near_edge_displacement_m"
                ),
                "near_edge_phase": summarize(records, "near_edge_phase"),
                "knee_penalty_weight": summarize(records, "knee_penalty_weight"),
                "plate_path_progress": summarize(records, "plate_path_progress"),
                "covered_geom_count": summarize(records, "covered_geom_count"),
                "arm_overlap_depth_m": summarize(records, "arm_overlap_depth_m"),
                "arm_extraction_reward": summarize(
                    records, "arm_extraction_reward"
                ),
                "supine_lateral_gain": summarize(records, "supine_lateral_gain"),
                "left_knee_rad": summarize(records, "left_knee_rad"),
                "right_knee_rad": summarize(records, "right_knee_rad"),
                "left_hip_pitch_rad": summarize(records, "left_hip_pitch_rad"),
                "right_hip_pitch_rad": summarize(records, "right_hip_pitch_rad"),
                "knee_bend_cost": summarize(records, "knee_bend_cost"),
                "pre_escape_knee_curl": summarize(records, "pre_escape_knee_curl"),
                "knee_extension_gain": summarize(records, "knee_extension_gain"),
                "leg_xy_progress": summarize(records, "leg_xy_progress"),
                "leg_release": summarize(records, "leg_release"),
                "leg_straightness": summarize(records, "leg_straightness"),
                "hip_bend_cost": summarize(records, "hip_bend_cost"),
                "moving_straight": summarize(records, "moving_straight"),
                "torso_travel": summarize(records, "torso_travel"),
                "headward_arms": summarize(records, "headward_arms"),
                "linear_leg_alignment": summarize(records, "linear_leg_alignment"),
                "hand_push": summarize(records, "hand_push"),
                "hand_drive": summarize(records, "hand_drive"),
                "hand_plant": summarize(records, "hand_plant"),
                "overcurl": summarize(records, "overcurl"),
                "no_progress": summarize(records, "no_progress"),
                "stalled_no_progress": summarize(records, "stalled_no_progress"),
                "plate_lift_cost": summarize(records, "plate_lift_cost"),
                "series": records,
            }
        result = {
            "checkpoint": str(args.checkpoint),
            "iteration": checkpoint.get("iteration"),
            "arm": checkpoint["arm"],
            "sample_action": args.sample_action,
            "teacher_iteration": teacher_checkpoint.get("iteration"),
            "checkpoint_reset_variant": checkpoint.get("a6_reset_variant"),
            "replay_reset_variant": A6_RESET_VARIANT,
            "beta": float(term.beta.mean()),
            "assist_force_n": 0.0,
            "guided_offset_m": args.guided_offset_m,
            "guided_horizontal_assist_n": args.guided_horizontal_assist_n,
            "guided_horizontal_assist_until_step": args.guided_horizontal_assist_until_step,
            "guided_horizontal_speed_target_mps": args.guided_horizontal_speed_target_mps,
            "teacher_after_escape": args.teacher_after_escape,
            "guided_knee_action": args.guided_knee_action,
            "guided_knee_target_rad": args.guided_knee_target_rad,
            "guided_zero_joints": args.guided_zero_joints,
            "guided_arm_scale": args.guided_arm_scale,
            "guided_hip_scale": args.guided_hip_scale,
            "checkpoint_safe_actions": safe_actions,
            "aligned_edge": aligned_edge,
            "lateral_actions": lateral_actions,
            "lift_guard": lift_guard,
            "knee_straight_prior": knee_straight_prior,
            "arm_extraction": arm_extraction,
            "prone_full_arms": prone_full_arms,
            "footprint_edge": footprint_edge,
            "footprint_overlap_m": footprint_overlap_m,
            "guided_lateral_scale": args.guided_lateral_scale if lateral_actions else 0.25,
            "auto_reset": False,
            "worlds": worlds,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        for name, world in worlds.items():
            print(name, {key: value for key, value in world.items() if key != "series"})
        print(f"report={args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
