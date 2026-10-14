"""Batch, no-assist A6 checkpoint evaluation in the four playback scenes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.a6_reward import GUIDED_MAX_LIFT_M, PLATE_FORCE_INVALID_N
from g1recovery_host.model import HOST29_JOINT_NAMES, HOST29_TARGET_JOINT_POS
from g1recovery_host.phase_switch import (
    reset_guided_history_after_escape,
    reset_prone_history_after_escape,
)
from g1recovery_host.ppo import HostPolicy


LABELS = ("guided_prone", "guided_supine", "free_supine", "flat_supine")
LAYOUT_ROWS = ((1, 1, 0), (1, 0, 0), (2, 0, 0), (0, 0, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--supine-teacher-checkpoint", type=Path)
    parser.add_argument("--postescape-teacher-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--per-scene", type=int, default=8)
    parser.add_argument("--all-postescape", action="store_true")
    parser.add_argument("--postescape-bank", type=Path)
    parser.add_argument("--postescape-live-obs-override", action="store_true")
    parser.add_argument("--postescape-motor-delay-override", action="store_true")
    parser.add_argument("--reset-policy-history-on-escape", action="store_true")
    parser.add_argument("--reset-supine-history-on-escape", action="store_true")
    parser.add_argument("--guided-offset-m", type=float, default=0.25)
    parser.add_argument("--sample-action", action="store_true")
    parser.add_argument("--mean-after-escape", action="store_true")
    parser.add_argument("--knee-diagnostics", action="store_true")
    parser.add_argument("--prone-full-arms-override", action="store_true")
    parser.add_argument("--arm-actuation-scale", type=float, default=None)
    parser.add_argument("--hip-actuation-scale", type=float, default=None)
    args = parser.parse_args()
    if args.per_scene < 8 or args.per_scene % 8:
        parser.error("--per-scene must be a positive multiple of 8")
    if args.all_postescape and args.per_scene < 32:
        parser.error("--all-postescape needs at least 32 worlds")
    if args.postescape_teacher_checkpoint is not None and not args.all_postescape:
        parser.error("--postescape-teacher-checkpoint requires --all-postescape")
    if args.postescape_bank is not None and not args.all_postescape:
        parser.error("--postescape-bank requires --all-postescape")
    if args.postescape_motor_delay_override and not args.all_postescape:
        parser.error("--postescape-motor-delay-override requires --all-postescape")
    if args.postescape_motor_delay_override and args.postescape_live_obs_override:
        parser.error("motor delay and live observation overrides cannot be combined")
    if args.reset_policy_history_on_escape and args.all_postescape:
        parser.error("history-on-escape reset requires a full escape scene")
    if args.reset_supine_history_on_escape and args.all_postescape:
        parser.error("supine history reset requires a full escape scene")
    if args.mean_after_escape and not args.sample_action:
        parser.error("--mean-after-escape requires --sample-action")
    if not 0 <= args.guided_offset_m <= 0.5:
        parser.error("--guided-offset-m must be in [0, 0.5]")
    if any(
        value is not None and not 0 < value <= 1
        for value in (args.arm_actuation_scale, args.hip_actuation_scale)
    ):
        parser.error("actuation scales must be in (0, 1]")
    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if state.get("task") != "a6" or state.get("model_source") != "unitree_g1_29":
        raise ValueError("A6 evaluation requires a 29-DoF A6 checkpoint")
    variant = state["reward_variant"]
    phase_switch = (
        "_phase_history_reset_v1" in variant
        or args.reset_policy_history_on_escape
    )
    staged_v5 = "_staged_escape_v5" in variant
    edge_curriculum = "_v8_edge_curriculum_v1" in variant
    if not staged_v5 or not edge_curriculum:
        raise ValueError("batch evaluation requires staged v5 edge curriculum")
    policy = HostPolicy(
        612,
        29,
        log_std_bounds=(-2.3, math.log(state.get("action_std_max", 0.7))),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval()
    postescape_teacher = None
    if args.postescape_teacher_checkpoint is not None:
        source = torch.load(
            args.postescape_teacher_checkpoint,
            map_location=args.device,
            weights_only=False,
        )
        if source.get("task") != "flat29" or source.get("model_source") != "unitree_g1_29":
            raise ValueError("postescape teacher must be a 29-DoF flat checkpoint")
        postescape_teacher = HostPolicy(
            564,
            29,
            log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
            upper_residual=True,
            waist_residual_scale=5.0,
        ).to(args.device)
        postescape_teacher.load_state_dict(source["policy"], strict=True)
        postescape_teacher.eval()
    supine_teacher = None
    if args.supine_teacher_checkpoint is not None:
        source = torch.load(
            args.supine_teacher_checkpoint, map_location=args.device, weights_only=False
        )
        if source.get("model_source") != "unitree_g1_29" or not (
            "_staged_escape_v5" in source.get("reward_variant", "")
        ):
            raise ValueError("supine teacher must be a 29-DoF staged-v5 A6 checkpoint")
        supine_teacher = HostPolicy(
            612,
            29,
            log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
            upper_residual=True,
            waist_residual_scale=5.0,
        ).to(args.device)
        supine_teacher.load_state_dict(source["policy"], strict=True)
        supine_teacher.eval()
    layout = (
        np.tile(np.asarray((1, 1, 3), dtype=np.int64), (args.per_scene, 1))
        if args.all_postescape
        else np.repeat(np.asarray(LAYOUT_ROWS, dtype=np.int64), args.per_scene, axis=0)
    )
    cfg = host_a6_env_cfg(
        len(layout),
        args.seed,
        arm=state["arm"],
        edge_curriculum=True,
        plate_observation=True,
        safe_actions="_safe_actions_v1" in variant,
        prone_full_arms=(
            "_prone_full_arms_v1" in variant or args.prone_full_arms_override
        ),
        supine_full_arms="_supine_full_arms_v1" in variant,
        aligned_edge="_aligned_edge_v1" in variant,
        lateral_actions="_lateral_actions_v1" in variant,
        lift_guard="_lift_guard_v1" in variant,
        footprint_edge="_footprint_edge_v1" in variant,
        footprint_overlap_m=state.get("a6_footprint_overlap_m", 0.25),
    )
    cfg.events["reset_scene"].params["layout"] = layout
    if args.arm_actuation_scale is not None:
        cfg.actions["joint_pos"].guided_arm_scale = args.arm_actuation_scale
    if args.hip_actuation_scale is not None:
        cfg.actions["joint_pos"].guided_hip_scale = args.hip_actuation_scale
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps * 0.02
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        if args.postescape_bank is not None:
            env._a6_postescape_bank_path = args.postescape_bank
        env._a6_postescape_live_obs_v1 = (
            "_postescape_live_obs_v1" in variant
            or args.postescape_live_obs_override
        )
        env._a6_staged_escape_v5 = staged_v5
        env._a6_prone_recovery_v1 = "_prone_recovery_v1" in variant
        env._a6_bidirectional_recovery_v1 = "_bidirectional_recovery_v1" in variant
        env._a6_knee_straight_prior_v1 = "_knee_straight_prior_v1" in variant
        env._a6_arm_extraction_v1 = "_arm_extraction_v1" in variant
        env._a6_edge_offset_m = args.guided_offset_m
        obs, _ = env.reset()
        if args.postescape_motor_delay_override:
            # Diagnostic: recover the original HoST 30-step motor wait even
            # for replayed escape states, while leaving their poses unchanged.
            env._a6_postescape_start.zero_()
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(state["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        term.guided_escape_assist_force.zero_()
        term.guided_escape_assist_speed_target.zero_()
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        first_escape = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        if args.all_postescape:
            first_escape[env._a6_escaped] = 1
        first_invalid = torch.zeros_like(first_escape)
        first_stand = torch.zeros_like(first_escape)
        max_clearance = torch.zeros(env.num_envs, device=env.device)
        max_guided_force = torch.zeros_like(max_clearance)
        max_guided_lift = torch.zeros_like(max_clearance)
        invalid_by_force = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        invalid_by_lift = torch.zeros_like(invalid_by_force)
        invalid_other = torch.zeros_like(invalid_by_force)
        final_height = torch.zeros_like(max_clearance)
        final_upright = torch.zeros_like(max_clearance)
        knee_trace = None
        if args.knee_diagnostics:
            knee_ids = [
                HOST29_JOINT_NAMES.index(f"{side}_knee_joint")
                for side in ("left", "right")
            ]
            standing_knees = torch.tensor(
                [HOST29_TARGET_JOINT_POS[f"{side}_knee_joint"] for side in ("left", "right")],
                device=env.device,
            )
            knee_trace = torch.full(
                (args.steps, env.num_envs), float("nan"), device=env.device
            )
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                escaped_before = env._a6_escaped.clone()
                mean_action = policy.deterministic_action(obs["actor"])
                action = (
                    policy.distribution(obs["actor"]).sample()
                    if args.sample_action else mean_action
                )
                if args.mean_after_escape:
                    escaped_guided = (env._a6_scene == 1) & env._a6_escaped
                    action = torch.where(escaped_guided[:, None], mean_action, action)
                if supine_teacher is not None:
                    guided_supine = (env._a6_scene == 1) & (env._a6_direction == 0)
                    teacher_action = supine_teacher.deterministic_action(obs["actor"])
                    action = torch.where(guided_supine[:, None], teacher_action, action)
                if postescape_teacher is not None:
                    action = postescape_teacher.deterministic_action(obs["actor"][:, :564])
                obs, _, terminated, truncated, _ = env.step(action)
                guided = env._a6_scene == 1
                force = env.scene["guided_contact"].data.force_history.norm(dim=-1).amax(dim=(1, 2))
                lift = env._a6_plate_lift_peak
                max_guided_force = torch.where(
                    active & guided,
                    torch.maximum(max_guided_force, force),
                    max_guided_force,
                )
                max_guided_lift = torch.where(
                    active & guided,
                    torch.maximum(max_guided_lift, lift),
                    max_guided_lift,
                )
                new_invalid = active & env._a6_invalid & (first_invalid == 0)
                force_trip = force > PLATE_FORCE_INVALID_N
                lift_trip = lift > GUIDED_MAX_LIFT_M
                invalid_by_force |= new_invalid & force_trip
                invalid_by_lift |= new_invalid & lift_trip
                invalid_other |= new_invalid & ~(force_trip | lift_trip)
                if phase_switch:
                    obs, _ = reset_prone_history_after_escape(
                        env, escaped_before, terminated | truncated
                    )
                if args.reset_supine_history_on_escape:
                    obs, _ = reset_guided_history_after_escape(
                        env, escaped_before, terminated | truncated, direction=0
                    )
                robot = env.scene["robot"].data
                if knee_trace is not None:
                    knee_deviation = (
                        robot.joint_pos[:, knee_ids] - standing_knees
                    ).abs().amax(-1)
                    knee_trace[step - 1] = torch.where(
                        active & guided & ~escaped_before,
                        knee_deviation,
                        float("nan"),
                    )
                height = robot.root_link_pos_w[:, 2]
                upright = -robot.projected_gravity_b[:, 2]
                first_escape = torch.where(
                    active & env._a6_escaped & (first_escape == 0),
                    step,
                    first_escape,
                )
                first_invalid = torch.where(
                    active & env._a6_invalid & (first_invalid == 0),
                    step,
                    first_invalid,
                )
                stand = (height > 0.6) & (upright > 0.8)
                first_stand = torch.where(
                    active & stand & (first_stand == 0), step, first_stand
                )
                max_clearance = torch.where(
                    active,
                    torch.maximum(max_clearance, env._a6_best_clearance),
                    max_clearance,
                )
                final_height = torch.where(active, height, final_height)
                final_upright = torch.where(active, upright, final_upright)
                done = (terminated | truncated) & active
                active &= ~done
                if not active.any():
                    break
                # Inactive worlds still advance in the batched simulator.
                # Reset every finished world so later steps stay well-defined.
                finished = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(finished):
                    obs, _ = env.reset(env_ids=finished)
        result = {
            "checkpoint": str(args.checkpoint),
            "iteration": state["iteration"],
            "sample_action": args.sample_action,
            "mean_after_escape": args.mean_after_escape,
            "supine_teacher_checkpoint": (
                str(args.supine_teacher_checkpoint)
                if args.supine_teacher_checkpoint is not None else None
            ),
            "postescape_teacher_checkpoint": (
                str(args.postescape_teacher_checkpoint)
                if args.postescape_teacher_checkpoint is not None else None
            ),
            "seed": args.seed,
            "guided_offset_m": args.guided_offset_m,
            "steps": args.steps,
            "per_scene": args.per_scene,
            "all_postescape": args.all_postescape,
            "postescape_bank": (
                str(args.postescape_bank) if args.postescape_bank is not None else None
            ),
            "postescape_live_obs_v1": env._a6_postescape_live_obs_v1,
            "postescape_motor_delay_override": args.postescape_motor_delay_override,
            "reset_policy_history_on_escape": phase_switch,
            "reset_supine_history_on_escape": args.reset_supine_history_on_escape,
            "arm_actuation_scale": cfg.actions["joint_pos"].guided_arm_scale,
            "prone_arm_actuation_scale": cfg.actions["joint_pos"].guided_prone_arm_scale,
            "supine_arm_actuation_scale": cfg.actions["joint_pos"].guided_supine_arm_scale,
            "hip_actuation_scale": cfg.actions["joint_pos"].guided_hip_scale,
            "assistance_n": 0.0,
            "knee_diagnostics": args.knee_diagnostics,
            "scenes": {},
        }
        labels = ("guided_prone_postescape",) if args.all_postescape else LABELS
        for index, label in enumerate(labels):
            rows = (
                slice(0, args.per_scene) if args.all_postescape
                else slice(index * args.per_scene, (index + 1) * args.per_scene)
            )
            escaped = first_escape[rows].cpu().tolist()
            invalid = first_invalid[rows].cpu().tolist()
            stood = first_stand[rows].cpu().tolist()
            heights = final_height[rows].cpu().tolist()
            uprights = final_upright[rows].cpu().tolist()
            result["scenes"][label] = {
                "escaped": sum(value > 0 for value in escaped),
                "invalid": sum(value > 0 for value in invalid),
                "invalid_by_force": int(invalid_by_force[rows].sum()),
                "invalid_by_lift": int(invalid_by_lift[rows].sum()),
                "invalid_other": int(invalid_other[rows].sum()),
                "stood_at_end": sum(
                    height > 0.6 and upright > 0.8
                    for height, upright in zip(heights, uprights, strict=True)
                ),
                "escaped_then_stood_at_end": sum(
                    escaped_step > 0 and height > 0.6 and upright > 0.8
                    for escaped_step, height, upright in zip(
                        escaped, heights, uprights, strict=True
                    )
                ),
                "first_escape_steps": escaped,
                "first_invalid_steps": invalid,
                "first_stand_steps": stood,
                "final_height_m": [round(value, 4) for value in heights],
                "final_upright": [round(value, 4) for value in uprights],
                "max_xy_clearance_m": [
                    round(value, 4) for value in max_clearance[rows].cpu().tolist()
                ],
                "max_guided_force_n": [
                    round(value, 1) for value in max_guided_force[rows].cpu().tolist()
                ],
                "max_guided_lift_m": [
                    round(value, 4) for value in max_guided_lift[rows].cpu().tolist()
                ],
            }
            if knee_trace is not None and label.startswith("guided_"):
                traces = knee_trace[:, rows].cpu().numpy().T
                tails = [values[np.isfinite(values)][-100:] for values in traces]
                active_escape = [
                    values[30:][np.isfinite(values[30:])] for values in traces
                ]
                median_deviation = [
                    float(np.median(values)) if len(values) else None for values in tails
                ]
                above_20deg = [
                    float(np.mean(values > math.radians(20))) if len(values) else None
                    for values in tails
                ]
                valid_medians = [value for value in median_deviation if value is not None]
                valid_fractions = [value for value in above_20deg if value is not None]
                active_fractions = [
                    float(np.mean(values > math.radians(20)))
                    for values in active_escape if len(values)
                ]
                active_p90 = [
                    float(np.percentile(values, 90))
                    for values in active_escape if len(values)
                ]
                result["scenes"][label].update({
                    "preescape_last100_knee_deviation_median_rad": [
                        round(value, 4) if value is not None else None
                        for value in median_deviation
                    ],
                    "preescape_last100_above20_fraction": [
                        round(value, 4) if value is not None else None
                        for value in above_20deg
                    ],
                    "preescape_knee_deviation_p50_rad": round(
                        float(np.median(valid_medians)), 4
                    ) if valid_medians else None,
                    "preescape_above20_mean_fraction": round(
                        float(np.mean(valid_fractions)), 4
                    ) if valid_fractions else None,
                    "preescape_after30_above20_mean_fraction": round(
                        float(np.mean(active_fractions)), 4
                    ) if active_fractions else None,
                    "preescape_after30_knee_deviation_p90_rad": round(
                        float(np.median(active_p90)), 4
                    ) if active_p90 else None,
                })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: {k: v for k, v in scene.items() if not isinstance(v, list)}
                          for key, scene in result["scenes"].items()}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
