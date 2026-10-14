"""Collect physically reached prone guided-board escape states for reset training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.ppo import HostPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--total-steps", type=int, default=1500)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--guided-offset-m", type=float, default=0.15)
    parser.add_argument("--mean-action", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 32 or args.num_envs % 32:
        parser.error("--num-envs must be a positive multiple of 32")
    if args.total_steps <= 0 or args.max_samples <= 0:
        parser.error("step and sample limits must be positive")
    if not 0 <= args.guided_offset_m <= 0.5:
        parser.error("--guided-offset-m must be in [0, 0.5]")
    torch.manual_seed(args.seed)
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    variant = checkpoint["reward_variant"]
    if checkpoint.get("model_source") != "unitree_g1_29" or not (
        "_staged_escape_v5" in variant and "_v8_edge_curriculum_v1" in variant
    ):
        raise ValueError("expected a 29-DoF staged-v5 A6 checkpoint")
    policy = HostPolicy(
        612,
        29,
        log_std_bounds=(-2.3, math.log(checkpoint.get("action_std_max", 0.7))),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    layout = np.tile(np.asarray((1, 1, 0), dtype=np.int64), (args.num_envs, 1))
    cfg = host_a6_env_cfg(
        args.num_envs,
        args.seed,
        arm=checkpoint["arm"],
        edge_curriculum=True,
        plate_observation=True,
        safe_actions="_safe_actions_v1" in variant,
        prone_full_arms="_prone_full_arms_v1" in variant,
        aligned_edge="_aligned_edge_v1" in variant,
        lateral_actions="_lateral_actions_v1" in variant,
        lift_guard="_lift_guard_v1" in variant,
        footprint_edge="_footprint_edge_v1" in variant,
        footprint_overlap_m=checkpoint.get("a6_footprint_overlap_m", 0.25),
    )
    cfg.events["reset_scene"].params["layout"] = layout
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    poses: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    plates: list[np.ndarray] = []
    clearances: list[np.ndarray] = []
    actor_histories: list[np.ndarray] = []
    real_step_counts: list[np.ndarray] = []
    try:
        env._a6_staged_escape_v5 = True
        env._a6_knee_straight_prior_v1 = "_knee_straight_prior_v1" in variant
        env._a6_arm_extraction_v1 = "_arm_extraction_v1" in variant
        env._a6_edge_offset_m = args.guided_offset_m
        obs, _ = env.reset()
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(checkpoint["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        term.guided_escape_assist_force.zero_()
        term.guided_escape_assist_speed_target.zero_()
        steps_run = 0
        for step in range(args.total_steps):
            with torch.no_grad():
                before = env._a6_escaped.clone()
                action = (
                    policy.deterministic_action(obs["actor"])
                    if args.mean_action
                    else policy.distribution(obs["actor"]).sample()
                )
                obs, _, terminated, truncated, _ = env.step(action)
                selected = (
                    ~before
                    & env._a6_escaped
                    & ~env._a6_invalid
                    & (env._a6_clear_age >= 25)
                ).nonzero(as_tuple=False).squeeze(-1)
                if len(selected):
                    selected = selected[: args.max_samples - sum(len(x) for x in poses)]
                    robot = env.scene["robot"].data
                    origins = env.scene.env_origins[selected]
                    qpos = torch.cat(
                        (
                            robot.root_link_pos_w[selected] - origins,
                            robot.root_link_quat_w[selected],
                            robot.joint_pos[selected],
                        ),
                        dim=-1,
                    )
                    qvel = torch.cat(
                        (
                            robot.root_link_lin_vel_w[selected],
                            robot.root_link_ang_vel_w[selected],
                            robot.joint_vel[selected],
                        ),
                        dim=-1,
                    )
                    plate = (
                        env.sim.data.geom_xpos[
                            selected, env._a6_plate_geoms[0]
                        ] - origins
                    )
                    poses.append(qpos.cpu().numpy().copy())
                    velocities.append(qvel.cpu().numpy().copy())
                    plates.append(plate.cpu().numpy().copy())
                    clearances.append(
                        env._a6_best_clearance[selected].cpu().numpy().copy()
                    )
                    actor_histories.append(
                        obs["actor"][selected].cpu().numpy().copy()
                    )
                    real_step_counts.append(
                        term.real_steps[selected].cpu().numpy().copy()
                    )
                steps_run = step + 1
                if sum(len(x) for x in poses) >= args.max_samples:
                    break
                finished = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(finished):
                    obs, _ = env.reset(env_ids=finished)
        if not poses:
            raise RuntimeError("no guided-prone escapes were collected")
        qpos = np.concatenate(poses)[: args.max_samples]
        qvel = np.concatenate(velocities)[: args.max_samples]
        plate_xyz = np.concatenate(plates)[: args.max_samples]
        best_xy_clearance = np.concatenate(clearances)[: args.max_samples]
        actor_obs = np.concatenate(actor_histories)[: args.max_samples]
        real_steps = np.concatenate(real_step_counts)[: args.max_samples]
        if qpos.shape[1] != 36 or qvel.shape[1] != 35:
            raise RuntimeError(f"unexpected 29-DoF state shape: {qpos.shape}, {qvel.shape}")
        if actor_obs.shape != (len(qpos), 612):
            raise RuntimeError(f"unexpected actor history shape: {actor_obs.shape}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            qpos=qpos,
            qvel=qvel,
            plate_xyz=plate_xyz,
            best_xy_clearance=best_xy_clearance,
            actor_obs=actor_obs,
            real_steps=real_steps,
        )
        summary = {
            "source_checkpoint": str(args.checkpoint),
            "source_iteration": checkpoint["iteration"],
            "source_reward_variant": variant,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps_run": steps_run,
            "num_samples": len(qpos),
            "guided_offset_m": args.guided_offset_m,
            "mean_action": args.mean_action,
            "min_xy_clearance_m": float(best_xy_clearance.min()),
            "max_xy_clearance_m": float(best_xy_clearance.max()),
            "mean_root_height_m": float(qpos[:, 2].mean()),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
