"""Record successful centered-board side escape demonstrations for proprio BC.

The 612D board-aware policy is queried only while creating training data.
Saved student inputs contain the ordinary 738D body and motor history.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_distill import (
    effective_guided_side_teacher_action,
    legacy_teacher_observation,
    validate_legacy_teacher,
)
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.ppo import HostPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-side", type=int, default=256)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--sample-stride", type=int, default=5)
    args = parser.parse_args()
    if args.per_side < 1 or args.steps < 100 or args.sample_stride < 1:
        parser.error("need positive per-side/stride and at least 100 control steps")

    torch.manual_seed(args.seed)
    state = torch.load(args.student, map_location=args.device, weights_only=False)
    source = torch.load(args.teacher, map_location=args.device, weights_only=False)
    if state.get("observation_contract") != "proprio_torque_actor_privileged_critic_v2":
        raise ValueError("student must have 738D proprio actor observations")
    validate_legacy_teacher(
        source, state["physics_revision"], state["a6_reset_variant"],
    )
    teacher = HostPolicy(
        612, 29,
        log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
        upper_residual=True, waist_residual_scale=5.0,
    ).to(args.device)
    teacher.load_state_dict(source["policy"], strict=True)
    teacher.eval().requires_grad_(False)
    layout = np.concatenate([
        np.tile((1, direction, 0), (args.per_side, 1))
        for direction in (2, 3)
    ])
    layout = layout[np.random.default_rng(args.seed).permutation(len(layout))]
    cfg = host_a6_env_cfg(
        len(layout), args.seed, arm="G+", edge_curriculum=True,
        critic_plate_observation=True, proprio_motor_torque=True,
        aligned_edge=True, footprint_edge=True,
        footprint_overlap_m=state.get("a6_footprint_overlap_m", 0.03),
        privileged_guided_actions=False,
    )
    cfg.events["reset_scene"].params["layout"] = layout
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps * 0.02
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env._a6_staged_escape_v5 = True
        env._a6_knee_straight_prior_v1 = "_knee_straight_prior_v1" in state["reward_variant"]
        env._a6_arm_extraction_v1 = "_arm_extraction_v1" in state["reward_variant"]
        env._a6_prone_flexible_exit_v1 = "_prone_flexible_exit_v1" in state["reward_variant"]
        env._a6_prone_recovery_v1 = True
        env._a6_bidirectional_recovery_v1 = True
        env._a6_edge_offset_m = 0.0
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(source["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        torch.manual_seed(args.seed)
        obs, _ = env.reset()
        initial_direction = env._a6_direction.clone()
        active = torch.ones(env.num_envs, device=args.device, dtype=torch.bool)
        ever_escaped = torch.zeros_like(active)
        successful = torch.zeros_like(active)
        recorded = torch.zeros_like(active)
        obs_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        step_parts: list[np.ndarray] = []
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                raw = teacher.deterministic_action(
                    legacy_teacher_observation(obs["critic"])
                )
                side_action = effective_guided_side_teacher_action(raw)
                preescape = active & ~env._a6_escaped & ~env._a6_invalid
                if step % args.sample_stride == 0:
                    sample = preescape & (term.real_steps > 30)
                    if sample.any():
                        obs_parts.append(
                            obs["actor"][sample].cpu().numpy().astype(np.float16)
                        )
                        action_parts.append(
                            side_action[sample].cpu().numpy().astype(np.float16)
                        )
                        row_parts.append(
                            sample.nonzero(as_tuple=False).squeeze(-1).cpu().numpy().astype(np.int32)
                        )
                        step_parts.append(
                            np.full(int(sample.sum()), step, dtype=np.int16)
                        )
                action = torch.where(env._a6_escaped[:, None], raw, side_action)
                obs, _, terminated, truncated, _ = env.step(action)
                ever_escaped |= active & env._a6_escaped
                complete = active & (terminated | truncated | (step == args.steps))
                if complete.any():
                    robot = env.scene["robot"].data
                    stand = ((robot.root_link_pos_w[:, 2] > 0.6)
                             & (-robot.projected_gravity_b[:, 2] > 0.8))
                    successful |= complete & ever_escaped & stand & ~env._a6_invalid
                    recorded |= complete
                    active &= ~complete
                if not active.any():
                    break
                finished = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(finished):
                    obs, _ = env.reset(env_ids=finished)

        if not obs_parts:
            raise RuntimeError("teacher produced no active side observations")
        rows = np.concatenate(row_parts)
        success = successful.cpu().numpy()
        archive = dict(
            actor_obs=np.concatenate(obs_parts),
            target_action=np.concatenate(action_parts),
            row=rows,
            control_step=np.concatenate(step_parts),
            successful_episode=success[rows],
            direction=initial_direction.cpu().numpy()[rows],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **archive)
        report = {
            "student": str(args.student), "teacher": str(args.teacher),
            "seed": args.seed, "per_side": args.per_side,
            "steps": args.steps, "sample_stride": args.sample_stride,
            "recorded_episodes": int(recorded.sum()),
            "escaped_episodes": int(ever_escaped.sum()),
            "successful_episodes": int(successful.sum()),
            "successful_left": int((successful & (initial_direction == 2)).sum()),
            "successful_right": int((successful & (initial_direction == 3)).sum()),
            "samples": len(rows),
            "successful_samples": int(archive["successful_episode"].sum()),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
