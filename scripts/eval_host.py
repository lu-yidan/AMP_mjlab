"""Evaluate flat-ground get-up success with HoST's height/fall criterion."""

import argparse
import json
import math
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host import host_ground29_env_cfg, host_ground_env_cfg
from g1recovery_host.action import PHYSICS_REVISION
from g1recovery_host.ppo import HostPolicy


def evaluate(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu or fix GPU access")
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    is_29 = args.task == "flat29"
    model_source = "unitree_g1_29" if is_29 else args.model_source
    physics_revision = f"{PHYSICS_REVISION}_flat29_29_v1" if is_29 else PHYSICS_REVISION
    checkpoint_source = checkpoint.get("model_source", "mjlab")
    if checkpoint_source != model_source and not args.allow_model_transfer:
        raise ValueError(
            f"checkpoint model_source={checkpoint_source!r}, requested "
            f"{model_source!r}; use --allow-model-transfer only for a "
            "cross-geometry diagnostic"
        )
    checkpoint_revision = checkpoint.get("physics_revision", "single_torso_force_v1")
    if checkpoint_revision != physics_revision and not args.allow_physics_transfer:
        raise ValueError(
            f"checkpoint physics_revision={checkpoint_revision!r}, current "
            f"{physics_revision!r}; use --allow-physics-transfer only for a "
            "cross-physics diagnostic"
        )
    checkpoint_reward = checkpoint.get("reward_variant", "paper")
    pose_version = (
        "v6"
        if checkpoint_reward.endswith("pose_v6")
        else "v5"
        if checkpoint_reward.endswith("pose_v5")
        else "v4"
        if checkpoint_reward.endswith("pose_v4")
        else "v3"
        if checkpoint_reward.endswith("pose_v3")
        else "v2"
    )
    cfg = (
        host_ground29_env_cfg(
            args.num_envs,
            args.seed,
            pose_version=pose_version,
        )
        if is_29
        else host_ground_env_cfg(
            args.num_envs,
            args.seed,
            model_source=args.model_source,
            legacy_style=checkpoint_reward in ("paper", "height_guard_v1"),
        )
    )
    cfg.episode_length_s = args.steps * 0.02
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    policy = (
        HostPolicy(
            564,
            29,
            log_std_bounds=(-2.3, math.log(0.7))
            if pose_version in ("v3", "v4", "v5", "v6")
            else None,
            upper_residual=pose_version in ("v4", "v5", "v6"),
            waist_residual_scale=5.0 if pose_version in ("v5", "v6") else 2.0,
        ).to(args.device)
        if is_29
        else HostPolicy().to(args.device)
    )
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    term = env.action_manager.get_term("joint_pos")
    successes = []
    stood_counts = []
    fell_counts = []
    max_heights = []
    max_stand_steps = []
    final_heights = []
    sustained_last_second = []
    successes_sustained_last_second = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            term.beta.fill_(args.action_beta)
            term.assist_force.fill_(args.assist_force_n)
            ever_stood = torch.zeros(
                args.num_envs, dtype=torch.bool, device=args.device
            )
            fell_after = torch.zeros_like(ever_stood)
            max_height = torch.full((args.num_envs,), float("-inf"), device=args.device)
            stand_streak = torch.zeros(
                args.num_envs, dtype=torch.long, device=args.device
            )
            longest_streak = torch.zeros_like(stand_streak)
            last_second_min_height = torch.full_like(max_height, float("inf"))
            for step in range(args.steps):
                with torch.no_grad():
                    action = (
                        policy.distribution(obs["actor"]).sample()
                        if args.stochastic
                        else policy.deterministic_action(obs["actor"])
                    )
                obs, _, _, _, _ = env.step(action)
                height = env.scene["robot"].data.root_link_pos_w[:, 2]
                max_height = torch.maximum(max_height, height)
                stand_streak = torch.where(
                    height > 0.7, stand_streak + 1, torch.zeros_like(stand_streak)
                )
                longest_streak = torch.maximum(longest_streak, stand_streak)
                if step >= args.steps - 50:
                    last_second_min_height = torch.minimum(
                        last_second_min_height, height
                    )
                ever_stood |= height > 0.7
                fell_after |= ever_stood & (height < 0.5)
            success = ever_stood & ~fell_after
            successes.extend(success.cpu().tolist())
            stood_counts.extend(ever_stood.cpu().tolist())
            fell_counts.extend(fell_after.cpu().tolist())
            max_heights.extend(max_height.cpu().tolist())
            max_stand_steps.extend(longest_streak.cpu().tolist())
            final_heights.extend(height.cpu().tolist())
            sustained_last_second.extend((last_second_min_height > 0.7).cpu().tolist())
            successes_sustained_last_second.extend(
                (success & (last_second_min_height > 0.7)).cpu().tolist()
            )
            print(
                f"episode={episode} successes={int(success.sum())}/{args.num_envs}",
                flush=True,
            )
        result = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_source": model_source,
            "checkpoint_model_source": checkpoint.get("model_source", "mjlab"),
            "physics_revision": physics_revision,
            "checkpoint_physics_revision": checkpoint_revision,
            "checkpoint_reward_variant": checkpoint.get("reward_variant", "paper"),
            "checkpoint_ppo_schedule": checkpoint.get("ppo_schedule", "fixed"),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "episodes_per_env": args.episodes,
            "steps_per_episode": args.steps,
            "criterion": "ever pelvis z > 0.7 m and never pelvis z < 0.5 m after standing",
            "assistance_force_n": args.assist_force_n,
            "action_beta": args.action_beta,
            "stochastic": args.stochastic,
            "ever_stood": sum(stood_counts),
            "fell_after_standing": sum(fell_counts),
            "successes": sum(successes),
            "total": len(successes),
            "success_rate": sum(successes) / len(successes),
            "mean_max_pelvis_height_m": sum(max_heights) / len(max_heights),
            "max_pelvis_height_m": max(max_heights),
            "mean_longest_stand_s": 0.02 * sum(max_stand_steps) / len(max_stand_steps),
            "longest_stand_s": 0.02 * max(max_stand_steps),
            "mean_final_pelvis_height_m": sum(final_heights) / len(final_heights),
            "ended_above_0_7m": sum(value > 0.7 for value in final_heights),
            "sustained_last_second_above_0_7m": sum(sustained_last_second),
            "successes_sustained_last_second_above_0_7m": sum(
                successes_sustained_last_second
            ),
        }
        print(json.dumps(result, indent=2))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("flat", "flat29"), default="flat")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-source",
        choices=("mjlab", "host_urdf", "host_urdf_capsule"),
        default="mjlab",
    )
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-beta", type=float, default=0.25)
    parser.add_argument("--assist-force-n", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--allow-physics-transfer", action="store_true")
    parser.add_argument("--allow-model-transfer", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.num_envs, args.episodes, args.steps) <= 0:
        parser.error("counts must be positive")
    if not 0.0 < args.action_beta <= 1.0 or args.assist_force_n < 0.0:
        parser.error("action beta must be in (0, 1] and assist force nonnegative")
    evaluate(args)


if __name__ == "__main__":
    main()
