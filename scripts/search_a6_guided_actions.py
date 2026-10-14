"""Search simple joint-action biases for unassisted guided-board motion.

Each candidate sees the same reset pose within a posture. This is a physics
feasibility diagnostic, not a trained policy or a substitute for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_reward import guided_exit_margin
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.model import HOST29_JOINT_NAMES
from g1recovery_host.ppo import HostPolicy


def candidate_actions(params: torch.Tensor) -> torch.Tensor:
    bias = torch.zeros(len(params), 29, device=params.device)
    groups = (
        ("hip_roll", 0, 0.9, (1.0, -1.0)),
        ("ankle_roll", 1, 0.9, (1.0, -1.0)),
        ("shoulder_roll", 2, 1.2, (1.0, -1.0)),
        ("waist_roll", 3, 0.7, (1.0,)),
        ("hip_yaw", 4, 0.8, (1.0, -1.0)),
        ("shoulder_yaw", 5, 1.0, (1.0, -1.0)),
        ("shoulder_pitch", 6, 1.0, (1.0, 1.0)),
        ("elbow", 7, 1.0, (1.0, 1.0)),
        ("knee", 8, 0.9, (1.0, 1.0)),
        ("hip_pitch", 9, 0.9, (1.0, 1.0)),
    )
    for stem, column, amplitude, signs in groups:
        names = (
            (f"left_{stem}_joint", f"right_{stem}_joint")
            if len(signs) == 2
            else (f"{stem}_joint",)
        )
        for name, sign in zip(names, signs, strict=True):
            bias[:, HOST29_JOINT_NAMES.index(name)] = sign * amplitude * params[:, column]
    return bias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--bias-until-step", type=int, default=120)
    parser.add_argument("--guided-offset-m", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=1)
    args = parser.parse_args()
    if args.candidates < 1 or (2 * args.candidates) % 32:
        parser.error("twice the candidate count must be divisible by 32")
    if not 30 < args.bias_until_step <= args.steps:
        parser.error("bias end must be between motor activation and episode end")
    if args.generations < 1:
        parser.error("generations must be positive")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    variant = state.get("reward_variant", "")
    if "_staged_escape_v5" not in variant or "_aligned_edge_v1" not in variant:
        parser.error("search requires a staged-v5, aligned-edge A6 checkpoint")
    torch.manual_seed(args.seed)
    policy = HostPolicy(
        612,
        29,
        log_std_bounds=(-2.3, math.log(0.7)),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval()
    count = args.candidates
    layout = np.zeros((2 * count, 3), dtype=np.int64)
    layout[:, 0] = 1
    layout[count:, 1] = 1
    cfg = host_a6_env_cfg(
        2 * count,
        args.seed,
        arm="G+",
        edge_curriculum=True,
        plate_observation=True,
        safe_actions=True,
        aligned_edge=True,
    )
    cfg.events["reset_scene"].params["layout"] = layout
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps * 0.02
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env._a6_staged_escape_v5 = True
        env._a6_edge_offset_m = args.guided_offset_m
        env.reset()
        for key, pool in env._a6_pools.items():
            env._a6_pools[key] = pool[:1]
        env._a6_source.zero_()
        action_term = env.action_manager.get_term("joint_pos")
        rows = torch.arange(2 * count, device=args.device)
        plate = env._a6_plate_geoms[0]
        sobol = 2.0 * torch.quasirandom.SobolEngine(10, scramble=True, seed=args.seed).draw(count) - 1.0
        sobol[0] = 0.0
        params = sobol.to(args.device).repeat(2, 1)
        result = {"supine": [], "prone": []}
        for generation in range(args.generations):
            obs, _ = env.reset()
            obs = obs["actor"]
            action_term.beta.fill_(float(state["curriculum"]["beta"].float().mean()))
            action_term.assist_force.zero_()
            action_term.guided_escape_assist_force.zero_()
            bias = candidate_actions(params)
            best_progress = torch.zeros(2 * count, device=args.device)
            lift_peak = torch.zeros_like(best_progress)
            invalid = torch.zeros(2 * count, device=args.device, dtype=torch.bool)
            escaped = torch.zeros_like(invalid)
            alive = torch.ones_like(invalid)
            for step in range(args.steps):
                with torch.no_grad():
                    action = policy.deterministic_action(obs)
                    if 30 <= step < args.bias_until_step:
                        action += bias
                    action[~alive] = 0
                    obs, _, terminated, truncated, _ = env.step(action)
                    obs = obs["actor"]
                    robot = env.scene["robot"].data
                    plate_xy = env.sim.data.geom_xpos[rows, plate, :2]
                    margin = guided_exit_margin(robot.root_link_pos_w[:, :2], plate_xy)
                    progress = (env._a6_initial_exit_margin - margin).clamp_min(0)
                    best_progress = torch.where(
                        alive, torch.maximum(best_progress, progress), best_progress
                    )
                    lift_peak = torch.where(
                        alive, torch.maximum(lift_peak, env._a6_plate_lift_peak), lift_peak
                    )
                    invalid |= alive & env._a6_invalid
                    escaped |= alive & env._a6_escaped
                    done = terminated | truncated
                    alive &= ~done
                    if done.any():
                        reset_obs, _ = env.reset(env_ids=done.nonzero().squeeze(-1))
                        obs = reset_obs["actor"]
            next_params = params.clone()
            for posture, ids in (("supine", slice(0, count)), ("prone", slice(count, 2 * count))):
                score = best_progress[ids] - invalid[ids].float() + 2.0 * escaped[ids].float()
                top = torch.topk(score, min(12, count)).indices
                result[posture].append(
                    {
                        "generation": generation,
                        "escaped": int(escaped[ids].sum()),
                        "invalid": int(invalid[ids].sum()),
                        "best_progress_max_m": float(best_progress[ids].max()),
                        "top": [
                            {
                                "candidate": int(i),
                                "progress_m": float(best_progress[ids][i]),
                                "invalid": bool(invalid[ids][i]),
                                "escaped": bool(escaped[ids][i]),
                                "plate_lift_peak_m": float(lift_peak[ids][i]),
                                "parameters": [round(float(x), 3) for x in params[ids][i]],
                            }
                            for i in top
                        ],
                    }
                )
                elite = params[ids][top]
                sample = (
                    elite.mean(0)
                    + torch.randn(count, 10, device=args.device)
                    * elite.std(0).clamp_min(0.15)
                ).clamp(-1.0, 1.0)
                sample[: min(4, count)] = elite[: min(4, count)]
                next_params[ids] = sample
            params = next_params
            print(
                json.dumps(
                    {
                        posture: {
                            key: result[posture][-1][key]
                            for key in ("escaped", "invalid", "best_progress_max_m")
                        }
                        for posture in result
                    }
                ),
                flush=True,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
