"""Search joint-action pulses for safe, unassisted centred guided-side escape.

This is a simulator feasibility diagnostic. It does not change the saved PPO
actor, use privileged observations, or count as a deployment policy result.
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


GROUPS = (
    ("hip_roll", .45, (1., -1.)),
    ("ankle_roll", .45, (1., -1.)),
    ("shoulder_roll", .60, (1., -1.)),
    ("waist_roll", .35, (1.,)),
    ("hip_yaw", .40, (1., -1.)),
    ("shoulder_yaw", .50, (1., -1.)),
    ("shoulder_pitch", .45, (1., 1.)),
    ("elbow", .45, (1., 1.)),
    ("knee", .45, (1., 1.)),
    ("hip_pitch", .45, (1., 1.)),
)


def candidate_bias(params: torch.Tensor) -> torch.Tensor:
    bias = torch.zeros(len(params), 29, device=params.device)
    for column, (stem, amplitude, signs) in enumerate(GROUPS):
        names = ((f"left_{stem}_joint", f"right_{stem}_joint")
                 if len(signs) == 2 else (f"{stem}_joint",))
        for name, sign in zip(names, signs, strict=True):
            bias[:, HOST29_JOINT_NAMES.index(name)] = sign * amplitude * params[:, column]
    return bias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--candidates-per-side", type=int, default=128)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--pulse-until", type=int, default=120)
    args = parser.parse_args()
    n = args.candidates_per_side
    if n < 16 or (2 * n) % 32 or args.generations < 1:
        parser.error("candidate count must be >=16, total divisible by32")
    if not 30 < args.pulse_until <= args.steps:
        parser.error("pulse must end after motor startup and within episode")
    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if state.get("actor_obs_dim") != 738 or state.get("action_contract") != "scene_independent_v1":
        parser.error("requires a 738D proprio actor with scene-independent actions")
    policy = HostPolicy(738, 29, critic_obs_dim=786,
                        log_std_bounds=(-2.3, math.log(state.get("action_std_max", .7))),
                        upper_residual=True, waist_residual_scale=5.).to(args.device)
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval()
    layout = np.zeros((2*n, 3), dtype=np.int64)
    layout[:, 0] = 1
    layout[:n, 1] = 2
    layout[n:, 1] = 3
    cfg = host_a6_env_cfg(2*n, args.seed, arm="G+", edge_curriculum=True,
                           critic_plate_observation=True, proprio_motor_torque=True,
                           aligned_edge=True, footprint_edge=True,
                           footprint_overlap_m=.03, privileged_guided_actions=False)
    cfg.events["reset_scene"].params["layout"] = layout
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps * .02
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env._a6_staged_escape_v5 = True
        env._a6_prone_flexible_exit_v1 = True
        env._a6_arm_extraction_v1 = True
        env._a6_edge_offset_m = 0.
        obs, _ = env.reset()
        for direction in (2, 3):
            env._a6_pools[direction, 0] = env._a6_pools[direction, 0][:1]
        env._a6_source.zero_()
        term = env.action_manager.get_term("joint_pos")
        rows = torch.arange(2*n, device=args.device)
        sobol = torch.quasirandom.SobolEngine(len(GROUPS), scramble=True, seed=args.seed)
        params = (2*sobol.draw(n)-1).to(args.device).repeat(2, 1)
        params[0] = 0
        params[n] = 0
        records = []
        for generation in range(args.generations):
            obs, _ = env.reset()
            obs = obs["actor"]
            term.beta.fill_(float(state["curriculum"]["beta"].float().mean()))
            term.assist_force.zero_()
            term.guided_escape_assist_force.zero_()
            bias = candidate_bias(params)
            best_progress = torch.zeros(2*n, device=args.device)
            invalid = torch.zeros(2*n, device=args.device, dtype=torch.bool)
            escaped = invalid.clone()
            alive = ~invalid
            lift = torch.zeros_like(best_progress)
            for step in range(1, args.steps+1):
                with torch.no_grad():
                    action = policy.deterministic_action(obs)
                    if 31 <= step <= args.pulse_until:
                        action += bias
                    action[~alive] = 0
                    result, _, terminated, truncated, _ = env.step(action)
                    obs = result["actor"]
                    margin = guided_exit_margin(
                        env.scene["robot"].data.root_link_pos_w[:, :2],
                        env.sim.data.geom_xpos[rows, env._a6_plate_geoms[0], :2],
                    )
                    progress = (env._a6_initial_exit_margin-margin).clamp_min(0)
                    best_progress = torch.where(alive, torch.maximum(best_progress, progress), best_progress)
                    lift = torch.where(alive, torch.maximum(lift, env._a6_plate_lift_peak), lift)
                    invalid |= alive & env._a6_invalid
                    escaped |= alive & env._a6_escaped & ~env._a6_invalid
                    done = (terminated | truncated) & alive
                    alive &= ~done
                    if done.any():
                        reset_obs, _ = env.reset(env_ids=done.nonzero().squeeze(-1))
                        obs = reset_obs["actor"]
            next_params = params.clone()
            row = {"generation": generation}
            for label, sl in (("left", slice(0,n)), ("right", slice(n,2*n))):
                score = best_progress[sl] - 3*invalid[sl].float() + 4*escaped[sl].float()
                top = torch.topk(score, min(12,n)).indices
                elites = params[sl][top]
                row[label] = {
                    "escaped": int(escaped[sl].sum()),
                    "invalid": int(invalid[sl].sum()),
                    "best_progress_max_m": float(best_progress[sl].max()),
                    "top": [dict(candidate=int(i), progress_m=float(best_progress[sl][i]),
                                 escaped=bool(escaped[sl][i]), invalid=bool(invalid[sl][i]),
                                 lift_m=float(lift[sl][i]), params=params[sl][i].tolist())
                            for i in top[:5]],
                }
                sample = (elites.mean(0) + torch.randn(n,len(GROUPS),device=args.device)
                          * elites.std(0).clamp_min(.15)).clamp(-1.,1.)
                sample[:min(len(elites),4)] = elites[:min(len(elites),4)]
                next_params[sl] = sample
            records.append(row)
            params = next_params
            print(json.dumps({k: {t: row[k][t] for t in ("escaped","invalid","best_progress_max_m")}
                              for k in ("left","right")}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records,indent=2),encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
