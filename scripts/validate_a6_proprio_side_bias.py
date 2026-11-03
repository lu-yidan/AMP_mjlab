"""Cross-pose validation of fixed side-lying action pulses found by CEM."""

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
from search_a6_proprio_side_actions import candidate_bias


def candidates(path: Path):
    search = json.loads(path.read_text())
    selected = {}
    for side in ("left", "right"):
        items = [(f"{side}_zero", [0.0] * 10)]
        seen = {tuple([0.0] * 10)}
        for row in search[-3:]:
            for rank, entry in enumerate(row[side]["top"]):
                params = entry["params"]
                key = tuple(round(value, 5) for value in params)
                if key not in seen:
                    items.append((f"{side}_g{row['generation']}_r{rank}", params))
                    seen.add(key)
        selected[side] = items
    return selected


def evaluate(args, seed: int, selected: dict):
    n = args.per_candidate
    labels, parameters, layout = [], [], []
    for side, direction in (("left", 2), ("right", 3)):
        for name, params in selected[side]:
            labels.extend([name] * n)
            parameters.extend([params] * n)
            layout.extend([(1, direction, 0)] * n)
    layout_np = np.array(layout, dtype=np.int64)
    total = len(layout_np)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy = HostPolicy(
        738, 29, critic_obs_dim=786,
        log_std_bounds=(-2.3, math.log(state.get("action_std_max", 0.7))),
        upper_residual=True, waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval().requires_grad_(False)
    cfg = host_a6_env_cfg(
        total, seed, arm="G+", edge_curriculum=True,
        critic_plate_observation=True, proprio_motor_torque=True,
        aligned_edge=True, footprint_edge=True, footprint_overlap_m=0.03,
        privileged_guided_actions=False,
    )
    cfg.events["reset_scene"].params["layout"] = layout_np
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps * 0.02
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env._a6_staged_escape_v5 = True
        env._a6_prone_flexible_exit_v1 = True
        env._a6_arm_extraction_v1 = True
        env._a6_edge_offset_m = 0.0
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(state["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        torch.manual_seed(seed)
        obs, _ = env.reset()
        bias = candidate_bias(torch.tensor(parameters, device=args.device))
        active = torch.ones(total, dtype=torch.bool, device=args.device)
        escaped = torch.zeros_like(active)
        invalid = torch.zeros_like(active)
        stood = torch.zeros_like(active)
        first_escape = torch.zeros(total, dtype=torch.int32, device=args.device)
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                action = policy.deterministic_action(obs["actor"])
                if 31 <= step <= args.pulse_until:
                    action += bias
                action[~active] = 0
                obs, _, terminated, truncated, _ = env.step(action)
                new_escape = active & env._a6_escaped & ~escaped
                first_escape = torch.where(new_escape, step, first_escape)
                escaped |= active & env._a6_escaped
                invalid |= active & env._a6_invalid
                robot = env.scene["robot"].data
                final_stand = ((robot.root_link_pos_w[:, 2] > 0.6)
                               & (-robot.projected_gravity_b[:, 2] > 0.8))
                stood = torch.where(active, final_stand, stood)
                done = active & (terminated | truncated | (step == args.steps))
                active &= ~done
                if not active.any():
                    break
                finished = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(finished):
                    obs, _ = env.reset(env_ids=finished)
        escaped_np = escaped.cpu().numpy()
        invalid_np = invalid.cpu().numpy()
        stood_np = stood.cpu().numpy()
        first_np = first_escape.cpu().numpy()
        result = {}
        labels_np = np.array(labels)
        for name in dict.fromkeys(labels):
            mask = labels_np == name
            success = escaped_np & stood_np & ~invalid_np
            result[name] = {
                "episodes": int(mask.sum()),
                "escaped": int((escaped_np & mask).sum()),
                "escaped_then_stood": int((success & mask).sum()),
                "invalid": int((invalid_np & mask).sum()),
                "first_escape_steps": first_np[mask].tolist(),
            }
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 48])
    parser.add_argument("--per-candidate", type=int, default=32)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--pulse-until", type=int, default=350)
    args = parser.parse_args()
    if args.per_candidate < 4 or not 30 < args.pulse_until < args.steps:
        parser.error("invalid candidate count or pulse interval")
    selected = candidates(args.search)
    report = {
        "checkpoint": str(args.checkpoint), "search": str(args.search),
        "per_candidate": args.per_candidate,
        "pulse_until": args.pulse_until,
        "candidates": selected,
        "results": {},
    }
    for seed in args.seeds:
        report["results"][str(seed)] = evaluate(args, seed, selected)
        print(f"seed={seed} finished", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
