"""Deterministic four-posture evaluation for the 29-joint flat HoST baseline."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.host_recovery.config.g1.rl_cfg import unitree_g1_host_standup_ppo_runner_cfg
from src.tasks.host_recovery.flat29 import flat29_env_cfg
from src.tasks.host_recovery.mdp.events import POSTURE_QUATS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


POSTURES = ("prone", "supine", "left_side", "right_side")


def _fraction(value: torch.Tensor) -> float:
    return value.float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--hold-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    cfg = flat29_env_cfg(play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    agent = unitree_g1_host_standup_ppo_runner_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    try:
        runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
        runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        robot = env.scene["robot"]
        torso_id = robot.find_bodies("torso_link")[0][0]
        foot_ids = torch.tensor(
            robot.find_sites(("left_foot", "right_foot"))[0], device=env.device
        )
        reset_cfg = env.event_manager.get_term_cfg("reset_base")
        results = {}

        for posture in POSTURES:
            reset_cfg.params["posture"] = posture
            wrapper.reset()
            expected = torch.tensor(POSTURE_QUATS[posture], device=env.device)
            alignment = (robot.data.root_link_quat_w * expected).sum(-1).abs()
            if not (alignment > 0.999).all():
                raise RuntimeError(f"{posture} reset mismatch: {alignment.min().item()}")

            obs = wrapper.get_observations()
            valid = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
            ever_standing = torch.zeros_like(valid)
            held = torch.zeros_like(valid)
            consecutive = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
            first_stand = torch.full((args.num_envs,), -1, dtype=torch.long, device=env.device)
            first_hold = torch.full_like(first_stand, -1)
            final_standing = torch.zeros_like(valid)
            final_upright = torch.zeros_like(valid)

            with torch.inference_mode():
                for step in range(args.steps):
                    actions = policy(obs["actor"])
                    obs, _, dones, _ = wrapper.step(actions)
                    valid &= ~dones
                    torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
                    feet_z = robot.data.site_pos_w[:, foot_ids, 2].mean(dim=1)
                    height = torso_z - feet_z
                    upright = robot.data.projected_gravity_b[:, 2] < -0.8
                    standing = (height > 0.6) & upright & valid
                    newly_standing = standing & (first_stand < 0)
                    first_stand[newly_standing] = step
                    ever_standing |= standing
                    consecutive = torch.where(standing, consecutive + 1, 0)
                    newly_held = (consecutive >= args.hold_steps) & ~held
                    first_hold[newly_held] = step - args.hold_steps + 1
                    held |= newly_held
                    final_standing = standing
                    final_upright = upright & valid

            held_times = first_hold[first_hold >= 0].float() * cfg.step_dt
            results[posture] = {
                "valid_fraction": _fraction(valid),
                "termination_fraction": _fraction(~valid),
                "ever_standing_fraction": _fraction(ever_standing),
                "held_fraction": _fraction(held & valid),
                "final_standing_fraction": _fraction(final_standing),
                "final_upright_fraction": _fraction(final_upright),
                "first_hold_seconds_median": (
                    held_times.median().item() if held_times.numel() else None
                ),
                "final_root_linear_speed_mean": robot.data.root_link_lin_vel_w.norm(dim=-1)[valid].mean().item()
                if valid.any() else None,
                "final_joint_speed_rms": robot.data.joint_vel[valid].square().mean().sqrt().item()
                if valid.any() else None,
            }
            print(posture, json.dumps(results[posture], sort_keys=True), flush=True)

        report = {
            "checkpoint": str(args.checkpoint),
            "num_envs_per_posture": args.num_envs,
            "steps": args.steps,
            "step_dt": cfg.step_dt,
            "hold_steps": args.hold_steps,
            "success_definition": "torso-minus-feet > 0.6 m and projected gravity z < -0.8 for consecutive hold_steps",
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
