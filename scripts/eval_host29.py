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
    parser.add_argument(
        "--action-scale", default="checkpoint",
        help="checkpoint to restore the saved curriculum value, or a numeric scale",
    )
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
        if args.action_scale == "checkpoint":
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            saved_scale = saved.get("infos", {}).get("env_state", {}).get("host_action_rescale")
            if saved_scale is None:
                raise RuntimeError("checkpoint has no host_action_rescale state")
            action_scale = float(saved_scale.float().mean())
        else:
            action_scale = float(args.action_scale)
        action_term = env.action_manager.get_term("joint_pos")
        if isinstance(action_term.cfg.scale, dict):
            action_term.cfg.scale = {key: action_scale for key in action_term.cfg.scale}
        else:
            action_term.cfg.scale = action_scale
        if isinstance(action_term._scale, torch.Tensor):
            action_term._scale[:] = action_scale
        else:
            action_term._scale = action_scale
        env._host_action_rescale = torch.full(
            (args.num_envs, 1), action_scale, device=env.device
        )
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
            standing_sample_count = torch.zeros((), device=env.device)
            standing_yaw_rate_sum = torch.zeros((), device=env.device)
            standing_yaw_rate_abs_sum = torch.zeros((), device=env.device)
            standing_yaw_rate_sq_sum = torch.zeros((), device=env.device)
            standing_horizontal_speed_sq_sum = torch.zeros((), device=env.device)
            standing_root_height_sum = torch.zeros((), device=env.device)
            standing_joint_speed_sq_sum = torch.zeros(
                len(robot.joint_names), device=env.device
            )

            # Environment managers update persistent buffers in-place between
            # postures, so no_grad is required here instead of inference_mode.
            with torch.no_grad():
                for step in range(args.steps):
                    actions = policy(obs["actor"])
                    obs, _, dones, _ = wrapper.step(actions)
                    valid &= ~dones.bool()
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
                    standing_float = standing.float()
                    standing_sample_count += standing_float.sum()
                    standing_yaw_rate_sum += (
                        robot.data.root_link_ang_vel_w[:, 2] * standing_float
                    ).sum()
                    standing_yaw_rate_abs_sum += (
                        robot.data.root_link_ang_vel_w[:, 2].abs()
                        * standing_float
                    ).sum()
                    standing_yaw_rate_sq_sum += (
                        robot.data.root_link_ang_vel_w[:, 2].square()
                        * standing_float
                    ).sum()
                    standing_horizontal_speed_sq_sum += (
                        robot.data.root_link_lin_vel_w[:, :2].square().sum(dim=-1)
                        * standing_float
                    ).sum()
                    standing_root_height_sum += (
                        robot.data.root_link_pos_w[:, 2] * standing_float
                    ).sum()
                    standing_joint_speed_sq_sum += (
                        robot.data.joint_vel.square() * standing_float[:, None]
                    ).sum(dim=0)

            held_times = first_hold[first_hold >= 0].float() * env.step_dt
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
                "final_root_height_mean": robot.data.root_link_pos_w[:, 2][valid].mean().item()
                if valid.any() else None,
                "final_root_yaw_rate_rms": robot.data.root_link_ang_vel_w[:, 2][valid].square().mean().sqrt().item()
                if valid.any() else None,
                "standing_yaw_rate_rms": (
                    (standing_yaw_rate_sq_sum / standing_sample_count).sqrt().item()
                    if standing_sample_count.item() > 0 else None
                ),
                "standing_yaw_rate_mean": (
                    (standing_yaw_rate_sum / standing_sample_count).item()
                    if standing_sample_count.item() > 0 else None
                ),
                "standing_yaw_rate_abs_mean": (
                    (standing_yaw_rate_abs_sum / standing_sample_count).item()
                    if standing_sample_count.item() > 0 else None
                ),
                "standing_horizontal_speed_rms": (
                    (standing_horizontal_speed_sq_sum / standing_sample_count).sqrt().item()
                    if standing_sample_count.item() > 0 else None
                ),
                "standing_root_height_mean": (
                    (standing_root_height_sum / standing_sample_count).item()
                    if standing_sample_count.item() > 0 else None
                ),
                "final_joint_speed_rms": robot.data.joint_vel[valid].square().mean().sqrt().item()
                if valid.any() else None,
                "standing_joint_speed_rms": (
                    {
                        name: value
                        for name, value in zip(
                            robot.joint_names,
                            (standing_joint_speed_sq_sum / standing_sample_count)
                            .sqrt()
                            .tolist(),
                            strict=True,
                        )
                    }
                    if standing_sample_count.item() > 0 else None
                ),
            }
            print(posture, json.dumps(results[posture], sort_keys=True), flush=True)

        report = {
            "checkpoint": str(args.checkpoint),
            "num_envs_per_posture": args.num_envs,
            "steps": args.steps,
            "step_dt": env.step_dt,
            "hold_steps": args.hold_steps,
            "action_scale": action_scale,
            "success_definition": "torso-minus-feet > 0.6 m and projected gravity z < -0.8 for consecutive hold_steps",
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
