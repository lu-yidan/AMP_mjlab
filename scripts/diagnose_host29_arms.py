"""Measure late-standing joint posture and motor usage of a flat29 checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host import host_ground29_env_cfg
from g1recovery_host.model import (
    HOST29_JOINT_NAMES,
    HOST29_STAND_ARM_JOINT_POS,
    HOST29_STAND_WAIST_JOINT_NAMES,
    HOST29_TARGET_JOINT_POS,
)
from g1recovery_host.ppo import HostPolicy
from g1recovery_host.stand_guard import apply_stand_pose_guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--late-start", type=int, default=350)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--posture-takeover-step", type=int)
    parser.add_argument("--zero-upper-actions", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.late_start < args.steps <= 500:
        parser.error("require 0 <= --late-start < --steps <= 500")

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    if checkpoint.get("task") != "flat29":
        parser.error("checkpoint is not a flat29 policy")
    checkpoint_reward = checkpoint.get("reward_variant", "")
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
    policy = HostPolicy(
        564,
        29,
        upper_residual=pose_version in ("v4", "v5", "v6"),
        waist_residual_scale=5.0 if pose_version in ("v5", "v6") else 2.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    cfg = host_ground29_env_cfg(
        args.num_envs,
        args.seed,
        pose_version=pose_version,
    )
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        names = tuple(robot.joint_names)
        if names != HOST29_JOINT_NAMES:
            raise ValueError("joint order differs from the trained 29-DoF model")
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(0.25)
        term.assist_force.zero_()
        samples = {key: [] for key in ("q", "vel", "torque", "action", "height")}
        snapshots = {}
        for step in range(args.steps):
            with torch.no_grad():
                action = policy.deterministic_action(obs["actor"])
                if args.zero_upper_actions:
                    action[:, 12:] = 0.0
                if args.posture_takeover_step is not None:
                    action = apply_stand_pose_guard(
                        env, action, min_steps=args.posture_takeover_step
                    )
                obs, _, _, _, _ = env.step(action)
            if step in (0, 30, 31, 100, 200, 350, args.steps - 1):
                snapshots[str(step)] = {
                    "mean_pelvis_height_m": round(
                        float(robot.data.root_link_pos_w[:, 2].mean()), 4
                    ),
                    "mean_joint_pos_rad": {
                        name: round(float(robot.data.joint_pos[:, i].mean()), 4)
                        for i, name in enumerate(names)
                        if any(
                            part in name
                            for part in ("waist_pitch", "elbow", "wrist_pitch")
                        )
                    },
                }
            if step >= args.late_start:
                data = robot.data
                samples["q"].append(data.joint_pos.detach().cpu().clone())
                samples["vel"].append(data.joint_vel.detach().cpu().clone())
                samples["torque"].append(data.qfrc_actuator.detach().cpu().clone())
                samples["action"].append(action.detach().cpu().clone())
                samples["height"].append(
                    data.root_link_pos_w[:, 2].detach().cpu().clone()
                )
        series = {key: torch.cat(value) for key, value in samples.items()}
        mask = series["height"] > 0.7
        if not mask.any():
            raise RuntimeError("no late-standing frames with pelvis above 0.7 m")
        target = torch.tensor([HOST29_TARGET_JOINT_POS[name] for name in names])
        limits = term._effort_limit.detach().cpu()
        model = env.sim.mj_model
        joints = {}
        for i, name in enumerate(names):
            q = series["q"][mask, i]
            v = series["vel"][mask, i]
            torque = series["torque"][mask, i]
            action = series["action"][mask, i]
            lower, upper = map(float, model.joint(f"robot/{name}").range)
            joints[name] = {
                "target_rad": round(float(target[i]), 4),
                "mean_rad": round(float(q.mean()), 4),
                "std_rad": round(float(q.std()), 4),
                "mean_abs_error_rad": round(float((q - target[i]).abs().mean()), 4),
                "mean_abs_velocity_rad_s": round(float(v.abs().mean()), 4),
                "mean_action": round(float(action.mean()), 4),
                "mean_abs_action": round(float(action.abs().mean()), 4),
                "mean_torque_nm": round(float(torque.mean()), 4),
                "mean_abs_torque_nm": round(float(torque.abs().mean()), 4),
                "torque_limit_nm": round(float(limits[i]), 4),
                "joint_limits_rad": [round(lower, 4), round(upper, 4)],
                "near_lower_limit_frac": round(
                    float((q < lower + 0.05).float().mean()), 4
                ),
                "near_upper_limit_frac": round(
                    float((q > upper - 0.05).float().mean()), 4
                ),
                "near_action_bound_frac": round(
                    float((action.abs() > 0.98).float().mean()), 4
                ),
                "near_torque_limit_frac": round(
                    float((torque.abs() > 0.95 * limits[i]).float().mean()), 4
                ),
            }
        standing_q = series["q"][mask]
        arm_ids = [names.index(name) for name in HOST29_STAND_ARM_JOINT_POS]
        waist_ids = [names.index(name) for name in HOST29_STAND_WAIST_JOINT_NAMES]
        arm_pose_score = torch.exp(
            -0.75 * (standing_q[:, arm_ids] - target[arm_ids]).square().sum(-1)
        )
        waist_pose_score = torch.exp(
            -3.0 * (standing_q[:, waist_ids] - target[waist_ids]).square().sum(-1)
        )
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "late_start_step": args.late_start,
            "steps": args.steps,
            "num_envs": args.num_envs,
            "posture_takeover_step": args.posture_takeover_step,
            "zero_upper_actions": args.zero_upper_actions,
            "standing_samples": int(mask.sum()),
            "total_late_samples": int(mask.numel()),
            "mean_standing_pelvis_height_m": round(
                float(series["height"][mask].mean()), 4
            ),
            "mean_arm_pose_score": float(arm_pose_score.mean()),
            "mean_waist_pose_score": float(waist_pose_score.mean()),
            "snapshots": snapshots,
            "joints": joints,
        }
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
