"""Compare flat standing posture of pose-v6 and A6 policies in the same A6 worlds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.model import (
    HOST29_JOINT_NAMES,
    HOST29_STAND_ARM_JOINT_POS,
    HOST29_STAND_WAIST_JOINT_NAMES,
    HOST29_TARGET_JOINT_POS,
)
from g1recovery_host.ppo import HostPolicy

FOUR_VIEW_LAYOUT = ((2, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))


def evaluate(label: str, path: Path, args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    checkpoint = torch.load(path, map_location=args.device, weights_only=False)
    if checkpoint.get("model_source") != "unitree_g1_29":
        raise ValueError(f"{path} is not a native 29-DoF G1 checkpoint")
    policy = HostPolicy(
        564,
        29,
        log_std_bounds=(-2.3, math.log(0.7)),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    cfg = host_a6_env_cfg(args.num_envs, args.seed, arm="G+")
    layout = (
        FOUR_VIEW_LAYOUT
        if args.layout == "four-view"
        else tuple((0, i % 4, 0) for i in range(args.num_envs))
    )
    cfg.events["reset_scene"].params = {"layout": layout}
    # Match Viser's reset-on-termination behavior for obstacle worlds; the
    # flat-only experiment uses one uninterrupted episode for posture scoring.
    cfg.auto_reset = args.layout == "four-view"
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        names = tuple(robot.joint_names)
        if names != HOST29_JOINT_NAMES:
            raise ValueError("A6 robot joint order differs from the 29-DoF expert")
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(0.25)
        term.assist_force.zero_()
        saved = {key: [] for key in ("q", "action", "torque", "height", "upright")}
        ever_stood = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        max_height = torch.full((args.num_envs,), float("-inf"), device=env.device)
        ever_escaped = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        for step in range(args.steps):
            with torch.no_grad():
                action = policy.deterministic_action(obs["actor"])
                obs, _, _, _, _ = env.step(action)
            height = robot.data.root_link_pos_w[:, 2]
            upright = robot.data.projected_gravity_b[:, 2] < -0.8
            ever_stood |= (height > 0.7) & upright
            max_height = torch.maximum(max_height, height)
            ever_escaped |= env._a6_escaped
            if step >= args.late_start:
                for key, value in (
                    ("q", robot.data.joint_pos),
                    ("action", action),
                    ("torque", robot.data.qfrc_actuator),
                    ("height", height),
                    ("upright", upright),
                ):
                    saved[key].append(value.detach().cpu().clone())
        series = {key: torch.cat(values) for key, values in saved.items()}
        standing = (series["height"] > 0.7) & series["upright"]
        target = torch.tensor([HOST29_TARGET_JOINT_POS[name] for name in names])
        arm_ids = [names.index(name) for name in HOST29_STAND_ARM_JOINT_POS]
        waist_ids = [names.index(name) for name in HOST29_STAND_WAIST_JOINT_NAMES]
        result = {
            "label": label,
            "layout": args.layout,
            "checkpoint": str(path),
            "iteration": checkpoint.get("iteration"),
            "parent_sha256": checkpoint.get("parent_sha256"),
            "standing_worlds": int(ever_stood.sum()),
            "late_standing_frames": int(standing.sum()),
            "late_total_frames": int(standing.numel()),
            "late_standing_fraction": float(standing.float().mean()),
            "late_mean_pelvis_height_m": float(series["height"].mean()),
            "late_upright_fraction": float(series["upright"].float().mean()),
            "worlds": [
                {
                    "scene": int(env._a6_scene[i]),
                    "direction": int(env._a6_direction[i]),
                    "ever_stood": bool(ever_stood[i]),
                    "ever_escaped": bool(ever_escaped[i]),
                    "max_pelvis_height_m": round(float(max_height[i]), 4),
                    "late_mean_pelvis_height_m": round(
                        float(series["height"].reshape(-1, args.num_envs)[:, i].mean()),
                        4,
                    ),
                    "late_upper_joint_means_rad": {
                        name: round(
                            float(
                                series["q"]
                                .reshape(-1, args.num_envs, 29)[:, i, names.index(name)]
                                .mean()
                            ),
                            4,
                        )
                        for name in (
                            "waist_pitch_joint",
                            "left_shoulder_pitch_joint",
                            "left_elbow_joint",
                            "left_wrist_pitch_joint",
                            "right_shoulder_pitch_joint",
                            "right_elbow_joint",
                            "right_wrist_pitch_joint",
                        )
                    },
                }
                for i in range(args.num_envs)
            ],
        }
        all_q = series["q"]
        all_action = series["action"]
        hard_limits = robot.data.joint_pos_limits[0].cpu()
        result["late_upper_joints"] = {
            name: {
                "mean_rad": round(float(all_q[:, i].mean()), 4),
                "target_rad": round(float(target[i]), 4),
                "mean_abs_error_rad": round(
                    float((all_q[:, i] - target[i]).abs().mean()), 4
                ),
                "mean_action": round(float(all_action[:, i].mean()), 4),
                "near_joint_limit_fraction": round(
                    float(
                        (
                            (all_q[:, i] < hard_limits[i, 0] + 0.05)
                            | (all_q[:, i] > hard_limits[i, 1] - 0.05)
                        )
                        .float()
                        .mean()
                    ),
                    4,
                ),
            }
            for i, name in enumerate(names)
            if i >= 12
        }
        if standing.any():
            q = series["q"][standing]
            action = series["action"][standing]
            torque = series["torque"][standing]
            hard_limits = robot.data.joint_pos_limits[0].cpu()
            effort = term._effort_limit.cpu()
            result["arm_pose_score"] = float(
                torch.exp(
                    -0.75 * (q[:, arm_ids] - target[arm_ids]).square().sum(-1)
                ).mean()
            )
            result["waist_pose_score"] = float(
                torch.exp(
                    -3.0 * (q[:, waist_ids] - target[waist_ids]).square().sum(-1)
                ).mean()
            )
            result["joints"] = {
                name: {
                    "mean_rad": round(float(q[:, i].mean()), 4),
                    "target_rad": round(float(target[i]), 4),
                    "mean_abs_error_rad": round(
                        float((q[:, i] - target[i]).abs().mean()), 4
                    ),
                    "mean_action": round(float(action[:, i].mean()), 4),
                    "near_joint_limit_fraction": round(
                        float(
                            (
                                (q[:, i] < hard_limits[i, 0] + 0.05)
                                | (q[:, i] > hard_limits[i, 1] - 0.05)
                            )
                            .float()
                            .mean()
                        ),
                        4,
                    ),
                    "near_torque_limit_fraction": round(
                        float((torque[:, i].abs() > 0.95 * effort[i]).float().mean()), 4
                    ),
                }
                for i, name in enumerate(names)
            }
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", required=True, help="label:path"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument(
        "--layout", choices=("flat-directions", "four-view"), default="flat-directions"
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--late-start", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_envs < 4 or not 0 <= args.late_start < args.steps <= 500:
        parser.error("require num-envs >= 4 and 0 <= late-start < steps <= 500")
    if args.layout == "four-view" and args.num_envs != 4:
        parser.error("four-view layout requires --num-envs 4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    results = []
    for spec in args.checkpoint:
        if ":" not in spec:
            parser.error("each --checkpoint must be label:path")
        label, filename = spec.split(":", 1)
        path = Path(filename)
        if not path.is_file():
            parser.error(f"checkpoint missing: {path}")
        result = evaluate(label, path, args)
        results.append(result)
        print(
            f"{label}: standing={result['late_standing_frames']}/{result['late_total_frames']} "
            f"arm={result.get('arm_pose_score', float('nan')):.4f} "
            f"waist={result.get('waist_pose_score', float('nan')):.4f}",
            flush=True,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
