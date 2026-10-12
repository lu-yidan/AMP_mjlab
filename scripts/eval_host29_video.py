"""Record 29-DoF flat recovery rollout videos for the long-train checkpoint.

Forces every environment into one of the four postures, then records environment
0 with :class:`VideoRecorder`. Extra-environment rendering is disabled so the
offscreen renderer draws a single robot instead of stacking neighbors on top of
environment 0.
"""

from __future__ import annotations

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
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.wrappers import VideoRecorder

from src.tasks.host_recovery.config.g1.rl_cfg import (
    unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.flat29 import flat29_env_cfg
from src.tasks.host_recovery.mdp.events import POSTURE_QUATS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner
from scripts.eval_host29_upper_override import UPPER_JOINT_NAMES, UPPER_TARGET


POSTURES = ("prone", "supine", "left_side", "right_side")


def record_posture(
    env: ManagerBasedRlEnv,
    checkpoint: Path,
    agent_cfg,
    output_dir: Path,
    posture: str,
    steps: int,
    device: str,
    override_blend: float,
    override_gain: float,
    balance_blend: float,
    balance_gain: float,
    override_delay_steps: int,
    override_ramp_steps: int,
    foot_align_blend: float,
    foot_align_gain: float,
    foot_align_max_action: float,
    foot_align_mode: str,
    foot_align_delay_steps: int,
    foot_align_ramp_steps: int,
    sagittal_symmetry_blend: float,
    sagittal_symmetry_gain: float,
    sagittal_symmetry_integral_gain: float,
    sagittal_symmetry_integral_limit: float,
    sagittal_symmetry_max_action: float,
    sagittal_symmetry_mode: str,
    sagittal_symmetry_delay_steps: int,
    sagittal_symmetry_ramp_steps: int,
    sagittal_target_blend: float,
    sagittal_target_gain: float,
    sagittal_target_max_action: float,
    sagittal_target: tuple[float, float, float],
    sagittal_target_side: str,
    sagittal_target_delay_steps: int,
    sagittal_target_ramp_steps: int,
) -> Path:
    env.event_manager.get_term_cfg("reset_base").params["posture"] = posture

    recorder = VideoRecorder(
        env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=steps,
        name_prefix=f"host29_{posture}",
        disable_logger=True,
    )
    wrapper = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_optimizer=False, map_location=device)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_scale = saved.get("infos", {}).get("env_state", {}).get("host_action_rescale")
    if saved_scale is None:
        raise RuntimeError("checkpoint has no host_action_rescale state")
    action_scale = float(saved_scale.float().mean())

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
        (env.num_envs, 1), action_scale, device=env.device
    )

    policy = runner.get_inference_policy(device=device)
    robot = env.scene["robot"]
    torso_id = robot.find_bodies("torso_link")[0][0]
    foot_ids = torch.tensor(
        robot.find_sites(("left_foot", "right_foot"))[0], device=env.device
    )
    upper_joint_ids = torch.tensor(
        robot.find_joints(UPPER_JOINT_NAMES, preserve_order=True)[0],
        device=env.device,
    )
    target_ids = list(action_term._target_ids)
    upper_action_ids = torch.tensor(
        [target_ids.index(int(joint_id)) for joint_id in upper_joint_ids.tolist()],
        device=env.device,
    )
    hip_pitch_joint_ids = torch.tensor(
        robot.find_joints(
            ("left_hip_pitch_joint", "right_hip_pitch_joint"),
            preserve_order=True,
        )[0],
        device=env.device,
    )
    hip_pitch_action_ids = torch.tensor(
        [target_ids.index(int(joint_id)) for joint_id in hip_pitch_joint_ids.tolist()],
        device=env.device,
    )
    sagittal_joint_ids = torch.tensor(
        robot.find_joints(
            (
                "left_hip_pitch_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "right_hip_pitch_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
            ),
            preserve_order=True,
        )[0],
        device=env.device,
    ).view(2, 3)
    sagittal_action_ids = torch.tensor(
        [target_ids.index(int(joint_id)) for joint_id in sagittal_joint_ids.flatten().tolist()],
        device=env.device,
    ).view(2, 3)
    sagittal_target_tensor = torch.tensor(sagittal_target, device=env.device)
    sagittal_target_side_mask = torch.tensor(
        {
            "both": (1.0, 1.0),
            "left": (1.0, 0.0),
            "right": (0.0, 1.0),
        }[sagittal_target_side],
        device=env.device,
    )
    upper_target = torch.tensor(UPPER_TARGET, device=env.device)
    override_count = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    symmetry_integral = torch.zeros((env.num_envs, 3), device=env.device)
    obs = wrapper.get_observations()
    with torch.no_grad():
        for _ in range(steps):
            actions = policy(obs["actor"])
            if (
                override_blend > 0.0
                or balance_blend > 0.0
                or foot_align_blend > 0.0
                or sagittal_symmetry_blend > 0.0
                or sagittal_target_blend > 0.0
            ):
                torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
                feet_z = robot.data.site_pos_w[:, foot_ids, 2].mean(dim=1)
                standing = (
                    (torso_z - feet_z > 0.6)
                    & (robot.data.projected_gravity_b[:, 2] < -0.8)
                )
                override_count = torch.where(
                    standing, override_count + 1, 0
                )
                selected = actions[:, upper_action_ids]
            if override_blend > 0.0:
                correction = override_gain * (
                    upper_target - robot.data.joint_pos[:, upper_joint_ids]
                ) / action_scale
                correction = correction.clamp(-1.0, 1.0)
                ramp = (
                    (override_count - override_delay_steps).float()
                    / max(override_ramp_steps, 1)
                ).clamp(0.0, 1.0)
                mix = (override_blend * ramp)[:, None]
                actions[:, upper_action_ids] = selected + mix * (
                    correction - selected
                )
            if balance_blend > 0.0:
                world_down = torch.zeros(
                    (env.num_envs, 3), device=env.device
                )
                world_down[:, 2] = -1.0
                gravity_torso = quat_apply_inverse(
                    robot.data.body_link_quat_w[:, torso_id], world_down
                )
                balance_delta = torch.stack(
                    (gravity_torso[:, 1], -gravity_torso[:, 0]), dim=1
                )
                balance_correction = (
                    balance_gain * balance_delta / action_scale
                ).clamp(-1.0, 1.0)
                ramp = (
                    (override_count - override_delay_steps).float()
                    / max(override_ramp_steps, 1)
                ).clamp(0.0, 1.0)
                mix = (balance_blend * ramp)[:, None]
                actions[:, upper_action_ids[:2]] = selected[:, :2] + mix * (
                    balance_correction - selected[:, :2]
                )
            if foot_align_blend > 0.0:
                root_pos = robot.data.root_link_pos_w
                root_quat = robot.data.root_link_quat_w
                feet_w = robot.data.site_pos_w[:, foot_ids]
                left_foot_b = quat_apply_inverse(root_quat, feet_w[:, 0] - root_pos)
                right_foot_b = quat_apply_inverse(root_quat, feet_w[:, 1] - root_pos)
                forward_error = left_foot_b[:, 0] - right_foot_b[:, 0]
                correction = (
                    foot_align_gain * forward_error / action_scale
                ).clamp(-foot_align_max_action, foot_align_max_action)
                foot_ramp = (
                    (override_count - foot_align_delay_steps).float()
                    / max(foot_align_ramp_steps, 1)
                ).clamp(0.0, 1.0)
                foot_mix = foot_align_blend * foot_ramp
                selected_hip = actions[:, hip_pitch_action_ids]
                if foot_align_mode == "add":
                    actions[:, hip_pitch_action_ids[0]] += foot_mix * correction
                    actions[:, hip_pitch_action_ids[1]] -= foot_mix * correction
                else:
                    common = selected_hip.mean(dim=1)
                    desired = torch.stack(
                        (common + correction, common - correction), dim=1
                    )
                    actions[:, hip_pitch_action_ids] = selected_hip + foot_mix[:, None] * (
                        desired - selected_hip
                    )
            if sagittal_symmetry_blend > 0.0:
                joint_delta = (
                    robot.data.joint_pos[:, sagittal_joint_ids[0]]
                    - robot.data.joint_pos[:, sagittal_joint_ids[1]]
                )
                integrate = override_count > sagittal_symmetry_delay_steps
                symmetry_integral = torch.where(
                    integrate[:, None],
                    (symmetry_integral + env.step_dt * joint_delta).clamp(
                        -sagittal_symmetry_integral_limit,
                        sagittal_symmetry_integral_limit,
                    ),
                    torch.zeros_like(symmetry_integral),
                )
                symmetry_correction = (
                    -0.5
                    * (
                        sagittal_symmetry_gain * joint_delta
                        + sagittal_symmetry_integral_gain * symmetry_integral
                    )
                    / action_scale
                ).clamp(-sagittal_symmetry_max_action, sagittal_symmetry_max_action)
                symmetry_ramp = (
                    (override_count - sagittal_symmetry_delay_steps).float()
                    / max(sagittal_symmetry_ramp_steps, 1)
                ).clamp(0.0, 1.0)
                symmetry_mix = sagittal_symmetry_blend * symmetry_ramp[:, None]
                if sagittal_symmetry_mode == "add":
                    actions[:, sagittal_action_ids[0]] += symmetry_mix * symmetry_correction
                    actions[:, sagittal_action_ids[1]] -= symmetry_mix * symmetry_correction
                else:
                    selected_sagittal = torch.stack(
                        (
                            actions[:, sagittal_action_ids[0]],
                            actions[:, sagittal_action_ids[1]],
                        ),
                        dim=1,
                    )
                    common_sagittal = selected_sagittal.mean(dim=1)
                    desired_sagittal = torch.stack(
                        (
                            common_sagittal + symmetry_correction,
                            common_sagittal - symmetry_correction,
                        ),
                        dim=1,
                    )
                    mixed_sagittal = selected_sagittal + symmetry_mix[:, None] * (
                        desired_sagittal - selected_sagittal
                    )
                    actions[:, sagittal_action_ids[0]] = mixed_sagittal[:, 0]
                    actions[:, sagittal_action_ids[1]] = mixed_sagittal[:, 1]
            if sagittal_target_blend > 0.0:
                target_correction = (
                    sagittal_target_gain
                    * (
                        sagittal_target_tensor[None, None, :]
                        - robot.data.joint_pos[:, sagittal_joint_ids]
                    )
                    / action_scale
                ).clamp(-sagittal_target_max_action, sagittal_target_max_action)
                target_ramp = (
                    (override_count - sagittal_target_delay_steps).float()
                    / max(sagittal_target_ramp_steps, 1)
                ).clamp(0.0, 1.0)
                target_mix = (
                    sagittal_target_blend
                    * target_ramp[:, None, None]
                    * sagittal_target_side_mask[None, :, None]
                )
                selected_sagittal = torch.stack(
                    (
                        actions[:, sagittal_action_ids[0]],
                        actions[:, sagittal_action_ids[1]],
                    ),
                    dim=1,
                )
                mixed_sagittal = selected_sagittal + target_mix * (
                    target_correction - selected_sagittal
                )
                actions[:, sagittal_action_ids[0]] = mixed_sagittal[:, 0]
                actions[:, sagittal_action_ids[1]] = mixed_sagittal[:, 1]
            obs, _, _, _ = wrapper.step(actions)

    return output_dir / f"host29_{posture}-step-0.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--override-blend", type=float, default=0.0)
    parser.add_argument("--override-gain", type=float, default=0.5)
    parser.add_argument("--balance-blend", type=float, default=0.0)
    parser.add_argument("--balance-gain", type=float, default=0.5)
    parser.add_argument("--override-delay-steps", type=int, default=25)
    parser.add_argument("--override-ramp-steps", type=int, default=25)
    parser.add_argument("--foot-align-blend", type=float, default=0.0)
    parser.add_argument("--foot-align-gain", type=float, default=0.05)
    parser.add_argument("--foot-align-max-action", type=float, default=0.10)
    parser.add_argument("--foot-align-mode", choices=("add", "pair"), default="add")
    parser.add_argument("--foot-align-delay-steps", type=int, default=50)
    parser.add_argument("--foot-align-ramp-steps", type=int, default=150)
    parser.add_argument("--sagittal-symmetry-blend", type=float, default=0.0)
    parser.add_argument("--sagittal-symmetry-gain", type=float, default=0.5)
    parser.add_argument("--sagittal-symmetry-integral-gain", type=float, default=0.0)
    parser.add_argument("--sagittal-symmetry-integral-limit", type=float, default=1.0)
    parser.add_argument("--sagittal-symmetry-max-action", type=float, default=0.5)
    parser.add_argument("--sagittal-symmetry-mode", choices=("add", "pair"), default="add")
    parser.add_argument("--sagittal-symmetry-delay-steps", type=int, default=50)
    parser.add_argument("--sagittal-symmetry-ramp-steps", type=int, default=150)
    parser.add_argument("--sagittal-target-blend", type=float, default=0.0)
    parser.add_argument("--sagittal-target-gain", type=float, default=1.0)
    parser.add_argument("--sagittal-target-max-action", type=float, default=1.0)
    parser.add_argument(
        "--sagittal-target", type=float, nargs=3, default=(0.0, 0.0, 0.0)
    )
    parser.add_argument(
        "--sagittal-target-side", choices=("both", "left", "right"), default="both"
    )
    parser.add_argument("--sagittal-target-delay-steps", type=int, default=50)
    parser.add_argument("--sagittal-target-ramp-steps", type=int, default=150)
    args = parser.parse_args()

    if not 0.0 <= args.override_blend <= 1.0:
        raise ValueError("override-blend must lie in [0, 1]")
    if not 0.0 <= args.balance_blend <= 1.0:
        raise ValueError("balance-blend must lie in [0, 1]")
    if args.balance_gain == 0.0:
        raise ValueError("balance-gain must be non-zero")
    if not 0.0 <= args.foot_align_blend <= 1.0:
        raise ValueError("foot-align-blend must lie in [0, 1]")
    if args.foot_align_gain < 0.0:
        raise ValueError("foot-align-gain must be non-negative")
    if args.foot_align_max_action <= 0.0:
        raise ValueError("foot-align-max-action must be positive")
    if not 0.0 <= args.sagittal_symmetry_blend <= 1.0:
        raise ValueError("sagittal-symmetry-blend must lie in [0, 1]")
    if args.sagittal_symmetry_gain < 0.0:
        raise ValueError("sagittal-symmetry-gain must be non-negative")
    if args.sagittal_symmetry_integral_gain < 0.0:
        raise ValueError("sagittal-symmetry-integral-gain must be non-negative")
    if args.sagittal_symmetry_integral_limit <= 0.0:
        raise ValueError("sagittal-symmetry-integral-limit must be positive")
    if args.sagittal_symmetry_max_action <= 0.0:
        raise ValueError("sagittal-symmetry-max-action must be positive")
    if not 0.0 <= args.sagittal_target_blend <= 1.0:
        raise ValueError("sagittal-target-blend must lie in [0, 1]")
    if args.sagittal_target_gain <= 0.0:
        raise ValueError("sagittal-target-gain must be positive")
    if args.sagittal_target_max_action <= 0.0:
        raise ValueError("sagittal-target-max-action must be positive")

    cfg = flat29_env_cfg(play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    # The recovery scene keeps every environment origin at the world origin, so
    # the offscreen renderer would otherwise draw neighboring environments on
    # top of environment 0. Render only environment 0 for a clean video.
    cfg.viewer.max_extra_envs = 0

    agent_cfg = unitree_g1_host_standup_ppo_runner_cfg()
    device = "cuda:0"
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest = []
        for posture in POSTURES:
            path = record_posture(
                env,
                args.checkpoint,
                agent_cfg,
                args.output,
                posture,
                args.steps,
                device,
                args.override_blend,
                args.override_gain,
                args.balance_blend,
                args.balance_gain,
                args.override_delay_steps,
                args.override_ramp_steps,
                args.foot_align_blend,
                args.foot_align_gain,
                args.foot_align_max_action,
                args.foot_align_mode,
                args.foot_align_delay_steps,
                args.foot_align_ramp_steps,
                args.sagittal_symmetry_blend,
                args.sagittal_symmetry_gain,
                args.sagittal_symmetry_integral_gain,
                args.sagittal_symmetry_integral_limit,
                args.sagittal_symmetry_max_action,
                args.sagittal_symmetry_mode,
                args.sagittal_symmetry_delay_steps,
                args.sagittal_symmetry_ramp_steps,
                args.sagittal_target_blend,
                args.sagittal_target_gain,
                args.sagittal_target_max_action,
                tuple(args.sagittal_target),
                args.sagittal_target_side,
                args.sagittal_target_delay_steps,
                args.sagittal_target_ramp_steps,
            )
            manifest.append({"posture": posture, "video": str(path), "exists": path.exists()})
            print("RECORDED", path, flush=True)

        (args.output / "video_manifest.json").write_text(
            json.dumps(
                {
                    "override": {
                        "blend": args.override_blend,
                        "gain": args.override_gain,
                        "balance_blend": args.balance_blend,
                        "balance_gain": args.balance_gain,
                        "delay_steps": args.override_delay_steps,
                        "ramp_steps": args.override_ramp_steps,
                        "foot_align_blend": args.foot_align_blend,
                        "foot_align_gain": args.foot_align_gain,
                        "foot_align_max_action": args.foot_align_max_action,
                        "foot_align_mode": args.foot_align_mode,
                        "foot_align_delay_steps": args.foot_align_delay_steps,
                        "foot_align_ramp_steps": args.foot_align_ramp_steps,
                        "sagittal_symmetry_blend": args.sagittal_symmetry_blend,
                        "sagittal_symmetry_gain": args.sagittal_symmetry_gain,
                        "sagittal_symmetry_integral_gain": args.sagittal_symmetry_integral_gain,
                        "sagittal_symmetry_integral_limit": args.sagittal_symmetry_integral_limit,
                        "sagittal_symmetry_max_action": args.sagittal_symmetry_max_action,
                        "sagittal_symmetry_mode": args.sagittal_symmetry_mode,
                        "sagittal_symmetry_delay_steps": args.sagittal_symmetry_delay_steps,
                        "sagittal_symmetry_ramp_steps": args.sagittal_symmetry_ramp_steps,
                        "sagittal_target_blend": args.sagittal_target_blend,
                        "sagittal_target_gain": args.sagittal_target_gain,
                        "sagittal_target_max_action": args.sagittal_target_max_action,
                        "sagittal_target": args.sagittal_target,
                        "sagittal_target_side": args.sagittal_target_side,
                        "sagittal_target_delay_steps": args.sagittal_target_delay_steps,
                        "sagittal_target_ramp_steps": args.sagittal_target_ramp_steps,
                    },
                    "videos": manifest,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("HOST29_VIDEO_COMPLETE", len(manifest), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
