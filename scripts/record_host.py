"""Render one deterministic 23-DoF HoST get-up episode to MP4."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host import host_ground_env_cfg
from g1recovery_host.action import PHYSICS_REVISION
from g1recovery_host.ppo import HostPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-source", default="host_urdf_capsule")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--action-beta", type=float, default=0.25)
    parser.add_argument("--assist-force-n", type=float, default=0.0)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.render_stride <= 0:
        parser.error("steps and render stride must be positive")
    if not 0.0 < args.action_beta <= 1.0 or args.assist_force_n < 0.0:
        parser.error("action beta must be in (0, 1] and assist force nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    if checkpoint.get("model_source", "mjlab") != args.model_source:
        parser.error("checkpoint and requested model source differ")
    if checkpoint.get("physics_revision", "single_torso_force_v1") != PHYSICS_REVISION:
        parser.error("checkpoint and current physics revision differ")
    cfg = host_ground_env_cfg(
        1,
        args.seed,
        model_source=args.model_source,
        legacy_style=checkpoint.get("reward_variant", "paper")
        in ("paper", "height_guard_v1"),
    )
    cfg.episode_length_s = args.steps * 0.02
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=args.device)
    output = Path(args.output)
    try:
        obs, _ = env.reset()
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(args.action_beta)
        term.assist_force.fill_(args.assist_force_n)
        policy = HostPolicy().to(args.device)
        policy.load_state_dict(checkpoint["policy"])
        policy.eval()
        model = env.sim.mj_model
        data = env.sim.mj_data
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        initial_pos = env.sim.data.qpos[0, :3].cpu().numpy()
        camera.lookat[:] = (initial_pos[0], initial_pos[1], 0.6)
        camera.distance = 2.4
        camera.azimuth = 120
        camera.elevation = -18
        frames = []

        def render_frame(renderer):
            data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())

        heights = []
        ever_stood = False
        fell_after = False
        streak = 0
        longest_streak = 0
        with mujoco.Renderer(model, width=640, height=480) as renderer:
            render_frame(renderer)
            for step in range(args.steps):
                with torch.no_grad():
                    action = policy.deterministic_action(obs["actor"])
                obs, _, _, _, _ = env.step(action)
                height = float(env.scene["robot"].data.root_link_pos_w[0, 2])
                heights.append(height)
                streak = streak + 1 if height > 0.7 else 0
                longest_streak = max(longest_streak, streak)
                ever_stood |= height > 0.7
                fell_after |= ever_stood and height < 0.5
                if (step + 1) % args.render_stride == 0:
                    render_frame(renderer)
        success = ever_stood and not fell_after
        result = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "seed": args.seed,
            "model_source": args.model_source,
            "action_beta": args.action_beta,
            "assistance_force_n": args.assist_force_n,
            "steps": args.steps,
            "ever_stood": ever_stood,
            "fell_after_standing": fell_after,
            "paper_height_criterion_success": success,
            "max_pelvis_height_m": max(heights),
            "final_pelvis_height_m": heights[-1],
            "longest_continuous_above_0_7m_s": longest_streak * 0.02,
            "last_second_above_0_7m": min(heights[-50:]) > 0.7,
            "video_saved": success or not args.require_success,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        if result["video_saved"]:
            imageio.mimsave(
                output,
                frames,
                fps=50 / args.render_stride,
                quality=8,
                macro_block_size=16,
            )
        output.with_suffix(".json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
