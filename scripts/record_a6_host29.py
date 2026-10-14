"""Record the four HoST A6 playback scenes as MP4s and a labeled contact sheet."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import OffscreenRenderer
from PIL import Image, ImageDraw, ImageFont

from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.phase_switch import reset_prone_history_after_escape
from g1recovery_host.ppo import HostPolicy


LAYOUT = ((2, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))
LABELS = ("Free / supine", "Guided / prone", "Guided / supine", "Flat / supine")
SLUGS = ("free_supine", "guided_prone", "guided_supine", "flat_supine")


def make_environment(checkpoint: dict, seed: int, device: str) -> ManagerBasedRlEnv:
    variant = checkpoint["reward_variant"]
    if (
        checkpoint.get("task") != "a6"
        or checkpoint.get("model_source") != "unitree_g1_29"
        or "_staged_escape_v5" not in variant
        or "_v8_edge_curriculum_v1" not in variant
    ):
        raise ValueError("recording requires a staged-v5, 29-DoF HoST A6 checkpoint")
    cfg = host_a6_env_cfg(
        4,
        seed,
        arm=checkpoint["arm"],
        edge_curriculum=True,
        plate_observation=True,
        safe_actions="_safe_actions_v1" in variant,
        prone_full_arms="_prone_full_arms_v1" in variant,
        supine_full_arms="_supine_full_arms_v1" in variant,
        aligned_edge="_aligned_edge_v1" in variant,
        lateral_actions="_lateral_actions_v1" in variant,
        lift_guard="_lift_guard_v1" in variant,
        footprint_edge="_footprint_edge_v1" in variant,
        footprint_overlap_m=checkpoint.get("a6_footprint_overlap_m", 0.25),
    )
    cfg.events["reset_scene"].params["layout"] = LAYOUT
    cfg.episode_length_s = 15.0
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=device)
    env._a6_staged_escape_v5 = True
    env._a6_knee_straight_prior_v1 = "_knee_straight_prior_v1" in variant
    env._a6_arm_extraction_v1 = "_arm_extraction_v1" in variant
    env._a6_prone_recovery_v1 = "_prone_recovery_v1" in variant
    env._a6_bidirectional_recovery_v1 = "_bidirectional_recovery_v1" in variant
    env._a6_postescape_live_obs_v1 = "_postescape_live_obs_v1" in variant
    env._a6_edge_offset_m = 0.0
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.steps > 750 or args.render_stride <= 0:
        parser.error("steps must be in [1, 750] and render stride must be positive")
    if args.width % 16 or args.height % 16 or args.width < 320 or args.height < 240:
        parser.error("width/height must be multiples of 16 and at least 320x240")
    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required when recording")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    variant = checkpoint["reward_variant"]
    policy = HostPolicy(
        obs_dim=612,
        action_dim=29,
        log_std_bounds=(-2.3, math.log(checkpoint.get("action_std_max", 0.7))),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()

    prefix = f"HoST_model_{checkpoint['iteration']:05d}_seed{args.seed}"
    paths: list[Path] = []
    if not args.dry_run:
        assert args.output_dir is not None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = [args.output_dir / f"{prefix}_{slug}.mp4" for slug in SLUGS]
        paths += [args.output_dir / f"{prefix}_four_scenes.mp4"]
        paths += [args.output_dir / f"{prefix}_poster.png"]
        paths += [args.output_dir / f"{prefix}_metrics.json"]
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing output: {existing}")

    renderers: list[OffscreenRenderer] = []
    writers = []
    env = make_environment(checkpoint, args.seed, args.device)
    camera_centers: list[np.ndarray] = []
    camera_origins: list[np.ndarray] = []
    try:
        obs, _ = env.reset()
        actual = torch.stack(
            (env._a6_scene, env._a6_direction, env._a6_stratum), dim=1
        )
        expected = torch.tensor(LAYOUT, device=args.device)
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Scene layout changed: {actual.tolist()}")
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(checkpoint["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        term.guided_escape_assist_force.zero_()
        term.guided_escape_assist_speed_target.zero_()

        if not args.dry_run:
            for index in range(4):
                view = copy.deepcopy(env.cfg.viewer)
                view.env_idx = index
                view.max_extra_envs = 0
                view.width = args.width
                view.height = args.height
                view.distance = 2.8
                view.azimuth = 120.0
                view.elevation = -18.0
                origin = env.scene.env_origins[index].cpu().numpy()
                view.lookat = tuple((origin + np.array((0.0, 0.0, 0.6))).tolist())
                view.enable_shadows = False
                renderer = OffscreenRenderer(
                    env.sim.mj_model,
                    view,
                    env.scene,
                    sim_model=env.sim.model,
                    expanded_fields=env.sim.expanded_fields,
                )
                renderer.initialize()
                renderers.append(renderer)
                camera_origins.append(origin)
                camera_centers.append(origin + np.array((0.0, 0.0, 0.6)))
            for path in paths[:5]:
                size = (
                    (args.width * 2, args.height * 2)
                    if path == paths[4]
                    else (args.width, args.height)
                )
                writer = imageio_ffmpeg.write_frames(
                    str(path),
                    size,
                    fps=50 / args.render_stride,
                    quality=7,
                    codec="libx264",
                    output_params=["-movflags", "+faststart"],
                )
                writer.send(None)
                writers.append(writer)

        first_escape = [0] * 4
        first_invalid = [0] * 4
        ended_early = [False] * 4
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
        recorded = 0
        executed = 0
        for step in range(1, args.steps + 1):
            with torch.inference_mode():
                action = policy.deterministic_action(obs["actor"])
                if not torch.isfinite(action).all():
                    raise FloatingPointError("Non-finite policy action")
                escaped_before = env._a6_escaped.clone()
                obs, _, terminated, truncated, _ = env.step(action)
                if "_phase_history_reset_v1" in variant:
                    obs, _ = reset_prone_history_after_escape(
                        env, escaped_before, terminated | truncated
                    )
            executed = step
            for index in range(4):
                if bool(env._a6_escaped[index]) and not first_escape[index]:
                    first_escape[index] = step
                if bool(env._a6_invalid[index]) and not first_invalid[index]:
                    first_invalid[index] = step
                if (bool(terminated[index]) or bool(truncated[index])) and step < args.steps:
                    ended_early[index] = True

            if renderers and step % args.render_stride == 0:
                frames = []
                roots = env.scene["robot"].data.root_link_pos_w.detach().cpu().numpy()
                for index, (renderer, writer) in enumerate(zip(renderers, writers[:4], strict=True)):
                    target = camera_origins[index].copy()
                    target[:2] = 0.8 * roots[index, :2] + 0.2 * camera_origins[index][:2]
                    target[2] += 0.6
                    camera_centers[index] += 0.12 * (target - camera_centers[index])
                    renderer._cam.lookat[:] = camera_centers[index]
                    renderer.update(env.sim.data)
                    frame = np.ascontiguousarray(renderer.render())
                    writer.send(frame)
                    frames.append(frame)
                panel = Image.new("RGB", (args.width * 2, args.height * 2))
                for index, frame in enumerate(frames):
                    tile = Image.fromarray(frame)
                    draw = ImageDraw.Draw(tile)
                    draw.rectangle((0, 0, args.width, 34), fill=(20, 25, 30))
                    draw.text((12, 5), f"{LABELS[index]}   t={step / 50:.1f}s", font=font, fill="white")
                    panel.paste(tile, ((index % 2) * args.width, (index // 2) * args.height))
                writers[4].send(np.ascontiguousarray(np.asarray(panel)))
                if recorded == 0:
                    panel.save(paths[5])
                recorded += 1
            if any(ended_early):
                break
            if step % 100 == 0:
                print(f"seed={args.seed} step={step}/{args.steps}", flush=True)

        robot = env.scene["robot"].data
        heights = robot.root_link_pos_w[:, 2].detach().cpu().tolist()
        upright = (-robot.projected_gravity_b[:, 2]).detach().cpu().tolist()
        scenes = {}
        for index, slug in enumerate(SLUGS):
            stood = heights[index] > 0.6 and upright[index] > 0.8
            scenes[slug] = {
                "escaped": first_escape[index] > 0,
                "first_escape_step": first_escape[index],
                "invalid": first_invalid[index] > 0,
                "first_invalid_step": first_invalid[index],
                "ended_early": ended_early[index],
                "stood_at_end": stood,
                "success": stood and not ended_early[index]
                and (index == 3 or first_escape[index] > 0),
                "final_pelvis_height_m": round(heights[index], 4),
                "final_upright": round(upright[index], 4),
            }
        result = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checksum,
            "seed": args.seed,
            "layout": list(SLUGS),
            "control_steps": executed,
            "video_frames": recorded,
            "fps": 50 / args.render_stride,
            "guided_offset_m": 0.0,
            "assist_force_n": 0.0,
            "camera": "smooth follow: 80% robot root, 20% scene origin",
            "all_success": all(scene["success"] for scene in scenes.values()),
            "scenes": scenes,
        }
        if paths:
            paths[6].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        for writer in writers:
            writer.close()
        for renderer in renderers:
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()
