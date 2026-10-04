"""Record one named A6 recovery scene from a checkpoint without a viewer.

The current 30/50/20 reset allocation gives four environments in this order:
flat/supine, guided/supine, guided/prone, free/supine.  The ordinary mjlab
``play --video`` records environment 0 only; selecting the renderer's primary
environment here makes each saved video genuinely correspond to its label.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from g1recovery_amp.tasks.recovery import LOW_TORQUE_CAP_ONLY_90_TASK_ID
from g1recovery_amp.tasks.recovery.mdp.a6_resets import (
    balanced_three_scene_curriculum_groups,
)


SCENES = {
    "flat": (0, 0, 0),
    "guided_supine": (1, 1, 0),
    "guided_prone": (2, 1, 1),
    "free": (3, 2, 0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500, help="500 steps = 10 s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--shadows", action="store_true", help="off by default")
    args = parser.parse_args()

    checkpoint = args.checkpoint_file.resolve(strict=True)
    if args.steps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("steps, width and height must be positive")

    prefix = f"{checkpoint.stem}_{args.scene}_seed{args.seed}"
    output_dir = args.output_dir.resolve()
    output_path = output_dir / f"{prefix}-step-0.mp4"
    if output_path.exists():
        raise FileExistsError(f"video already exists; choose another directory: {output_path}")

    configure_torch_backends()
    cfg = load_env_cfg(LOW_TORQUE_CAP_ONLY_90_TASK_ID, play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = 4
    cfg.viewer.width = args.width
    cfg.viewer.height = args.height
    cfg.viewer.enable_shadows = args.shadows
    cfg.viewer.max_extra_envs = 0

    env_idx, expected_scene, expected_direction = SCENES[args.scene]
    cfg.viewer.env_idx = env_idx
    reset_cfg = cfg.events["a6_three_scene_curriculum_reset"]
    scene, _, direction, _ = balanced_three_scene_curriculum_groups(
        cfg.scene.num_envs, tuple(reset_cfg.params["scene_weights"])
    )
    if int(scene[env_idx]) != expected_scene or int(direction[env_idx]) != expected_direction:
        raise RuntimeError("scene allocation changed; refusing to mislabel the video")

    agent_cfg = load_rl_cfg(LOW_TORQUE_CAP_ONLY_90_TASK_ID)
    runner_cls = load_runner_cls(LOW_TORQUE_CAP_ONLY_90_TASK_ID)
    if runner_cls is None:
        raise RuntimeError("the recovery task has no registered runner")

    raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
    recorder = VideoRecorder(
        raw_env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix=prefix,
    )
    env = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)
    try:
        runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        observations = env.get_observations()
        print(
            f"Recording {args.scene}: env={env_idx}, seed={args.seed}, "
            f"duration={args.steps * raw_env.step_dt:.2f} s, output={output_path}"
        )
        with torch.inference_mode():
            for _ in range(args.steps):
                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    raise FloatingPointError("policy action contains NaN/Inf")
                observations, _, _, _ = env.step(actions)
    finally:
        env.close()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"video was not saved: {output_path}")
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1048576:.1f} MiB)")


if __name__ == "__main__":
    main()
