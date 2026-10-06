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
from mjlab.utils.wrappers import VideoRecorder

from src.tasks.host_recovery.config.g1.rl_cfg import (
    unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.flat29 import flat29_env_cfg
from src.tasks.host_recovery.mdp.events import POSTURE_QUATS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


POSTURES = ("prone", "supine", "left_side", "right_side")


def record_posture(
    env: ManagerBasedRlEnv,
    checkpoint: Path,
    agent_cfg,
    output_dir: Path,
    posture: str,
    steps: int,
    device: str,
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
    obs = wrapper.get_observations()
    with torch.no_grad():
        for _ in range(steps):
            obs, _, _, _ = wrapper.step(policy(obs["actor"]))

    return output_dir / f"host29_{posture}-step-0.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
            )
            manifest.append({"posture": posture, "video": str(path), "exists": path.exists()})
            print("RECORDED", path, flush=True)

        (args.output / "video_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print("HOST29_VIDEO_COMPLETE", len(manifest), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
