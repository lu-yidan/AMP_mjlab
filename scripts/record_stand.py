"""Headless video: prone -> stand up -> stable standing, zero pull force."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401


def main():
    task = "Unitree-G1-HoST-StandUp"
    ckpt = os.environ["CKPT"]
    action_scale = float(os.environ.get("ACTION_SCALE", "0.30"))
    frames = int(os.environ.get("FRAMES", "800"))
    out = Path(os.environ.get("VIDEO_DIR", "logs/rsl_rl/g1_host_standup/2026-09-24_01-18-23/videos/play"))

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", "64"))
    env_cfg.actions["joint_pos"].scale = action_scale
    agent_cfg = load_rl_cfg(task)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = VideoRecorder(
        env,
        video_folder=out,
        step_trigger=lambda step: step == 0,
        video_length=frames,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(ckpt, load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    with torch.no_grad():
        for _ in range(frames):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    env.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
