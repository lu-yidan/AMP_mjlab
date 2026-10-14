"""Record flat near-standing A6 rollouts at a checkpoint's training scale."""

from __future__ import annotations

import argparse
import hashlib
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

from scripts.train_a6_host_v6 import action_scale_indices
from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime import a6_env_cfg
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.mdp.a6_geometry import DIRECTIONS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def record_direction(
  checkpoint: Path,
  output_dir: Path,
  direction_id: int,
  action_scale: float,
  num_envs: int,
  steps: int,
  seed: int,
) -> Path:
  cfg = a6_env_cfg(
    play=True,
    seed=seed,
    dynamics=False,
    group="G+",
    bank_split="train",
  )
  cfg.scene.num_envs = num_envs
  cfg.viewer.max_extra_envs = 0
  cfg.terminations.pop("invalid_plate", None)
  for event in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
    cfg.events.pop(event, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  direction = DIRECTIONS[direction_id]
  prefix = f"a6_29_flat_{direction}_near_scale_{action_scale:.6f}"
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  try:
    env.reset()
    env._a6_scene.zero_()
    env._a6_direction.fill_(direction_id)
    env._a6_stratum.fill_(2)
    env._a6_source.zero_()
    env._reset_idx(torch.arange(num_envs, device=env.device))
    set_action_scale(env, action_scale)

    recorder = VideoRecorder(
      env,
      video_folder=output_dir,
      step_trigger=lambda step: step == 0,
      video_length=steps,
      name_prefix=prefix,
      disable_logger=True,
    )
    wrapper = RslRlVecEnvWrapper(recorder, clip_actions=agent.clip_actions)
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
    runner.load(str(checkpoint), load_optimizer=False, map_location="cuda:0")
    set_action_scale(env, action_scale)
    indices = action_scale_indices(env, runner.obs_normalizer._mean.shape[-1])
    runner.obs_normalizer._mean[:, indices] = action_scale
    policy = runner.get_inference_policy(device="cuda:0")

    observations = wrapper.get_observations()
    with torch.no_grad():
      for _ in range(steps):
        observations, _, _, _ = wrapper.step(policy(observations["actor"]))
    return output_dir / f"{prefix}-step-0.mp4"
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--action-scale", type=float, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=1150)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if args.num_envs < 32 or args.num_envs % 32:
    raise ValueError("--num-envs must be >= 32 and divisible by 32")

  args.output.mkdir(parents=True, exist_ok=False)
  manifest = []
  for direction_id, direction in enumerate(DIRECTIONS):
    video = record_direction(
      args.checkpoint,
      args.output,
      direction_id,
      args.action_scale,
      args.num_envs,
      args.steps,
      args.seed,
    )
    manifest.append(
      {
        "scene": "flat",
        "direction": direction,
        "stratum": "near",
        "source": "natural",
        "action_scale": args.action_scale,
        "video": str(video),
        "exists": video.exists(),
      }
    )
    print("RECORDED", video, flush=True)

  payload = {
    "checkpoint": str(args.checkpoint),
    "checkpoint_sha256": sha256(args.checkpoint),
    "stratum": "near",
    "action_scale": args.action_scale,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "seed": args.seed,
    "videos": manifest,
  }
  (args.output / "video_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
  )
  print("A6_29_NEAR_VIDEO_COMPLETE", len(manifest), flush=True)


if __name__ == "__main__":
  main()
