"""Record 23-DoF A6 rollout videos for the final checkpoint.

Each video forces every environment into one scene/direction/source cell, then
records environment 0 with :class:`VideoRecorder`. The board scenes keep their
historical play-mode settings, so a single rollout is a full 23 s (1150 control
steps at 0.02 s) without the 10 s training timeout.
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

from src.tasks.host_recovery.a6_runtime_23 import (
  SCENE_NAMES,
  a6_env_cfg,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.mdp.a6_geometry import DIRECTIONS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def force_cell(
  env: ManagerBasedRlEnv,
  scene_id: int,
  direction_id: int,
  source_id: int,
) -> None:
  """Route every environment to one bank cell for the next reset."""
  env._a6_scene.fill_(scene_id)
  env._a6_direction.fill_(direction_id)
  env._a6_stratum.fill_(0)
  env._a6_source.fill_(source_id)


def record_cell(
  env: ManagerBasedRlEnv,
  checkpoint: Path,
  agent_cfg,
  output_dir: Path,
  scene_id: int,
  direction_id: int,
  source_id: int,
  steps: int,
  device: str,
) -> Path:
  scene_name = SCENE_NAMES[scene_id]
  direction_name = DIRECTIONS[direction_id]
  source_name = "procedural" if source_id else "natural"
  prefix = f"a6_23_{scene_name}_{direction_name}_{source_name}"

  force_cell(env, scene_id, direction_id, source_id)

  recorder = VideoRecorder(
    env,
    video_folder=output_dir,
    step_trigger=lambda step: step == 0,
    video_length=steps,
    name_prefix=prefix,
    disable_logger=True,
  )
  wrapper = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)
  runner = HoSTOnPolicyRunner(wrapper, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_optimizer=False, map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs = wrapper.get_observations()
  with torch.no_grad():
    for _ in range(steps):
      obs, _, _, _ = wrapper.step(policy(obs["actor"]))
  return output_dir / f"{prefix}-step-0.mp4"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--group", choices=("G-", "G+"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=1150)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  if args.num_envs < 32 or args.num_envs % 32:
    raise ValueError("--num-envs must be >= 32 and divisible by 32")

  cfg = a6_env_cfg(
    play=True,
    seed=args.seed,
    dynamics=False,
    group=args.group,
    bank_split="train",
  )
  cfg.scene.num_envs = args.num_envs
  # A6 keeps every environment origin at the world origin (env_spacing=0), so the
  # offscreen renderer would otherwise draw neighboring environments stacked on top
  # of environment 0. Disable context rendering for a clean single-robot video.
  cfg.viewer.max_extra_envs = 0
  cfg.terminations.pop("invalid_plate", None)
  for event in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
    cfg.events.pop(event, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  agent_cfg = unitree_g1_host_standup_ppo_runner_cfg()
  device = "cuda:0"
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  env.reset()

  cells = []
  for scene_id in range(len(SCENE_NAMES)):
    for direction_id in range(len(DIRECTIONS)):
      cells.append((scene_id, direction_id, 0))
  # Additional documented failure cells for the vertical board.
  cells.append((1, 0, 1))
  cells.append((1, 1, 1))

  args.output.mkdir(parents=True, exist_ok=True)
  manifest = []
  for scene_id, direction_id, source_id in cells:
    path = record_cell(
      env,
      args.checkpoint,
      agent_cfg,
      args.output,
      scene_id,
      direction_id,
      source_id,
      args.steps,
      device,
    )
    manifest.append(
      {
        "scene": SCENE_NAMES[scene_id],
        "direction": DIRECTIONS[direction_id],
        "source": "procedural" if source_id else "natural",
        "video": str(path),
        "exists": path.exists(),
      }
    )
    print("RECORDED", path, flush=True)

  (args.output / "video_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
  )
  env.close()
  print("A6_23_VIDEO_COMPLETE", len(manifest), flush=True)


if __name__ == "__main__":
  main()
