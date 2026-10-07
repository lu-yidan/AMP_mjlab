"""Isolate why the 23-DoF HoST actor loses flat-ground standing under A6.

Runs a deterministic (mean-policy) rollout over the A6 environment and reports
head height / uprightness / flat primary-success, with the training dynamics
randomisation and the interval push toggleable independently.
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

from src.tasks.host_recovery.a6_runtime_23 import (
  _a6_ground_force,
  _a6_height,
  _a6_upright,
  a6_env_cfg,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def primary(env: ManagerBasedRlEnv) -> torch.Tensor:
  height = _a6_height(env)
  upright = _a6_upright(env)
  load = _a6_ground_force(env, "quality_feet")[..., 2].abs().amin(-1)
  return (height >= 1.15) & (upright >= 0.90) & (load > 20.0)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--group", choices=("G-", "G+"), default="G-")
  p.add_argument("--num-envs", type=int, default=128)
  p.add_argument("--steps", type=int, default=1150)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--dynamics", action="store_true")
  p.add_argument("--push", action="store_true")
  p.add_argument("--flat-only", action="store_true")
  p.add_argument("--episode-length-s", type=float, default=None)
  a = p.parse_args()

  cfg = a6_env_cfg(play=False, seed=a.seed, dynamics=a.dynamics, group=a.group)
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  if a.episode_length_s is not None:
    cfg.episode_length_s = a.episode_length_s
  cfg.terminations.pop("invalid_plate", None)
  if not a.push:
    cfg.events.pop("push_robot", None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  env.reset()
  if a.flat_only:
    env._a6_scene[:] = 0
    env._a6_stratum[:] = 0
    env._a6_source[:] = 0
    env._reset_idx(torch.arange(env.num_envs, device=env.device))

  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
  runner.load(str(a.checkpoint), load_optimizer=False, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  obs = wrapper.get_observations()

  hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  rows = {}
  with torch.no_grad():
    for step in range(a.steps):
      obs, _, _, _ = wrapper.step(policy(obs["actor"]))
      if step % 100 == 0 or step == a.steps - 1:
        height = _a6_height(env)
        upright = _a6_upright(env)
        rows[step] = {
          "head_height_mean": float(height.mean()),
          "head_height_min": float(height.amin()),
          "head_height_max": float(height.amax()),
          "upright_mean": float(upright.mean()),
        }
      now = primary(env)
      hold = torch.where(now, hold + 1, torch.zeros_like(hold))
      success |= hold >= 150

  flat = env._a6_scene == 0
  total_flat = int(flat.sum())
  flat_success = int(success[flat].sum()) if total_flat else 0
  print(
    json.dumps(
      {
        "checkpoint": str(a.checkpoint),
        "dynamics": a.dynamics,
        "push": a.push,
        "flat_only": a.flat_only,
        "episode_length_s": cfg.episode_length_s,
        "num_envs": a.num_envs,
        "steps": a.steps,
        "flat_n": total_flat,
        "flat_primary_success": (
          f"{flat_success}/{total_flat}" if total_flat else "n/a"
        ),
        "flat_primary_rate": (
          flat_success / total_flat if total_flat else None
        ),
        "all_primary_rate": float(success.float().mean()),
        "head_height_trajectory": rows,
      },
      indent=2,
    ),
    flush=True,
  )
  env.close()


if __name__ == "__main__":
  main()
