"""Measure flat-ground skill retention for the 29-DoF V4 A6 transfer."""

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

from src.tasks.host_recovery.a6_runtime import (
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
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--group", choices=("G-", "G+"), default="G-")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=1150)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--dynamics", action="store_true")
  parser.add_argument("--push", action="store_true")
  args = parser.parse_args()

  cfg = a6_env_cfg(
    play=False,
    seed=args.seed,
    dynamics=args.dynamics,
    group=args.group,
  )
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.terminations.pop("invalid_plate", None)
  if not args.push:
    cfg.events.pop("push_robot", None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  env.reset()
  env._a6_scene[:] = 0
  env._a6_stratum[:] = 0
  env._a6_source[:] = 0
  env._reset_idx(torch.arange(env.num_envs, device=env.device))

  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
  runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  observations = wrapper.get_observations()
  robot = env.scene["robot"]
  torso_id = robot.find_bodies("torso_link")[0][0]
  foot_ids = torch.tensor(
    robot.find_sites(("left_foot", "right_foot"))[0], device=env.device
  )

  hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  host_hold = torch.zeros_like(hold)
  host_success = torch.zeros_like(success)
  host_final = torch.zeros_like(success)
  trajectory = {}
  with torch.no_grad():
    for step in range(args.steps):
      observations, _, _, _ = wrapper.step(policy(observations["actor"]))
      if step % 100 == 0 or step == args.steps - 1:
        height = _a6_height(env)
        upright = _a6_upright(env)
        trajectory[step] = {
          "head_height_mean": float(height.mean()),
          "head_height_min": float(height.amin()),
          "head_height_max": float(height.amax()),
          "upright_mean": float(upright.mean()),
        }
      now = primary(env)
      hold = torch.where(now, hold + 1, torch.zeros_like(hold))
      success |= hold >= 150
      torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
      feet_z = robot.data.site_pos_w[:, foot_ids, 2].mean(dim=1)
      host_upright = robot.data.projected_gravity_b[:, 2] < -0.8
      host_final = ((torso_z - feet_z) > 0.6) & host_upright
      host_hold = torch.where(
        host_final, host_hold + 1, torch.zeros_like(host_hold)
      )
      host_success |= host_hold >= 150

  result = {
    "checkpoint": str(args.checkpoint),
    "group": args.group,
    "dynamics": args.dynamics,
    "push": args.push,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "flat_primary_success": f"{int(success.sum())}/{env.num_envs}",
    "flat_primary_rate": float(success.float().mean()),
    "host_held_success": f"{int(host_success.sum())}/{env.num_envs}",
    "host_held_rate": float(host_success.float().mean()),
    "host_final_standing_rate": float(host_final.float().mean()),
    "head_height_trajectory": trajectory,
  }
  print(json.dumps(result, indent=2), flush=True)
  env.close()


if __name__ == "__main__":
  main()
