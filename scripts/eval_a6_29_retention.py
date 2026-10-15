"""Evaluate uninterrupted near-standing retention for a 29-DoF A6 policy."""

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

from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime import _a6_height, _a6_upright, a6_env_cfg
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.mdp.observations import UNACTUATED_STEPS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--group", choices=("G-", "G+"), default="G-")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=1150)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--action-scale", type=float, required=True)
  parser.add_argument("--height-threshold", type=float, default=0.70)
  parser.add_argument("--upright-threshold", type=float, default=0.80)
  parser.add_argument("--required-step-fraction", type=float, default=0.80)
  args = parser.parse_args()

  if args.output.exists():
    raise FileExistsError(args.output)
  if args.num_envs < 32 or args.num_envs % 32:
    raise ValueError("num-envs must be >= 32 and divisible by 32")
  if args.steps <= UNACTUATED_STEPS:
    raise ValueError("steps must include a controlled interval")

  cfg = a6_env_cfg(
    play=False,
    seed=args.seed,
    dynamics=False,
    group=args.group,
  )
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = int(1e9)
  cfg.curriculum.pop("action_scale", None)
  cfg.curriculum.pop("pull_force", None)
  cfg.terminations.pop("invalid_plate", None)
  for event in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
    cfg.events.pop(event, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  try:
    env.reset()
    env._a6_scene[:] = 0
    env._a6_stratum[:] = 2
    env._a6_source[:] = 0
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    set_action_scale(env, args.action_scale)

    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
    runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
    set_action_scale(env, args.action_scale)

    action = env.action_manager.get_term("joint_pos")
    num_joints = env.scene["robot"].data.joint_pos.shape[-1]
    single_step_dim = 7 + 2 * num_joints + action.action_dim
    scale_indices = torch.arange(
      single_step_dim - 1,
      runner.obs_normalizer._mean.shape[-1],
      single_step_dim,
      device=runner.obs_normalizer._mean.device,
    )
    runner.obs_normalizer._mean[:, scale_indices] = args.action_scale

    policy = runner.get_inference_policy(device="cuda:0")
    observations = wrapper.get_observations()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    good_steps = torch.zeros(env.num_envs, device=env.device)
    controlled_steps = 0
    final_retained = torch.zeros_like(alive)
    trajectory = {}
    capture_steps = {
      UNACTUATED_STEPS,
      50,
      100,
      200,
      400,
      800,
      args.steps - 1,
    }

    with torch.inference_mode():
      for step in range(args.steps):
        actions = policy(observations["actor"])
        observations, _, dones, _ = wrapper.step(actions)
        alive &= ~dones.bool()
        height = _a6_height(env)
        upright = _a6_upright(env)
        retained = (
          alive
          & (height > args.height_threshold)
          & (upright > args.upright_threshold)
        )
        if step + 1 > UNACTUATED_STEPS:
          good_steps += retained.float()
          controlled_steps += 1
          final_retained = retained
        if step in capture_steps:
          trajectory[str(step)] = {
            "alive_rate": float(alive.float().mean()),
            "retained_rate": float(retained.float().mean()),
            "head_height_mean": float(height.mean()),
            "upright_mean": float(upright.mean()),
          }

    retained_step_fraction = good_steps / controlled_steps
    cohort = (
      alive
      & final_retained
      & (retained_step_fraction >= args.required_step_fraction)
    )
    result = {
      "checkpoint": str(args.checkpoint.resolve()),
      "group": args.group,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "seed": args.seed,
      "action_scale_requested": args.action_scale,
      "action_scale": float(env._host_action_rescale.mean()),
      "unactuated_steps": UNACTUATED_STEPS,
      "height_threshold": args.height_threshold,
      "upright_threshold": args.upright_threshold,
      "required_step_fraction": args.required_step_fraction,
      "alive_rate": float(alive.float().mean()),
      "retained_cohort_success": f"{int(cohort.sum())}/{env.num_envs}",
      "retained_cohort_rate": float(cohort.float().mean()),
      "retained_step_fraction_mean": float(retained_step_fraction.mean()),
      "retained_step_fraction_min": float(retained_step_fraction.amin()),
      "final_retained_rate": float(final_retained.float().mean()),
      "trajectory": trajectory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print("A6_29_RETENTION_EVAL_COMPLETE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
