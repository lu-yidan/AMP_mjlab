"""Collect motor-on near-standing failure states for A6 recovery training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from scripts.train_a6_host_v6 import action_scale_indices
from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime import (
  _a6_height,
  _a6_upright,
  a6_env_cfg,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.mdp.observations import UNACTUATED_STEPS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=20261014)
  parser.add_argument("--action-scale", type=float, default=0.6223878264)
  parser.add_argument("--last-controlled-step", type=int, default=420)
  parser.add_argument(
    "--capture-controlled-steps",
    type=int,
    nargs="+",
    default=(40, 60, 80, 100, 120, 140, 160),
  )
  parser.add_argument("--failure-height", type=float, default=0.70)
  parser.add_argument("--failure-upright", type=float, default=0.80)
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  if args.output.exists():
    raise FileExistsError(args.output)

  cfg = a6_env_cfg(
    play=True,
    seed=args.seed,
    dynamics=False,
    group="G+",
    bank_split="train",
  )
  cfg.scene.num_envs = args.num_envs
  cfg.terminations.pop("invalid_plate", None)
  for event in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
    cfg.events.pop(event, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.reset()
    env._a6_scene.zero_()
    env._a6_stratum.fill_(2)
    env._a6_source.zero_()
    env._a6_direction.copy_(
      torch.arange(args.num_envs, device=env.device) % 4
    )
    env._reset_idx(torch.arange(args.num_envs, device=env.device))
    set_action_scale(env, args.action_scale)

    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
    runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
    set_action_scale(env, args.action_scale)
    indices = action_scale_indices(env, runner.obs_normalizer._mean.shape[-1])
    runner.obs_normalizer._mean[:, indices] = args.action_scale
    policy = runner.get_inference_policy(device="cuda:0")
    observations = wrapper.get_observations()
    robot = env.scene["robot"]

    captures: list[dict[str, torch.Tensor]] = []
    capture_steps = set(args.capture_controlled_steps)
    recovery_at_end = None
    with torch.no_grad():
      for rollout_step in range(UNACTUATED_STEPS + args.last_controlled_step + 1):
        actions = policy(observations["actor"])
        observations, _, _, _ = wrapper.step(actions)
        controlled_step = rollout_step - UNACTUATED_STEPS + 1
        height = _a6_height(env)
        upright = _a6_upright(env)
        if controlled_step in capture_steps:
          failed = (height < args.failure_height) | (
            upright < args.failure_upright
          )
          finite = (
            torch.isfinite(robot.data.root_link_pos_w).all(-1)
            & torch.isfinite(robot.data.root_link_quat_w).all(-1)
            & torch.isfinite(robot.data.root_link_lin_vel_w).all(-1)
            & torch.isfinite(robot.data.root_link_ang_vel_w).all(-1)
            & torch.isfinite(robot.data.joint_pos).all(-1)
            & torch.isfinite(robot.data.joint_vel).all(-1)
          )
          ids = torch.nonzero(failed & finite, as_tuple=False).flatten()
          if len(ids):
            root_pos = (
              robot.data.root_link_pos_w[ids] - env.scene.env_origins[ids]
            )
            root_state = torch.cat(
              (
                root_pos,
                robot.data.root_link_quat_w[ids],
                robot.data.root_link_lin_vel_w[ids],
                robot.data.root_link_ang_vel_w[ids],
              ),
              dim=-1,
            )
            captures.append(
              {
                "root_state": root_state.cpu(),
                "joint_pos": robot.data.joint_pos[ids].cpu(),
                "joint_vel": robot.data.joint_vel[ids].cpu(),
                "direction": env._a6_direction[ids].cpu(),
                "controlled_step": torch.full(
                  (len(ids),), controlled_step, dtype=torch.long
                ),
                "height": height[ids].cpu(),
                "upright": upright[ids].cpu(),
                "teacher_action": actions[ids].cpu(),
                "env_id": ids.cpu(),
              }
            )
        if controlled_step == args.last_controlled_step:
          recovery_at_end = (
            (height >= args.failure_height)
            & (upright >= args.failure_upright)
          ).cpu()

    if not captures or recovery_at_end is None:
      raise RuntimeError("no finite early-failure states were collected")
    merged = {
      key: torch.cat([entry[key] for entry in captures], dim=0)
      for key in captures[0]
    }
    merged["eventually_recovered"] = recovery_at_end[merged["env_id"]]
    arrays = {key: value.numpy() for key, value in merged.items()}
    np.savez_compressed(args.output, **arrays)

    direction_counts = np.bincount(arrays["direction"], minlength=4).tolist()
    step_values, step_counts = np.unique(
      arrays["controlled_step"], return_counts=True
    )
    manifest = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "output": str(args.output.resolve()),
      "output_sha256": sha256(args.output),
      "seed": args.seed,
      "num_envs": args.num_envs,
      "action_scale": args.action_scale,
      "capture_controlled_steps": sorted(capture_steps),
      "last_controlled_step": args.last_controlled_step,
      "failure_height": args.failure_height,
      "failure_upright": args.failure_upright,
      "rows": int(len(arrays["height"])),
      "direction_counts": direction_counts,
      "step_counts": {
        str(step): int(count)
        for step, count in zip(step_values, step_counts, strict=True)
      },
      "eventually_recovered_rate": float(
        arrays["eventually_recovered"].mean()
      ),
      "height_range": [
        float(arrays["height"].min()),
        float(arrays["height"].max()),
      ],
      "upright_range": [
        float(arrays["upright"].min()),
        float(arrays["upright"].max()),
      ],
      "finite": bool(
        all(
          np.isfinite(value).all()
          for key, value in arrays.items()
          if value.dtype.kind == "f"
        )
      ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
      json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("A6_29_RECOVERY_BANK_COMPLETE", json.dumps(manifest), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
