"""Evaluate a recovery policy followed by a post-standing policy."""

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
from mjlab.utils.lab_api.math import quat_apply

from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.flat29 import flat29_env_cfg
from src.tasks.host_recovery.mdp.events import POSTURE_QUATS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


POSTURES = ("prone", "supine", "left_side", "right_side")
NUM_HISTORY = 6
A6_HEAD_OFFSET = 0.43


def checkpoint_scale(path: Path) -> float:
  saved = torch.load(path, map_location="cpu", weights_only=False)
  return float(saved["infos"]["env_state"]["host_action_rescale"].float().mean())


def policy_observation(observation: torch.Tensor, action_scale: float) -> torch.Tensor:
  adjusted = observation.clone()
  if adjusted.shape[1] % NUM_HISTORY:
    raise RuntimeError(f"unexpected actor observation width: {adjusted.shape[1]}")
  one_step_obs = adjusted.shape[1] // NUM_HISTORY
  adjusted[:, one_step_obs - 1 :: one_step_obs] = action_scale
  return adjusted


def fraction(value: torch.Tensor) -> float:
  return float(value.float().mean())


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--recovery-checkpoint", type=Path, required=True)
  parser.add_argument("--standing-checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=1150)
  parser.add_argument("--switch-step", type=int, default=250)
  parser.add_argument("--hold-steps", type=int, default=50)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  cfg = flat29_env_cfg(play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  try:
    recovery_runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
    recovery_runner.load(
      str(args.recovery_checkpoint), load_optimizer=False, map_location="cuda:0"
    )
    recovery_policy = recovery_runner.get_inference_policy(device="cuda:0")
    recovery_scale = checkpoint_scale(args.recovery_checkpoint)

    standing_runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
    standing_runner.load(
      str(args.standing_checkpoint), load_optimizer=False, map_location="cuda:0"
    )
    standing_policy = standing_runner.get_inference_policy(device="cuda:0")
    standing_scale = checkpoint_scale(args.standing_checkpoint)

    robot = env.scene["robot"]
    torso_id = robot.find_bodies("torso_link")[0][0]
    foot_ids = torch.tensor(
      robot.find_sites(("left_foot", "right_foot"))[0], device=env.device
    )
    reset_cfg = env.event_manager.get_term_cfg("reset_base")
    results = {}
    for posture in POSTURES:
      set_action_scale(env, recovery_scale)
      reset_cfg.params["posture"] = posture
      wrapper.reset()
      observations = wrapper.get_observations()
      valid = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
      standing_held = torch.zeros_like(valid)
      standing_consecutive = torch.zeros(
        args.num_envs, dtype=torch.long, device=env.device
      )
      formal_ever = torch.zeros_like(valid)
      formal_held = torch.zeros_like(valid)
      formal_consecutive = torch.zeros_like(standing_consecutive)
      final_standing = torch.zeros_like(valid)
      final_formal = torch.zeros_like(valid)
      final_head = torch.zeros(args.num_envs, device=env.device)

      with torch.no_grad():
        for step in range(args.steps):
          if step == args.switch_step:
            set_action_scale(env, standing_scale)
          if step < args.switch_step:
            actor_observation = policy_observation(
              observations["actor"], recovery_scale
            )
            actions = recovery_policy(actor_observation)
          else:
            actor_observation = policy_observation(
              observations["actor"], standing_scale
            )
            actions = standing_policy(actor_observation)
          observations, _, terminated, _ = wrapper.step(actions)
          valid &= ~terminated.bool()

          feet_z = robot.data.site_pos_w[:, foot_ids, 2]
          relative_height = robot.data.body_link_pos_w[:, torso_id, 2] - feet_z.mean(dim=1)
          upright = -robot.data.projected_gravity_b[:, 2]
          standing = (relative_height > 0.6) & (upright > 0.8) & valid
          standing_consecutive = torch.where(
            standing, standing_consecutive + 1, torch.zeros_like(standing_consecutive)
          )
          standing_held |= standing_consecutive >= args.hold_steps

          local = torch.zeros((args.num_envs, 3), device=env.device)
          local[:, 2] = A6_HEAD_OFFSET
          head = (
            robot.data.body_link_pos_w[:, torso_id, 2]
            + quat_apply(robot.data.body_link_quat_w[:, torso_id], local)[:, 2]
            - env.scene.env_origins[:, 2]
          )
          formal = (head >= 1.15) & (upright >= 0.90) & valid
          formal_ever |= formal
          formal_consecutive = torch.where(
            formal, formal_consecutive + 1, torch.zeros_like(formal_consecutive)
          )
          formal_held |= formal_consecutive >= args.hold_steps
          if step == args.steps - 1:
            final_standing = standing
            final_formal = formal
            final_head = head

      results[posture] = {
        "valid_fraction": fraction(valid),
        "held_fraction": fraction(standing_held),
        "final_standing_fraction": fraction(final_standing),
        "a6_formal_ever_fraction": fraction(formal_ever),
        "a6_formal_held_fraction": fraction(formal_held),
        "a6_formal_final_fraction": fraction(final_formal),
        "a6_head_height_final_mean_m": float(final_head.mean()),
        "a6_head_height_final_min_m": float(final_head.amin()),
      }
      print(posture, json.dumps(results[posture], sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
      json.dumps(
        {
          "recovery_checkpoint": str(args.recovery_checkpoint),
          "standing_checkpoint": str(args.standing_checkpoint),
          "recovery_scale": recovery_scale,
          "standing_scale": standing_scale,
          "switch_step": args.switch_step,
          "steps": args.steps,
          "results": results,
        },
        indent=2,
      ),
      encoding="utf-8",
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
