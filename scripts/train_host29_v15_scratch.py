"""Train the extreme V15 posture objective from random initialization."""

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.posture29_v15 import posture29_v15_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=12000)
  parser.add_argument("--seed", type=int, default=20261009)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v15_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v15_scratch"
  agent.max_iterations = args.updates
  agent.save_interval = 500
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    observations = wrapper.get_observations()
    assert observations["actor"].shape[-1] == 564
    assert wrapper.num_actions == 29

    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), str(args.log_dir), "cuda:0")
    launch = {
      "checkpoint": None,
      "from_scratch": True,
      "actor_trainable_scope": "full actor MLP",
      "fresh_actor": True,
      "fresh_critic": True,
      "fresh_actor_normalizer": True,
      "fresh_critic_normalizer": True,
      "fresh_optimizer": True,
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "learning_rate": agent.algorithm.learning_rate,
      "schedule": agent.algorithm.schedule,
      "learning_epochs": agent.algorithm.num_learning_epochs,
      "entropy_coef": agent.algorithm.entropy_coef,
      "noise_std": agent.actor.distribution_cfg["init_std"],
      "action_scale_curriculum": True,
      "pull_force_curriculum": True,
      "reward_adjustments": {
        "hands_beside_hips_error": -5.00,
        "torso_vertical": 1.50,
        "torso_tilt": -4.00,
        "torso_support_error": -8.00,
        "upper_pose_linear": -2.50,
        "elbow_up": -0.10,
        "arm_twist": -0.08,
        "wrist_flip": -0.05,
        "elbow_flexion": -0.50,
        "foot_horizontal_speed": -0.20,
        "leg_velocity_penalty": -0.0025,
        "upper_velocity_penalty": -0.0020,
      },
      "git_commit": subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True
      ).strip(),
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

    policy = runner.get_inference_policy(device="cuda:0")
    with torch.inference_mode():
      for _ in range(100):
        actions = policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(actions)
        assert actions.shape == (args.num_envs, 29)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
    print("HOST29_V15_SCRATCH_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V15_SCRATCH_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
