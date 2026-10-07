"""Low-disturbance continuation for post-stand arm alignment and leg stability."""

import argparse
import copy
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from src.tasks.host_recovery.config.g1.rl_cfg import unitree_g1_host_standup_ppo_runner_cfg
from src.tasks.host_recovery.directional29 import directional29_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=3000)
  parser.add_argument("--seed", type=int, default=20261007)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = directional29_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_directional"
  agent.max_iterations = args.updates
  agent.save_interval = 500
  agent.algorithm.learning_rate = 3.0e-5
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 3
  agent.algorithm.entropy_coef = 0.001
  agent.actor.distribution_cfg.update(init_std=0.05, max_std=0.08)
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), str(args.log_dir), "cuda:0")
    fresh_critic = copy.deepcopy(runner.alg.policy.critic.state_dict())
    fresh_critic_normalizer = copy.deepcopy(
      runner.privileged_obs_normalizer.state_dict()
    )
    runner.load(
      str(args.checkpoint),
      load_optimizer=False,
      load_cfg={"actor": True, "critic": False},
      map_location="cuda:0",
    )
    runner.alg.policy.critic.load_state_dict(fresh_critic)
    runner.privileged_obs_normalizer.load_state_dict(fresh_critic_normalizer)
    runner.current_learning_iteration = 0
    runner.obs_normalizer.until = int(runner.obs_normalizer.count.item())
    with torch.no_grad():
      runner.alg.policy.std.fill_(0.05)
    runner.alg.policy.std.requires_grad_(True)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "zero_pull_force": True,
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 3.0e-5,
      "entropy_coef": 0.001,
      "noise_std": 0.05,
      "max_noise_std": 0.08,
      "reward_weights": {
        "upper_forearm_alignment": 0.04,
        "forearm_hand_alignment": 0.04,
        "leg_vertical": 0.08,
        "leg_stillness": 0.06,
        "leg_velocity_penalty": -0.0012,
      },
    }
    (args.log_dir / "launch.json").write_text(json.dumps(launch, indent=2))

    observations = wrapper.get_observations()
    policy = runner.get_inference_policy(device="cuda:0")
    with torch.inference_mode():
      for _ in range(100):
        actions = policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(actions)
        assert actions.shape == (args.num_envs, 29)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
    print("HOST29_DIRECTIONAL_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_DIRECTIONAL_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
