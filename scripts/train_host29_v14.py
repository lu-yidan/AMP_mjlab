"""Continue V13 with dense, non-saturating torso and arm constraints."""

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from scripts.train_host29_v7 import set_action_scale, sha256
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.posture29_v8 import POST_STAND_HEIGHT
from src.tasks.host_recovery.posture29_v14 import posture29_v14_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=2000)
  parser.add_argument("--seed", type=int, default=20261009)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v14_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v14"
  agent.max_iterations = args.updates
  agent.save_interval = 250
  agent.algorithm.learning_rate = 1.0e-5
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 2
  agent.algorithm.entropy_coef = 0.0
  agent.actor.distribution_cfg.update(init_std=0.015, max_std=0.015)
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

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy_state = runner.alg.policy.state_dict()
    for name, value in source["actor_state_dict"].items():
      if name == "distribution.std_param":
        continue
      mapped = "actor." + name.removeprefix("mlp.")
      assert torch.equal(policy_state[mapped].detach().cpu(), value.cpu()), mapped
    assert len(runner.alg.optimizer.state) == 0

    action_scale = float(
      source["infos"]["env_state"]["host_action_rescale"].float().mean()
    )
    set_action_scale(env, action_scale)
    with torch.no_grad():
      runner.alg.policy.std.fill_(0.015)
    runner.alg.policy.std.requires_grad_(False)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "actor_exact_at_launch": True,
      "actor_trainable_scope": "full actor MLP",
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 1.0e-5,
      "learning_epochs": 2,
      "entropy_coef": 0.0,
      "noise_std": 0.015,
      "action_std_frozen": True,
      "action_scale_frozen": action_scale,
      "post_stand_height": POST_STAND_HEIGHT,
      "reason": (
        "Replace V13's saturated exponential geometry rewards with dense "
        "distance costs and give the full actor enough optimization capacity."
      ),
      "reward_adjustments": {
        "hands_beside_hips_error": -1.50,
        "torso_vertical": 0.75,
        "torso_tilt": -1.20,
        "torso_support_error": -2.00,
        "upper_pose_linear": -0.80,
        "elbow_flexion": -0.08,
        "foot_horizontal_speed": -0.12,
        "leg_velocity_penalty": -0.0020,
        "upper_velocity_penalty": -0.0012,
      },
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

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
    print("HOST29_V14_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V14_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
