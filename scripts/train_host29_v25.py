"""Correct V24 rotation with observable, critic-friendly rewards."""

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
from src.tasks.host_recovery.posture29_v25 import posture29_v25_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=20261010)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v25_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v25"
  agent.max_iterations = args.updates
  agent.save_interval = 100
  agent.algorithm.learning_rate = 2e-5
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 2
  agent.algorithm.entropy_coef = 0.0
  agent.actor.distribution_cfg.update(init_std=0.01, max_std=0.01)
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

    for parameter in runner.alg.policy.actor.parameters():
      parameter.requires_grad_(False)
    for layer_index in (4, 6):
      for parameter in runner.alg.policy.actor[layer_index].parameters():
        parameter.requires_grad_(True)
    frozen_actor = {
      name: parameter.detach().clone()
      for name, parameter in runner.alg.policy.actor.named_parameters()
      if not parameter.requires_grad
    }

    action_scale = float(
      source["infos"]["env_state"]["host_action_rescale"].float().mean()
    )
    set_action_scale(env, action_scale)
    with torch.no_grad():
      runner.alg.policy.std.fill_(0.01)
    runner.alg.policy.std.requires_grad_(False)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "actor_exact_at_launch": True,
      "actor_trainable_scope": "final Linear(256, 128) and Linear(128, 29)",
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 2e-5,
      "learning_epochs": 2,
      "entropy_coef": 0.0,
      "noise_std": 0.01,
      "action_std_frozen": True,
      "action_scale_frozen": action_scale,
      "action_scale_curriculum": False,
      "pull_force_curriculum": False,
      "reason": (
        "V24's absolute heading, world-foot orientation, and foot-anchor costs "
        "are absent from the actor observation and produced a 2.4e6 critic "
        "loss. V25 removes those partially observable, unbounded objectives "
        "and targets the observed left-hip-yaw cycle with bounded stance, "
        "neutral lateral-joint pose, stillness, and action costs."
      ),
      "reward_adjustments": {
        "yaw_rate": -1.50,
        "foot_horizontal_speed": -0.35,
        "leg_velocity": -0.0035,
        "hip_yaw_velocity": -0.06,
        "left_hip_yaw_velocity": -0.06,
        "lateral_balance_velocity": -0.010,
        "torso_vertical": 0.75,
        "torso_tilt": -2.00,
        "feet_side_by_side_huber": -0.40,
        "torso_over_support": 0.50,
        "pelvis_over_support": 0.40,
        "lateral_joint_pose": -0.30,
        "lateral_joint_stillness": 0.30,
        "lateral_action": -0.15,
        "left_hip_yaw_action": -0.10,
      },
      "removed_unobservable_rewards": (
        "heading_error",
        "feet_world_orientation",
        "world_foot_anchor",
        "root_horizontal_speed",
      ),
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

    observations = wrapper.get_observations()
    policy = runner.get_inference_policy(device="cuda:0")
    reward_abs_sum = torch.zeros(
      len(env.reward_manager.active_terms), device=env.device
    )
    with torch.inference_mode():
      for _ in range(100):
        actions = policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(actions)
        reward_abs_sum += env.reward_manager._step_reward.abs().mean(dim=0)
        assert actions.shape == (args.num_envs, 29)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
    reward_abs_mean = {
      name: float(value)
      for name, value in zip(
        env.reward_manager.active_terms, reward_abs_sum.cpu() / 100.0, strict=True
      )
      if name.startswith("target_")
    }
    largest_target = max(reward_abs_mean.values(), default=0.0)
    assert largest_target < 2.0, reward_abs_mean
    print("HOST29_V25_PREFLIGHT_REWARD_ABS", json.dumps(reward_abs_mean), flush=True)
    print("HOST29_V25_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    current_actor = dict(runner.alg.policy.actor.named_parameters())
    for name, value in frozen_actor.items():
      assert torch.equal(current_actor[name], value), name
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V25_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
