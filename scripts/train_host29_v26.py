"""Correct V25's residual torso lean and asymmetric raised arm."""

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
from src.tasks.host_recovery.posture29_v26 import posture29_v26_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


POSTURE_JOINTS = (
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
EXPECTED_ROWS = list(range(13, 29))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=800)
  parser.add_argument("--seed", type=int, default=20261011)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v26_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v26"
  agent.max_iterations = args.updates
  agent.save_interval = 100
  agent.algorithm.learning_rate = 3e-5
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 2
  agent.algorithm.entropy_coef = 0.0
  agent.actor.distribution_cfg.update(init_std=0.005, max_std=0.005)
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
    output_layer = runner.alg.policy.actor[6]
    output_layer.weight.requires_grad_(True)
    output_layer.bias.requires_grad_(True)
    joint_names = list(env.scene["robot"].joint_names)
    trainable_rows = [joint_names.index(name) for name in POSTURE_JOINTS]
    assert trainable_rows == EXPECTED_ROWS, (trainable_rows, joint_names)
    weight_mask = torch.zeros_like(output_layer.weight)
    bias_mask = torch.zeros_like(output_layer.bias)
    weight_mask[trainable_rows] = 1.0
    bias_mask[trainable_rows] = 1.0
    output_layer.weight.register_hook(lambda gradient: gradient * weight_mask)
    output_layer.bias.register_hook(lambda gradient: gradient * bias_mask)
    frozen_weight = output_layer.weight.detach().clone()
    frozen_bias = output_layer.bias.detach().clone()

    action_scale = float(
      source["infos"]["env_state"]["host_action_rescale"].float().mean()
    )
    set_action_scale(env, action_scale)
    with torch.no_grad():
      runner.alg.policy.std.fill_(0.005)
    runner.alg.policy.std.requires_grad_(False)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "actor_exact_at_launch": True,
      "actor_trainable_scope": "final actor rows 13 through 28 only",
      "trainable_action_outputs": list(POSTURE_JOINTS),
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 3e-5,
      "learning_epochs": 2,
      "entropy_coef": 0.0,
      "noise_std": 0.005,
      "action_std_frozen": True,
      "action_scale_frozen": action_scale,
      "action_scale_curriculum": False,
      "pull_force_curriculum": False,
      "reason": (
        "V25 removes continuous turning and corrective stepping, but its "
        "videos retain a repeatable torso lean and one raised asymmetric arm. "
        "V26 freezes every leg and waist-yaw output that produced the stable "
        "stance and trains only waist roll/pitch and arm rows, with separate "
        "left/right pose targets so one arm cannot hide in a 17-joint mean."
      ),
      "reward_adjustments": {
        "torso_vertical": 1.25,
        "torso_tilt": -3.00,
        "torso_over_support": 0.75,
        "waist_roll_pitch": -1.00,
        "hands_beside_hips": -5.00,
        "elbows_outside_torso": 0.40,
        "elbow_flexion": -0.25,
        "left_arm_pose": -1.25,
        "right_arm_pose": -1.25,
        "waist_pose": -1.25,
        "arm_stillness": 0.30,
      },
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
    assert all(
      not parameter.requires_grad
      for index, parameter in enumerate(runner.alg.policy.actor.parameters())
      if index not in (6, 7)
    )
    print("HOST29_V26_PREFLIGHT_REWARD_ABS", json.dumps(reward_abs_mean), flush=True)
    print("HOST29_V26_PREFLIGHT_ROWS", trainable_rows, flush=True)
    print("HOST29_V26_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    frozen_rows = [index for index in range(29) if index not in trainable_rows]
    assert torch.equal(output_layer.weight[frozen_rows], frozen_weight[frozen_rows])
    assert torch.equal(output_layer.bias[frozen_rows], frozen_bias[frozen_rows])
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V26_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
