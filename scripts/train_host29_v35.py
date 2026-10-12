"""Train observable lateral stillness while freezing sagittal support."""

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
from src.tasks.host_recovery.posture29_v35 import posture29_v35_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


TRAINABLE_ROWS = [1, 2, 5, 7, 8, 11, 12]
TRAIN_STD = 0.012
FROZEN_STD = 0.001
MAX_WEIGHT_DELTA = 0.004
MAX_BIAS_DELTA = 0.025


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=400)
  parser.add_argument("--seed", type=int, default=20261012)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v35_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v35"
  agent.max_iterations = args.updates
  agent.save_interval = 50
  agent.algorithm.learning_rate = 2e-5
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 2
  agent.algorithm.entropy_coef = 0.0
  agent.actor.distribution_cfg.update(init_std=FROZEN_STD, max_std=TRAIN_STD)
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
    frozen_rows = [index for index in range(29) if index not in TRAINABLE_ROWS]
    weight_mask = torch.zeros_like(output_layer.weight)
    bias_mask = torch.zeros_like(output_layer.bias)
    weight_mask[TRAINABLE_ROWS] = 1.0
    bias_mask[TRAINABLE_ROWS] = 1.0
    output_layer.weight.register_hook(lambda gradient: gradient * weight_mask)
    output_layer.bias.register_hook(lambda gradient: gradient * bias_mask)
    source_weight = output_layer.weight.detach().clone()
    source_bias = output_layer.bias.detach().clone()

    def clamp_actor_after_step(_optimizer, _args, _kwargs) -> None:
      with torch.no_grad():
        output_layer.weight[frozen_rows] = source_weight[frozen_rows]
        output_layer.bias[frozen_rows] = source_bias[frozen_rows]
        weight_delta = (
          output_layer.weight[TRAINABLE_ROWS] - source_weight[TRAINABLE_ROWS]
        ).clamp(-MAX_WEIGHT_DELTA, MAX_WEIGHT_DELTA)
        bias_delta = (
          output_layer.bias[TRAINABLE_ROWS] - source_bias[TRAINABLE_ROWS]
        ).clamp(-MAX_BIAS_DELTA, MAX_BIAS_DELTA)
        output_layer.weight[TRAINABLE_ROWS] = (
          source_weight[TRAINABLE_ROWS] + weight_delta
        )
        output_layer.bias[TRAINABLE_ROWS] = source_bias[TRAINABLE_ROWS] + bias_delta

    runner.alg.optimizer.register_step_post_hook(clamp_actor_after_step)

    action_scale = float(
      source["infos"]["env_state"]["host_action_rescale"].float().mean()
    )
    set_action_scale(env, action_scale)
    with torch.no_grad():
      runner.alg.policy.std.fill_(FROZEN_STD)
      runner.alg.policy.std[TRAINABLE_ROWS] = TRAIN_STD
    runner.alg.policy.std.requires_grad_(False)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "actor_exact_at_launch": True,
      "actor_trainable_scope": "bounded lateral lower-body output rows only",
      "trainable_action_outputs": [
        env.scene["robot"].joint_names[index] for index in TRAINABLE_ROWS
      ],
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 2e-5,
      "learning_epochs": 2,
      "entropy_coef": 0.0,
      "train_std": TRAIN_STD,
      "frozen_std": FROZEN_STD,
      "max_weight_delta": MAX_WEIGHT_DELTA,
      "max_bias_delta": MAX_BIAS_DELTA,
      "action_scale_frozen": action_scale,
      "posture_controller": None,
      "reason": (
        "V34 becomes quieter but remains crouched and tilted because the upper "
        "controller changes the load without changing the V25 posture attractor. "
        "V35 returns to V25, freezes all sagittal and upper outputs, and optimizes "
        "only observation-identifiable yaw and joint stillness objectives."
      ),
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

    observations = wrapper.get_observations()
    normalized = runner.obs_normalizer(observations["actor"])
    probe = runner.alg.policy.actor(normalized)
    runner.alg.optimizer.zero_grad(set_to_none=True)
    probe[:, TRAINABLE_ROWS].square().mean().backward()
    assert output_layer.weight.grad[TRAINABLE_ROWS].abs().sum() > 0
    assert output_layer.bias.grad[TRAINABLE_ROWS].abs().sum() > 0
    assert torch.count_nonzero(output_layer.weight.grad[frozen_rows]) == 0
    assert torch.count_nonzero(output_layer.bias.grad[frozen_rows]) == 0
    runner.alg.optimizer.zero_grad(set_to_none=True)

    policy = runner.get_inference_policy(device="cuda:0")
    reward_abs_sum = torch.zeros(
      len(env.reward_manager.active_terms), device=env.device
    )
    with torch.inference_mode():
      for _ in range(200):
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
        env.reward_manager.active_terms, reward_abs_sum.cpu() / 200.0, strict=True
      )
      if name.startswith("target_")
    }
    assert max(reward_abs_mean.values(), default=0.0) < 3.0, reward_abs_mean
    for name in (
      "target_yaw_rate",
      "target_leg_dof_vel",
      "target_lateral_joint_stillness",
      "target_lateral_action",
    ):
      assert reward_abs_mean[name] > 0.0, (name, reward_abs_mean)
    print("HOST29_V35_PREFLIGHT_REWARD_ABS", json.dumps(reward_abs_mean), flush=True)
    print("HOST29_V35_PREFLIGHT_ROWS", TRAINABLE_ROWS, flush=True)
    print("HOST29_V35_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    assert torch.equal(output_layer.weight[frozen_rows], source_weight[frozen_rows])
    assert torch.equal(output_layer.bias[frozen_rows], source_bias[frozen_rows])
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V35_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
