"""Acquire formal-scale control through a gated near-standing homotopy."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from scripts.train_a6_host_v2 import (
  CONTROL_DT,
  FORMAL_ACTION_SCALE,
  height_margin_reward,
  primary_stance_reward,
  sha256,
  smooth_gate,
)
from scripts.train_a6_host_v5 import stage_progress_reward
from scripts.train_a6_host_v6 import (
  action_scale_indices,
  configure_flat_near_resets,
)
from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime import (
  A6_POSITIVE_NAMES,
  _a6_height,
  _a6_upright,
  a6_env_cfg,
  a6_reset_counts,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


TRAIN_STD = 0.08
LEARNING_RATE = 2e-5
ACTION_SCALE_DECAY = 0.01
ACTION_SCALE_HEIGHT_GATE = 0.65


def near_retention_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  height = smooth_gate(_a6_height(env), 0.72, 1.08)
  upright = smooth_gate(_a6_upright(env), 0.55, 0.90)
  return height * upright


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=1800)
  parser.add_argument("--seed", type=int, default=20261013)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  source_action_scale = float(
    source["infos"]["env_state"]["host_action_rescale"].float().mean()
  )

  cfg = a6_env_cfg(seed=args.seed, dynamics=False, group="G+")
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.curriculum["action_scale"].params.update(
    threshold_height=ACTION_SCALE_HEIGHT_GATE,
    decay=ACTION_SCALE_DECAY,
    min_scale=FORMAL_ACTION_SCALE,
  )
  cfg.curriculum.pop("pull_force", None)
  cfg.terminations.pop("invalid_plate", None)
  for name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(name, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  for name, term in cfg.rewards.items():
    if not name.startswith("a6_"):
      term.weight = 0.0
  for name in A6_POSITIVE_NAMES:
    cfg.rewards[f"a6_{name}"].params["dense_guide"] = False
  cfg.rewards["a6_stage_pose"].weight = 1.50 * CONTROL_DT
  cfg.rewards["a6_head_velocity"].weight = 0.50 * CONTROL_DT
  cfg.rewards["a6_height"].weight = 2.00 * CONTROL_DT
  cfg.rewards["a6_upright"].weight = 2.00 * CONTROL_DT
  cfg.rewards["a6_stage_progress"] = RewardTermCfg(
    func=stage_progress_reward, weight=0.50 * CONTROL_DT
  )
  cfg.rewards["a6_height_margin"] = RewardTermCfg(
    func=height_margin_reward, weight=2.00 * CONTROL_DT
  )
  cfg.rewards["a6_primary_stance"] = RewardTermCfg(
    func=primary_stance_reward, weight=2.00 * CONTROL_DT
  )
  cfg.rewards["a6_near_retention"] = RewardTermCfg(
    func=near_retention_reward, weight=4.00 * CONTROL_DT
  )

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host_a6_v8_near_scale_homotopy"
  agent.max_iterations = args.updates
  agent.num_steps_per_env = 24
  agent.save_interval = 50
  agent.algorithm.learning_rate = LEARNING_RATE
  agent.algorithm.schedule = "fixed"
  agent.algorithm.entropy_coef = 0.0
  agent.algorithm.num_learning_epochs = 2
  agent.actor.distribution_cfg.update(init_std=TRAIN_STD, max_std=TRAIN_STD)
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.reset()
    configure_flat_near_resets(env)
    set_action_scale(env, source_action_scale)

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
    set_action_scale(env, source_action_scale)

    policy_state = runner.alg.policy.state_dict()
    for name, value in source["actor_state_dict"].items():
      if name == "distribution.std_param":
        continue
      mapped = "actor." + name.removeprefix("mlp.")
      assert torch.equal(policy_state[mapped].detach().cpu(), value.cpu()), mapped
    assert len(runner.alg.optimizer.state) == 0
    assert all(
      parameter.requires_grad for parameter in runner.alg.policy.actor.parameters()
    )

    scale_indices = action_scale_indices(
      env, runner.obs_normalizer._mean.shape[-1]
    )
    runner.obs_normalizer.until = int(runner.obs_normalizer.count.item())

    def sync_action_scale_mean(module, _inputs) -> None:
      with torch.no_grad():
        module._mean[:, scale_indices] = env._host_action_rescale.mean()

    runner.obs_normalizer.register_forward_pre_hook(sync_action_scale_mean)
    sync_action_scale_mean(runner.obs_normalizer, ())

    with torch.no_grad():
      runner.alg.policy.std.fill_(TRAIN_STD)
    runner.alg.policy.std.requires_grad_(False)
    env._a6_adaptation_start_step = int(env.common_step_counter)

    reset_counts = a6_reset_counts(env)
    assert reset_counts["scene"] == [args.num_envs, 0, 0], reset_counts
    assert reset_counts["stratum"] == [0, 0, args.num_envs], reset_counts
    assert reset_counts["source"] == [args.num_envs, 0], reset_counts

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "method": "gated near-standing action-scale homotopy",
      "group": "G+",
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "action_scale_start": source_action_scale,
      "action_scale_target": FORMAL_ACTION_SCALE,
      "action_scale_decay": ACTION_SCALE_DECAY,
      "action_scale_height_gate": ACTION_SCALE_HEIGHT_GATE,
      "reset_mix": reset_counts,
      "nominal_dynamics": True,
      "observation_noise": False,
      "invalid_plate_termination": False,
      "train_std": TRAIN_STD,
      "learning_rate": LEARNING_RATE,
      "learning_epochs": 2,
      "actor_trainable_scope": "full actor",
      "actor_normalizer": "frozen except scale means synchronized to runtime scale",
      "actor_normalizer_scale_indices": scale_indices.tolist(),
      "legacy_style_target_rewards": "disabled",
      "obstacle_rewards": "disabled during acquisition",
      "fresh_critic": True,
      "fresh_optimizer": True,
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

    observations = wrapper.get_observations()
    normalized = runner.obs_normalizer(observations["actor"])
    probe = runner.alg.policy.actor(normalized)
    runner.alg.optimizer.zero_grad(set_to_none=True)
    probe.square().mean().backward()
    actor_layers = (runner.alg.policy.actor[0], runner.alg.policy.actor[6])
    assert all(layer.weight.grad.abs().sum() > 0 for layer in actor_layers)
    assert all(layer.bias.grad.abs().sum() > 0 for layer in actor_layers)
    runner.alg.optimizer.zero_grad(set_to_none=True)

    reward_abs_sum = torch.zeros(
      len(env.reward_manager.active_terms), device=env.device
    )
    policy = runner.get_inference_policy(device="cuda:0")
    rollout_steps = 600 if args.preflight else 200
    start_scale = float(env._host_action_rescale.mean())
    with torch.inference_mode():
      for _ in range(rollout_steps):
        actions = policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(actions)
        reward_abs_sum += env.reward_manager._step_reward.abs().mean(dim=0)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
    reward_abs_mean = {
      name: float(value)
      for name, value in zip(
        env.reward_manager.active_terms,
        reward_abs_sum.cpu() / rollout_steps,
        strict=True,
      )
    }
    for name in (
      "a6_stage_pose",
      "a6_height",
      "a6_upright",
      "a6_stage_progress",
      "a6_height_margin",
      "a6_primary_stance",
      "a6_near_retention",
    ):
      assert 0.0 < reward_abs_mean[name] < 0.20, (name, reward_abs_mean[name])
    end_scale = float(env._host_action_rescale.mean())
    synced_mean = runner.obs_normalizer._mean[:, scale_indices]
    assert torch.allclose(
      synced_mean, torch.full_like(synced_mean, end_scale), atol=1e-6
    )
    if args.preflight:
      assert end_scale < start_scale, (start_scale, end_scale)
      assert end_scale >= start_scale - ACTION_SCALE_DECAY - 1e-5
    print(
      "A6_V8_PREFLIGHT_REWARD_ABS",
      json.dumps(
        {
          name: reward_abs_mean[name]
          for name in (
            "a6_stage_pose",
            "a6_head_velocity",
            "a6_height",
            "a6_upright",
            "a6_stage_progress",
            "a6_height_margin",
            "a6_primary_stance",
            "a6_near_retention",
          )
        }
      ),
      flush=True,
    )
    print(
      "A6_V8_PREFLIGHT_SCALE",
      json.dumps({"start": start_scale, "end": end_scale}),
      flush=True,
    )
    print("A6_V8_PREFLIGHT_PASS", json.dumps(launch), flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    assert runner.alg.policy.training
    assert runner.obs_normalizer.training
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("A6_V8_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
