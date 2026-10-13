"""Train A6 at formal scale with a mixed-reset full-actor curriculum."""

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

from scripts.train_a6_host_v3 import (
  CONTROL_DT,
  FORMAL_ACTION_SCALE,
  height_margin_reward,
  primary_stance_reward,
  sha256,
  smooth_gate,
)
from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime import (
  A6_POSITIVE_NAMES,
  SCENE_NAMES,
  _a6_height,
  _a6_upright,
  a6_env_cfg,
  a6_reset_counts,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.mdp import a6_geometry
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


TRAIN_STD = 0.15
LEARNING_RATE = 3e-5
LOW_FRACTION = 0.25
MIDDLE_FRACTION = 0.375


def stage_progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  height = _a6_height(env)
  upright = _a6_upright(env)
  stage = env._a6_stage.clamp(0, 3)
  height_low = height.new_tensor((0.18, 0.48, 0.72, 0.90))[stage]
  height_high = height.new_tensor((0.62, 0.86, 1.15, 1.15))[stage]
  upright_low = height.new_tensor((0.15, 0.40, 0.62, 0.75))[stage]
  upright_high = height.new_tensor((0.60, 0.76, 0.93, 0.93))[stage]
  height_progress = smooth_gate(height, height_low, height_high)
  upright_progress = smooth_gate(upright, upright_low, upright_high)
  return 0.60 * height_progress + 0.40 * upright_progress


def configure_reverse_mix(env: ManagerBasedRlEnv) -> None:
  env._a6_source.zero_()
  for scene_id in range(len(SCENE_NAMES)):
    for direction_id in range(len(a6_geometry.DIRECTIONS)):
      ids = torch.nonzero(
        (env._a6_scene == scene_id)
        & (env._a6_direction == direction_id),
        as_tuple=False,
      ).flatten()
      count = int(ids.numel())
      low = round(count * LOW_FRACTION)
      middle = round(count * MIDDLE_FRACTION)
      middle = min(middle, count - low)
      env._a6_stratum[ids[:low]] = 0
      env._a6_stratum[ids[low : low + middle]] = 1
      env._a6_stratum[ids[low + middle :]] = 2
      procedural = round(low * 0.25)
      env._a6_source[ids[:procedural]] = 1
  env._reset_idx(torch.arange(env.num_envs, device=env.device))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=1200)
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
  cfg.curriculum.clear()
  cfg.terminations.pop("invalid_plate", None)
  for name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(name, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False
  for name in A6_POSITIVE_NAMES:
    cfg.rewards[f"a6_{name}"].params["dense_guide"] = False
  cfg.rewards["a6_stage_pose"].weight = 0.44 * CONTROL_DT
  cfg.rewards["a6_head_velocity"].weight = 0.30 * CONTROL_DT
  cfg.rewards["a6_height"].weight = 0.25 * CONTROL_DT
  cfg.rewards["a6_upright"].weight = 0.25 * CONTROL_DT
  cfg.rewards["a6_stage_progress"] = RewardTermCfg(
    func=stage_progress_reward, weight=1.50 * CONTROL_DT
  )
  cfg.rewards["a6_height_margin"] = RewardTermCfg(
    func=height_margin_reward, weight=0.80 * CONTROL_DT
  )
  cfg.rewards["a6_primary_stance"] = RewardTermCfg(
    func=primary_stance_reward, weight=1.20 * CONTROL_DT
  )
  cfg.rewards["plate_completion"].weight = 1.00 * CONTROL_DT
  cfg.rewards["plate_geometry_progress"].weight = 0.75 * CONTROL_DT
  cfg.rewards["plate_clearance"].weight = 0.10 * CONTROL_DT
  cfg.rewards["plate_separation"].weight = 0.02 * CONTROL_DT
  cfg.rewards["path_quiet_feet"].weight = -0.01 * CONTROL_DT

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host_a6_v5_formal_reverse_curriculum"
  agent.max_iterations = args.updates
  agent.num_steps_per_env = 24
  agent.save_interval = 50
  agent.algorithm.learning_rate = LEARNING_RATE
  agent.algorithm.schedule = "fixed"
  agent.algorithm.entropy_coef = 0.0
  agent.algorithm.num_learning_epochs = 4
  agent.actor.distribution_cfg.update(init_std=TRAIN_STD, max_std=TRAIN_STD)
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.reset()
    configure_reverse_mix(env)
    set_action_scale(env, FORMAL_ACTION_SCALE)

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

    with torch.no_grad():
      runner.alg.policy.std.fill_(TRAIN_STD)
    runner.alg.policy.std.requires_grad_(False)
    env._a6_adaptation_start_step = int(env.common_step_counter)
    set_action_scale(env, FORMAL_ACTION_SCALE)

    reset_counts = a6_reset_counts(env)
    assert sum(reset_counts["stratum"]) == args.num_envs
    assert all(count > 0 for count in reset_counts["stratum"]), reset_counts
    low_ratio = reset_counts["stratum"][0] / args.num_envs
    assert 0.20 <= low_ratio <= 0.30, reset_counts

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "method": "fixed-formal-scale mixed-reset full-actor curriculum",
      "group": "G+",
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "formal_action_scale": FORMAL_ACTION_SCALE,
      "source_action_scale": source_action_scale,
      "nominal_dynamics": True,
      "observation_noise": False,
      "invalid_plate_termination": False,
      "common_task_alpha": 1.0,
      "reset_mix": reset_counts,
      "train_std": TRAIN_STD,
      "learning_rate": LEARNING_RATE,
      "learning_epochs": 4,
      "actor_trainable_scope": "full actor",
      "actor_normalizer_frozen": True,
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
    with torch.inference_mode():
      for _ in range(300):
        actions = policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(actions)
        reward_abs_sum += env.reward_manager._step_reward.abs().mean(dim=0)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
    reward_abs_mean = {
      name: float(value)
      for name, value in zip(
        env.reward_manager.active_terms, reward_abs_sum.cpu() / 300.0, strict=True
      )
    }
    for name in (
      "a6_stage_pose",
      "a6_height",
      "a6_upright",
      "a6_stage_progress",
      "a6_height_margin",
      "a6_primary_stance",
    ):
      assert 0.0 < reward_abs_mean[name] < 0.10, (name, reward_abs_mean[name])
    assert "invalid_plate" not in env.termination_manager.active_terms
    assert float(env._host_action_rescale.amin()) == FORMAL_ACTION_SCALE
    assert float(env._host_action_rescale.amax()) == FORMAL_ACTION_SCALE
    print(
      "A6_V5_PREFLIGHT_REWARD_ABS",
      json.dumps(
        {
          name: reward_abs_mean[name]
          for name in (
            "a6_stage_pose",
            "a6_height",
            "a6_upright",
            "a6_stage_progress",
            "a6_height_margin",
            "a6_primary_stance",
            "plate_geometry_progress",
            "plate_completion",
          )
        }
      ),
      flush=True,
    )
    print("A6_V5_PREFLIGHT_PASS", json.dumps(launch), flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("A6_V5_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
