"""Raise V38's standing equilibrium to the verified SMP/A6 head contract."""

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
from src.tasks.host_recovery.posture29_v39 import (
  a6_formal_contract,
  a6_virtual_head_height,
  posture29_v39_env_cfg,
)
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


LATERAL_ROWS = (1, 2, 5, 7, 8, 11, 12)
SAGITTAL_ROWS = (0, 3, 4, 6, 9, 10)
UPPER_ROWS = tuple(range(13, 29))
ROW_GROUPS = {
  "lateral": LATERAL_ROWS,
  "sagittal": SAGITTAL_ROWS,
  "upper": UPPER_ROWS,
}
ROW_STD = {"lateral": 0.006, "sagittal": 0.012, "upper": 0.004}
BASE_WEIGHT_DELTA = {"lateral": 0.0015, "sagittal": 0.0030, "upper": 0.0010}
BASE_BIAS_DELTA = {"lateral": 0.008, "sagittal": 0.015, "upper": 0.005}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=1200)
  parser.add_argument("--save-interval", type=int, default=50)
  parser.add_argument("--seed", type=int, default=20261018)
  parser.add_argument("--trust-scale", type=float, default=1.0)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()
  if args.trust_scale <= 0.0:
    raise ValueError("trust-scale must be positive")

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v39_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v39"
  agent.max_iterations = args.updates
  agent.save_interval = args.save_interval
  agent.algorithm.learning_rate = 8e-6
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 2
  agent.algorithm.entropy_coef = 0.0
  agent.actor.distribution_cfg.update(init_std=0.004, max_std=0.012)
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
    source_weight = output_layer.weight.detach().clone()
    source_bias = output_layer.bias.detach().clone()
    max_weight_delta = {
      name: value * args.trust_scale for name, value in BASE_WEIGHT_DELTA.items()
    }
    max_bias_delta = {
      name: value * args.trust_scale for name, value in BASE_BIAS_DELTA.items()
    }

    def clamp_actor_after_step(_optimizer, _args, _kwargs) -> None:
      with torch.no_grad():
        for group, rows in ROW_GROUPS.items():
          row_ids = list(rows)
          weight_delta = (
            output_layer.weight[row_ids] - source_weight[row_ids]
          ).clamp(-max_weight_delta[group], max_weight_delta[group])
          bias_delta = (
            output_layer.bias[row_ids] - source_bias[row_ids]
          ).clamp(-max_bias_delta[group], max_bias_delta[group])
          output_layer.weight[row_ids] = source_weight[row_ids] + weight_delta
          output_layer.bias[row_ids] = source_bias[row_ids] + bias_delta

    runner.alg.optimizer.register_step_post_hook(clamp_actor_after_step)

    action_scale = float(
      source["infos"]["env_state"]["host_action_rescale"].float().mean()
    )
    set_action_scale(env, action_scale)
    with torch.no_grad():
      for group, rows in ROW_GROUPS.items():
        runner.alg.policy.std[list(rows)] = ROW_STD[group]
    runner.alg.policy.std.requires_grad_(False)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "trust_scale": args.trust_scale,
      "actor_exact_at_launch": True,
      "actor_trainable_scope": "bounded final actor layer",
      "row_groups": {name: list(rows) for name, rows in ROW_GROUPS.items()},
      "row_std": ROW_STD,
      "max_weight_delta": max_weight_delta,
      "max_bias_delta": max_bias_delta,
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 8e-6,
      "learning_epochs": 2,
      "action_scale_frozen": action_scale,
      "a6_contract": {
        "head_site": "torso_link + rotated [0, 0, 0.43] m",
        "height_m": 1.15,
        "upright": 0.90,
        "training_target_height_m": 1.17,
        "post_stand_only": True,
      },
      "reason": (
        "V38 reaches HoST's torso-relative standing predicate but settles at "
        "about 1.09 m under the SMP/A6 head-site metric. V39 preserves the "
        "recovery actor at launch and trains a bounded state-dependent output "
        "layer against the authoritative 0.43 m head-site height contract."
      ),
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

    observations = wrapper.get_observations()
    normalized = runner.obs_normalizer(observations["actor"])
    probe = runner.alg.policy.actor(normalized)
    runner.alg.optimizer.zero_grad(set_to_none=True)
    probe.square().mean().backward()
    assert output_layer.weight.grad.abs().sum() > 0
    assert output_layer.bias.grad.abs().sum() > 0
    runner.alg.optimizer.zero_grad(set_to_none=True)

    policy = runner.get_inference_policy(device="cuda:0")
    preflight_steps = 450
    reward_abs_sum = torch.zeros(
      len(env.reward_manager.active_terms), device=env.device
    )
    final_height = None
    final_contract = None
    with torch.inference_mode():
      for _ in range(preflight_steps):
        actions = policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(actions)
        reward_abs_sum += env.reward_manager._step_reward.abs().mean(dim=0)
        final_height = a6_virtual_head_height(
          env, env.reward_manager.get_term_cfg("target_a6_height_progress").params["torso_cfg"]
        )
        final_contract = a6_formal_contract(
          env, env.reward_manager.get_term_cfg("target_a6_formal_contract").params["torso_cfg"]
        )
        assert actions.shape == (args.num_envs, 29)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
    reward_abs_mean = {
      name: float(value)
      for name, value in zip(
        env.reward_manager.active_terms,
        reward_abs_sum.cpu() / preflight_steps,
        strict=True,
      )
      if name.startswith("target_a6_")
    }
    assert reward_abs_mean["target_a6_height_progress"] > 0.0
    assert reward_abs_mean["target_a6_height_shortfall"] > 0.0
    assert final_height is not None and final_contract is not None
    assert torch.isfinite(final_height).all()
    print("HOST29_V39_PREFLIGHT_REWARD_ABS", json.dumps(reward_abs_mean), flush=True)
    print(
      "HOST29_V39_PREFLIGHT_A6",
      json.dumps(
        {
          "head_height_mean_m": float(final_height.mean()),
          "head_height_min_m": float(final_height.amin()),
          "formal_contract_fraction": float(final_contract.mean()),
        }
      ),
      flush=True,
    )
    print("HOST29_V39_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    for group, rows in ROW_GROUPS.items():
      row_ids = list(rows)
      assert (
        output_layer.weight[row_ids] - source_weight[row_ids]
      ).abs().amax() <= max_weight_delta[group] + 1e-7
      assert (
        output_layer.bias[row_ids] - source_bias[row_ids]
      ).abs().amax() <= max_bias_delta[group] + 1e-7
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V39_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
