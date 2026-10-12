"""Train paired hip/knee output biases while preserving V28 feedback."""

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
from src.tasks.host_recovery.posture29_v30 import posture29_v30_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


EXPECTED_ROWS = [0, 3, 6, 9]
PAIRED_ROWS = ((0, 6), (3, 9))
TRAIN_STD = 0.04
FROZEN_STD = 0.005
MAX_BIAS_DELTA = 0.05


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=400)
  parser.add_argument("--seed", type=int, default=20261011)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v30_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v30"
  agent.max_iterations = args.updates
  agent.save_interval = 50
  agent.algorithm.learning_rate = 2e-4
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 4
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
    output_layer.bias.requires_grad_(True)
    trainable_rows = EXPECTED_ROWS
    frozen_rows = [row for row in range(29) if row not in trainable_rows]
    source_weight = output_layer.weight.detach().clone()
    source_bias = output_layer.bias.detach().clone()

    def project_bias_gradient(gradient: torch.Tensor) -> torch.Tensor:
      projected = torch.zeros_like(gradient)
      for left_row, right_row in PAIRED_ROWS:
        pair_gradient = 0.5 * (gradient[left_row] - gradient[right_row])
        projected[left_row] = pair_gradient
        projected[right_row] = -pair_gradient
      return projected

    output_layer.bias.register_hook(project_bias_gradient)

    def clamp_bias_after_step(_optimizer, _args, _kwargs) -> None:
      with torch.no_grad():
        output_layer.weight.copy_(source_weight)
        output_layer.bias[frozen_rows] = source_bias[frozen_rows]
        for left_row, right_row in PAIRED_ROWS:
          left_delta = output_layer.bias[left_row] - source_bias[left_row]
          right_delta = output_layer.bias[right_row] - source_bias[right_row]
          paired_delta = (0.5 * (left_delta - right_delta)).clamp(
            -MAX_BIAS_DELTA, MAX_BIAS_DELTA
          )
          output_layer.bias[left_row] = source_bias[left_row] + paired_delta
          output_layer.bias[right_row] = source_bias[right_row] - paired_delta

    runner.alg.optimizer.register_step_post_hook(clamp_bias_after_step)

    action_scale = float(
      source["infos"]["env_state"]["host_action_rescale"].float().mean()
    )
    set_action_scale(env, action_scale)
    with torch.no_grad():
      runner.alg.policy.std.fill_(FROZEN_STD)
      runner.alg.policy.std[trainable_rows] = TRAIN_STD
    runner.alg.policy.std.requires_grad_(False)
    assert torch.all(runner.alg.policy.std[trainable_rows] == TRAIN_STD)
    assert torch.all(runner.alg.policy.std[frozen_rows] == FROZEN_STD)

    action_term = env.action_manager.get_term("joint_pos")
    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "actor_exact_at_launch": True,
      "actor_trainable_scope": "paired final-layer hip-pitch/knee biases only",
      "trainable_action_rows": trainable_rows,
      "trainable_action_outputs": [
        env.scene["robot"].joint_names[row] for row in trainable_rows
      ],
      "paired_gradient_rows": PAIRED_ROWS,
      "max_bias_delta": MAX_BIAS_DELTA,
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 2e-4,
      "learning_epochs": 4,
      "entropy_coef": 0.0,
      "trainable_noise_std": TRAIN_STD,
      "frozen_noise_std": FROZEN_STD,
      "action_std_frozen": True,
      "action_scale_frozen": action_scale,
      "action_scale_curriculum": False,
      "pull_force_curriculum": False,
      "posture_controller": {
        "arm_blend": action_term.cfg.arm_blend,
        "arm_gain": action_term.cfg.arm_gain,
        "balance_blend": action_term.cfg.balance_blend,
        "balance_gain": action_term.cfg.balance_gain,
        "standing_height": action_term.cfg.standing_height,
        "delay_steps": action_term.cfg.delay_steps,
        "ramp_steps": action_term.cfg.ramp_steps,
      },
      "reason": (
        "V29 proves a monotonic geometry-motion trade-off: learned feature "
        "weights reduce fore-aft foot offset but progressively destroy the "
        "stable closed-loop response. V30 returns to V28, keeps every actor "
        "weight fixed, removes the unnecessary ankle and joint-symmetry "
        "objectives, and permits only bounded antisymmetric hip-pitch/knee "
        "bias shifts so the stable state feedback remains unchanged."
      ),
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

    observations = wrapper.get_observations()
    normalized = runner.obs_normalizer(observations["actor"])
    probe = runner.alg.policy.actor(normalized)
    runner.alg.optimizer.zero_grad(set_to_none=True)
    probe[:, trainable_rows].square().mean().backward()
    assert output_layer.weight.grad is None
    assert output_layer.bias.grad[trainable_rows].abs().sum() > 0
    assert torch.count_nonzero(output_layer.bias.grad[frozen_rows]) == 0
    for left_row, right_row in PAIRED_ROWS:
      assert torch.equal(
        output_layer.bias.grad[left_row], -output_layer.bias.grad[right_row]
      )
    runner.alg.optimizer.zero_grad(set_to_none=True)

    policy = runner.get_inference_policy(device="cuda:0")
    reward_abs_sum = torch.zeros(
      len(env.reward_manager.active_terms), device=env.device
    )
    max_controller_mix = 0.0
    with torch.inference_mode():
      for _ in range(100):
        actions = policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(actions)
        reward_abs_sum += env.reward_manager._step_reward.abs().mean(dim=0)
        max_controller_mix = max(
          max_controller_mix, float(action_term._last_controller_mix.max())
        )
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
    assert max(reward_abs_mean.values(), default=0.0) < 3.0, reward_abs_mean
    assert reward_abs_mean["target_feet_forward_alignment_error"] > 0.0
    assert reward_abs_mean["target_sagittal_leg_symmetry_error"] == 0.0
    assert max_controller_mix >= 0.5, max_controller_mix
    print("HOST29_V30_PREFLIGHT_REWARD_ABS", json.dumps(reward_abs_mean), flush=True)
    print("HOST29_V30_PREFLIGHT_ROWS", trainable_rows, flush=True)
    print("HOST29_V30_PREFLIGHT_CONTROLLER", max_controller_mix, flush=True)
    print("HOST29_V30_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    assert torch.equal(output_layer.weight, source_weight)
    assert torch.equal(output_layer.bias[frozen_rows], source_bias[frozen_rows])
    for left_row, right_row in PAIRED_ROWS:
      left_delta = output_layer.bias[left_row] - source_bias[left_row]
      right_delta = output_layer.bias[right_row] - source_bias[right_row]
      assert torch.allclose(left_delta, -right_delta, atol=1e-7, rtol=0.0)
      assert left_delta.abs() <= MAX_BIAS_DELTA
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V30_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
