"""Train an observable recovery residual around the frozen V38 actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from scripts.train_a6_host_v6 import action_scale_indices
from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime import (
  A6_POSITIVE_NAMES,
  _a6_ground_force,
  _a6_height,
  _a6_upright,
  a6_env_cfg,
  a6_reset_counts,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.residual_control import (
  TeacherResidualVecEnvWrapper,
  obstacle_residual_gate,
  residual_effort,
)
from src.tasks.host_recovery.mdp.observations import UNACTUATED_STEPS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


CONTROL_DT = 0.02


def smooth_gate(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
  scaled = ((value - low) / (high - low)).clamp(0.0, 1.0)
  return scaled.square() * (3.0 - 2.0 * scaled)


def post_escape_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  gate = (
    (env._a6_scene > 0)
    & env._a6_escaped
    & (env.episode_length_buf > UNACTUATED_STEPS)
  )
  return gate.float() * smooth_gate(_a6_height(env), 1.04, 1.17)


def post_escape_stance_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  gate = (
    (env._a6_scene > 0)
    & env._a6_escaped
    & (env.episode_length_buf > UNACTUATED_STEPS)
  )
  height = smooth_gate(_a6_height(env), 1.07, 1.17)
  upright = smooth_gate(_a6_upright(env), 0.84, 0.93)
  load = _a6_ground_force(env, "quality_feet")[..., 2].abs().amin(-1)
  support = smooth_gate(load, 10.0, 30.0)
  return gate.float() * height * upright * support


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--residual-checkpoint", type=Path)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--group", choices=("G-", "G+"), required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=1200)
  parser.add_argument("--seed", type=int, default=20261016)
  parser.add_argument("--action-scale", type=float, default=0.5723877549171448)
  parser.add_argument("--residual-scale", type=float, default=0.25)
  parser.add_argument("--residual-after-escape", action="store_true")
  parser.add_argument("--learning-rate", type=float, default=1e-5)
  parser.add_argument("--train-std", type=float, default=0.06)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--episode-length-steps", type=int, default=1150)
  parser.add_argument("--post-escape-height-weight", type=float, default=0.0)
  parser.add_argument("--post-escape-stance-weight", type=float, default=0.0)
  parser.add_argument("--preflight", action="store_true")
  parser.add_argument("--preflight-steps", type=int, default=1150)
  parser.add_argument(
    "--method-name",
    default="obstacle-scene-isolated residual around frozen base",
  )
  parser.add_argument(
    "--experiment-name", default="g1_host_a6_v19_v38_recovery_residual"
  )
  parser.add_argument("--log-tag", default="A6_V19")
  args = parser.parse_args()

  if args.log_dir.exists():
    raise FileExistsError(args.log_dir)
  if args.num_envs < 32 or args.num_envs % 32:
    raise ValueError("--num-envs must be >= 32 and divisible by 32")
  if not 0.0 < args.action_scale <= 1.0:
    raise ValueError(args.action_scale)

  cfg = a6_env_cfg(seed=args.seed, dynamics=False, group=args.group)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = args.episode_length_steps * CONTROL_DT
  cfg.curriculum.pop("action_scale", None)
  cfg.curriculum.pop("pull_force", None)
  cfg.terminations.pop("invalid_plate", None)
  for event in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
    cfg.events.pop(event, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False
  for name, term in cfg.rewards.items():
    if not name.startswith(("a6_", "plate_", "path_")):
      term.weight = 0.0
  for name in A6_POSITIVE_NAMES:
    cfg.rewards[f"a6_{name}"].params["dense_guide"] = False
  cfg.rewards["a6_residual_effort"] = RewardTermCfg(
    func=residual_effort,
    weight=-0.20 * CONTROL_DT,
  )
  if args.post_escape_height_weight:
    cfg.rewards["a6_post_escape_height"] = RewardTermCfg(
      func=post_escape_height_reward,
      weight=args.post_escape_height_weight * CONTROL_DT,
    )
  if args.post_escape_stance_weight:
    cfg.rewards["a6_post_escape_stance"] = RewardTermCfg(
      func=post_escape_stance_reward,
      weight=args.post_escape_stance_weight * CONTROL_DT,
    )

  args.log_dir.mkdir(parents=True, exist_ok=False)
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = args.experiment_name
  agent.max_iterations = args.updates
  agent.save_interval = args.save_interval
  agent.algorithm.learning_rate = args.learning_rate
  agent.algorithm.num_learning_epochs = 2
  agent.actor.distribution_cfg.update(
    init_std=args.train_std,
    max_std=args.train_std,
  )
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.reset()
    env._a6_scene.copy_(
      1 + torch.arange(env.num_envs, device=env.device, dtype=torch.long) % 2
    )
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    set_action_scale(env, args.action_scale)
    teacher_wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    teacher_runner = HoSTOnPolicyRunner(
      teacher_wrapper, asdict(agent), device="cuda:0"
    )
    teacher_runner.load(
      str(args.base_checkpoint), load_optimizer=False, map_location="cuda:0"
    )
    set_action_scale(env, args.action_scale)
    teacher_scale_indices = action_scale_indices(
      env, teacher_runner.obs_normalizer._mean.shape[-1]
    )
    teacher_runner.obs_normalizer._mean[:, teacher_scale_indices] = args.action_scale
    teacher_runner.alg.policy.eval()
    teacher_runner.obs_normalizer.eval()
    for parameter in teacher_runner.alg.policy.parameters():
      parameter.requires_grad_(False)
    base_policy = teacher_runner.get_inference_policy(device="cuda:0")

    wrapper = TeacherResidualVecEnvWrapper(
      env,
      base_policy=base_policy,
      residual_scale=args.residual_scale,
      clip_actions=agent.clip_actions,
      apply_after_escape=args.residual_after_escape,
    )
    if args.preflight:
      env_ids = torch.arange(env.num_envs, device=env.device)
      env._a6_scene.copy_((env_ids * 3 // env.num_envs).clamp(max=2))
      env._reset_idx(env_ids)
    set_action_scale(env, args.action_scale)
    runner = HoSTOnPolicyRunner(
      wrapper, asdict(agent), str(args.log_dir), "cuda:0"
    )
    output_layer = runner.alg.policy.actor[6]
    if args.residual_checkpoint is not None:
      runner.load(
        str(args.residual_checkpoint),
        load_optimizer=False,
        map_location="cuda:0",
      )
    with torch.no_grad():
      if args.residual_checkpoint is None:
        output_layer.weight.zero_()
        output_layer.bias.zero_()
      runner.alg.policy.std.fill_(args.train_std)
    runner.alg.policy.std.requires_grad_(False)
    for param_group in runner.alg.optimizer.param_groups:
      param_group["lr"] = args.learning_rate

    launch = {
      "base_checkpoint": str(args.base_checkpoint.resolve()),
      "base_checkpoint_sha256": sha256(args.base_checkpoint),
      "base_actor": "frozen V38",
      "residual_checkpoint": (
        str(args.residual_checkpoint.resolve())
        if args.residual_checkpoint is not None
        else None
      ),
      "residual_checkpoint_sha256": (
        sha256(args.residual_checkpoint)
        if args.residual_checkpoint is not None
        else None
      ),
      "method": args.method_name,
      "group": args.group,
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "action_scale": args.action_scale,
      "residual_scale": args.residual_scale,
      "residual_after_escape": args.residual_after_escape,
      "post_escape_height_weight": args.post_escape_height_weight,
      "post_escape_stance_weight": args.post_escape_stance_weight,
      "episode_length_steps": args.episode_length_steps,
      "learning_rate": args.learning_rate,
      "train_std": args.train_std,
      "reset_mix": a6_reset_counts(env),
      "training_scenes": "obstacle-only, balanced guided/free",
      "teacher_scale_history_center": args.action_scale,
      "flat_action_contract": "exact frozen V38 action",
      "residual_gate": (
        "scene>0 and motor-on"
        if args.residual_after_escape
        else "scene>0 and not escaped and motor-on"
      ),
      "legacy_style_target_rewards": "disabled",
      "a6_and_obstacle_rewards": "enabled",
      "common_task_alpha": 1.0,
      "nominal_dynamics": True,
      "observation_noise": False,
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2) + "\n", encoding="utf-8"
    )

    observations = wrapper.get_observations()
    normalized = runner.obs_normalizer(observations["actor"])
    probe = runner.alg.policy.actor(normalized)
    runner.alg.optimizer.zero_grad(set_to_none=True)
    probe.sum().backward()
    assert output_layer.weight.grad is not None
    assert output_layer.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher_runner.alg.policy.parameters())
    runner.alg.optimizer.zero_grad(set_to_none=True)

    rollout_steps = args.preflight_steps if args.preflight else 200
    residual_policy = runner.get_inference_policy(device="cuda:0")
    gate_sum = 0.0
    flat_applied_residual_max = 0.0
    obstacle_gate_count = 0
    finite_reward = True
    reward_abs_sum = torch.zeros(
      len(env.reward_manager.active_terms), device=env.device
    )
    gate_contract = obstacle_residual_gate(
      torch.tensor([0, 1, 1, 2], device=env.device),
      torch.tensor([False, False, True, True], device=env.device),
      torch.tensor([True, True, True, False], device=env.device),
      apply_after_escape=args.residual_after_escape,
    ).tolist()
    with torch.inference_mode():
      for _ in range(rollout_steps):
        residual_actions = residual_policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(residual_actions)
        reward_abs_sum += env.reward_manager._step_reward.abs().mean(dim=0)
        gate = env._a6_residual_gate.squeeze(-1)
        gate_sum += float(gate.mean())
        flat = env._a6_scene == 0
        if flat.any():
          flat_applied_residual_max = max(
            flat_applied_residual_max,
            float(env._a6_gated_residual_action[flat].abs().max()),
          )
        obstacle_gate_count += int(gate.sum())
        finite_reward &= bool(torch.isfinite(rewards).all())
        assert torch.isfinite(env._a6_combined_action).all()
        assert torch.isfinite(observations["actor"]).all()

    reward_abs_mean = {
      name: float(value)
      for name, value in zip(
        env.reward_manager.active_terms,
        reward_abs_sum.cpu() / rollout_steps,
        strict=True,
      )
    }
    preflight = {
      "steps": rollout_steps,
      "mean_gate": gate_sum / rollout_steps,
      "flat_applied_residual_max": flat_applied_residual_max,
      "obstacle_gate_count": obstacle_gate_count,
      "gate_contract_flat_active_escaped_motor_off": gate_contract,
      "residual_mean_abs": float(env._a6_residual_action.abs().mean()),
      "gated_residual_mean_abs": float(
        env._a6_gated_residual_action.abs().mean()
      ),
      "finite_reward": finite_reward,
      "action_scale": float(env._host_action_rescale.mean()),
      "reset_mix": a6_reset_counts(env),
      "post_escape_reward_abs_mean": {
        name: reward_abs_mean[name]
        for name in ("a6_post_escape_height", "a6_post_escape_stance")
        if name in reward_abs_mean
      },
    }
    print(f"{args.log_tag}_PREFLIGHT", json.dumps(preflight), flush=True)
    assert finite_reward
    if args.residual_checkpoint is None:
      assert preflight["gated_residual_mean_abs"] < 1e-7
    assert preflight["flat_applied_residual_max"] == 0.0
    if args.preflight:
      assert preflight["reset_mix"]["scene"][0] > 0
    assert preflight["obstacle_gate_count"] > 0
    expected_gate_contract = [False, True, args.residual_after_escape, False]
    assert preflight["gate_contract_flat_active_escaped_motor_off"] == (
      expected_gate_contract
    )
    assert abs(preflight["action_scale"] - args.action_scale) < 1e-6
    if args.post_escape_height_weight:
      assert reward_abs_mean["a6_post_escape_height"] > 0.0
    if args.post_escape_stance_weight:
      assert reward_abs_mean["a6_post_escape_stance"] > 0.0
    if args.preflight:
      print(f"{args.log_tag}_PREFLIGHT_PASS", json.dumps(launch), flush=True)
      return

    runner.alg.policy.train()
    runner.obs_normalizer.train()
    if runner.privileged_obs_normalizer is not None:
      runner.privileged_obs_normalizer.train()
    runner.save(str(args.log_dir / "model_0.pt"), infos={"residual_control": launch})
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
    runner.save(
      str(args.log_dir / f"model_{args.updates}_final.pt"),
      infos={"residual_control": launch},
    )
    print(f"{args.log_tag}_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
