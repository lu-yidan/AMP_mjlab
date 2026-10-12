"""Train A6 with a forced, reset-synchronized action-scale bridge."""

from __future__ import annotations

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
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime import (
  A6_POSITIVE_NAMES,
  SCENE_NAMES,
  _a6_ground_force,
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


FORMAL_ACTION_SCALE = 0.25
BRIDGE_HOLD_STEPS = 500
BRIDGE_DURATION_STEPS = 17_000
TRAIN_STD = 0.12
MAX_WEIGHT_DELTA = 0.05
MAX_BIAS_DELTA = 0.25
CONTROL_DT = 0.02


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth_gate(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
  scaled = ((value - low) / (high - low)).clamp(0.0, 1.0)
  return scaled.square() * (3.0 - 2.0 * scaled)


def height_margin_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return smooth_gate(_a6_height(env), 1.04, 1.16)


def primary_stance_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  height = smooth_gate(_a6_height(env), 1.07, 1.16)
  upright = smooth_gate(_a6_upright(env), 0.84, 0.92)
  load = _a6_ground_force(env, "quality_feet")[..., 2].abs().amin(-1)
  support = smooth_gate(load, 10.0, 30.0)
  return height * upright * support


def scheduled_action_scale_bridge(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  start_scale: float,
  min_scale: float = FORMAL_ACTION_SCALE,
  hold_steps: int = BRIDGE_HOLD_STEPS,
  duration_steps: int = BRIDGE_DURATION_STEPS,
  term_name: str = "joint_pos",
) -> torch.Tensor:
  """Linearly bridge action scale on reset, independent of task success."""
  if env_ids is None or len(env_ids) == 0:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  elapsed = max(int(env.common_step_counter) - hold_steps, 0)
  progress = min(elapsed / duration_steps, 1.0)
  scheduled_scale = start_scale + (min_scale - start_scale) * progress

  rescale = getattr(env, "_host_action_rescale", None)
  if rescale is None:
    rescale = torch.full(
      (env.num_envs, 1), start_scale, device=env.device, dtype=torch.float
    )
    env._host_action_rescale = rescale
  rescale[env_ids] = scheduled_scale

  action_term = env.action_manager.get_term(term_name)
  scale_cfg = action_term.cfg.scale
  scale_value = float(rescale.mean())
  if isinstance(scale_cfg, dict):
    action_term.cfg.scale = {key: scale_value for key in scale_cfg}
  else:
    action_term.cfg.scale = scale_value
  if isinstance(action_term._scale, torch.Tensor):
    action_term._scale[:] = scale_value
  else:
    action_term._scale = scale_value
  return rescale.mean()


def configure_formal_low(env: ManagerBasedRlEnv) -> None:
  env._a6_stratum.zero_()
  env._a6_source.zero_()
  for scene_id in range(len(SCENE_NAMES)):
    for direction_id in range(len(a6_geometry.DIRECTIONS)):
      ids = torch.nonzero(
        (env._a6_scene == scene_id)
        & (env._a6_direction == direction_id),
        as_tuple=False,
      ).flatten()
      procedural = round(int(ids.numel()) * 0.25)
      env._a6_source[ids[:procedural]] = 1
  env._reset_idx(torch.arange(env.num_envs, device=env.device))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=1600)
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
  cfg.curriculum["action_scale"].func = scheduled_action_scale_bridge
  cfg.curriculum["action_scale"].params = {
    "start_scale": source_action_scale,
    "min_scale": FORMAL_ACTION_SCALE,
    "hold_steps": BRIDGE_HOLD_STEPS,
    "duration_steps": BRIDGE_DURATION_STEPS,
  }
  cfg.curriculum.pop("pull_force", None)
  cfg.terminations.pop("invalid_plate", None)
  for name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(name, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False
  for name in A6_POSITIVE_NAMES:
    cfg.rewards[f"a6_{name}"].params["dense_guide"] = False
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
  agent.experiment_name = "g1_host_a6_v3_forced_bridge"
  agent.max_iterations = args.updates
  agent.num_steps_per_env = 24
  agent.save_interval = 100
  agent.algorithm.learning_rate = 5e-5
  agent.algorithm.schedule = "fixed"
  agent.algorithm.entropy_coef = 0.0
  agent.algorithm.num_learning_epochs = 4
  agent.actor.distribution_cfg.update(init_std=TRAIN_STD, max_std=TRAIN_STD)
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.reset()
    configure_formal_low(env)
    set_action_scale(env, source_action_scale)

    all_env_ids = torch.arange(env.num_envs, device=env.device)
    initial_counter = int(env.common_step_counter)
    schedule_checks = {}
    for label, step in (
      ("start", 0),
      ("midpoint", BRIDGE_HOLD_STEPS + BRIDGE_DURATION_STEPS // 2),
      ("target", BRIDGE_HOLD_STEPS + BRIDGE_DURATION_STEPS),
    ):
      env.common_step_counter = step
      value = scheduled_action_scale_bridge(
        env,
        all_env_ids,
        start_scale=source_action_scale,
        min_scale=FORMAL_ACTION_SCALE,
        hold_steps=BRIDGE_HOLD_STEPS,
        duration_steps=BRIDGE_DURATION_STEPS,
      )
      schedule_checks[label] = float(value)
    expected_midpoint = 0.5 * (source_action_scale + FORMAL_ACTION_SCALE)
    assert abs(schedule_checks["start"] - source_action_scale) < 1e-6
    assert abs(schedule_checks["midpoint"] - expected_midpoint) < 1e-6
    assert abs(schedule_checks["target"] - FORMAL_ACTION_SCALE) < 1e-6
    env.common_step_counter = initial_counter
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
    runner.obs_normalizer.until = int(runner.obs_normalizer.count.item())

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

    def clamp_actor_after_step(_optimizer, _args, _kwargs) -> None:
      with torch.no_grad():
        output_layer.weight.copy_(
          source_weight
          + (output_layer.weight - source_weight).clamp(
            -MAX_WEIGHT_DELTA, MAX_WEIGHT_DELTA
          )
        )
        output_layer.bias.copy_(
          source_bias
          + (output_layer.bias - source_bias).clamp(
            -MAX_BIAS_DELTA, MAX_BIAS_DELTA
          )
        )

    runner.alg.optimizer.register_step_post_hook(clamp_actor_after_step)
    with torch.no_grad():
      runner.alg.policy.std.fill_(TRAIN_STD)
    runner.alg.policy.std.requires_grad_(False)
    env._a6_adaptation_start_step = int(env.common_step_counter)
    set_action_scale(env, source_action_scale)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "method": "forced reset-synchronized action-scale bridge",
      "group": "G+",
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "formal_low_resets": True,
      "nominal_dynamics": True,
      "observation_noise": False,
      "invalid_plate_termination": False,
      "common_task_alpha": 1.0,
      "action_scale_start": source_action_scale,
      "action_scale_target": FORMAL_ACTION_SCALE,
      "bridge_hold_steps": BRIDGE_HOLD_STEPS,
      "bridge_duration_steps": BRIDGE_DURATION_STEPS,
      "schedule_checks": schedule_checks,
      "train_std": TRAIN_STD,
      "learning_rate": 5e-5,
      "learning_epochs": 4,
      "actor_trainable_scope": "bounded final output layer",
      "max_weight_delta": MAX_WEIGHT_DELTA,
      "max_bias_delta": MAX_BIAS_DELTA,
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "reset_counts": a6_reset_counts(env),
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
    assert reward_abs_mean["a6_height_margin"] > 0.0
    assert reward_abs_mean["a6_primary_stance"] > 0.0
    assert reward_abs_mean["a6_height_margin"] < 0.05
    assert reward_abs_mean["a6_primary_stance"] < 0.05
    assert a6_reset_counts(env)["stratum"] == [args.num_envs, 0, 0]
    assert "invalid_plate" not in env.termination_manager.active_terms
    assert float(env._host_action_rescale.amin()) >= FORMAL_ACTION_SCALE
    assert float(env._host_action_rescale.amax()) <= source_action_scale
    print(
      "A6_V3_PREFLIGHT_REWARD_ABS",
      json.dumps(
        {
          name: reward_abs_mean[name]
          for name in (
            "a6_stage_pose",
            "a6_height",
            "a6_upright",
            "a6_height_margin",
            "a6_primary_stance",
            "plate_geometry_progress",
            "plate_clearance",
            "plate_completion",
            "path_quiet_feet",
            "path_joint_stall",
          )
        }
      ),
      flush=True,
    )
    print("A6_V3_PREFLIGHT_PASS", json.dumps(launch), flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("A6_V3_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
