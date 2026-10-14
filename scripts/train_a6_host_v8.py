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
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from scripts.a6_recovery_bank import (
  activate_a6_recovery_resets,
  mixed_a6_recovery_reset,
)
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
from src.tasks.host_recovery.mdp.metrics import HoSTMetrics
from src.tasks.host_recovery.mdp.observations import UNACTUATED_STEPS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


TRAIN_STD = 0.08
LEARNING_RATE = 2e-5
ACTION_SCALE_DECAY = 0.01
ACTION_SCALE_STANDING_HEIGHT = 0.60
ACTION_SCALE_STANDING_FRACTION = 0.90
RETENTION_UPRIGHT = 0.80
RETENTION_STEP_FRACTION = 0.80


def _ensure_retention_state(env: ManagerBasedRlEnv) -> None:
  if not hasattr(env, "_a6_retention_good_steps"):
    env._a6_retention_good_steps = torch.zeros(
      env.num_envs, device=env.device
    )
    env._a6_retention_control_steps = torch.zeros(
      env.num_envs, device=env.device
    )
    env._a6_last_retention_cohort_fraction = 0.0
    env._a6_last_retention_step_fraction = 0.0


def near_retention_reward(
  env: ManagerBasedRlEnv,
  track_retention: bool = False,
  gate_height: float = ACTION_SCALE_STANDING_HEIGHT,
  gate_upright: float = RETENTION_UPRIGHT,
) -> torch.Tensor:
  raw_height = _a6_height(env)
  raw_upright = _a6_upright(env)
  if track_retention:
    _ensure_retention_state(env)
    controlled = env.episode_length_buf > UNACTUATED_STEPS
    retained = (
      controlled
      & (raw_height > gate_height)
      & (raw_upright > gate_upright)
    )
    env._a6_retention_good_steps += retained.float()
    env._a6_retention_control_steps += controlled.float()
  height = smooth_gate(raw_height, 0.72, 1.08)
  upright = smooth_gate(raw_upright, 0.55, 0.90)
  return height * upright


def early_recovery_reward(
  env: ManagerBasedRlEnv,
  window_steps: int = 120,
) -> torch.Tensor:
  controlled_step = env.episode_length_buf - UNACTUATED_STEPS
  active = (controlled_step > 0) & (controlled_step <= window_steps)
  height = smooth_gate(_a6_height(env), 0.62, 0.98)
  upright = smooth_gate(_a6_upright(env), 0.62, 0.88)
  progress = controlled_step.float().clamp(1, window_steps) / window_steps
  time_weight = 1.0 - 0.5 * progress
  return active.float() * height * upright * time_weight


def near_action_scale_decay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  standing_height: float = ACTION_SCALE_STANDING_HEIGHT,
  min_fraction: float = ACTION_SCALE_STANDING_FRACTION,
  decay: float = ACTION_SCALE_DECAY,
  min_scale: float = FORMAL_ACTION_SCALE,
  required_successes: int = 1,
  require_retention: bool = False,
  retention_upright: float = RETENTION_UPRIGHT,
  retention_step_fraction: float = RETENTION_STEP_FRACTION,
  min_scale_hold_steps: int = 0,
  term_name: str = "joint_pos",
) -> torch.Tensor:
  if env_ids is None or len(env_ids) == 0:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  state = HoSTMetrics.get()
  reached = False
  if require_retention:
    _ensure_retention_state(env)
    if state.initialized and state.current_head_height is not None:
      final_retained = (
        (state.current_head_height[env_ids] > standing_height)
        & (_a6_upright(env)[env_ids] > retention_upright)
      )
      total_steps = env._a6_retention_control_steps[env_ids].clamp_min(1.0)
      step_fraction = env._a6_retention_good_steps[env_ids] / total_steps
      episode_retained = final_retained & (
        step_fraction >= retention_step_fraction
      )
      cohort_fraction = float(episode_retained.float().mean())
      env._a6_last_retention_cohort_fraction = cohort_fraction
      env._a6_last_retention_step_fraction = float(step_fraction.mean())
      reached = cohort_fraction >= min_fraction
    env._a6_retention_good_steps[env_ids] = 0.0
    env._a6_retention_control_steps[env_ids] = 0.0
  elif state.initialized and state.last_episode_head_height is not None:
    standing = state.last_episode_head_height[env_ids] > standing_height
    reached = bool(standing.float().mean() >= min_fraction)

  current_step = int(env.common_step_counter)
  last_change_step = int(
    getattr(env, "_a6_scale_last_change_step", current_step)
  )
  hold_complete = current_step - last_change_step >= min_scale_hold_steps
  reached = reached and hold_complete

  success_streak = int(getattr(env, "_a6_scale_success_streak", 0))
  success_streak = success_streak + 1 if reached else 0
  env._a6_scale_success_streak = success_streak

  rescale = getattr(env, "_host_action_rescale", None)
  if rescale is None:
    term = env.action_manager.get_term(term_name)
    scale = term.cfg.scale
    value = (
      float(sum(scale.values()) / len(scale))
      if isinstance(scale, dict)
      else float(scale)
    )
    rescale = torch.full((env.num_envs, 1), value, device=env.device)
    env._host_action_rescale = rescale

  if success_streak >= required_successes:
    rescale[env_ids] = (rescale[env_ids] - decay).clamp(min=min_scale)
    env._a6_scale_success_streak = 0
    env._a6_scale_last_change_step = current_step

  term = env.action_manager.get_term(term_name)
  new_value = float(rescale.mean())
  if isinstance(term.cfg.scale, dict):
    term.cfg.scale = {key: new_value for key in term.cfg.scale}
  else:
    term.cfg.scale = new_value
  if isinstance(term._scale, torch.Tensor):
    term._scale[:] = new_value
  else:
    term._scale = new_value
  return rescale.mean()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=1800)
  parser.add_argument("--seed", type=int, default=20261013)
  parser.add_argument("--preflight", action="store_true")
  parser.add_argument("--recovery-bank", type=Path)
  parser.add_argument("--recovery-fraction", type=float, default=0.0)
  parser.add_argument("--recovery-successful-only", action="store_true")
  parser.add_argument("--action-scale-decay", type=float, default=ACTION_SCALE_DECAY)
  parser.add_argument(
    "--standing-height", type=float, default=ACTION_SCALE_STANDING_HEIGHT
  )
  parser.add_argument(
    "--standing-fraction", type=float, default=ACTION_SCALE_STANDING_FRACTION
  )
  parser.add_argument("--required-successes", type=int, default=1)
  parser.add_argument("--require-retention-gate", action="store_true")
  parser.add_argument(
    "--retention-upright", type=float, default=RETENTION_UPRIGHT
  )
  parser.add_argument(
    "--retention-step-fraction",
    type=float,
    default=RETENTION_STEP_FRACTION,
  )
  parser.add_argument("--retention-reward-weight", type=float, default=4.0)
  parser.add_argument("--early-recovery-weight", type=float, default=0.0)
  parser.add_argument("--early-recovery-window", type=int, default=120)
  parser.add_argument(
    "--actor-scope", choices=("full", "final-layer"), default="full"
  )
  parser.add_argument("--trainable-output-rows", type=int, nargs="*")
  parser.add_argument("--max-weight-delta", type=float, default=0.0)
  parser.add_argument("--max-bias-delta", type=float, default=0.0)
  parser.add_argument("--frozen-std", type=float)
  parser.add_argument("--reset-optimizer-state", action="store_true")
  parser.add_argument("--min-scale-hold-steps", type=int, default=0)
  parser.add_argument("--train-std", type=float, default=TRAIN_STD)
  parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
  parser.add_argument("--resume-training-state", action="store_true")
  parser.add_argument(
    "--method-name",
    default="gated near-standing action-scale homotopy",
  )
  parser.add_argument(
    "--experiment-name",
    default="g1_host_a6_v8_near_scale_homotopy",
  )
  parser.add_argument("--log-tag", default="A6_V8")
  args = parser.parse_args()

  if args.recovery_bank is None and args.recovery_fraction != 0.0:
    raise ValueError("--recovery-fraction requires --recovery-bank")
  if args.recovery_bank is not None:
    if not args.recovery_bank.is_file():
      raise FileNotFoundError(args.recovery_bank)
    if not 0.0 < args.recovery_fraction < 1.0:
      raise ValueError("mixed recovery training requires fraction in (0, 1)")

  args.log_dir.mkdir(parents=True, exist_ok=False)
  source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  source_action_scale = float(
    source["infos"]["env_state"]["host_action_rescale"].float().mean()
  )

  cfg = a6_env_cfg(seed=args.seed, dynamics=False, group="G+")
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.curriculum["action_scale"].func = near_action_scale_decay
  cfg.curriculum["action_scale"].params = {
    "standing_height": args.standing_height,
    "min_fraction": args.standing_fraction,
    "decay": args.action_scale_decay,
    "min_scale": FORMAL_ACTION_SCALE,
    "required_successes": args.required_successes,
    "require_retention": args.require_retention_gate,
    "retention_upright": args.retention_upright,
    "retention_step_fraction": args.retention_step_fraction,
    "min_scale_hold_steps": args.min_scale_hold_steps,
  }
  cfg.curriculum.pop("pull_force", None)
  cfg.terminations.pop("invalid_plate", None)
  for name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(name, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False
  if args.recovery_bank is not None:
    cfg.events["a6_reset"] = EventTermCfg(
      func=mixed_a6_recovery_reset,
      mode="reset",
      params={
        "bank_split": "train",
        "recovery_bank": str(args.recovery_bank.resolve()),
        "recovery_fraction": args.recovery_fraction,
        "successful_only": args.recovery_successful_only,
      },
    )
    cfg.events["a6_recovery_activate"] = EventTermCfg(
      func=activate_a6_recovery_resets,
      mode="step",
      params={},
    )

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
    func=near_retention_reward,
    weight=args.retention_reward_weight * CONTROL_DT,
    params={
      "track_retention": args.require_retention_gate,
      "gate_height": args.standing_height,
      "gate_upright": args.retention_upright,
    },
  )
  if args.early_recovery_weight > 0.0:
    cfg.rewards["a6_early_recovery"] = RewardTermCfg(
      func=early_recovery_reward,
      weight=args.early_recovery_weight * CONTROL_DT,
      params={"window_steps": args.early_recovery_window},
    )

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = args.experiment_name
  agent.max_iterations = args.updates
  agent.num_steps_per_env = 24
  agent.save_interval = 50
  agent.algorithm.learning_rate = args.learning_rate
  agent.algorithm.schedule = "fixed"
  agent.algorithm.entropy_coef = 0.0
  agent.algorithm.num_learning_epochs = 2
  agent.actor.distribution_cfg.update(
    init_std=args.train_std, max_std=args.train_std
  )
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.reset()
    configure_flat_near_resets(env)
    set_action_scale(env, source_action_scale)

    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    if args.recovery_bank is not None:
      activate_a6_recovery_resets(env)
      env.scene.write_data_to_sim()
      env.sim.forward()
      env.sim.sense()
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), str(args.log_dir), "cuda:0")
    fresh_critic = copy.deepcopy(runner.alg.policy.critic.state_dict())
    fresh_critic_normalizer = copy.deepcopy(
      runner.privileged_obs_normalizer.state_dict()
    )
    runner.load(
      str(args.checkpoint),
      load_optimizer=args.resume_training_state,
      load_cfg={"actor": True, "critic": args.resume_training_state},
      map_location="cuda:0",
    )
    if not args.resume_training_state:
      runner.alg.policy.critic.load_state_dict(fresh_critic)
      runner.privileged_obs_normalizer.load_state_dict(fresh_critic_normalizer)
    if args.reset_optimizer_state:
      runner.alg.optimizer.state.clear()
    runner.current_learning_iteration = 0
    set_action_scale(env, source_action_scale)
    env._a6_scale_success_streak = 0
    env._a6_scale_last_change_step = int(env.common_step_counter)
    for param_group in runner.alg.optimizer.param_groups:
      param_group["lr"] = args.learning_rate

    policy_state = runner.alg.policy.state_dict()
    for name, value in source["actor_state_dict"].items():
      if name == "distribution.std_param":
        continue
      mapped = "actor." + name.removeprefix("mlp.")
      assert torch.equal(policy_state[mapped].detach().cpu(), value.cpu()), mapped
    if not args.resume_training_state:
      assert len(runner.alg.optimizer.state) == 0
    trainable_rows = None
    frozen_rows = None
    if args.actor_scope == "final-layer":
      for parameter in runner.alg.policy.actor.parameters():
        parameter.requires_grad_(False)
      output_layer = runner.alg.policy.actor[6]
      for parameter in output_layer.parameters():
        parameter.requires_grad_(True)
      source_weight = output_layer.weight.detach().clone()
      source_bias = output_layer.bias.detach().clone()
      if args.trainable_output_rows:
        trainable_rows = sorted(set(args.trainable_output_rows))
        assert all(0 <= row < output_layer.out_features for row in trainable_rows)
        frozen_rows = [
          row for row in range(output_layer.out_features) if row not in trainable_rows
        ]
        weight_mask = torch.zeros_like(output_layer.weight)
        bias_mask = torch.zeros_like(output_layer.bias)
        weight_mask[trainable_rows] = 1.0
        bias_mask[trainable_rows] = 1.0
        output_layer.weight.register_hook(lambda gradient: gradient * weight_mask)
        output_layer.bias.register_hook(lambda gradient: gradient * bias_mask)

      def constrain_output_layer(_optimizer, _args, _kwargs) -> None:
        with torch.no_grad():
          if frozen_rows:
            output_layer.weight[frozen_rows] = source_weight[frozen_rows]
            output_layer.bias[frozen_rows] = source_bias[frozen_rows]
          if args.max_weight_delta > 0.0:
            output_layer.weight.copy_(
              source_weight
              + (output_layer.weight - source_weight).clamp(
                -args.max_weight_delta, args.max_weight_delta
              )
            )
          if args.max_bias_delta > 0.0:
            output_layer.bias.copy_(
              source_bias
              + (output_layer.bias - source_bias).clamp(
                -args.max_bias_delta, args.max_bias_delta
              )
            )

      runner.alg.optimizer.register_step_post_hook(constrain_output_layer)
    else:
      assert all(
        parameter.requires_grad
        for parameter in runner.alg.policy.actor.parameters()
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
      if trainable_rows is not None and args.frozen_std is not None:
        runner.alg.policy.std.fill_(args.frozen_std)
        runner.alg.policy.std[trainable_rows] = args.train_std
      else:
        runner.alg.policy.std.fill_(args.train_std)
    runner.alg.policy.std.requires_grad_(False)
    env._a6_adaptation_start_step = int(env.common_step_counter)

    reset_counts = a6_reset_counts(env)
    assert reset_counts["scene"] == [args.num_envs, 0, 0], reset_counts
    assert reset_counts["stratum"] == [0, 0, args.num_envs], reset_counts
    assert reset_counts["source"] == [args.num_envs, 0], reset_counts
    recovery_mask = None
    near_mask = None
    recovery_reset_count = 0
    if args.recovery_bank is not None:
      recovery_mask = env._a6_recovery_reset.clone()
      near_mask = ~recovery_mask
      recovery_reset_count = int(recovery_mask.sum())
      expected = round(args.num_envs * args.recovery_fraction)
      assert recovery_reset_count == expected, (
        recovery_reset_count,
        expected,
      )
      assert not env._a6_recovery_pending.any()
      assert torch.all(
        env.episode_length_buf[recovery_mask] > UNACTUATED_STEPS
      )
      assert torch.all(env.episode_length_buf[near_mask] == 0)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "method": args.method_name,
      "group": "G+",
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "action_scale_start": source_action_scale,
      "action_scale_target": FORMAL_ACTION_SCALE,
      "action_scale_decay": args.action_scale_decay,
      "action_scale_standing_height": args.standing_height,
      "action_scale_standing_fraction": args.standing_fraction,
      "action_scale_required_successes": args.required_successes,
      "action_scale_retention_gate": args.require_retention_gate,
      "action_scale_retention_upright": args.retention_upright,
      "action_scale_retention_step_fraction": (
        args.retention_step_fraction
      ),
      "action_scale_min_hold_steps": args.min_scale_hold_steps,
      "reset_mix": reset_counts,
      "recovery_bank": (
        str(args.recovery_bank.resolve())
        if args.recovery_bank is not None
        else None
      ),
      "recovery_fraction": args.recovery_fraction,
      "recovery_successful_only": args.recovery_successful_only,
      "recovery_reset_count": recovery_reset_count,
      "recovery_motor_off_skipped": args.recovery_bank is not None,
      "nominal_dynamics": True,
      "observation_noise": False,
      "invalid_plate_termination": False,
      "train_std": args.train_std,
      "learning_rate": args.learning_rate,
      "learning_epochs": 2,
      "actor_trainable_scope": args.actor_scope,
      "trainable_output_rows": trainable_rows,
      "max_weight_delta": args.max_weight_delta,
      "max_bias_delta": args.max_bias_delta,
      "frozen_std": args.frozen_std,
      "optimizer_state_reset": args.reset_optimizer_state,
      "actor_normalizer": "frozen except scale means synchronized to runtime scale",
      "actor_normalizer_scale_indices": scale_indices.tolist(),
      "legacy_style_target_rewards": "disabled",
      "obstacle_rewards": "disabled during acquisition",
      "fresh_critic": not args.resume_training_state,
      "fresh_optimizer": not args.resume_training_state,
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

    observations = wrapper.get_observations()
    normalized = runner.obs_normalizer(observations["actor"])
    probe = runner.alg.policy.actor(normalized)
    runner.alg.optimizer.zero_grad(set_to_none=True)
    probe.square().mean().backward()
    if args.actor_scope == "final-layer":
      assert runner.alg.policy.actor[0].weight.grad is None
      assert runner.alg.policy.actor[0].bias.grad is None
      actor_layers = (runner.alg.policy.actor[6],)
    else:
      actor_layers = (runner.alg.policy.actor[0], runner.alg.policy.actor[6])
    assert all(layer.weight.grad.abs().sum() > 0 for layer in actor_layers)
    assert all(layer.bias.grad.abs().sum() > 0 for layer in actor_layers)
    if trainable_rows is not None:
      assert output_layer.weight.grad[frozen_rows].abs().sum() == 0
      assert output_layer.bias.grad[frozen_rows].abs().sum() == 0
      assert output_layer.weight.grad[trainable_rows].abs().sum() > 0
      assert output_layer.bias.grad[trainable_rows].abs().sum() > 0
    runner.alg.optimizer.zero_grad(set_to_none=True)

    reward_abs_sum = torch.zeros(
      len(env.reward_manager.active_terms), device=env.device
    )
    policy = runner.get_inference_policy(device="cuda:0")
    rollout_steps = 1100 if args.preflight else 200
    start_scale = float(env._host_action_rescale.mean())
    recovery_baseline = {}
    probe_alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)

    def capture_recovery_baseline(step: int) -> None:
      if recovery_mask is None or near_mask is None:
        return
      height = _a6_height(env)
      upright = _a6_upright(env)
      for label, mask in (("recovery", recovery_mask), ("near", near_mask)):
        alive = mask & probe_alive
        retained = (height[alive] > args.standing_height) & (
          upright[alive] > args.retention_upright
        )
        recovery_baseline[f"{label}_step{step}"] = {
          "count": int(mask.sum()),
          "alive_count": int(alive.sum()),
          "survival_fraction": float(alive.sum() / mask.sum()),
          "retained_fraction": (
            float(retained.float().mean()) if alive.any() else 0.0
          ),
          "height_mean": float(height[alive].mean()) if alive.any() else 0.0,
          "upright_mean": (
            float(upright[alive].mean()) if alive.any() else 0.0
          ),
        }

    capture_recovery_baseline(0)
    with torch.inference_mode():
      for rollout_step in range(1, rollout_steps + 1):
        actions = policy(observations["actor"])
        observations, rewards, terminated, timeouts = wrapper.step(actions)
        probe_alive &= ~(terminated | timeouts)
        reward_abs_sum += env.reward_manager._step_reward.abs().mean(dim=0)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
        if rollout_step in (100, 400):
          capture_recovery_baseline(rollout_step)
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
    if args.early_recovery_weight > 0.0:
      value = reward_abs_mean["a6_early_recovery"]
      assert 0.0 < value < 0.20, ("a6_early_recovery", value)
    end_scale = float(env._host_action_rescale.mean())
    metric_state = HoSTMetrics.get()
    assert metric_state.last_episode_head_height is not None
    last_standing_fraction = float(
      (
        metric_state.last_episode_head_height
        > args.standing_height
      ).float().mean()
    )
    synced_mean = runner.obs_normalizer._mean[:, scale_indices]
    assert torch.allclose(
      synced_mean, torch.full_like(synced_mean, end_scale), atol=1e-6
    )
    if args.preflight:
      assert FORMAL_ACTION_SCALE <= end_scale <= start_scale
      assert end_scale >= start_scale - args.action_scale_decay - 1e-5
    print(
      f"{args.log_tag}_PREFLIGHT_REWARD_ABS",
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
      f"{args.log_tag}_PREFLIGHT_SCALE",
      json.dumps(
        {
          "start": start_scale,
          "end": end_scale,
          "last_standing_fraction": last_standing_fraction,
          "last_retention_cohort_fraction": float(
            getattr(env, "_a6_last_retention_cohort_fraction", 0.0)
          ),
          "last_retention_step_fraction": float(
            getattr(env, "_a6_last_retention_step_fraction", 0.0)
          ),
        }
      ),
      flush=True,
    )
    if recovery_baseline:
      print(
        f"{args.log_tag}_RECOVERY_BASELINE",
        json.dumps(recovery_baseline),
        flush=True,
      )
    print(f"{args.log_tag}_PREFLIGHT_PASS", json.dumps(launch), flush=True)
    if args.preflight:
      return

    env._a6_scale_success_streak = 0
    env._a6_scale_last_change_step = int(env.common_step_counter)
    if args.require_retention_gate:
      _ensure_retention_state(env)
      env._a6_retention_good_steps.zero_()
      env._a6_retention_control_steps.zero_()

    runner.alg.policy.train()
    runner.obs_normalizer.train()
    if runner.privileged_obs_normalizer is not None:
      runner.privileged_obs_normalizer.train()
    runner.save(str(args.log_dir / "model_0.pt"))
    assert runner.alg.policy.training
    assert runner.obs_normalizer.training
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print(f"{args.log_tag}_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
