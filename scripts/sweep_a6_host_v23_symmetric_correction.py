"""Scan static symmetric sagittal corrections after A6 obstacle escape."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from scripts.eval_a6_23_formal import abnormal_termination
from scripts.train_a6_host_v6 import action_scale_indices
from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime import (
  _a6_completion_clearance,
  _a6_ground_force,
  _a6_height,
  _a6_upright,
  a6_env_cfg,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.mdp.a6_geometry import DIRECTIONS
from src.tasks.host_recovery.mdp.observations import UNACTUATED_STEPS
from src.tasks.host_recovery.residual_control import (
  FrozenRecoveryPostEscapeCorrectionVecEnvWrapper,
  post_escape_correction_gate,
)
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


SCENE_NAMES = ("vertical_plate", "free_plate")
DEFAULT_GRID = (-0.15, -0.075, 0.0, 0.075, 0.15)
CONTROL_DT = 0.02


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _paired_layout(num_per_combo: int, device: torch.device) -> dict[str, torch.Tensor]:
  if num_per_combo != 32:
    raise ValueError("V23 paired layout currently requires 32 environments per combo")
  local = torch.arange(num_per_combo, device=device)
  within_scene = local % 16
  return {
    "scene": 1 + (local >= 16).long(),
    "direction": within_scene // 4,
    "source": ((within_scene % 4) == 3).long(),
    "stratum": torch.zeros(num_per_combo, dtype=torch.long, device=device),
  }


def _configure_paired_resets(
  env: ManagerBasedRlEnv,
  combo_count: int,
  num_per_combo: int,
) -> tuple[float, float]:
  layout = _paired_layout(num_per_combo, env.device)
  for name, values in layout.items():
    getattr(env, f"_a6_{name}").copy_(values.repeat(combo_count))

  low_rng_state = env._a6_low_rng.get_state()
  natural_rng_state = env._a6_nat_rng.get_state()
  dynamics_rng_state = env._a6_dynamics_rng.get_state()
  for combo_id in range(combo_count):
    start = combo_id * num_per_combo
    ids = torch.arange(start, start + num_per_combo, device=env.device)
    env._a6_low_rng.set_state(low_rng_state)
    env._a6_nat_rng.set_state(natural_rng_state)
    env._a6_dynamics_rng.set_state(dynamics_rng_state)
    env._reset_idx(ids)

  robot = env.scene["robot"]
  relative_root_pos = robot.data.root_link_pos_w - env.scene.env_origins
  initial_state = torch.cat(
    (
      relative_root_pos,
      robot.data.root_link_quat_w,
      robot.data.root_link_lin_vel_w,
      robot.data.root_link_ang_vel_w,
      robot.data.joint_pos,
      robot.data.joint_vel,
    ),
    dim=-1,
  ).view(combo_count, num_per_combo, -1)
  paired_state_error = float(
    (initial_state - initial_state[:1]).abs().amax().item()
  )

  obs = env.observation_manager.compute()["actor"]
  paired_observation_error = float(
    (obs.view(combo_count, num_per_combo, -1) - obs[:num_per_combo]).abs().amax()
  )
  return paired_state_error, paired_observation_error


def _static_actions(
  env: ManagerBasedRlEnv,
  wrapper: FrozenRecoveryPostEscapeCorrectionVecEnvWrapper,
  combos: list[tuple[float, float, float]],
  num_per_combo: int,
) -> torch.Tensor:
  action_dim = env.action_manager.get_term("joint_pos").action_dim
  actions = torch.zeros((env.num_envs, action_dim), device=env.device)
  pair_ids = wrapper._pair_action_ids
  for combo_id, values in enumerate(combos):
    ids = slice(combo_id * num_per_combo, (combo_id + 1) * num_per_combo)
    for pair_index, value in enumerate(values):
      actions[ids, pair_ids[pair_index]] = value
  return actions


def _primary_condition(env: ManagerBasedRlEnv) -> torch.Tensor:
  height = _a6_height(env)
  upright = _a6_upright(env)
  foot_load = _a6_ground_force(env, "quality_feet")[..., 2].abs().amin(-1)
  return (height >= 1.15) & (upright >= 0.90) & (foot_load > 20.0)


def _escape_condition(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (env._a6_scene > 0) & (_a6_completion_clearance(env) >= 0.04)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float | None:
  if not bool(mask.any()):
    return None
  return float(value[mask].float().mean())


def _combo_row(
  combo_id: int,
  combo: tuple[float, float, float],
  ids: torch.Tensor,
  env: ManagerBasedRlEnv,
  primary_success: torch.Tensor,
  escaped_seen: torch.Tensor,
  abnormal: torch.Tensor,
  other_early_done: torch.Tensor,
  final_height: torch.Tensor,
  final_upright: torch.Tensor,
  final_min_foot_load: torch.Tensor,
  post_escape_height_max: torch.Tensor,
  height_upright_hold_max: torch.Tensor,
  primary_hold_max: torch.Tensor,
  post_escape_steps: torch.Tensor,
  combined_clip_sum: torch.Tensor,
) -> dict[str, object]:
  escaped = escaped_seen[ids]
  escaped_ids = ids[escaped]
  steps = post_escape_steps[escaped_ids].clamp_min(1.0)
  row: dict[str, object] = {
    "combo_id": combo_id,
    "hip_pitch": combo[0],
    "knee": combo[1],
    "ankle_pitch": combo[2],
    "n": int(ids.numel()),
    "primary": int(primary_success[ids].sum()),
    "escape": int(escaped.sum()),
    "abnormal": int(abnormal[ids].sum()),
    "other_early_done": int(other_early_done[ids].sum()),
    "post_escape_max_head_ge_1_15": int(
      (post_escape_height_max[ids] >= 1.15).sum()
    ),
    "post_escape_height_upright_nonzero": int(
      (height_upright_hold_max[ids] > 0).sum()
    ),
    "post_escape_height_upright_hold_3s": int(
      (height_upright_hold_max[ids] >= 150).sum()
    ),
    "post_escape_primary_hold_nonzero": int(
      (primary_hold_max[ids] > 0).sum()
    ),
    "post_escape_primary_hold_3s": int(
      (primary_hold_max[ids] >= 150).sum()
    ),
    "escaped_final_head_mean_m": _masked_mean(final_height[ids], escaped),
    "escaped_final_upright_mean": _masked_mean(final_upright[ids], escaped),
    "escaped_final_min_foot_load_mean_n": _masked_mean(
      final_min_foot_load[ids], escaped
    ),
    "escaped_max_head_mean_m": _masked_mean(
      post_escape_height_max[ids], escaped
    ),
    "escaped_height_upright_hold_max_mean_s": (
      _masked_mean(height_upright_hold_max[ids], escaped) * CONTROL_DT
      if bool(escaped.any())
      else None
    ),
    "escaped_height_upright_hold_max_s": (
      float(height_upright_hold_max[escaped_ids].max() * CONTROL_DT)
      if escaped_ids.numel()
      else None
    ),
    "escaped_primary_hold_max_mean_s": (
      _masked_mean(primary_hold_max[ids], escaped) * CONTROL_DT
      if bool(escaped.any())
      else None
    ),
    "escaped_primary_hold_max_s": (
      float(primary_hold_max[escaped_ids].max() * CONTROL_DT)
      if escaped_ids.numel()
      else None
    ),
    "combined_action_clip_fraction": (
      float((combined_clip_sum[escaped_ids] / steps).mean())
      if escaped_ids.numel()
      else None
    ),
  }
  by_scene = {}
  for scene_id, scene_name in ((1, "vertical_plate"), (2, "free_plate")):
    scene_ids = ids[env._a6_scene[ids] == scene_id]
    by_scene[scene_name] = {
      "n": int(scene_ids.numel()),
      "primary": int(primary_success[scene_ids].sum()),
      "escape": int(escaped_seen[scene_ids].sum()),
      "abnormal": int(abnormal[scene_ids].sum()),
    }
  row["by_scene"] = by_scene
  return row


def _rank_key(row: dict[str, object]) -> tuple[float, ...]:
  return (
    float(row["primary"]),
    float(row["post_escape_height_upright_hold_3s"]),
    float(row["post_escape_height_upright_nonzero"]),
    float(row["post_escape_max_head_ge_1_15"]),
    float(row["escaped_max_head_mean_m"] or -1.0),
    float(row["escape"]),
    -float(row["abnormal"]),
    -float(row["combined_action_clip_fraction"] or 0.0),
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-checkpoint", type=Path, required=True)
  parser.add_argument("--recovery-checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--group", choices=("G-", "G+"), required=True)
  parser.add_argument("--num-per-combo", type=int, default=32)
  parser.add_argument("--steps", type=int, default=1150)
  parser.add_argument("--seed", type=int, default=20261018)
  parser.add_argument("--action-scale", type=float, default=0.9323874)
  parser.add_argument("--recovery-scale", type=float, default=0.25)
  parser.add_argument("--grid-values", type=float, nargs="+", default=DEFAULT_GRID)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  if args.output.exists():
    raise FileExistsError(args.output)
  if args.num_per_combo != 32:
    raise ValueError("--num-per-combo must be 32 for the paired formal layout")
  if args.steps != 1150:
    raise ValueError("V23 scan uses the formal 1150-step horizon")

  if args.preflight:
    combos = [(0.0, 0.0, 0.0), (-0.075, -0.075, 0.075)]
  else:
    values = tuple(float(value) for value in args.grid_values)
    combos = list(itertools.product(values, repeat=3))
  combo_count = len(combos)
  num_envs = combo_count * args.num_per_combo
  if num_envs < 32 or num_envs % 32:
    raise ValueError(num_envs)

  cfg = a6_env_cfg(
    play=True,
    seed=args.seed,
    dynamics=False,
    group=args.group,
    bank_split="train",
  )
  cfg.scene.num_envs = num_envs
  cfg.seed = args.seed
  cfg.terminations.pop("invalid_plate", None)
  for event in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
    cfg.events.pop(event, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  try:
    env.reset()
    set_action_scale(env, args.action_scale)
    paired_state_error, paired_observation_error = _configure_paired_resets(
      env, combo_count, args.num_per_combo
    )
    if paired_state_error != 0.0 or paired_observation_error != 0.0:
      raise AssertionError(
        ("paired reset mismatch", paired_state_error, paired_observation_error)
      )

    base_wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    base_runner = HoSTOnPolicyRunner(base_wrapper, asdict(agent), device="cuda:0")
    base_runner.load(
      str(args.base_checkpoint), load_optimizer=False, map_location="cuda:0"
    )
    set_action_scale(env, args.action_scale)
    base_scale_indices = action_scale_indices(
      env, base_runner.obs_normalizer._mean.shape[-1]
    )
    base_runner.obs_normalizer._mean[:, base_scale_indices] = args.action_scale
    base_runner.alg.policy.eval()
    base_runner.obs_normalizer.eval()
    base_policy = base_runner.get_inference_policy(device="cuda:0")

    recovery_runner = HoSTOnPolicyRunner(
      base_wrapper, asdict(agent), device="cuda:0"
    )
    recovery_runner.load(
      str(args.recovery_checkpoint),
      load_optimizer=False,
      map_location="cuda:0",
    )
    recovery_scale_indices = action_scale_indices(
      env, recovery_runner.obs_normalizer._mean.shape[-1]
    )
    recovery_runner.obs_normalizer._mean[
      :, recovery_scale_indices
    ] = args.action_scale
    recovery_runner.alg.policy.eval()
    recovery_runner.obs_normalizer.eval()
    recovery_policy = recovery_runner.get_inference_policy(device="cuda:0")

    wrapper = FrozenRecoveryPostEscapeCorrectionVecEnvWrapper(
      env,
      base_policy=base_policy,
      recovery_policy=recovery_policy,
      recovery_scale=args.recovery_scale,
      correction_scale=1.0,
      clip_actions=agent.clip_actions,
    )
    static_actions = _static_actions(
      env, wrapper, combos, args.num_per_combo
    )
    projected = wrapper.project_correction(static_actions)
    if not torch.equal(projected, static_actions):
      raise AssertionError("static correction projection changed paired actions")
    active_ids = torch.unique(wrapper._pair_action_ids)
    inactive_mask = torch.ones(static_actions.shape[-1], dtype=torch.bool, device=env.device)
    inactive_mask[active_ids] = False
    if float(static_actions[:, inactive_mask].abs().max()) != 0.0:
      raise AssertionError("inactive correction rows are nonzero")

    gate_audit = post_escape_correction_gate(
      torch.tensor([0, 1, 2, 1], device=env.device),
      torch.tensor([True, False, True, True], device=env.device),
      torch.tensor([True, True, True, False], device=env.device),
    ).tolist()
    if gate_audit != [False, False, True, False]:
      raise AssertionError(gate_audit)

    obs = wrapper.get_observations()
    n = env.num_envs
    device = env.device
    primary_success = torch.zeros(n, dtype=torch.bool, device=device)
    escaped_seen = torch.zeros(n, dtype=torch.bool, device=device)
    ever_abnormal = torch.zeros(n, dtype=torch.bool, device=device)
    other_early_done = torch.zeros(n, dtype=torch.bool, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    primary_hold = torch.zeros(n, dtype=torch.long, device=device)
    escape_hold = torch.zeros(n, dtype=torch.long, device=device)
    final_height = torch.zeros(n, device=device)
    final_upright = torch.zeros(n, device=device)
    final_min_foot_load = torch.zeros(n, device=device)
    post_escape_height_max = torch.zeros(n, device=device)
    height_upright_hold = torch.zeros(n, dtype=torch.long, device=device)
    height_upright_hold_max = torch.zeros(n, dtype=torch.long, device=device)
    post_escape_primary_hold = torch.zeros(n, dtype=torch.long, device=device)
    post_escape_primary_hold_max = torch.zeros(n, dtype=torch.long, device=device)
    post_escape_steps = torch.zeros(n, device=device)
    combined_clip_sum = torch.zeros(n, device=device)
    recovery_bitwise_max_error = 0.0
    preescape_correction_max = 0.0
    motor_off_correction_max = 0.0
    postescape_correction_seen = False
    nonzero_combo_mask = static_actions.abs().amax(-1) > 0

    with torch.no_grad():
      for step in range(args.steps):
        expected_recovery = recovery_policy(obs["actor"])
        motor_on_before_step = env.episode_length_buf > UNACTUATED_STEPS
        escaped_before_step = env._a6_escaped.clone()
        expected_correction_gate = post_escape_correction_gate(
          env._a6_scene, escaped_before_step, motor_on_before_step
        )
        obs, _, dones, _ = wrapper.step(static_actions)
        recovery_bitwise_max_error = max(
          recovery_bitwise_max_error,
          float(
            (
              env._a6_recovery_residual_action - expected_recovery
            ).abs().max()
          ),
        )
        gated_correction = env._a6_gated_residual_action
        actual_correction_gate = env._a6_residual_gate.squeeze(-1).bool()
        if not torch.equal(expected_correction_gate, actual_correction_gate):
          raise AssertionError("stored correction gate differs from pre-step gate")
        preescape = (
          (env._a6_scene > 0) & ~escaped_before_step & motor_on_before_step
        )
        motor_off = ~motor_on_before_step
        if bool(preescape.any()):
          preescape_correction_max = max(
            preescape_correction_max,
            float(gated_correction[preescape].abs().max()),
          )
        if bool(motor_off.any()):
          motor_off_correction_max = max(
            motor_off_correction_max,
            float(gated_correction[motor_off].abs().max()),
          )
        active_nonzero = expected_correction_gate & nonzero_combo_mask
        if bool(active_nonzero.any()):
          expected = static_actions[active_nonzero]
          actual = gated_correction[active_nonzero]
          if not torch.equal(expected, actual):
            raise AssertionError(
              ("post-escape correction mismatch", float((expected - actual).abs().max()))
            )
          postescape_correction_seen = True

        abnormal, _, _ = abnormal_termination(env)
        newly_abnormal = abnormal & alive
        ever_abnormal |= newly_abnormal
        early_done = dones.bool() & alive & ~abnormal
        other_early_done |= early_done
        alive &= ~(abnormal | early_done)

        primary_now = _primary_condition(env) & alive
        primary_hold = torch.where(
          primary_now, primary_hold + 1, torch.zeros_like(primary_hold)
        )
        primary_start_step = step + 2 - primary_hold
        primary_success |= (
          alive & (primary_hold >= 150) & (primary_start_step <= 1000)
        )

        escape_now = _escape_condition(env) & alive
        escape_hold = torch.where(
          escape_now, escape_hold + 1, torch.zeros_like(escape_hold)
        )
        escaped_seen |= alive & (escape_hold >= 25)

        height = _a6_height(env)
        upright = _a6_upright(env)
        min_foot_load = _a6_ground_force(
          env, "quality_feet"
        )[..., 2].abs().amin(-1)
        final_height = torch.where(alive, height, final_height)
        final_upright = torch.where(alive, upright, final_upright)
        final_min_foot_load = torch.where(
          alive, min_foot_load, final_min_foot_load
        )
        post_escape_alive = escaped_seen & alive
        post_escape_height_max = torch.where(
          post_escape_alive,
          torch.maximum(post_escape_height_max, height),
          post_escape_height_max,
        )
        height_upright_now = (
          post_escape_alive & (height >= 1.15) & (upright >= 0.90)
        )
        height_upright_hold = torch.where(
          height_upright_now,
          height_upright_hold + 1,
          torch.zeros_like(height_upright_hold),
        )
        height_upright_hold_max = torch.maximum(
          height_upright_hold_max, height_upright_hold
        )
        post_escape_primary_hold = torch.where(
          post_escape_alive & primary_now,
          post_escape_primary_hold + 1,
          torch.zeros_like(post_escape_primary_hold),
        )
        post_escape_primary_hold_max = torch.maximum(
          post_escape_primary_hold_max, post_escape_primary_hold
        )
        active = post_escape_alive.float()
        post_escape_steps += active
        preclip = (
          env._a6_base_action
          + env._a6_gated_recovery_residual_action
          + env._a6_gated_residual_action
        )
        combined_clip_sum += (
          (preclip.abs() > 1.0).float().mean(-1) * active
        )

    if recovery_bitwise_max_error != 0.0:
      raise AssertionError(("frozen recovery mismatch", recovery_bitwise_max_error))
    if preescape_correction_max != 0.0 or motor_off_correction_max != 0.0:
      raise AssertionError(
        ("correction gate leak", preescape_correction_max, motor_off_correction_max)
      )
    if args.preflight and not postescape_correction_seen:
      raise AssertionError("preflight did not observe an escaped nonzero-correction world")
    finite_tensors = (
      final_height,
      final_upright,
      final_min_foot_load,
      post_escape_height_max,
      env._a6_base_action,
      env._a6_gated_recovery_residual_action,
      env._a6_gated_residual_action,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_tensors):
      raise FloatingPointError("non-finite V23 rollout value")

    rows = []
    for combo_id, combo in enumerate(combos):
      ids = torch.arange(
        combo_id * args.num_per_combo,
        (combo_id + 1) * args.num_per_combo,
        device=device,
      )
      rows.append(
        _combo_row(
          combo_id,
          combo,
          ids,
          env,
          primary_success,
          escaped_seen,
          ever_abnormal,
          other_early_done,
          final_height,
          final_upright,
          final_min_foot_load,
          post_escape_height_max,
          height_upright_hold_max,
          post_escape_primary_hold_max,
          post_escape_steps,
          combined_clip_sum,
        )
      )
    ranked = sorted(rows, key=_rank_key, reverse=True)
    promising = [
      row
      for row in rows
      if int(row["primary"]) > 0
      or int(row["post_escape_height_upright_nonzero"]) > 0
      or int(row["post_escape_max_head_ge_1_15"]) > 0
    ]

    cpu_layout = _paired_layout(32, torch.device("cpu"))
    initial_layout = {
      "scene": [int(value) for value in cpu_layout["scene"]],
      "direction": [int(value) for value in cpu_layout["direction"]],
      "source": [int(value) for value in cpu_layout["source"]],
    }
    summary = {
      "method": "V23 static post-escape symmetric sagittal action response scan",
      "adapter_label": "29-DoF HoST A6 adapter",
      "native_scale_diagnostic": True,
      "formal_0_25_claim": False,
      "group": args.group,
      "preflight": args.preflight,
      "base_checkpoint": str(args.base_checkpoint.resolve()),
      "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
      "recovery_checkpoint": str(args.recovery_checkpoint.resolve()),
      "recovery_checkpoint_sha256": sha256_file(args.recovery_checkpoint),
      "action_scale": args.action_scale,
      "recovery_scale": args.recovery_scale,
      "correction_values_are_actual_additive_actions": True,
      "grid_values": list(args.grid_values),
      "combo_count": combo_count,
      "num_per_combo": args.num_per_combo,
      "num_envs": num_envs,
      "steps": args.steps,
      "seconds": args.steps * CONTROL_DT,
      "seed": args.seed,
      "paired_layout": initial_layout,
      "paired_initial_state_max_error": paired_state_error,
      "paired_initial_actor_observation_max_error": paired_observation_error,
      "gate_truth": gate_audit,
      "frozen_recovery_bitwise_max_error": recovery_bitwise_max_error,
      "preescape_correction_max_abs": preescape_correction_max,
      "motor_off_correction_max_abs": motor_off_correction_max,
      "postescape_nonzero_correction_seen": postescape_correction_seen,
      "total_primary": int(primary_success.sum()),
      "total_escape": int(escaped_seen.sum()),
      "total_abnormal": int(ever_abnormal.sum()),
      "promising_combo_count": len(promising),
      "promising_combo_ids": [int(row["combo_id"]) for row in promising],
      "best_combos": ranked[:10],
    }

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
      json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "combos.jsonl").open("w", encoding="utf-8") as stream:
      for row in rows:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.output / "per_env.jsonl").open("w", encoding="utf-8") as stream:
      for env_id in range(n):
        combo_id = env_id // args.num_per_combo
        row = {
          "env_id": env_id,
          "combo_id": combo_id,
          "local_pair_id": env_id % args.num_per_combo,
          "scene": SCENE_NAMES[int(env._a6_scene[env_id]) - 1],
          "direction": DIRECTIONS[int(env._a6_direction[env_id])],
          "source": "procedural" if int(env._a6_source[env_id]) else "natural",
          "hip_pitch": combos[combo_id][0],
          "knee": combos[combo_id][1],
          "ankle_pitch": combos[combo_id][2],
          "primary": bool(primary_success[env_id]),
          "escape": bool(escaped_seen[env_id]),
          "abnormal": bool(ever_abnormal[env_id]),
          "other_early_done": bool(other_early_done[env_id]),
          "final_head_m": float(final_height[env_id]),
          "final_upright": float(final_upright[env_id]),
          "final_min_foot_load_n": float(final_min_foot_load[env_id]),
          "post_escape_max_head_m": float(post_escape_height_max[env_id]),
          "height_upright_hold_max_s": float(
            height_upright_hold_max[env_id] * CONTROL_DT
          ),
          "primary_hold_max_s": float(
            post_escape_primary_hold_max[env_id] * CONTROL_DT
          ),
        }
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
      "files": {
        name: sha256_file(args.output / name)
        for name in ("summary.json", "combos.jsonl", "per_env.jsonl")
      }
    }
    (args.output / "manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("A6_V23_SWEEP_COMPLETE", json.dumps(summary), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
