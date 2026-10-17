"""Formal 23-DoF A6 evaluation: 2048 single-attempt worlds, 23 s, no 10 s timeout.

The success predicate is intentionally separate from the historical escape
predicate. A trial succeeds when, before 20 s, the robot continuously satisfies
head height >= 1.15 m, uprightness >= 0.90, and both ankle loads > 20 N for 3 s,
and is then observed through 23 s. Escape is a diagnostic only and is computed
with the frozen A6 clearance geometry at 4 cm for 0.5 s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.a6_runtime_23 import (
  SCENE_NAMES,
  _a6_completion_clearance,
  _a6_ground_force,
  _a6_height,
  _a6_upright,
  a6_env_cfg,
  a6_reset_counts,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.mdp.a6_geometry import DIRECTIONS
from src.tasks.host_recovery.residual_control import (
  FrozenRecoveryPostEscapeCorrectionVecEnvWrapper,
  SymmetricSagittalCorrectionProjection,
  TeacherResidualVecEnvWrapper,
)
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


BANK_FILES = (
  "outputs/multiterrain_bank/train.npz",
  "datasets/reset_banks/natural_curriculum_v1/train.npz",
  "datasets/reset_banks/procedural_low_v1/train.npz",
)

REPORT_TITLE = "23-DoF A6 formal evaluation"
COMPLETE_MARKER = "A6_23_FORMAL_EVAL_COMPLETE"


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
  array = value.detach().cpu().contiguous().numpy()
  return hashlib.sha256(array.tobytes()).hexdigest()


def configure_all_low(env: ManagerBasedRlEnv) -> None:
  """Force the final protocol's low natural/procedural initial-state split.

  Training keeps a 40/40/20 low/middle/near split inside flat scenes. The
  formal report requires every scene/direction cell to draw only from the low
  bank, with 75% natural and 25% procedural rows.
  """
  env._a6_stratum[:] = 0
  env._a6_source[:] = 0
  for scene_id in range(len(SCENE_NAMES)):
    for direction_id in range(len(DIRECTIONS)):
      ids = torch.nonzero(
        (env._a6_scene == scene_id)
        & (env._a6_direction == direction_id)
        & (env._a6_stratum == 0),
        as_tuple=False,
      ).flatten()
      procedural = round(int(ids.numel()) * 0.25)
      env._a6_source[ids[:procedural]] = 1


def abnormal_termination(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return force/depth-based physical aborts using the formal thresholds."""
  active = env._a6_scene > 0
  which = env._a6_scene == 2
  free = env.scene["free_contact"].data
  guided = env.scene["guided_contact"].data
  force = torch.where(
    which,
    free.force.norm(dim=-1).amax(-1),
    guided.force.norm(dim=-1).amax(-1),
  )
  depth = torch.where(
    which,
    free.dist.amin(-1),
    guided.dist.amin(-1),
  )
  abnormal = active & ((force > 5000.0) | (depth < -0.05))
  return abnormal, force, depth


def primary_condition(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Head height, uprightness and both-ankle load from the formal protocol."""
  height = _a6_height(env)
  upright = _a6_upright(env)
  foot_load = _a6_ground_force(env, "quality_feet")[..., 2].abs().amin(-1)
  return (height >= 1.15) & (upright >= 0.90) & (foot_load > 20.0)


def escape_condition(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Horizontal clearance at least 4 cm for the active board scenes."""
  active = env._a6_scene > 0
  clearance = _a6_completion_clearance(env)
  return active & (clearance >= 0.04)


def _fmt_frac(num: int, den: int) -> str:
  return f"{num}/{den} ({100.0 * num / den:.2f}%)" if den else "n/a"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--group", choices=("G-", "G+"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--steps", type=int, default=1150)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--bank-split", choices=("train", "validation"), default="train")
  parser.add_argument("--action-scale", type=float)
  parser.add_argument("--residual-checkpoint", type=Path)
  parser.add_argument("--residual-scale", type=float, default=0.25)
  parser.add_argument("--residual-after-escape", action="store_true")
  parser.add_argument("--frozen-recovery-checkpoint", type=Path)
  parser.add_argument("--correction-checkpoint", type=Path)
  parser.add_argument("--correction-scale", type=float, default=0.15)
  args = parser.parse_args()

  if args.num_envs < 32 or args.num_envs % 32:
    raise ValueError("--num-envs must be >= 32 and divisible by 32")
  dual_residual = (
    args.frozen_recovery_checkpoint is not None
    or args.correction_checkpoint is not None
  )
  if dual_residual and (
    args.frozen_recovery_checkpoint is None
    or args.correction_checkpoint is None
  ):
    raise ValueError(
      "--frozen-recovery-checkpoint and --correction-checkpoint are required together"
    )
  if dual_residual and args.residual_checkpoint is not None:
    raise ValueError("single and dual residual checkpoints are mutually exclusive")

  cfg = a6_env_cfg(
    play=True,
    seed=args.seed,
    dynamics=False,
    group=args.group,
    bank_split=args.bank_split,
  )
  cfg.scene.num_envs = args.num_envs
  # Play mode already removes the 10 s training timeout, pull-force curriculum,
  # action-scale curriculum and push events. The physical-abort termination is
  # removed here so a threshold crossing can be captured before any automatic
  # reset; the formal force/depth thresholds are computed below.
  cfg.terminations.pop("invalid_plate", None)
  for event in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
    cfg.events.pop(event, None)
  cfg.observations["actor"].terms["host"].params["add_noise"] = False

  bank_hashes = {name: sha256_file(Path(name)) for name in BANK_FILES}
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  env.reset()
  configure_all_low(env)
  if args.action_scale is not None:
    set_action_scale(env, args.action_scale)

  base_wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  try:
    runner = HoSTOnPolicyRunner(base_wrapper, asdict(agent), device="cuda:0")
    runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
    if args.action_scale is not None:
      set_action_scale(env, args.action_scale)
      action = env.action_manager.get_term("joint_pos")
      num_joints = env.scene["robot"].data.joint_pos.shape[-1]
      single_step_dim = 7 + 2 * num_joints + action.action_dim
      scale_indices = torch.arange(
        single_step_dim - 1,
        runner.obs_normalizer._mean.shape[-1],
        single_step_dim,
        device=runner.obs_normalizer._mean.device,
      )
      runner.obs_normalizer._mean[:, scale_indices] = args.action_scale
    base_policy = runner.get_inference_policy(device="cuda:0")
    residual_runner = None
    recovery_runner = None
    if dual_residual:
      recovery_runner = HoSTOnPolicyRunner(
        base_wrapper, asdict(agent), device="cuda:0"
      )
      recovery_runner.load(
        str(args.frozen_recovery_checkpoint),
        load_optimizer=False,
        map_location="cuda:0",
      )
      if args.action_scale is not None:
        recovery_runner.obs_normalizer._mean[:, scale_indices] = args.action_scale
      recovery_policy = recovery_runner.get_inference_policy(device="cuda:0")
      wrapper = FrozenRecoveryPostEscapeCorrectionVecEnvWrapper(
        env,
        base_policy=base_policy,
        recovery_policy=recovery_policy,
        recovery_scale=args.residual_scale,
        correction_scale=args.correction_scale,
        clip_actions=agent.clip_actions,
      )
      residual_runner = HoSTOnPolicyRunner(
        wrapper, asdict(agent), device="cuda:0"
      )
      residual_runner.alg.policy.actor.append(
        SymmetricSagittalCorrectionProjection(wrapper._pair_action_ids)
      )
      residual_runner.load(
        str(args.correction_checkpoint),
        load_optimizer=False,
        map_location="cuda:0",
      )
      policy = residual_runner.get_inference_policy(device="cuda:0")
    elif args.residual_checkpoint is None:
      wrapper = base_wrapper
      policy = base_policy
    else:
      wrapper = TeacherResidualVecEnvWrapper(
        env,
        base_policy=base_policy,
        residual_scale=args.residual_scale,
        clip_actions=agent.clip_actions,
        apply_after_escape=args.residual_after_escape,
      )
      residual_runner = HoSTOnPolicyRunner(
        wrapper, asdict(agent), device="cuda:0"
      )
      residual_runner.load(
        str(args.residual_checkpoint),
        load_optimizer=False,
        map_location="cuda:0",
      )
      policy = residual_runner.get_inference_policy(device="cuda:0")
    obs = wrapper.get_observations()

    robot = env.scene["robot"]
    n = env.num_envs
    device = env.device
    initial_qpos = env.sim.data.qpos.detach().cpu().contiguous().numpy()
    initial_qpos_sha256 = hashlib.sha256(initial_qpos.tobytes()).hexdigest()
    per_env_initial_sha256 = [
      hashlib.sha256(initial_qpos[index].tobytes()).hexdigest()
      for index in range(n)
    ]

    primary_success = torch.zeros(n, dtype=torch.bool, device=device)
    escaped_seen = torch.zeros(n, dtype=torch.bool, device=device)
    ever_abnormal = torch.zeros(n, dtype=torch.bool, device=device)
    abnormal_force = torch.zeros(n, dtype=torch.bool, device=device)
    abnormal_depth = torch.zeros(n, dtype=torch.bool, device=device)
    other_early_done = torch.zeros(n, dtype=torch.bool, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)

    primary_hold = torch.zeros(n, dtype=torch.long, device=device)
    escape_hold = torch.zeros(n, dtype=torch.long, device=device)
    primary_first_second = torch.full((n,), -1.0, device=device)
    escape_first_second = torch.full((n,), -1.0, device=device)
    final_height = torch.zeros(n, device=device)
    final_upright = torch.zeros(n, device=device)
    final_min_foot_load = torch.zeros(n, device=device)
    post_escape_height_max = torch.zeros(n, device=device)
    post_escape_upright_max = torch.zeros(n, device=device)
    post_escape_height_upright_hold = torch.zeros(
      n, dtype=torch.long, device=device
    )
    post_escape_height_upright_hold_max = torch.zeros(
      n, dtype=torch.long, device=device
    )
    post_escape_primary_hold = torch.zeros(n, dtype=torch.long, device=device)
    post_escape_primary_hold_max = torch.zeros(
      n, dtype=torch.long, device=device
    )
    post_escape_residual_steps = torch.zeros(n, device=device)
    post_escape_residual_abs_sum = torch.zeros(n, device=device)
    post_escape_residual_saturation_sum = torch.zeros(n, device=device)
    post_escape_combined_clip_sum = torch.zeros(n, device=device)
    post_escape_residual_joint_sum = torch.zeros(
      n, robot.data.joint_pos.shape[-1], device=device
    )
    post_escape_residual_joint_abs_sum = torch.zeros_like(
      post_escape_residual_joint_sum
    )

    with torch.no_grad():
      for step in range(args.steps):
        action = policy(obs["actor"])
        obs, _, dones, _ = wrapper.step(action)

        abnormal, force, depth = abnormal_termination(env)
        newly_abnormal = abnormal & alive
        abnormal_force |= newly_abnormal & (force > 5000.0)
        abnormal_depth |= newly_abnormal & (depth < -0.05)
        ever_abnormal |= newly_abnormal

        is_early_done = dones.bool() & alive & ~abnormal
        other_early_done |= is_early_done

        alive &= ~(abnormal | is_early_done)

        primary_now = primary_condition(env) & alive
        primary_hold = torch.where(
          primary_now, primary_hold + 1, torch.zeros_like(primary_hold)
        )
        primary_start_step = (step + 2 - primary_hold).float()
        started_by_20s = primary_start_step <= 1000.0
        primary_new = alive & (primary_hold >= 150) & started_by_20s
        first_primary_new = primary_new & (primary_first_second < 0)
        primary_first_second = torch.where(
          first_primary_new,
          primary_start_step * env.step_dt,
          primary_first_second,
        )
        primary_success |= primary_new

        escape_now = escape_condition(env) & alive
        escape_hold = torch.where(
          escape_now, escape_hold + 1, torch.zeros_like(escape_hold)
        )
        escape_new = alive & (escape_hold >= 25)
        first_escape_new = escape_new & (escape_first_second < 0)
        escape_start_step = (step + 2 - escape_hold).float()
        escape_first_second = torch.where(
          first_escape_new,
          escape_start_step * env.step_dt,
          escape_first_second,
        )
        escaped_seen |= escape_new

        height = _a6_height(env)
        upright = _a6_upright(env)
        min_foot_load = _a6_ground_force(env, "quality_feet")[..., 2].abs().amin(-1)
        final_height = torch.where(alive, height, final_height)
        final_upright = torch.where(alive, upright, final_upright)
        final_min_foot_load = torch.where(alive, min_foot_load, final_min_foot_load)
        post_escape_alive = escaped_seen & alive
        post_escape_height_max = torch.where(
          post_escape_alive,
          torch.maximum(post_escape_height_max, height),
          post_escape_height_max,
        )
        post_escape_upright_max = torch.where(
          post_escape_alive,
          torch.maximum(post_escape_upright_max, upright),
          post_escape_upright_max,
        )
        height_upright_now = (
          post_escape_alive & (height >= 1.15) & (upright >= 0.90)
        )
        post_escape_height_upright_hold = torch.where(
          height_upright_now,
          post_escape_height_upright_hold + 1,
          torch.zeros_like(post_escape_height_upright_hold),
        )
        post_escape_height_upright_hold_max = torch.maximum(
          post_escape_height_upright_hold_max,
          post_escape_height_upright_hold,
        )
        post_escape_primary_hold = torch.where(
          post_escape_alive & primary_now,
          post_escape_primary_hold + 1,
          torch.zeros_like(post_escape_primary_hold),
        )
        post_escape_primary_hold_max = torch.maximum(
          post_escape_primary_hold_max,
          post_escape_primary_hold,
        )
        if residual_runner is not None:
          residual_action = env._a6_residual_action
          gated_residual = env._a6_gated_residual_action
          recovery_residual = getattr(
            env,
            "_a6_gated_recovery_residual_action",
            torch.zeros_like(gated_residual),
          )
          preclip_action = (
            env._a6_base_action + recovery_residual + gated_residual
          )
          active = post_escape_alive.float()
          post_escape_residual_steps += active
          post_escape_residual_abs_sum += (
            residual_action.abs().mean(dim=-1) * active
          )
          post_escape_residual_saturation_sum += (
            (residual_action.abs() >= 0.95).float().mean(dim=-1) * active
          )
          post_escape_combined_clip_sum += (
            (preclip_action.abs() > 1.0).float().mean(dim=-1) * active
          )
          post_escape_residual_joint_sum += gated_residual * active.unsqueeze(-1)
          post_escape_residual_joint_abs_sum += (
            gated_residual.abs() * active.unsqueeze(-1)
          )

    timeout_events = int(
      (dones.bool() & (env.episode_length_buf >= int(env.max_episode_length))).sum()
    )

    def rows_for_mask(mask: torch.Tensor) -> dict:
      ids = torch.nonzero(mask, as_tuple=False).flatten()
      total = int(mask.sum())
      if total == 0:
        return {
          "n": 0,
          "primary_success": "0/0",
          "primary_success_rate": None,
          "escaped_seen": "0/0",
          "abnormal": 0,
          "abnormal_force": 0,
          "abnormal_depth": 0,
          "other_early_done": 0,
          "escaped_post_escape": None,
        }
      success = int(primary_success[ids].sum())
      escaped = int(escaped_seen[ids].sum())
      row = {
        "n": total,
        "primary_success": _fmt_frac(success, total),
        "primary_success_rate": success / total,
        "escaped_seen": _fmt_frac(escaped, total),
        "abnormal": int(ever_abnormal[ids].sum()),
        "abnormal_force": int(abnormal_force[ids].sum()),
        "abnormal_depth": int(abnormal_depth[ids].sum()),
        "other_early_done": int(other_early_done[ids].sum()),
      }
      escaped_ids = ids[escaped_seen[ids]]
      if escaped_ids.numel():
        residual_steps = post_escape_residual_steps[escaped_ids].clamp_min(1.0)
        row["escaped_post_escape"] = {
          "n": int(escaped_ids.numel()),
          "final_head_height_mean_m": float(final_height[escaped_ids].mean()),
          "final_upright_mean": float(final_upright[escaped_ids].mean()),
          "final_min_foot_load_mean_n": float(
            final_min_foot_load[escaped_ids].mean()
          ),
          "max_head_height_mean_m": float(
            post_escape_height_max[escaped_ids].mean()
          ),
          "max_upright_mean": float(post_escape_upright_max[escaped_ids].mean()),
          "height_upright_hold_max_mean_s": float(
            post_escape_height_upright_hold_max[escaped_ids].float().mean()
            * env.step_dt
          ),
          "primary_hold_max_mean_s": float(
            post_escape_primary_hold_max[escaped_ids].float().mean()
            * env.step_dt
          ),
          "residual_action_abs_mean": (
            float(
              (
                post_escape_residual_abs_sum[escaped_ids] / residual_steps
              ).mean()
            )
            if residual_runner is not None
            else None
          ),
          "residual_action_saturation_fraction": (
            float(
              (
                post_escape_residual_saturation_sum[escaped_ids]
                / residual_steps
              ).mean()
            )
            if residual_runner is not None
            else None
          ),
          "combined_action_clip_fraction": (
            float(
              (
                post_escape_combined_clip_sum[escaped_ids] / residual_steps
              ).mean()
            )
            if residual_runner is not None
            else None
          ),
        }
      else:
        row["escaped_post_escape"] = None
      return row

    summary_by_scene = {
      scene_name: rows_for_mask(env._a6_scene == scene_id)
      for scene_id, scene_name in enumerate(SCENE_NAMES)
    }
    direction_detail = {}
    for scene_id, scene_name in enumerate(SCENE_NAMES):
      for direction_id, direction_name in enumerate(DIRECTIONS):
        mask = (env._a6_scene == scene_id) & (env._a6_direction == direction_id)
        direction_detail[f"{scene_name}/{direction_name}"] = rows_for_mask(mask)
    source_detail = {}
    for source_id, source_name in ((0, "natural"), (1, "procedural")):
      mask = env._a6_source == source_id
      source_detail[source_name] = rows_for_mask(mask)
    scene_source_detail = {}
    for scene_id, scene_name in enumerate(SCENE_NAMES):
      for source_id, source_name in ((0, "natural"), (1, "procedural")):
        mask = (env._a6_scene == scene_id) & (env._a6_source == source_id)
        scene_source_detail[f"{scene_name}/{source_name}"] = rows_for_mask(mask)

    total = rows_for_mask(torch.ones(n, dtype=torch.bool, device=device))
    post_escape_residual_joint_response = None
    escaped_ids = torch.nonzero(escaped_seen, as_tuple=False).flatten()
    if residual_runner is not None and escaped_ids.numel():
      residual_steps = post_escape_residual_steps[escaped_ids].clamp_min(1.0)
      signed = (
        post_escape_residual_joint_sum[escaped_ids]
        / residual_steps.unsqueeze(-1)
      ).mean(dim=0)
      absolute = (
        post_escape_residual_joint_abs_sum[escaped_ids]
        / residual_steps.unsqueeze(-1)
      ).mean(dim=0)
      order = torch.argsort(absolute, descending=True)
      joint_names = list(robot.joint_names)
      post_escape_residual_joint_response = [
        {
          "joint": joint_names[int(index)],
          "signed_scaled_residual_mean": float(signed[index]),
          "abs_scaled_residual_mean": float(absolute[index]),
        }
        for index in order
      ]

    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256_file(args.checkpoint),
      "residual_checkpoint": (
        str(args.residual_checkpoint.resolve())
        if args.residual_checkpoint is not None
        else None
      ),
      "residual_checkpoint_sha256": (
        sha256_file(args.residual_checkpoint)
        if args.residual_checkpoint is not None
        else None
      ),
      "residual_scale": args.residual_scale if residual_runner is not None else None,
      "residual_after_escape": (
        args.residual_after_escape if residual_runner is not None else None
      ),
      "frozen_recovery_checkpoint": (
        str(args.frozen_recovery_checkpoint.resolve())
        if args.frozen_recovery_checkpoint is not None
        else None
      ),
      "frozen_recovery_checkpoint_sha256": (
        sha256_file(args.frozen_recovery_checkpoint)
        if args.frozen_recovery_checkpoint is not None
        else None
      ),
      "correction_checkpoint": (
        str(args.correction_checkpoint.resolve())
        if args.correction_checkpoint is not None
        else None
      ),
      "correction_checkpoint_sha256": (
        sha256_file(args.correction_checkpoint)
        if args.correction_checkpoint is not None
        else None
      ),
      "correction_scale": args.correction_scale if dual_residual else None,
      "action_scale_requested": args.action_scale,
      "action_scale": float(env._host_action_rescale.mean()),
      "group": args.group,
      "bank_split": args.bank_split,
      "seed": args.seed,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "seconds": args.steps * env.step_dt,
      "physics_dt": env.physics_dt,
      "control_dt": env.step_dt,
      "action_scale": float(env.action_manager.get_term("joint_pos").cfg.scale),
      "bank_sha256": bank_hashes,
      "initial_qpos_sha256": initial_qpos_sha256,
      "reset_counts": a6_reset_counts(env),
      "success_protocol": {
        "start_before_s": 20.0,
        "hold_s": 3.0,
        "observe_s": 23.0,
        "height_min_m": 1.15,
        "upright_min": 0.90,
        "both_ankle_load_min_n": 20.0,
      },
      "escape_diagnostic": {
        "clearance_min_m": 0.04,
        "hold_s": 0.5,
        "in_main_success": False,
      },
      "abnormal_termination": {
        "plate_force_max_n": 5000.0,
        "plate_penetration_max_m": 0.05,
        "no_initial_contact_tolerance": True,
      },
      "timeout_events": timeout_events,
      "total": total,
      "post_escape_residual_joint_response": post_escape_residual_joint_response,
      "by_scene": summary_by_scene,
      "by_direction": direction_detail,
      "by_source": source_detail,
      "by_scene_source": scene_source_detail,
    }
    (args.output / "summary.json").write_text(
      json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    abnormal_reason = np.where(
      abnormal_force.cpu().numpy(),
      "force",
      np.where(abnormal_depth.cpu().numpy(), "depth", ""),
    )
    with (args.output / "per_env.json").open("w", encoding="utf-8") as stream:
      for index in range(n):
        row = {
          "env_id": index,
          "scene": SCENE_NAMES[int(env._a6_scene[index])],
          "direction": DIRECTIONS[int(env._a6_direction[index])],
          "source": "procedural" if int(env._a6_source[index]) == 1 else "natural",
          "stratum": "low",
          "primary_success": bool(primary_success[index]),
          "mentor_stand_success_3s": bool(primary_success[index]),
          "escaped_seen": bool(escaped_seen[index]),
          "abnormal_termination": bool(ever_abnormal[index]),
          "abnormal_reason": abnormal_reason[index],
          "other_early_done": bool(other_early_done[index]),
          "first_primary_second": float(primary_first_second[index]),
          "first_escape_second": float(escape_first_second[index]),
          "final_head_height_m": float(final_height[index]),
          "final_upright": float(final_upright[index]),
          "final_min_foot_load_n": float(final_min_foot_load[index]),
          "post_escape_head_height_max_m": float(post_escape_height_max[index]),
          "post_escape_upright_max": float(post_escape_upright_max[index]),
          "post_escape_height_upright_hold_max_s": float(
            post_escape_height_upright_hold_max[index] * env.step_dt
          ),
          "post_escape_primary_hold_max_s": float(
            post_escape_primary_hold_max[index] * env.step_dt
          ),
          "initial_sha256": per_env_initial_sha256[index],
        }
        stream.write(json.dumps(row, ensure_ascii=True) + "\n")

    def line(label: str, value: str) -> str:
      return f"{label:<36} {value}\n"

    with (args.output / "report.txt").open("w", encoding="utf-8") as stream:
      stream.write(f"{REPORT_TITLE}\n")
      stream.write(f"checkpoint {args.checkpoint}\n")
      stream.write(f"group {args.group}\n")
      stream.write(f"seed {args.seed}  num_envs {args.num_envs}  steps {args.steps}\n")
      stream.write(f"seconds {args.steps * env.step_dt:.2f}  control_dt {env.step_dt:.2f}\n")
      stream.write(f"initial_qpos_sha256 {initial_qpos_sha256}\n")
      stream.write(f"timeout_events {timeout_events}\n\n")
      stream.write("scene\n")
      for scene_name, row in summary_by_scene.items():
        stream.write(
          line(scene_name, row["primary_success"])
          + line("  escaped diagnostic", row["escaped_seen"])
          + line("  abnormal termination", str(row["abnormal"]))
          + line("  other early done", str(row["other_early_done"]))
        )
      stream.write("\ntotal primary success\n")
      stream.write(line("all", total["primary_success"]))
      stream.write("\nscene/direction\n")
      for key, row in direction_detail.items():
        stream.write(line(key, row["primary_success"]))
      stream.write("\nscene/source\n")
      for key, row in scene_source_detail.items():
        stream.write(line(key, row["primary_success"]))

    print(COMPLETE_MARKER, json.dumps(total), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
