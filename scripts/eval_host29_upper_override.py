"""Evaluate a checkpoint with a diagnostic post-standing upper-body controller."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.flat29 import flat29_env_cfg
from src.tasks.host_recovery.mdp.events import POSTURE_QUATS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


POSTURES = ("prone", "supine", "left_side", "right_side")
UPPER_JOINT_NAMES = (
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
UPPER_TARGET = (
  0.0,
  0.0,
  0.0,
  0.3,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  -0.3,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
)
SAGITTAL_TARGET = (-0.5287, 1.2749, -0.5947)
STEP_PULSE_JOINT_NAMES = (
  "right_hip_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "left_hip_roll_joint",
  "left_ankle_roll_joint",
)


def _fraction(value: torch.Tensor) -> float:
  return value.float().mean().item()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--hold-steps", type=int, default=50)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--blend", type=float, required=True)
  parser.add_argument("--gain", type=float, default=0.5)
  parser.add_argument("--balance-blend", type=float, default=0.0)
  parser.add_argument("--balance-gain", type=float, default=0.5)
  parser.add_argument("--delay-steps", type=int, default=25)
  parser.add_argument("--ramp-steps", type=int, default=25)
  parser.add_argument("--foot-align-blend", type=float, default=0.0)
  parser.add_argument("--foot-align-gain", type=float, default=0.05)
  parser.add_argument("--foot-align-max-action", type=float, default=0.10)
  parser.add_argument("--foot-align-mode", choices=("add", "pair"), default="add")
  parser.add_argument("--foot-align-delay-steps", type=int, default=50)
  parser.add_argument("--foot-align-ramp-steps", type=int, default=150)
  parser.add_argument("--sagittal-symmetry-blend", type=float, default=0.0)
  parser.add_argument("--sagittal-symmetry-gain", type=float, default=0.5)
  parser.add_argument("--sagittal-symmetry-integral-gain", type=float, default=0.0)
  parser.add_argument("--sagittal-symmetry-integral-limit", type=float, default=1.0)
  parser.add_argument("--sagittal-symmetry-max-action", type=float, default=0.5)
  parser.add_argument("--sagittal-symmetry-mode", choices=("add", "pair"), default="add")
  parser.add_argument("--sagittal-symmetry-delay-steps", type=int, default=50)
  parser.add_argument("--sagittal-symmetry-ramp-steps", type=int, default=150)
  parser.add_argument("--sagittal-target-blend", type=float, default=0.0)
  parser.add_argument("--sagittal-target-gain", type=float, default=1.0)
  parser.add_argument("--sagittal-target-max-action", type=float, default=1.0)
  parser.add_argument(
    "--sagittal-target", type=float, nargs=3, default=SAGITTAL_TARGET
  )
  parser.add_argument(
    "--sagittal-target-delta", type=float, nargs=3, default=(0.0, 0.0, 0.0)
  )
  parser.add_argument(
    "--sagittal-target-side", choices=("both", "left", "right"), default="both"
  )
  parser.add_argument("--sagittal-target-delay-steps", type=int, default=50)
  parser.add_argument("--sagittal-target-ramp-steps", type=int, default=150)
  parser.add_argument("--sagittal-target-hold-steps", type=int, default=0)
  parser.add_argument("--sagittal-target-release-steps", type=int, default=50)
  parser.add_argument("--step-pulse-actions", type=float, nargs=16)
  parser.add_argument("--step-pulse-delay-steps", type=int, default=75)
  parser.add_argument("--step-pulse-phase1-steps", type=int, default=20)
  parser.add_argument("--step-pulse-phase2-steps", type=int, default=25)
  parser.add_argument("--step-pulse-release-steps", type=int, default=35)
  args = parser.parse_args()

  if not 0.0 <= args.blend <= 1.0:
    raise ValueError("blend must lie in [0, 1]")
  if args.gain <= 0.0:
    raise ValueError("gain must be positive")
  if not 0.0 <= args.balance_blend <= 1.0:
    raise ValueError("balance-blend must lie in [0, 1]")
  if args.balance_gain == 0.0:
    raise ValueError("balance-gain must be non-zero")
  if not 0.0 <= args.foot_align_blend <= 1.0:
    raise ValueError("foot-align-blend must lie in [0, 1]")
  if args.foot_align_gain < 0.0:
    raise ValueError("foot-align-gain must be non-negative")
  if args.foot_align_max_action <= 0.0:
    raise ValueError("foot-align-max-action must be positive")
  if not 0.0 <= args.sagittal_symmetry_blend <= 1.0:
    raise ValueError("sagittal-symmetry-blend must lie in [0, 1]")
  if args.sagittal_symmetry_gain < 0.0:
    raise ValueError("sagittal-symmetry-gain must be non-negative")
  if args.sagittal_symmetry_integral_gain < 0.0:
    raise ValueError("sagittal-symmetry-integral-gain must be non-negative")
  if args.sagittal_symmetry_integral_limit <= 0.0:
    raise ValueError("sagittal-symmetry-integral-limit must be positive")
  if args.sagittal_symmetry_max_action <= 0.0:
    raise ValueError("sagittal-symmetry-max-action must be positive")
  if not 0.0 <= args.sagittal_target_blend <= 1.0:
    raise ValueError("sagittal-target-blend must lie in [0, 1]")
  if args.sagittal_target_gain <= 0.0:
    raise ValueError("sagittal-target-gain must be positive")
  if args.sagittal_target_max_action <= 0.0:
    raise ValueError("sagittal-target-max-action must be positive")
  if args.sagittal_target_hold_steps < 0:
    raise ValueError("sagittal-target-hold-steps must be non-negative")
  if args.sagittal_target_release_steps <= 0:
    raise ValueError("sagittal-target-release-steps must be positive")
  if args.step_pulse_actions is not None and (
    args.step_pulse_phase1_steps <= 0
    or args.step_pulse_phase2_steps <= 0
    or args.step_pulse_release_steps <= 0
  ):
    raise ValueError("step pulse phase lengths must be positive")

  cfg = flat29_env_cfg(play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  try:
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
    runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_scale = saved.get("infos", {}).get("env_state", {}).get(
      "host_action_rescale"
    )
    if saved_scale is None:
      raise RuntimeError("checkpoint has no host_action_rescale state")
    action_scale = float(saved_scale.float().mean())
    action_term = env.action_manager.get_term("joint_pos")
    if isinstance(action_term.cfg.scale, dict):
      action_term.cfg.scale = {key: action_scale for key in action_term.cfg.scale}
    else:
      action_term.cfg.scale = action_scale
    if isinstance(action_term._scale, torch.Tensor):
      action_term._scale[:] = action_scale
    else:
      action_term._scale = action_scale
    env._host_action_rescale = torch.full(
      (args.num_envs, 1), action_scale, device=env.device
    )

    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    torso_id = robot.find_bodies("torso_link")[0][0]
    foot_ids = torch.tensor(
      robot.find_sites(("left_foot", "right_foot"))[0], device=env.device
    )
    upper_joint_ids = torch.tensor(
      robot.find_joints(UPPER_JOINT_NAMES, preserve_order=True)[0],
      device=env.device,
    )
    target_ids = list(action_term._target_ids)
    upper_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in upper_joint_ids.tolist()],
      device=env.device,
    )
    hip_pitch_joint_ids = torch.tensor(
      robot.find_joints(
        ("left_hip_pitch_joint", "right_hip_pitch_joint"),
        preserve_order=True,
      )[0],
      device=env.device,
    )
    hip_pitch_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in hip_pitch_joint_ids.tolist()],
      device=env.device,
    )
    sagittal_joint_ids = torch.tensor(
      robot.find_joints(
        (
          "left_hip_pitch_joint",
          "left_knee_joint",
          "left_ankle_pitch_joint",
          "right_hip_pitch_joint",
          "right_knee_joint",
          "right_ankle_pitch_joint",
        ),
        preserve_order=True,
      )[0],
      device=env.device,
    ).view(2, 3)
    sagittal_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in sagittal_joint_ids.flatten().tolist()],
      device=env.device,
    ).view(2, 3)
    sagittal_target_center = torch.tensor(args.sagittal_target, device=env.device)
    sagittal_target_delta = torch.tensor(
      args.sagittal_target_delta, device=env.device
    )
    sagittal_target = torch.stack(
      (
        sagittal_target_center + 0.5 * sagittal_target_delta,
        sagittal_target_center - 0.5 * sagittal_target_delta,
      )
    )
    sagittal_target_side_mask = torch.tensor(
      {
        "both": (1.0, 1.0),
        "left": (1.0, 0.0),
        "right": (0.0, 1.0),
      }[args.sagittal_target_side],
      device=env.device,
    )
    step_pulse_joint_ids = torch.tensor(
      robot.find_joints(STEP_PULSE_JOINT_NAMES, preserve_order=True)[0],
      device=env.device,
    )
    step_pulse_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in step_pulse_joint_ids.tolist()],
      device=env.device,
    )
    step_pulse_actions = (
      torch.tensor(args.step_pulse_actions, device=env.device).view(2, -1)
      if args.step_pulse_actions is not None
      else None
    )
    upper_target = torch.tensor(UPPER_TARGET, device=env.device)
    reset_cfg = env.event_manager.get_term_cfg("reset_base")
    results = {}

    for posture in POSTURES:
      reset_cfg.params["posture"] = posture
      wrapper.reset()
      expected = torch.tensor(POSTURE_QUATS[posture], device=env.device)
      alignment = (robot.data.root_link_quat_w * expected).sum(-1).abs()
      if not (alignment > 0.999).all():
        raise RuntimeError(f"{posture} reset mismatch: {alignment.min().item()}")

      obs = wrapper.get_observations()
      valid = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
      ever_standing = torch.zeros_like(valid)
      held = torch.zeros_like(valid)
      consecutive = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
      override_count = torch.zeros_like(consecutive)
      step_pulse_clock = torch.full_like(consecutive, -1)
      symmetry_integral = torch.zeros(
        (args.num_envs, 3), device=env.device
      )
      first_hold = torch.full_like(consecutive, -1)
      final_standing = torch.zeros_like(valid)
      final_upright = torch.zeros_like(valid)
      standing_sample_count = torch.zeros((), device=env.device)
      standing_yaw_rate_sq_sum = torch.zeros((), device=env.device)
      standing_horizontal_speed_sq_sum = torch.zeros((), device=env.device)
      standing_joint_speed_sq_sum = torch.zeros(
        len(robot.joint_names), device=env.device
      )

      with torch.no_grad():
        for step in range(args.steps):
          torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
          feet_z = robot.data.site_pos_w[:, foot_ids, 2].mean(dim=1)
          pre_height = torso_z - feet_z
          pre_upright = robot.data.projected_gravity_b[:, 2] < -0.8
          pre_standing = (pre_height > 0.6) & pre_upright & valid
          override_count = torch.where(pre_standing, override_count + 1, 0)
          step_pulse_start = (
            (step_pulse_clock < 0)
            & (override_count >= args.step_pulse_delay_steps)
          )
          step_pulse_clock = torch.where(
            step_pulse_start, torch.zeros_like(step_pulse_clock), step_pulse_clock
          )

          actions = policy(obs["actor"])
          selected = actions[:, upper_action_ids]
          correction = args.gain * (
            upper_target - robot.data.joint_pos[:, upper_joint_ids]
          ) / action_scale
          correction = correction.clamp(-1.0, 1.0)
          ramp = (
            (override_count - args.delay_steps).float()
            / max(args.ramp_steps, 1)
          ).clamp(0.0, 1.0)
          mix = (args.blend * ramp)[:, None]
          actions[:, upper_action_ids] = selected + mix * (correction - selected)
          if args.balance_blend > 0.0:
            world_down = torch.zeros(
              (args.num_envs, 3), device=env.device
            )
            world_down[:, 2] = -1.0
            gravity_torso = quat_apply_inverse(
              robot.data.body_link_quat_w[:, torso_id], world_down
            )
            balance_delta = torch.stack(
              (gravity_torso[:, 1], -gravity_torso[:, 0]), dim=1
            )
            balance_correction = (
              args.balance_gain * balance_delta / action_scale
            ).clamp(-1.0, 1.0)
            balance_mix = (args.balance_blend * ramp)[:, None]
            actions[:, upper_action_ids[:2]] = selected[:, :2] + balance_mix * (
              balance_correction - selected[:, :2]
            )
          if args.foot_align_blend > 0.0:
            root_pos = robot.data.root_link_pos_w
            root_quat = robot.data.root_link_quat_w
            feet_w = robot.data.site_pos_w[:, foot_ids]
            left_foot_b = quat_apply_inverse(root_quat, feet_w[:, 0] - root_pos)
            right_foot_b = quat_apply_inverse(root_quat, feet_w[:, 1] - root_pos)
            forward_error = left_foot_b[:, 0] - right_foot_b[:, 0]
            correction = (
              args.foot_align_gain * forward_error / action_scale
            ).clamp(
              -args.foot_align_max_action,
              args.foot_align_max_action,
            )
            foot_ramp = (
              (override_count - args.foot_align_delay_steps).float()
              / max(args.foot_align_ramp_steps, 1)
            ).clamp(0.0, 1.0)
            foot_mix = args.foot_align_blend * foot_ramp
            selected_hip = actions[:, hip_pitch_action_ids]
            if args.foot_align_mode == "add":
              actions[:, hip_pitch_action_ids[0]] += foot_mix * correction
              actions[:, hip_pitch_action_ids[1]] -= foot_mix * correction
            else:
              common = selected_hip.mean(dim=1)
              desired = torch.stack(
                (common + correction, common - correction), dim=1
              )
              actions[:, hip_pitch_action_ids] = selected_hip + foot_mix[:, None] * (
                desired - selected_hip
              )
          if args.sagittal_symmetry_blend > 0.0:
            joint_delta = (
              robot.data.joint_pos[:, sagittal_joint_ids[0]]
              - robot.data.joint_pos[:, sagittal_joint_ids[1]]
            )
            integrate = override_count > args.sagittal_symmetry_delay_steps
            symmetry_integral = torch.where(
              integrate[:, None],
              (
                symmetry_integral + env.step_dt * joint_delta
              ).clamp(
                -args.sagittal_symmetry_integral_limit,
                args.sagittal_symmetry_integral_limit,
              ),
              torch.zeros_like(symmetry_integral),
            )
            symmetry_correction = (
              -0.5
              * (
                args.sagittal_symmetry_gain * joint_delta
                + args.sagittal_symmetry_integral_gain * symmetry_integral
              )
              / action_scale
            ).clamp(
              -args.sagittal_symmetry_max_action,
              args.sagittal_symmetry_max_action,
            )
            symmetry_ramp = (
              (override_count - args.sagittal_symmetry_delay_steps).float()
              / max(args.sagittal_symmetry_ramp_steps, 1)
            ).clamp(0.0, 1.0)
            symmetry_mix = args.sagittal_symmetry_blend * symmetry_ramp[:, None]
            if args.sagittal_symmetry_mode == "add":
              actions[:, sagittal_action_ids[0]] += symmetry_mix * symmetry_correction
              actions[:, sagittal_action_ids[1]] -= symmetry_mix * symmetry_correction
            else:
              selected_sagittal = torch.stack(
                (
                  actions[:, sagittal_action_ids[0]],
                  actions[:, sagittal_action_ids[1]],
                ),
                dim=1,
              )
              common_sagittal = selected_sagittal.mean(dim=1)
              desired_sagittal = torch.stack(
                (
                  common_sagittal + symmetry_correction,
                  common_sagittal - symmetry_correction,
                ),
                dim=1,
              )
              mixed_sagittal = selected_sagittal + symmetry_mix[:, None] * (
                desired_sagittal - selected_sagittal
              )
              actions[:, sagittal_action_ids[0]] = mixed_sagittal[:, 0]
              actions[:, sagittal_action_ids[1]] = mixed_sagittal[:, 1]
          if args.sagittal_target_blend > 0.0:
            target_correction = (
              args.sagittal_target_gain
              * (
                sagittal_target[None, :, :]
                - robot.data.joint_pos[:, sagittal_joint_ids]
              )
              / action_scale
            ).clamp(
              -args.sagittal_target_max_action,
              args.sagittal_target_max_action,
            )
            target_ramp = (
              (override_count - args.sagittal_target_delay_steps).float()
              / max(args.sagittal_target_ramp_steps, 1)
            ).clamp(0.0, 1.0)
            if args.sagittal_target_hold_steps > 0:
              release_start = (
                args.sagittal_target_delay_steps
                + args.sagittal_target_ramp_steps
                + args.sagittal_target_hold_steps
              )
              target_release = (
                1.0
                - (override_count - release_start).float()
                / args.sagittal_target_release_steps
              ).clamp(0.0, 1.0)
              target_ramp *= target_release
            target_mix = (
              args.sagittal_target_blend
              * target_ramp[:, None, None]
              * sagittal_target_side_mask[None, :, None]
            )
            selected_sagittal = torch.stack(
              (
                actions[:, sagittal_action_ids[0]],
                actions[:, sagittal_action_ids[1]],
              ),
              dim=1,
            )
            mixed_sagittal = selected_sagittal + target_mix * (
              target_correction - selected_sagittal
            )
            actions[:, sagittal_action_ids[0]] = mixed_sagittal[:, 0]
            actions[:, sagittal_action_ids[1]] = mixed_sagittal[:, 1]
          if step_pulse_actions is not None:
            phase1_mask = (
              (step_pulse_clock >= 0)
              & (step_pulse_clock < args.step_pulse_phase1_steps)
            )
            phase2_end = (
              args.step_pulse_phase1_steps + args.step_pulse_phase2_steps
            )
            phase2_mask = (
              (step_pulse_clock >= args.step_pulse_phase1_steps)
              & (step_pulse_clock < phase2_end)
            )
            release_mask = (
              (step_pulse_clock >= phase2_end)
              & (
                step_pulse_clock
                < phase2_end + args.step_pulse_release_steps
              )
            )
            selected_pulse = actions[:, step_pulse_action_ids]
            pulse = torch.where(
              phase1_mask[:, None], step_pulse_actions[0], selected_pulse
            )
            pulse = torch.where(
              phase2_mask[:, None], step_pulse_actions[1], pulse
            )
            release = (
              1.0
              - (step_pulse_clock - phase2_end).float()
              / args.step_pulse_release_steps
            ).clamp(0.0, 1.0)
            release_values = selected_pulse + release[:, None] * (
              step_pulse_actions[1] - selected_pulse
            )
            pulse = torch.where(release_mask[:, None], release_values, pulse)
            pulse_active = phase1_mask | phase2_mask | release_mask
            actions[:, step_pulse_action_ids] = torch.where(
              pulse_active[:, None], pulse, selected_pulse
            )

          obs, _, dones, _ = wrapper.step(actions)
          step_pulse_clock = torch.where(
            step_pulse_clock >= 0, step_pulse_clock + 1, step_pulse_clock
          )
          valid &= ~dones.bool()
          torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
          feet_z = robot.data.site_pos_w[:, foot_ids, 2].mean(dim=1)
          height = torso_z - feet_z
          upright = robot.data.projected_gravity_b[:, 2] < -0.8
          standing = (height > 0.6) & upright & valid
          ever_standing |= standing
          consecutive = torch.where(standing, consecutive + 1, 0)
          newly_held = (consecutive >= args.hold_steps) & ~held
          first_hold[newly_held] = step - args.hold_steps + 1
          held |= newly_held
          final_standing = standing
          final_upright = upright & valid
          standing_float = standing.float()
          standing_sample_count += standing_float.sum()
          standing_yaw_rate_sq_sum += (
            robot.data.root_link_ang_vel_w[:, 2].square() * standing_float
          ).sum()
          standing_horizontal_speed_sq_sum += (
            robot.data.root_link_lin_vel_w[:, :2].square().sum(dim=-1)
            * standing_float
          ).sum()
          standing_joint_speed_sq_sum += (
            robot.data.joint_vel.square() * standing_float[:, None]
          ).sum(dim=0)

      held_times = first_hold[first_hold >= 0].float() * env.step_dt
      results[posture] = {
        "valid_fraction": _fraction(valid),
        "termination_fraction": _fraction(~valid),
        "ever_standing_fraction": _fraction(ever_standing),
        "held_fraction": _fraction(held & valid),
        "final_standing_fraction": _fraction(final_standing),
        "final_upright_fraction": _fraction(final_upright),
        "first_hold_seconds_median": (
          held_times.median().item() if held_times.numel() else None
        ),
        "final_joint_speed_rms": (
          robot.data.joint_vel[valid].square().mean().sqrt().item()
          if valid.any()
          else None
        ),
        "standing_yaw_rate_rms": (
          (standing_yaw_rate_sq_sum / standing_sample_count).sqrt().item()
          if standing_sample_count.item() > 0
          else None
        ),
        "standing_horizontal_speed_rms": (
          (standing_horizontal_speed_sq_sum / standing_sample_count).sqrt().item()
          if standing_sample_count.item() > 0
          else None
        ),
        "standing_joint_speed_rms": (
          {
            name: value
            for name, value in zip(
              robot.joint_names,
              (standing_joint_speed_sq_sum / standing_sample_count).sqrt().tolist(),
              strict=True,
            )
          }
          if standing_sample_count.item() > 0
          else None
        ),
      }
      print(posture, json.dumps(results[posture], sort_keys=True), flush=True)

    report = {
      "checkpoint": str(args.checkpoint),
      "num_envs_per_posture": args.num_envs,
      "steps": args.steps,
      "step_dt": env.step_dt,
      "action_scale": action_scale,
      "override": {
        "blend": args.blend,
        "gain": args.gain,
        "balance_blend": args.balance_blend,
        "balance_gain": args.balance_gain,
        "delay_steps": args.delay_steps,
        "ramp_steps": args.ramp_steps,
        "joint_names": UPPER_JOINT_NAMES,
        "target": UPPER_TARGET,
        "foot_align_blend": args.foot_align_blend,
        "foot_align_gain": args.foot_align_gain,
        "foot_align_max_action": args.foot_align_max_action,
        "foot_align_mode": args.foot_align_mode,
        "foot_align_delay_steps": args.foot_align_delay_steps,
        "foot_align_ramp_steps": args.foot_align_ramp_steps,
        "sagittal_symmetry_blend": args.sagittal_symmetry_blend,
        "sagittal_symmetry_gain": args.sagittal_symmetry_gain,
        "sagittal_symmetry_integral_gain": args.sagittal_symmetry_integral_gain,
        "sagittal_symmetry_integral_limit": args.sagittal_symmetry_integral_limit,
        "sagittal_symmetry_max_action": args.sagittal_symmetry_max_action,
        "sagittal_symmetry_mode": args.sagittal_symmetry_mode,
        "sagittal_symmetry_delay_steps": args.sagittal_symmetry_delay_steps,
        "sagittal_symmetry_ramp_steps": args.sagittal_symmetry_ramp_steps,
        "sagittal_target_blend": args.sagittal_target_blend,
        "sagittal_target_gain": args.sagittal_target_gain,
        "sagittal_target_max_action": args.sagittal_target_max_action,
        "sagittal_target_delay_steps": args.sagittal_target_delay_steps,
        "sagittal_target_ramp_steps": args.sagittal_target_ramp_steps,
        "sagittal_target_hold_steps": args.sagittal_target_hold_steps,
        "sagittal_target_release_steps": args.sagittal_target_release_steps,
        "sagittal_target": args.sagittal_target,
        "sagittal_target_delta": args.sagittal_target_delta,
        "sagittal_target_side": args.sagittal_target_side,
        "step_pulse_joint_names": STEP_PULSE_JOINT_NAMES,
        "step_pulse_actions": args.step_pulse_actions,
        "step_pulse_delay_steps": args.step_pulse_delay_steps,
        "step_pulse_phase1_steps": args.step_pulse_phase1_steps,
        "step_pulse_phase2_steps": args.step_pulse_phase2_steps,
        "step_pulse_release_steps": args.step_pulse_release_steps,
      },
      "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
  finally:
    env.close()


if __name__ == "__main__":
  main()
