"""Quantify post-standing torso and arm posture for a 29-DoF checkpoint."""

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
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from scripts.train_host29_v7 import set_action_scale
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.flat29 import LEG_JOINTS, flat29_env_cfg
from src.tasks.host_recovery.mdp.events import POSTURE_QUATS
from src.tasks.host_recovery.posture29_v26 import (
  LEFT_ARM_JOINTS,
  LEFT_ARM_TARGET,
  RIGHT_ARM_JOINTS,
  RIGHT_ARM_TARGET,
)
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


POSTURES = ("prone", "supine", "left_side", "right_side")
WAIST_JOINTS = ("waist_roll_joint", "waist_pitch_joint")
HAND_TARGETS = ((0.0, 0.28, -0.30), (0.0, -0.28, -0.30))
SAGITTAL_TARGET = (-0.5287, 1.2749, -0.5947)
A6_HEAD_OFFSET = 0.43
A6_FORMAL_HEAD_HEIGHT = 1.15
A6_FORMAL_UPRIGHT = 0.93
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


def _ids(robot, names: tuple[str, ...], device: str) -> torch.Tensor:
  return torch.tensor(
    robot.find_joints(names, preserve_order=True)[0], device=device
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--override-blend", type=float, default=0.0)
  parser.add_argument("--override-gain", type=float, default=0.5)
  parser.add_argument("--balance-blend", type=float, default=0.0)
  parser.add_argument("--balance-gain", type=float, default=0.5)
  parser.add_argument("--override-delay-steps", type=int, default=25)
  parser.add_argument("--override-ramp-steps", type=int, default=25)
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
    "--sagittal-target-control-mode",
    choices=("position", "error"),
    default="error",
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

  if not 0.0 <= args.override_blend <= 1.0:
    raise ValueError("override-blend must lie in [0, 1]")
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
    set_action_scale(env, action_scale)

    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    torso_id = robot.find_bodies("torso_link")[0][0]
    pelvis_id = robot.find_bodies("pelvis")[0][0]
    ankle_ids = torch.tensor(
      robot.find_bodies(
        ("left_ankle_roll_link", "right_ankle_roll_link"),
        preserve_order=True,
      )[0],
      device=env.device,
    )
    hand_ids = torch.tensor(
      robot.find_bodies(
        ("left_wrist_yaw_link", "right_wrist_yaw_link"),
        preserve_order=True,
      )[0],
      device=env.device,
    )
    foot_ids = torch.tensor(
      robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0],
      device=env.device,
    )
    left_arm_ids = _ids(robot, LEFT_ARM_JOINTS, env.device)
    right_arm_ids = _ids(robot, RIGHT_ARM_JOINTS, env.device)
    waist_ids = _ids(robot, WAIST_JOINTS, env.device)
    leg_ids = _ids(robot, LEG_JOINTS, env.device)
    upper_joint_ids = torch.cat((waist_ids, left_arm_ids, right_arm_ids))
    action_term = env.action_manager.get_term("joint_pos")
    target_ids = list(action_term._target_ids)
    upper_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in upper_joint_ids.tolist()],
      device=env.device,
    )
    hip_pitch_joint_ids = _ids(
      robot,
      ("left_hip_pitch_joint", "right_hip_pitch_joint"),
      env.device,
    )
    hip_pitch_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in hip_pitch_joint_ids.tolist()],
      device=env.device,
    )
    sagittal_joint_ids = _ids(
      robot,
      (
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
      ),
      env.device,
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
    left_target = torch.tensor(LEFT_ARM_TARGET, device=env.device)
    right_target = torch.tensor(RIGHT_ARM_TARGET, device=env.device)
    waist_target = torch.zeros(len(WAIST_JOINTS), device=env.device)
    upper_target = torch.cat((waist_target, left_target, right_target))
    hand_targets = torch.tensor(HAND_TARGETS, device=env.device)
    world_down = torch.zeros((args.num_envs, 3), device=env.device)
    world_down[:, 2] = -1.0
    reset_cfg = env.event_manager.get_term_cfg("reset_base")
    report = {}

    for posture in POSTURES:
      reset_cfg.params["posture"] = posture
      wrapper.reset()
      expected = torch.tensor(POSTURE_QUATS[posture], device=env.device)
      alignment = (robot.data.root_link_quat_w * expected).sum(-1).abs()
      if not (alignment > 0.999).all():
        raise RuntimeError(f"{posture} reset mismatch: {alignment.min().item()}")

      observations = wrapper.get_observations()
      valid = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
      override_count = torch.zeros(
        args.num_envs, dtype=torch.long, device=env.device
      )
      step_pulse_clock = torch.full_like(override_count, -1)
      symmetry_integral = torch.zeros(
        (args.num_envs, 3), device=env.device
      )
      count = torch.zeros((), device=env.device)
      sums = {
        "torso_tilt_angle_rad": torch.zeros((), device=env.device),
        "torso_support_error_m": torch.zeros((), device=env.device),
        "pelvis_support_error_m": torch.zeros((), device=env.device),
        "left_hand_target_error_m": torch.zeros((), device=env.device),
        "right_hand_target_error_m": torch.zeros((), device=env.device),
        "left_arm_target_mae_rad": torch.zeros((), device=env.device),
        "right_arm_target_mae_rad": torch.zeros((), device=env.device),
        "waist_target_mae_rad": torch.zeros((), device=env.device),
        "upper_action_abs_mean": torch.zeros((), device=env.device),
        "a6_head_height_m": torch.zeros((), device=env.device),
        "a6_formal_contract": torch.zeros((), device=env.device),
        "feet_forward_offset_m": torch.zeros((), device=env.device),
        "feet_height_offset_m": torch.zeros((), device=env.device),
        "feet_lateral_separation_m": torch.zeros((), device=env.device),
      }
      left_position_sum = torch.zeros(len(left_arm_ids), device=env.device)
      right_position_sum = torch.zeros(len(right_arm_ids), device=env.device)
      waist_position_sum = torch.zeros(len(waist_ids), device=env.device)
      leg_position_sum = torch.zeros(len(leg_ids), device=env.device)
      left_foot_local_sum = torch.zeros(3, device=env.device)
      right_foot_local_sum = torch.zeros(3, device=env.device)
      left_hand_local_sum = torch.zeros(3, device=env.device)
      right_hand_local_sum = torch.zeros(3, device=env.device)
      feet_forward_offset_min = torch.full(
        (args.num_envs,), float("inf"), device=env.device
      )
      feet_height_offset_max = torch.zeros(args.num_envs, device=env.device)
      feet_forward_offset_final = torch.full(
        (args.num_envs,), float("nan"), device=env.device
      )
      a6_head_height_final = torch.full(
        (args.num_envs,), float("nan"), device=env.device
      )
      a6_head_height_max = torch.full(
        (args.num_envs,), -float("inf"), device=env.device
      )
      a6_contract_steps = torch.zeros(args.num_envs, device=env.device)

      with torch.no_grad():
        for _ in range(args.steps):
          torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
          feet_z = robot.data.site_pos_w[:, foot_ids, 2].mean(dim=1)
          pre_standing = (
            (torso_z - feet_z > 0.6)
            & (robot.data.projected_gravity_b[:, 2] < -0.8)
            & valid
          )
          override_count = torch.where(pre_standing, override_count + 1, 0)
          step_pulse_start = (
            (step_pulse_clock < 0)
            & (override_count >= args.step_pulse_delay_steps)
          )
          step_pulse_clock = torch.where(
            step_pulse_start, torch.zeros_like(step_pulse_clock), step_pulse_clock
          )
          actions = policy(observations["actor"])
          selected = actions[:, upper_action_ids]
          if args.override_blend > 0.0:
            correction = args.override_gain * (
              upper_target - robot.data.joint_pos[:, upper_joint_ids]
            ) / action_scale
            correction = correction.clamp(-1.0, 1.0)
            ramp = (
              (override_count - args.override_delay_steps).float()
              / max(args.override_ramp_steps, 1)
            ).clamp(0.0, 1.0)
            mix = (args.override_blend * ramp)[:, None]
            actions[:, upper_action_ids] = selected + mix * (
              correction - selected
            )
          if args.balance_blend > 0.0:
            gravity_torso = quat_apply_inverse(
              robot.data.body_link_quat_w[:, torso_id], world_down
            )
            balance_delta = torch.stack(
              (gravity_torso[:, 1], -gravity_torso[:, 0]), dim=1
            )
            balance_correction = (
              args.balance_gain * balance_delta / action_scale
            ).clamp(-1.0, 1.0)
            ramp = (
              (override_count - args.override_delay_steps).float()
              / max(args.override_ramp_steps, 1)
            ).clamp(0.0, 1.0)
            mix = (args.balance_blend * ramp)[:, None]
            actions[:, upper_action_ids[:2]] = selected[:, :2] + mix * (
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
            if args.sagittal_target_control_mode == "position":
              target_error = (
                sagittal_target[None, :, :]
                - robot.data.default_joint_pos[:, sagittal_joint_ids]
              )
            else:
              target_error = (
                sagittal_target[None, :, :]
                - robot.data.joint_pos[:, sagittal_joint_ids]
              )
            target_correction = (
              args.sagittal_target_gain * target_error / action_scale
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
          observations, _, dones, _ = wrapper.step(actions)
          step_pulse_clock = torch.where(
            step_pulse_clock >= 0, step_pulse_clock + 1, step_pulse_clock
          )
          valid &= ~dones.bool()
          torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
          feet_z = robot.data.site_pos_w[:, foot_ids, 2].mean(dim=1)
          standing = (
            (torso_z - feet_z > 0.6)
            & (robot.data.projected_gravity_b[:, 2] < -0.8)
            & valid
          )
          if not standing.any():
            continue
          mask = standing.float()
          sample_count = mask.sum()
          count += sample_count
          torso_pos = robot.data.body_link_pos_w[:, torso_id]
          torso_quat = robot.data.body_link_quat_w[:, torso_id]
          head_local = torch.zeros((args.num_envs, 3), device=env.device)
          head_local[:, 2] = A6_HEAD_OFFSET
          a6_head_height = (
            torso_pos[:, 2]
            + quat_apply(torso_quat, head_local)[:, 2]
            - env.scene.env_origins[:, 2]
          )
          upright = -robot.data.projected_gravity_b[:, 2]
          a6_contract = (
            (a6_head_height >= A6_FORMAL_HEAD_HEIGHT)
            & (upright >= A6_FORMAL_UPRIGHT)
            & standing
          )
          a6_head_height_final = torch.where(
            standing, a6_head_height, a6_head_height_final
          )
          a6_head_height_max = torch.where(
            standing,
            torch.maximum(a6_head_height_max, a6_head_height),
            a6_head_height_max,
          )
          a6_contract_steps += a6_contract.float()
          gravity_torso = quat_apply_inverse(torso_quat, world_down)
          tilt = torch.acos((-gravity_torso[:, 2]).clamp(-1.0, 1.0))
          support = robot.data.body_link_pos_w[:, ankle_ids].mean(dim=1)
          pelvis = robot.data.body_link_pos_w[:, pelvis_id]
          hands = robot.data.body_link_pos_w[:, hand_ids]
          feet = robot.data.site_pos_w[:, foot_ids]
          root_pos = robot.data.root_link_pos_w
          root_quat = robot.data.root_link_quat_w
          feet_local = torch.stack(
            (
              quat_apply_inverse(root_quat, feet[:, 0] - root_pos),
              quat_apply_inverse(root_quat, feet[:, 1] - root_pos),
            ),
            dim=1,
          )
          forward_offset = torch.abs(feet_local[:, 0, 0] - feet_local[:, 1, 0])
          height_offset = torch.abs(feet[:, 0, 2] - feet[:, 1, 2])
          feet_forward_offset_min = torch.where(
            standing,
            torch.minimum(feet_forward_offset_min, forward_offset),
            feet_forward_offset_min,
          )
          feet_height_offset_max = torch.where(
            standing,
            torch.maximum(feet_height_offset_max, height_offset),
            feet_height_offset_max,
          )
          feet_forward_offset_final = torch.where(
            standing, forward_offset, feet_forward_offset_final
          )
          hand_local = torch.stack(
            (
              quat_apply_inverse(torso_quat, hands[:, 0] - torso_pos),
              quat_apply_inverse(torso_quat, hands[:, 1] - torso_pos),
            ),
            dim=1,
          )
          left_q = robot.data.joint_pos[:, left_arm_ids]
          right_q = robot.data.joint_pos[:, right_arm_ids]
          waist_q = robot.data.joint_pos[:, waist_ids]
          leg_q = robot.data.joint_pos[:, leg_ids]

          values = {
            "torso_tilt_angle_rad": tilt,
            "torso_support_error_m": torch.linalg.vector_norm(
              torso_pos[:, :2] - support[:, :2], dim=-1
            ),
            "pelvis_support_error_m": torch.linalg.vector_norm(
              pelvis[:, :2] - support[:, :2], dim=-1
            ),
            "left_hand_target_error_m": torch.linalg.vector_norm(
              hand_local[:, 0] - hand_targets[0], dim=-1
            ),
            "right_hand_target_error_m": torch.linalg.vector_norm(
              hand_local[:, 1] - hand_targets[1], dim=-1
            ),
            "left_arm_target_mae_rad": torch.abs(left_q - left_target).mean(-1),
            "right_arm_target_mae_rad": torch.abs(right_q - right_target).mean(-1),
            "waist_target_mae_rad": torch.abs(waist_q - waist_target).mean(-1),
            "upper_action_abs_mean": torch.abs(actions[:, upper_action_ids]).mean(-1),
            "a6_head_height_m": a6_head_height,
            "a6_formal_contract": a6_contract.float(),
            "feet_forward_offset_m": forward_offset,
            "feet_height_offset_m": height_offset,
            "feet_lateral_separation_m": torch.abs(
              feet_local[:, 0, 1] - feet_local[:, 1, 1]
            ),
          }
          for name, value in values.items():
            sums[name] += (value * mask).sum()
          left_position_sum += (left_q * mask[:, None]).sum(0)
          right_position_sum += (right_q * mask[:, None]).sum(0)
          waist_position_sum += (waist_q * mask[:, None]).sum(0)
          leg_position_sum += (leg_q * mask[:, None]).sum(0)
          left_foot_local_sum += (feet_local[:, 0] * mask[:, None]).sum(0)
          right_foot_local_sum += (feet_local[:, 1] * mask[:, None]).sum(0)
          left_hand_local_sum += (hand_local[:, 0] * mask[:, None]).sum(0)
          right_hand_local_sum += (hand_local[:, 1] * mask[:, None]).sum(0)

      if count.item() == 0:
        raise RuntimeError(f"{posture} produced no standing samples")
      result = {name: float(value / count) for name, value in sums.items()}
      result.update(
        feet_forward_offset_min_m=float(
          feet_forward_offset_min[torch.isfinite(feet_forward_offset_min)].mean()
        ),
        feet_height_offset_max_m=float(feet_height_offset_max.mean()),
        feet_forward_offset_final_m=float(
          feet_forward_offset_final[torch.isfinite(feet_forward_offset_final)].mean()
        ),
        a6_head_height_final_mean_m=float(
          a6_head_height_final[torch.isfinite(a6_head_height_final)].mean()
        ),
        a6_head_height_final_min_m=float(
          a6_head_height_final[torch.isfinite(a6_head_height_final)].amin()
        ),
        a6_head_height_max_mean_m=float(
          a6_head_height_max[torch.isfinite(a6_head_height_max)].mean()
        ),
        a6_formal_contract_final_fraction=float(
          (
            torch.isfinite(a6_head_height_final)
            & (a6_head_height_final >= A6_FORMAL_HEAD_HEIGHT)
            & (-robot.data.projected_gravity_b[:, 2] >= A6_FORMAL_UPRIGHT)
            & valid
          ).float().mean()
        ),
        a6_formal_contract_any_fraction=float((a6_contract_steps > 0).float().mean()),
        a6_formal_contract_mean_steps=float(a6_contract_steps.mean()),
        left_arm_joint_mean={
          name: float(value)
          for name, value in zip(
            LEFT_ARM_JOINTS, left_position_sum / count, strict=True
          )
        },
        right_arm_joint_mean={
          name: float(value)
          for name, value in zip(
            RIGHT_ARM_JOINTS, right_position_sum / count, strict=True
          )
        },
        waist_joint_mean={
          name: float(value)
          for name, value in zip(WAIST_JOINTS, waist_position_sum / count, strict=True)
        },
        leg_joint_mean={
          name: float(value)
          for name, value in zip(LEG_JOINTS, leg_position_sum / count, strict=True)
        },
        left_foot_root_frame_mean=(left_foot_local_sum / count).tolist(),
        right_foot_root_frame_mean=(right_foot_local_sum / count).tolist(),
        left_hand_torso_frame_mean=(left_hand_local_sum / count).tolist(),
        right_hand_torso_frame_mean=(right_hand_local_sum / count).tolist(),
        standing_samples=int(count.item()),
      )
      report[posture] = result
      print(posture, json.dumps(result, sort_keys=True), flush=True)

    output = {
      "checkpoint": str(args.checkpoint),
      "num_envs_per_posture": args.num_envs,
      "steps": args.steps,
      "seed": args.seed,
      "action_scale": action_scale,
      "override": {
        "blend": args.override_blend,
        "gain": args.override_gain,
        "balance_blend": args.balance_blend,
        "balance_gain": args.balance_gain,
        "delay_steps": args.override_delay_steps,
        "ramp_steps": args.override_ramp_steps,
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
        "sagittal_target_control_mode": args.sagittal_target_control_mode,
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
      "results": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
  finally:
    env.close()


if __name__ == "__main__":
  main()
