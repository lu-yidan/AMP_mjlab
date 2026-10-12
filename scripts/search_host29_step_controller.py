"""Search a short post-standing step pulse while keeping the V28 policy fixed."""

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
PULSE_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
)


def _candidate_reduce(
  values: torch.Tensor, candidate_ids: torch.Tensor, num_candidates: int
) -> torch.Tensor:
  totals = torch.zeros(num_candidates, device=values.device)
  counts = torch.zeros(num_candidates, device=values.device)
  totals.scatter_add_(0, candidate_ids, values)
  counts.scatter_add_(0, candidate_ids, torch.ones_like(values))
  return totals / counts.clamp_min(1.0)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-candidates", type=int, default=64)
  parser.add_argument("--repeats", type=int, default=2)
  parser.add_argument("--generations", type=int, default=4)
  parser.add_argument("--elite-fraction", type=float, default=0.2)
  parser.add_argument("--steps", type=int, default=800)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--step-delay", type=int, default=75)
  parser.add_argument("--phase1-steps", type=int, default=20)
  parser.add_argument("--phase2-steps", type=int, default=25)
  parser.add_argument("--release-steps", type=int, default=35)
  parser.add_argument("--score-window", type=int, default=100)
  args = parser.parse_args()

  if args.num_candidates < 4 or args.repeats < 1:
    raise ValueError("search requires at least four candidates and one repeat")
  if not 0.0 < args.elite_fraction < 1.0:
    raise ValueError("elite-fraction must lie in (0, 1)")

  num_envs = args.num_candidates * args.repeats
  cfg = flat29_env_cfg(play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  try:
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
    runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_scale = saved.get("infos", {}).get("env_state", {}).get(
      "action_rescale_value", 1.0
    )
    if not isinstance(saved_scale, torch.Tensor):
      saved_scale = torch.tensor(saved_scale)
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
      (num_envs, 1), action_scale, device=env.device
    )

    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.scene["robot"]
    torso_id = robot.find_bodies("torso_link")[0][0]
    foot_ids = torch.tensor(
      robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0],
      device=env.device,
    )
    upper_joint_ids = torch.tensor(
      robot.find_joints(UPPER_JOINT_NAMES, preserve_order=True)[0],
      device=env.device,
    )
    pulse_joint_ids = torch.tensor(
      robot.find_joints(PULSE_JOINT_NAMES, preserve_order=True)[0],
      device=env.device,
    )
    target_ids = list(action_term._target_ids)
    upper_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in upper_joint_ids.tolist()],
      device=env.device,
    )
    pulse_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in pulse_joint_ids.tolist()],
      device=env.device,
    )
    upper_target = torch.tensor(UPPER_TARGET, device=env.device)
    world_down = torch.zeros((num_envs, 3), device=env.device)
    world_down[:, 2] = -1.0
    candidate_ids = torch.arange(args.num_candidates, device=env.device).repeat_interleave(
      args.repeats
    )
    reset_cfg = env.event_manager.get_term_cfg("reset_base")

    phase1_mean = torch.tensor(
      (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.55,
        0.0,
        0.0,
        0.30,
        -0.60,
        0.0,
        0.0,
      ),
      device=env.device,
    )
    phase2_mean = torch.tensor(
      (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.20,
        0.0,
        0.0,
        -0.50,
        0.40,
        0.0,
        0.0,
      ),
      device=env.device,
    )
    mean = torch.cat((phase1_mean, phase2_mean))
    std = torch.full_like(mean, 0.25)
    for phase_offset in (0, len(PULSE_JOINT_NAMES)):
      std[phase_offset + 6] = 0.45
      std[phase_offset + 9] = 0.45
      std[phase_offset + 10] = 0.45
    elite_count = max(2, round(args.num_candidates * args.elite_fraction))
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 3100)
    history = []
    best_score = float("inf")
    best_params = None

    for generation in range(args.generations):
      samples = mean + std * torch.randn(
        (args.num_candidates, mean.numel()),
        generator=generator,
        device=env.device,
      )
      samples = samples.clamp(-1.0, 1.0)
      if generation == 0:
        samples[0] = mean.clamp(-1.0, 1.0)
      candidate_scores = torch.zeros(args.num_candidates, device=env.device)
      posture_metrics = {}

      for posture in POSTURES:
        reset_cfg.params["posture"] = posture
        wrapper.reset()
        expected = torch.tensor(POSTURE_QUATS[posture], device=env.device)
        alignment = (robot.data.root_link_quat_w * expected).sum(-1).abs()
        if not (alignment > 0.999).all():
          raise RuntimeError(f"{posture} reset mismatch: {alignment.min().item()}")

        observations = wrapper.get_observations()
        valid = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        standing_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        step_clock = torch.full_like(standing_count, -1)
        score_count = torch.zeros(num_envs, device=env.device)
        offset_sum = torch.zeros(num_envs, device=env.device)
        height_sum = torch.zeros(num_envs, device=env.device)
        tilt_sum = torch.zeros(num_envs, device=env.device)
        speed_sum = torch.zeros(num_envs, device=env.device)
        yaw_sum = torch.zeros(num_envs, device=env.device)
        standing_sum = torch.zeros(num_envs, device=env.device)

        with torch.no_grad():
          for step in range(args.steps):
            torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
            feet_z = robot.data.site_pos_w[:, foot_ids, 2].mean(dim=1)
            upright = robot.data.projected_gravity_b[:, 2] < -0.8
            standing = (torso_z - feet_z > 0.6) & upright & valid
            standing_count = torch.where(standing, standing_count + 1, 0)
            start = (step_clock < 0) & (standing_count >= args.step_delay)
            step_clock = torch.where(start, torch.zeros_like(step_clock), step_clock)

            actions = policy(observations["actor"])
            selected_upper = actions[:, upper_action_ids]
            upper_correction = (
              0.5
              * (upper_target - robot.data.joint_pos[:, upper_joint_ids])
              / action_scale
            ).clamp(-1.0, 1.0)
            upper_ramp = ((standing_count - 25).float() / 25).clamp(0.0, 1.0)
            upper_mix = upper_ramp[:, None]
            actions[:, upper_action_ids] = selected_upper + upper_mix * (
              upper_correction - selected_upper
            )
            gravity_torso = quat_apply_inverse(
              robot.data.body_link_quat_w[:, torso_id], world_down
            )
            balance_delta = torch.stack(
              (gravity_torso[:, 1], -gravity_torso[:, 0]), dim=1
            )
            balance_correction = (2.0 * balance_delta / action_scale).clamp(-1.0, 1.0)
            actions[:, upper_action_ids[:2]] = selected_upper[:, :2] + upper_mix * (
              balance_correction - selected_upper[:, :2]
            )

            phase1 = samples[candidate_ids, : len(PULSE_JOINT_NAMES)]
            phase2 = samples[candidate_ids, len(PULSE_JOINT_NAMES) :]
            phase1_mask = (step_clock >= 0) & (step_clock < args.phase1_steps)
            phase2_end = args.phase1_steps + args.phase2_steps
            phase2_mask = (step_clock >= args.phase1_steps) & (step_clock < phase2_end)
            release_mask = (step_clock >= phase2_end) & (
              step_clock < phase2_end + args.release_steps
            )
            pulse = torch.where(phase1_mask[:, None], phase1, actions[:, pulse_action_ids])
            pulse = torch.where(phase2_mask[:, None], phase2, pulse)
            release = (
              1.0 - (step_clock - phase2_end).float() / max(args.release_steps, 1)
            ).clamp(0.0, 1.0)
            release_values = actions[:, pulse_action_ids] + release[:, None] * (
              phase2 - actions[:, pulse_action_ids]
            )
            pulse = torch.where(release_mask[:, None], release_values, pulse)
            active = phase1_mask | phase2_mask | release_mask
            actions[:, pulse_action_ids] = torch.where(
              active[:, None], pulse, actions[:, pulse_action_ids]
            )

            observations, _, dones, _ = wrapper.step(actions)
            valid &= ~dones.bool()
            step_clock = torch.where(step_clock >= 0, step_clock + 1, step_clock)

            if step < args.steps - args.score_window:
              continue
            torso_z = robot.data.body_link_pos_w[:, torso_id, 2]
            feet = robot.data.site_pos_w[:, foot_ids]
            feet_z = feet[:, :, 2].mean(dim=1)
            upright = robot.data.projected_gravity_b[:, 2] < -0.8
            standing = (torso_z - feet_z > 0.6) & upright & valid
            root_pos = robot.data.root_link_pos_w
            root_quat = robot.data.root_link_quat_w
            left_foot = quat_apply_inverse(root_quat, feet[:, 0] - root_pos)
            right_foot = quat_apply_inverse(root_quat, feet[:, 1] - root_pos)
            gravity_torso = quat_apply_inverse(
              robot.data.body_link_quat_w[:, torso_id], world_down
            )
            score_count += 1.0
            offset_sum += torch.abs(left_foot[:, 0] - right_foot[:, 0])
            height_sum += torch.abs(feet[:, 0, 2] - feet[:, 1, 2])
            tilt_sum += torch.acos((-gravity_torso[:, 2]).clamp(-1.0, 1.0))
            speed_sum += torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
            yaw_sum += torch.abs(robot.data.root_link_ang_vel_w[:, 2])
            standing_sum += standing.float()

        denom = score_count.clamp_min(1.0)
        offset = offset_sum / denom
        height = height_sum / denom
        tilt = tilt_sum / denom
        speed = speed_sum / denom
        yaw = yaw_sum / denom
        standing_rate = standing_sum / denom
        score = (
          offset
          + 2.0 * (1.0 - standing_rate)
          + 2.0 * (~valid).float()
          + 0.75 * height
          + 0.15 * tilt
          + 0.10 * speed
          + 0.05 * yaw
        )
        score += 0.005 * samples[candidate_ids].square().mean(dim=1)
        candidate_scores += _candidate_reduce(score, candidate_ids, args.num_candidates)
        posture_metrics[posture] = {
          "offset": _candidate_reduce(offset, candidate_ids, args.num_candidates),
          "standing": _candidate_reduce(
            standing_rate, candidate_ids, args.num_candidates
          ),
          "speed": _candidate_reduce(speed, candidate_ids, args.num_candidates),
          "yaw": _candidate_reduce(yaw, candidate_ids, args.num_candidates),
          "tilt": _candidate_reduce(tilt, candidate_ids, args.num_candidates),
        }

      candidate_scores /= len(POSTURES)
      elite_indices = torch.topk(candidate_scores, elite_count, largest=False).indices
      elite = samples[elite_indices]
      mean = elite.mean(dim=0)
      std = elite.std(dim=0, unbiased=False).clamp_min(0.05) * 0.8
      generation_best = int(candidate_scores.argmin().item())
      generation_score = float(candidate_scores[generation_best])
      if generation_score < best_score:
        best_score = generation_score
        best_params = samples[generation_best].clone()
      metrics = {
        posture: {
          name: float(values[generation_best]) for name, values in values_by_name.items()
        }
        for posture, values_by_name in posture_metrics.items()
      }
      record = {
        "generation": generation,
        "score": generation_score,
        "params": samples[generation_best].tolist(),
        "metrics": metrics,
        "mean": mean.tolist(),
        "std": std.tolist(),
      }
      history.append(record)
      print(json.dumps(record, sort_keys=True), flush=True)

    output = {
      "checkpoint": str(args.checkpoint),
      "seed": args.seed,
      "num_candidates": args.num_candidates,
      "repeats": args.repeats,
      "generations": args.generations,
      "pulse_joint_names": PULSE_JOINT_NAMES,
      "step_delay": args.step_delay,
      "phase1_steps": args.phase1_steps,
      "phase2_steps": args.phase2_steps,
      "release_steps": args.release_steps,
      "best_score": best_score,
      "best_params": best_params.tolist() if best_params is not None else None,
      "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
  finally:
    env.close()


if __name__ == "__main__":
  main()
