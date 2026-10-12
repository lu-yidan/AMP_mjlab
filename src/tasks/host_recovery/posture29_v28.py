"""V28 deterministic upper posture with learned lower-body stabilization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.host_recovery.mdp.actions import (
  IncrementalJointPositionAction,
  IncrementalJointPositionActionCfg,
)
from src.tasks.host_recovery.posture29_v25 import posture29_v25_env_cfg


UPPER_JOINTS = (
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


@dataclass(kw_only=True)
class StabilizedPostureActionCfg(IncrementalJointPositionActionCfg):
  arm_blend: float = 1.0
  arm_gain: float = 0.5
  balance_blend: float = 1.0
  balance_gain: float = 2.0
  standing_height: float = 0.6
  delay_steps: int = 25
  ramp_steps: int = 25

  def build(self, env):
    return StabilizedPostureAction(self, env)


class StabilizedPostureAction(IncrementalJointPositionAction):
  def __init__(self, cfg, env):
    super().__init__(cfg=cfg, env=env)
    robot = self._entity
    self._torso_id = robot.find_bodies("torso_link")[0][0]
    self._foot_ids = torch.tensor(
      robot.find_sites(("left_foot", "right_foot"))[0], device=self.device
    )
    self._upper_joint_ids = torch.tensor(
      robot.find_joints(UPPER_JOINTS, preserve_order=True)[0],
      device=self.device,
    )
    target_ids = list(self._target_ids)
    self._upper_action_ids = torch.tensor(
      [target_ids.index(int(joint_id)) for joint_id in self._upper_joint_ids],
      device=self.device,
    )
    self._upper_target = torch.tensor(UPPER_TARGET, device=self.device)
    self._standing_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._world_down = torch.zeros((self.num_envs, 3), device=self.device)
    self._world_down[:, 2] = -1.0
    self._last_policy_actions = torch.zeros(
      (self.num_envs, self.action_dim), device=self.device
    )
    self._last_applied_actions = torch.zeros_like(self._last_policy_actions)
    self._last_controller_mix = torch.zeros(self.num_envs, device=self.device)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._standing_count[ids] = 0
    self._last_controller_mix[ids] = 0.0

  def _selected_scale(self, action_ids: torch.Tensor):
    if not isinstance(self._scale, torch.Tensor):
      return self._scale
    if self._scale.ndim == 0:
      return self._scale
    return self._scale[action_ids]

  def process_actions(self, actions: torch.Tensor) -> None:
    applied = actions.clone()
    torso_z = self._entity.data.body_link_pos_w[:, self._torso_id, 2]
    feet_z = self._entity.data.site_pos_w[:, self._foot_ids, 2].mean(dim=1)
    standing = (
      (torso_z - feet_z > self.cfg.standing_height)
      & (self._entity.data.projected_gravity_b[:, 2] < -0.8)
    )
    self._standing_count = torch.where(
      standing, self._standing_count + 1, 0
    )
    ramp = (
      (self._standing_count - self.cfg.delay_steps).float()
      / max(self.cfg.ramp_steps, 1)
    ).clamp(0.0, 1.0)
    selected = actions[:, self._upper_action_ids]
    joint_pos = self._entity.data.joint_pos[:, self._upper_joint_ids]

    arm_ids = self._upper_action_ids[2:]
    arm_scale = self._selected_scale(arm_ids)
    arm_correction = (
      self.cfg.arm_gain * (self._upper_target[2:] - joint_pos[:, 2:]) / arm_scale
    ).clamp(-1.0, 1.0)
    arm_mix = (self.cfg.arm_blend * ramp)[:, None]
    applied[:, arm_ids] = selected[:, 2:] + arm_mix * (
      arm_correction - selected[:, 2:]
    )

    gravity_torso = quat_apply_inverse(
      self._entity.data.body_link_quat_w[:, self._torso_id], self._world_down
    )
    balance_delta = torch.stack(
      (gravity_torso[:, 1], -gravity_torso[:, 0]), dim=1
    )
    waist_ids = self._upper_action_ids[:2]
    waist_scale = self._selected_scale(waist_ids)
    balance_correction = (
      self.cfg.balance_gain * balance_delta / waist_scale
    ).clamp(-1.0, 1.0)
    balance_mix = (self.cfg.balance_blend * ramp)[:, None]
    applied[:, waist_ids] = selected[:, :2] + balance_mix * (
      balance_correction - selected[:, :2]
    )

    self._last_policy_actions[:] = actions
    self._last_applied_actions[:] = applied
    self._last_controller_mix[:] = ramp
    super().process_actions(applied)


def posture29_v28_env_cfg():
  cfg = posture29_v25_env_cfg()
  action_cfg = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = StabilizedPostureActionCfg(
    entity_name=action_cfg.entity_name,
    actuator_names=action_cfg.actuator_names,
    scale=action_cfg.scale,
  )

  for name in (
    "target_upper_pose_linear",
    "target_hands_beside_hips_error",
    "target_elbows_outside_torso",
    "target_elbow_flexion",
    "target_waist_roll_pitch_error",
  ):
    if name in cfg.rewards:
      cfg.rewards[name].weight = 0.0
  return cfg
