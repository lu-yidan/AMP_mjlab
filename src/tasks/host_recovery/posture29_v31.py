"""V31 fixed right-step target with learned lower-body stabilization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.host_recovery.mdp.actions import IncrementalJointPositionAction
from src.tasks.host_recovery.posture29_v28 import (
  StabilizedPostureAction,
  StabilizedPostureActionCfg,
)
from src.tasks.host_recovery.posture29_v30 import posture29_v30_env_cfg


RIGHT_SAGITTAL_JOINTS = (
  "right_hip_pitch_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
)
RIGHT_SAGITTAL_TARGET = (0.30, 2.10, -1.00)


@dataclass(kw_only=True)
class FixedRightStepPostureActionCfg(StabilizedPostureActionCfg):
  sagittal_blend: float = 1.0
  sagittal_gain: float = 1.0
  sagittal_max_action: float = 1.0
  sagittal_delay_steps: int = 50
  sagittal_ramp_steps: int = 100
  sagittal_target: tuple[float, float, float] = RIGHT_SAGITTAL_TARGET

  def build(self, env):
    return FixedRightStepPostureAction(self, env)


class FixedRightStepPostureAction(StabilizedPostureAction):
  def __init__(self, cfg, env):
    super().__init__(cfg=cfg, env=env)
    robot = self._entity
    self._right_sagittal_joint_ids = torch.tensor(
      robot.find_joints(RIGHT_SAGITTAL_JOINTS, preserve_order=True)[0],
      device=self.device,
    )
    target_ids = list(self._target_ids)
    self._right_sagittal_action_ids = torch.tensor(
      [
        target_ids.index(int(joint_id))
        for joint_id in self._right_sagittal_joint_ids
      ],
      device=self.device,
    )
    self._right_sagittal_target = torch.tensor(
      cfg.sagittal_target, device=self.device
    )
    self._last_sagittal_mix = torch.zeros(self.num_envs, device=self.device)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._last_sagittal_mix[ids] = 0.0

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

    upper_ramp = (
      (self._standing_count - self.cfg.delay_steps).float()
      / max(self.cfg.ramp_steps, 1)
    ).clamp(0.0, 1.0)
    selected_upper = actions[:, self._upper_action_ids]
    upper_joint_pos = self._entity.data.joint_pos[:, self._upper_joint_ids]

    arm_ids = self._upper_action_ids[2:]
    arm_scale = self._selected_scale(arm_ids)
    arm_correction = (
      self.cfg.arm_gain
      * (self._upper_target[2:] - upper_joint_pos[:, 2:])
      / arm_scale
    ).clamp(-1.0, 1.0)
    arm_mix = (self.cfg.arm_blend * upper_ramp)[:, None]
    applied[:, arm_ids] = selected_upper[:, 2:] + arm_mix * (
      arm_correction - selected_upper[:, 2:]
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
    balance_mix = (self.cfg.balance_blend * upper_ramp)[:, None]
    applied[:, waist_ids] = selected_upper[:, :2] + balance_mix * (
      balance_correction - selected_upper[:, :2]
    )

    sagittal_joint_pos = self._entity.data.joint_pos[
      :, self._right_sagittal_joint_ids
    ]
    sagittal_scale = self._selected_scale(self._right_sagittal_action_ids)
    sagittal_correction = (
      self.cfg.sagittal_gain
      * (self._right_sagittal_target - sagittal_joint_pos)
      / sagittal_scale
    ).clamp(-self.cfg.sagittal_max_action, self.cfg.sagittal_max_action)
    sagittal_ramp = (
      (self._standing_count - self.cfg.sagittal_delay_steps).float()
      / max(self.cfg.sagittal_ramp_steps, 1)
    ).clamp(0.0, 1.0)
    sagittal_mix = (self.cfg.sagittal_blend * sagittal_ramp)[:, None]
    selected_sagittal = actions[:, self._right_sagittal_action_ids]
    applied[:, self._right_sagittal_action_ids] = selected_sagittal + (
      sagittal_mix * (sagittal_correction - selected_sagittal)
    )

    self._last_policy_actions[:] = actions
    self._last_applied_actions[:] = applied
    self._last_controller_mix[:] = upper_ramp
    self._last_sagittal_mix[:] = sagittal_ramp
    IncrementalJointPositionAction.process_actions(self, applied)


def target_feet_forward_alignment_error(
  env,
  forward_scale: float = 0.08,
  max_normalized_error: float = 4.0,
  foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=()),
) -> torch.Tensor:
  asset = env.scene[foot_cfg.name]
  feet_w = asset.data.site_pos_w[:, foot_cfg.site_ids]
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  left_b = quat_apply_inverse(root_quat, feet_w[:, 0] - root_pos)
  right_b = quat_apply_inverse(root_quat, feet_w[:, 1] - root_pos)
  normalized = torch.abs(left_b[:, 0] - right_b[:, 0]) / forward_scale
  loss = torch.nn.functional.smooth_l1_loss(
    normalized, torch.zeros_like(normalized), reduction="none", beta=1.0
  )
  action_term = env.action_manager.get_term("joint_pos")
  controller_gate = (action_term._last_sagittal_mix >= 0.5).float()
  return loss.clamp(max=max_normalized_error) * controller_gate


def posture29_v31_env_cfg():
  cfg = posture29_v30_env_cfg()
  action_cfg = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = FixedRightStepPostureActionCfg(
    entity_name=action_cfg.entity_name,
    actuator_names=action_cfg.actuator_names,
    scale=action_cfg.scale,
  )
  cfg.rewards["target_feet_forward_alignment_error"] = RewardTermCfg(
    func=target_feet_forward_alignment_error,
    weight=-1.20,
    params={
      "forward_scale": 0.08,
      "max_normalized_error": 4.0,
      "foot_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot"), preserve_order=True
      ),
    },
  )
  cfg.rewards["target_sagittal_leg_symmetry_error"].weight = 0.0
  return cfg
