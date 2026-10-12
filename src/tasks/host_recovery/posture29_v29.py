"""V29 sagittal foot alignment under the fixed V28 posture controller."""

from __future__ import annotations

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.host_recovery.posture29_v28 import posture29_v28_env_cfg


SAGITTAL_LEFT_JOINTS = (
  "left_hip_pitch_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
)
SAGITTAL_RIGHT_JOINTS = (
  "right_hip_pitch_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
)


def _controller_gate(env) -> torch.Tensor:
  action_term = env.action_manager.get_term("joint_pos")
  return (action_term._last_controller_mix >= 0.5).float()


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
  return loss.clamp(max=max_normalized_error) * _controller_gate(env)


def target_sagittal_leg_symmetry_error(
  env,
  joint_scale: tuple[float, float, float] = (0.30, 0.30, 0.30),
  max_normalized_error: float = 3.0,
  left_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
  right_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  asset = env.scene[left_cfg.name]
  left_q = asset.data.joint_pos[:, left_cfg.joint_ids]
  right_q = asset.data.joint_pos[:, right_cfg.joint_ids]
  scale = torch.tensor(joint_scale, device=left_q.device, dtype=left_q.dtype)
  normalized = torch.abs(left_q - right_q) / scale
  loss = torch.nn.functional.smooth_l1_loss(
    normalized, torch.zeros_like(normalized), reduction="none", beta=1.0
  )
  return loss.clamp(max=max_normalized_error).mean(-1) * _controller_gate(env)


def posture29_v29_env_cfg():
  cfg = posture29_v28_env_cfg()
  cfg.rewards["target_feet_side_by_side_huber_error"].weight = 0.0
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
  cfg.rewards["target_sagittal_leg_symmetry_error"] = RewardTermCfg(
    func=target_sagittal_leg_symmetry_error,
    weight=-0.80,
    params={
      "joint_scale": (0.30, 0.30, 0.30),
      "max_normalized_error": 3.0,
      "left_cfg": SceneEntityCfg(
        "robot", joint_names=SAGITTAL_LEFT_JOINTS, preserve_order=True
      ),
      "right_cfg": SceneEntityCfg(
        "robot", joint_names=SAGITTAL_RIGHT_JOINTS, preserve_order=True
      ),
    },
  )
  return cfg
