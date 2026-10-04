"""Supine HoST fine-tuning with the get-up task rewards from SMP.

The 23-DoF HoST actor and its observations are unchanged. This uses SMP's
head-height and upward-head-velocity shapes, not its 29-DoF diffusion prior.
"""

import torch

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.assets.robots.unitree_g1.g1_23dof_constants import get_spec

from .config.g1.env_cfgs import unitree_g1_host_standup_env_cfg


def robot_spec():
  spec = get_spec()
  spec.body("torso_link").add_site(name="smp_head", pos=(0, 0, 0.43))
  return spec


def _head(env):
  robot = env.scene["robot"]
  idx = robot.find_sites("smp_head")[0][0]
  return robot.data.site_pos_w[:, idx, 2] - env.scene.env_origins[:, 2], robot.data.site_lin_vel_w[:, idx, 2]


def smp_head_height(env):
  z, _ = _head(env)
  return torch.exp(-torch.clamp(1.1 - z, min=0).square())


def smp_upward_velocity(env):
  z, vz = _head(env)
  shaped = torch.exp(-100.0 * torch.clamp(0.25 - vz, min=0).square())
  return torch.where(z < 0.9, shaped, torch.ones_like(shaped))


def head_world_height(env):
  return _head(env)[0]


def upright_standing(env):
  z, _ = _head(env)
  upright = env.scene["robot"].data.projected_gravity_b[:, 2] < -0.8
  return ((z > 1.0) & upright).float()


def supine_smp_env_cfg(play=False):
  cfg = unitree_g1_host_standup_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = robot_spec
  cfg.events["reset_base"].params.update(posture="supine", height=None, xy_range=0.0)
  cfg.events.pop("init_pull_force", None)
  cfg.events.pop("apply_pull_force", None)
  cfg.curriculum = {}
  cfg.actions["joint_pos"].scale = 0.25
  cfg.rewards["smp_head_height"] = RewardTermCfg(func=smp_head_height, weight=0.3)
  cfg.rewards["smp_upward_velocity"] = RewardTermCfg(func=smp_upward_velocity, weight=0.7)
  cfg.metrics["smp_head_world_height"] = MetricsTermCfg(func=head_world_height)
  cfg.metrics["upright_standing"] = MetricsTermCfg(func=upright_standing)
  return cfg
