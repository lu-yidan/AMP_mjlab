"""Replay collected A6 motor-on failure states through mixed resets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.tasks.host_recovery.a6_runtime import (
  _clear_a6_runtime_buffers,
  _park_and_place_plates,
  a6_reset,
)
from src.tasks.host_recovery.mdp.observations import UNACTUATED_STEPS


def _env_ids_tensor(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
  ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if env_ids is None:
    return ids
  if isinstance(env_ids, slice):
    return ids[env_ids]
  return env_ids.to(device=env.device, dtype=torch.long)


def _ensure_recovery_bank(
  env,
  recovery_bank: str,
  successful_only: bool,
) -> None:
  bank_path = str(Path(recovery_bank).resolve())
  current = getattr(env, "_a6_recovery_bank_path", None)
  if current is not None:
    if current != bank_path:
      raise RuntimeError("A6 recovery bank cannot change after initialization")
    return

  required = {
    "root_state",
    "joint_pos",
    "joint_vel",
    "direction",
    "controlled_step",
    "height",
    "upright",
    "eventually_recovered",
  }
  with np.load(bank_path, allow_pickle=False) as data:
    missing = required.difference(data.files)
    if missing:
      raise KeyError(f"recovery bank missing arrays: {sorted(missing)}")
    mask = np.ones(len(data["height"]), dtype=bool)
    if successful_only:
      mask &= data["eventually_recovered"].astype(bool)
    if not mask.any():
      raise RuntimeError("recovery bank has no eligible rows")
    arrays = {name: np.array(data[name][mask], copy=True) for name in required}

  float_names = ("root_state", "joint_pos", "joint_vel", "height", "upright")
  if not all(np.isfinite(arrays[name]).all() for name in float_names):
    raise ValueError("recovery bank contains non-finite state")
  if arrays["root_state"].shape[1] != 13:
    raise ValueError("recovery root_state must have width 13")
  if arrays["joint_pos"].shape != arrays["joint_vel"].shape:
    raise ValueError("recovery joint position/velocity shapes differ")
  if not np.isin(arrays["direction"], np.arange(4)).all():
    raise ValueError("recovery bank direction must be in [0, 3]")

  env._a6_recovery_bank_path = bank_path
  env._a6_recovery_bank_successful_only = successful_only
  env._a6_recovery_bank = {
    name: torch.as_tensor(value, device=env.device)
    for name, value in arrays.items()
  }
  env._a6_recovery_rng = torch.Generator(device=env.device).manual_seed(
    int(env.cfg.seed) + 141019
  )
  env._a6_recovery_reset = torch.zeros(
    env.num_envs, dtype=torch.bool, device=env.device
  )
  env._a6_recovery_pending = torch.zeros_like(env._a6_recovery_reset)
  env._a6_recovery_row = torch.full(
    (env.num_envs,), -1, dtype=torch.long, device=env.device
  )
  env._a6_recovery_initial_height = torch.zeros(
    env.num_envs, dtype=torch.float32, device=env.device
  )
  env._a6_recovery_initial_upright = torch.zeros_like(
    env._a6_recovery_initial_height
  )
  env._a6_recovery_controlled_step = torch.zeros(
    env.num_envs, dtype=torch.long, device=env.device
  )
  env._a6_recovery_reset_total = 0
  env._a6_recovery_activation_total = 0


def mixed_a6_recovery_reset(
  env,
  env_ids: torch.Tensor | slice | None = None,
  bank_split: str = "train",
  recovery_bank: str = "",
  recovery_fraction: float = 0.5,
  successful_only: bool = True,
) -> None:
  """Mix ordinary near resets with replayed motor-on failure states."""
  if not 0.0 <= recovery_fraction <= 1.0:
    raise ValueError("recovery_fraction must be in [0, 1]")
  ids = _env_ids_tensor(env, env_ids)
  a6_reset(env, ids, bank_split=bank_split)
  _ensure_recovery_bank(env, recovery_bank, successful_only)
  if not len(ids):
    return

  env._a6_recovery_reset[ids] = False
  env._a6_recovery_pending[ids] = False
  env._a6_recovery_row[ids] = -1
  num_recovery = int(round(len(ids) * recovery_fraction))
  if recovery_fraction > 0.0:
    num_recovery = max(1, num_recovery)
  num_recovery = min(len(ids), num_recovery)
  if num_recovery == 0:
    return

  order = torch.randperm(
    len(ids), generator=env._a6_recovery_rng, device=env.device
  )
  recovery_ids = ids[order[:num_recovery]]
  bank_size = len(env._a6_recovery_bank["height"])
  rows = torch.randint(
    bank_size,
    (num_recovery,),
    generator=env._a6_recovery_rng,
    device=env.device,
  )
  bank = env._a6_recovery_bank

  env._a6_scene[recovery_ids] = 0
  env._a6_stratum[recovery_ids] = 2
  env._a6_source[recovery_ids] = 0
  env._a6_direction[recovery_ids] = bank["direction"][rows].long()

  root_state = bank["root_state"][rows].float().clone()
  root_state[:, :3] += env.scene.env_origins[recovery_ids]
  robot = env.scene["robot"]
  robot.write_root_state_to_sim(root_state, env_ids=recovery_ids)
  robot.write_joint_state_to_sim(
    bank["joint_pos"][rows].float(),
    bank["joint_vel"][rows].float(),
    env_ids=recovery_ids,
  )
  _park_and_place_plates(env, recovery_ids)
  _clear_a6_runtime_buffers(env, recovery_ids)
  env._a6_ever_contact[recovery_ids] = False

  height = bank["height"][rows].float()
  upright = bank["upright"][rows].float()
  stage = torch.zeros(num_recovery, dtype=torch.long, device=env.device)
  stage = torch.where((height >= 0.55) & (upright >= 0.55), 1, stage)
  stage = torch.where((height >= 0.78) & (upright >= 0.72), 2, stage)
  stage = torch.where((height >= 1.08) & (upright >= 0.85), 3, stage)
  env._a6_stage[recovery_ids] = stage

  env._a6_recovery_reset[recovery_ids] = True
  env._a6_recovery_pending[recovery_ids] = True
  env._a6_recovery_row[recovery_ids] = rows
  env._a6_recovery_initial_height[recovery_ids] = height
  env._a6_recovery_initial_upright[recovery_ids] = upright
  env._a6_recovery_controlled_step[recovery_ids] = bank[
    "controlled_step"
  ][rows].long()
  env._a6_recovery_reset_total += num_recovery


def activate_a6_recovery_resets(
  env,
  env_ids: torch.Tensor | slice | None = None,
) -> None:
  """Skip the passive motor-off window for replayed motor-on states."""
  if not hasattr(env, "_a6_recovery_pending"):
    return
  ids = _env_ids_tensor(env, env_ids)
  ids = ids[env._a6_recovery_pending[ids]]
  if not len(ids):
    return
  env.episode_length_buf[ids] = UNACTUATED_STEPS + 1
  env._a6_recovery_pending[ids] = False
  env._a6_recovery_activation_total += len(ids)
