"""Historical A6 three-scene geometry for the HoST adapter.

This module intentionally contains only the A6 scene boundary: flat ground,
a vertically guided plate, and a free plate. Stairs, slopes, and other boxes
are not part of the historical A6 protocol.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np


DIRECTIONS = ("supine", "prone", "left_side_down", "right_side_down")
STRATA = (
  "flat",
  "vertical_plate",
  "free_plate",
  "stair_interior",
  "stair_edge",
  "stair_straddle",
  "slope_interior",
  "slope_edge",
)

# Kept explicit so later terrain experiments cannot silently enter A6.
BOXES: tuple[tuple[Any, Any, Any], ...] = ()
PLATE_HALF_SIZE = (0.45, 0.32, 0.035)
PLATE_FRICTION = (1.2, 0.01, 0.001)


def add_terrain(body: Any) -> None:
  """Add the unreachable terrain sensor target used by the scene contract."""
  body.add_geom(
    name="sensor_target_parked",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=(100.0, 100.0, -10.0),
    size=(0.01,),
    rgba=(0.0, 0.0, 0.0, 0.0),
  )


def terrain_spec() -> Any:
  """Return the flat A6 terrain specification."""
  spec = mujoco.MjSpec()
  add_terrain(spec.worldbody.add_body(name="surfaces"))
  return spec


def _plate_spec(*, free: bool, body_name: str, geom_name: str) -> Any:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=body_name)
  if free:
    body.add_freejoint(name=f"{body_name}_joint")
  else:
    body.add_joint(
      name=f"{body_name}_slide",
      type=mujoco.mjtJoint.mjJNT_SLIDE,
      axis=(0.0, 0.0, 1.0),
      limited=True,
      range=(-1.2, 0.0),
      damping=60.0,
    )
  geom = body.add_geom(
    name=geom_name,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=PLATE_HALF_SIZE,
    mass=8.0,
    friction=PLATE_FRICTION,
    rgba=(0.85, 0.45, 0.12, 0.8),
    solref=(0.01, 1.0),
  )
  if not free:
    geom.priority = 1
    geom.solimp = (0.98, 0.995, 0.001, 0.5, 2.0)
  return spec


def guided_plate_spec() -> Any:
  """Return the vertically guided historical plate entity."""
  return _plate_spec(
    free=False,
    body_name="escape_plate",
    geom_name="escape_plate_geom",
  )


def free_plate_spec() -> Any:
  """Return the free-jointed historical plate entity."""
  return _plate_spec(free=True, body_name="plate", geom_name="plate_geom")


def quotas(num_envs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return deterministic 50/25/25 scene and four-direction allocations."""
  if num_envs < 32 or num_envs % 32:
    raise ValueError("A6 requires num_envs >= 32 and divisible by 32")
  counts = (num_envs // 2, num_envs // 4, num_envs // 4)
  scene = np.repeat(np.arange(3), counts)
  direction = np.concatenate(
    [np.repeat(np.arange(4), count // 4) for count in counts]
  )
  return scene, direction, scene.copy()


def support_height(xy: Any) -> Any:
  """Return the top support height for the plane plus optional future boxes."""
  import torch

  is_tensor = torch.is_tensor(xy)
  output = (
    torch.zeros_like(xy[..., 0])
    if is_tensor
    else np.zeros(xy.shape[:-1], dtype=np.asarray(xy).dtype)
  )
  for position, size, quat in BOXES:
    angle = 2.0 * math.atan2(quat[2], quat[0])
    cosine, sine = math.cos(angle), math.sin(angle)
    for sign in (-1.0, 1.0):
      if abs(sine) > 1e-8:
        z = position[2] + (
          cosine * (xy[..., 0] - position[0]) - sign * size[0]
        ) / sine
        local_z = sine * (xy[..., 0] - position[0]) + cosine * (z - position[2])
        valid = (
          (abs(xy[..., 1] - position[1]) <= size[1])
          & (abs(local_z) <= size[2])
          & (z >= 0)
        )
        output = torch.where(valid, torch.maximum(output, z), output) if is_tensor else np.where(valid, np.maximum(output, z), output)
    z = position[2] + (size[2] - sine * (xy[..., 0] - position[0])) / cosine
    local_x = cosine * (xy[..., 0] - position[0]) - sine * (z - position[2])
    valid = (abs(local_x) <= size[0]) & (abs(xy[..., 1] - position[1]) <= size[1])
    output = torch.where(valid, torch.maximum(output, z), output) if is_tensor else np.where(valid, np.maximum(output, z), output)
  return output


def scene_spec(spec: Any) -> Any:
  """Add the flat terrain contract to an existing MuJoCo specification."""
  add_terrain(spec.worldbody.add_body(name="surfaces"))
  return spec
