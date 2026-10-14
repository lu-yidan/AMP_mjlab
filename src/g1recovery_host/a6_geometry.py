"""Historical A6 flat, vertical-slider plate, and free-plate scene fixtures."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

SCENES = ("flat", "vertical_plate", "free_plate")
DIRECTIONS = ("supine", "prone", "left_side_down", "right_side_down")
PLATE_HALF_SIZE = (0.45, 0.32, 0.035)
PLATE_FRICTION = (1.2, 0.01, 0.001)
PLATE_MASS_KG = 6.0
A6_PHYSICS_SUFFIX = "a6_29_v2"


def quotas(num_envs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign the frozen 50/25/25 scene split and balanced low directions."""

    if num_envs < 32 or num_envs % 32:
        raise ValueError("A6 num_envs must be at least 32 and divisible by 32")
    counts = (num_envs // 2, num_envs // 4, num_envs // 4)
    scene = np.repeat(np.arange(3), counts)
    direction = np.concatenate(
        [np.repeat(np.arange(4), count // 4) for count in counts]
    )
    # Obstacles always start low; flat resets retain the L/M/H 40/40/20 split.
    stratum = np.zeros(num_envs, dtype=np.int64)
    flat_per_direction = counts[0] // 4
    for direction_id in range(4):
        rows = np.flatnonzero((scene == 0) & (direction == direction_id))
        low = round(flat_per_direction * 0.4)
        late = round(flat_per_direction * 0.2)
        stratum[rows[low : flat_per_direction - late]] = 1
        stratum[rows[flat_per_direction - late :]] = 2
    return scene, direction, stratum


def guided_plate_spec() -> mujoco.MjSpec:
    """A massive plate constrained to one vertical translational DoF."""

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="escape_plate")
    body.add_joint(
        name="escape_plate_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(0.0, 0.0, 1.0),
        limited=True,
        range=(-1.2, 0.0),
        damping=60.0,
    )
    geom = body.add_geom(
        name="escape_plate_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=PLATE_HALF_SIZE,
        mass=PLATE_MASS_KG,
        friction=PLATE_FRICTION,
        rgba=(0.12, 0.72, 0.24, 0.8),
    )
    geom.priority = 1
    geom.solref = (0.01, 1.0)
    geom.solimp = (0.98, 0.995, 0.001, 0.5, 2.0)
    return spec


def free_plate_spec() -> mujoco.MjSpec:
    """The same-sized rigid plate with all six free-body DoFs."""

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="plate")
    body.add_freejoint(name="free_plate_joint")
    geom = body.add_geom(
        name="plate_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=PLATE_HALF_SIZE,
        mass=PLATE_MASS_KG,
        friction=PLATE_FRICTION,
        rgba=(0.85, 0.45, 0.12, 0.8),
    )
    geom.solref = (0.01, 1.0)
    return spec


def scene_spec(spec: mujoco.MjSpec) -> None:
    """Keep the unreachable terrain sensor target in the A6 scene contract."""

    surfaces = spec.worldbody.add_body(name="surfaces")
    surfaces.add_geom(
        name="sensor_target_parked",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(100.0, 100.0, -10.0),
        size=(0.01,),
        rgba=(0.0, 0.0, 0.0, 0.0),
    )


def robot_bounds(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservative AABBs of the HoST robot's contact geoms, in world metres."""

    ids = env._a6_collision_geoms
    pos = env.sim.data.geom_xpos[:, ids]
    mat = env.sim.data.geom_xmat[:, ids]
    size = env.sim.model.geom_size[:, ids]
    geom_type = env.sim.model.geom_type[ids]
    ext = torch.einsum("ngij,ngj->ngi", mat.abs(), size)
    sphere = geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    capsule = geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    ext = torch.where(sphere[None, :, None], size[:, :, :1], ext)
    capsule_ext = size[:, :, :1] + size[:, :, 1:2] * mat[:, :, :, 2].abs()
    ext = torch.where(capsule[None, :, None], capsule_ext, ext)
    return pos, ext


def footprint(
    pos: torch.Tensor,
    ext: torch.Tensor,
    plate_pos: torch.Tensor,
    plate_ext: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A6 coverage count, overlap-depth sum, and XY clearance (no height exemption)."""

    overlap = (
        ext[..., :2]
        + plate_ext[:, None, :2]
        - (pos[..., :2] - plate_pos[:, None, :2]).abs()
    )
    covered = (overlap > 0).all(-1)
    score = torch.where(covered, overlap.clamp_min(0).amin(-1), 0.0).sum(-1)
    clearance = torch.where(covered, 0.0, (-overlap).clamp_min(0).norm(dim=-1)).amin(-1)
    return covered.sum(-1), score, clearance


def plate_footprint_inputs(env):
    """Return robot and plate bounds for planar, height-independent rewards."""
    pos, ext = robot_bounds(env)
    rows = torch.arange(env.num_envs, device=env.device)
    plate_ids = env._a6_plate_geoms[env._a6_scene.clamp_min(1) - 1]
    plate_pos = env.sim.data.geom_xpos[rows, plate_ids]
    plate_mat = env.sim.data.geom_xmat[rows, plate_ids]
    plate_size = env.sim.model.geom_size[rows, plate_ids]
    plate_ext = torch.einsum("nij,nj->ni", plate_mat.abs(), plate_size)
    return pos, ext, plate_pos, plate_ext


def plate_geometry(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the selected plate into XY; vertical lifting alone cannot clear it."""

    return footprint(*plate_footprint_inputs(env))


def completion_clearance(env) -> torch.Tensor:
    """A6 completion clearance: body parts fully above the plate are free.

    G uses :func:`plate_geometry` instead, intentionally without this height
    exemption, so lifting the plate vertically earns no planar-progress credit.
    """
    pos, ext = robot_bounds(env)
    rows = torch.arange(env.num_envs, device=env.device)
    plate_ids = env._a6_plate_geoms[env._a6_scene.clamp_min(1) - 1]
    plate_pos = env.sim.data.geom_xpos[rows, plate_ids]
    plate_mat = env.sim.data.geom_xmat[rows, plate_ids]
    plate_size = env.sim.model.geom_size[rows, plate_ids]
    plate_ext = torch.einsum("nij,nj->ni", plate_mat.abs(), plate_size)
    separation = (
        (pos[..., :2] - plate_pos[:, None, :2]).abs()
        - ext[..., :2]
        - plate_ext[:, None, :2]
    ).amax(-1)
    above = (pos[..., 2] - ext[..., 2]) > (
        plate_pos[:, None, 2] + plate_ext[:, None, 2] + 0.025
    )
    return torch.where(above, 1.0, separation).amin(-1)
