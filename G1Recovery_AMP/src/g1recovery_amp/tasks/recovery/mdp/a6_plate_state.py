"""Contact, clearance, completion, and invalid-state tracking for A6 plates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

A6_MIN_PLANAR_CLEARANCE = 0.025
A6_CLEAR_HOLD_STEPS = 15
A6_MAX_PLATE_FORCE = 1500.0
A6_CATASTROPHIC_PLATE_FORCE = 2500.0
A6_MAX_PLATE_PENETRATION = 0.020
A6_MAX_NO_CONTACT_STEPS = 25
A6_HEAD_OFFSET_IN_TORSO = (0.0, 0.0, 0.43)
A6_MEANINGFUL_PROGRESS_DELTA = 0.001
A6_SEPARATION_PROGRESS_SCALE = 0.025
A6_LATERAL_PROGRESS_SCALE = 0.025


def advance_no_progress_time(
    *,
    current_time: torch.Tensor,
    coverage_delta: torch.Tensor,
    clearance_delta: torch.Tensor,
    separation_progress: torch.Tensor,
    eligible: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    """Track time since the last meaningful geometric escape improvement."""

    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    meaningful_progress = (
        (coverage_delta >= A6_MEANINGFUL_PROGRESS_DELTA)
        | (clearance_delta >= A6_MEANINGFUL_PROGRESS_DELTA)
        | (
            separation_progress
            >= A6_MEANINGFUL_PROGRESS_DELTA / A6_SEPARATION_PROGRESS_SCALE
        )
    )
    advanced = torch.where(
        meaningful_progress,
        torch.zeros_like(current_time),
        current_time + step_dt,
    )
    return torch.where(eligible, advanced, torch.zeros_like(current_time))


def plate_invalid_reasons(
    *,
    active: torch.Tensor,
    force: torch.Tensor,
    depth: torch.Tensor,
    ever_contact: torch.Tensor,
    episode_steps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return force, penetration, and no-contact invalid masks separately."""

    invalid_force = active & (force > A6_MAX_PLATE_FORCE)
    invalid_penetration = active & (depth < -A6_MAX_PLATE_PENETRATION)
    invalid_no_contact = (
        active & (episode_steps > A6_MAX_NO_CONTACT_STEPS) & ~ever_contact
    )
    return invalid_force, invalid_penetration, invalid_no_contact


def sustained_force_invalid_from_state(
    *,
    active: torch.Tensor,
    force: torch.Tensor,
    consecutive_steps: torch.Tensor,
    force_limit: float = A6_MAX_PLATE_FORCE,
    hold_steps: int = 1,
    catastrophic_force: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require a sustained force violation while retaining an emergency cap."""

    if force_limit <= 0.0:
        raise ValueError("plate force limit must be positive")
    if hold_steps <= 0:
        raise ValueError("plate force hold steps must be positive")
    if catastrophic_force is not None and catastrophic_force <= force_limit:
        raise ValueError("catastrophic plate force must exceed the sustained limit")
    over_limit = active & (force > force_limit)
    next_steps = torch.where(
        over_limit,
        consecutive_steps + 1,
        torch.zeros_like(consecutive_steps),
    )
    invalid = next_steps >= hold_steps
    if catastrophic_force is not None:
        invalid |= active & (force > catastrophic_force)
    return next_steps, invalid


def lateral_progress_from_state(
    *,
    displacement_xy: torch.Tensor,
    lateral_axis_xy: torch.Tensor,
    best_distance: torch.Tensor,
    progress_scale: float = A6_LATERAL_PROGRESS_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return best-so-far lateral distance and one-way progress reward."""

    if displacement_xy.shape != lateral_axis_xy.shape:
        raise ValueError("displacement and lateral axis shapes must match")
    if displacement_xy.shape[-1] != 2:
        raise ValueError("lateral displacement must use planar x/y vectors")
    if best_distance.shape != displacement_xy.shape[:-1]:
        raise ValueError("best lateral distance shape must match the batch")
    if progress_scale <= 0.0:
        raise ValueError("lateral progress scale must be positive")
    distance = (displacement_xy * lateral_axis_xy).sum(dim=-1).abs()
    progress = ((distance - best_distance) / progress_scale).clamp(0.0, 1.0)
    return torch.maximum(best_distance, distance), progress


def _robot_collision_bounds(
    env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor]:
    geom_ids = env._a6_robot_collision_geom_ids
    pos = env.sim.data.geom_xpos[:, geom_ids]
    mat = env.sim.data.geom_xmat[:, geom_ids]
    size = env.sim.model.geom_size[:, geom_ids]
    geom_type = env.sim.model.geom_type[geom_ids]
    extent = torch.einsum("ngij,ngj->ngi", mat.abs(), size)
    sphere = geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    capsule = geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    extent = torch.where(sphere[None, :, None], size[:, :, :1], extent)
    capsule_extent = size[:, :, :1] + size[:, :, 1:2] * mat[:, :, :, 2].abs()
    extent = torch.where(capsule[None, :, None], capsule_extent, extent)
    return pos, extent


def a6_plate_geometry(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return overlap score and conservative planar clearance for each world."""

    pos, extent = _robot_collision_bounds(env)
    free_scene = env._a6_reset_scene == 2
    rows = torch.arange(env.num_envs, device=env.device)
    plate_ids = torch.stack(env._a6_plate_geom_ids)[free_scene.long()]
    plate_pos = env.sim.data.geom_xpos[rows, plate_ids]
    plate_rot = env.sim.data.geom_xmat[rows, plate_ids]
    plate_size = env.sim.model.geom_size[rows, plate_ids]
    plate_extent = torch.einsum("nij,nj->ni", plate_rot.abs(), plate_size)

    delta = (
        (pos[:, :, :2] - plate_pos[:, None, :2]).abs()
        - extent[:, :, :2]
        - plate_extent[:, None, :2]
    )
    separation = delta.amax(dim=-1)
    # Completion may ignore a body part that has fully risen above the board.
    above = (
        pos[:, :, 2] - extent[:, :, 2]
        > plate_pos[:, None, 2] + plate_extent[:, None, 2] + 0.025
    )
    separation = torch.where(above, torch.ones_like(separation), separation)
    score = -(-separation).clamp_min(0).sum(dim=-1)
    score += 0.5 * separation.amin(dim=-1).clamp(0.0, 0.04)
    return score, separation.amin(dim=-1)


def path_footprint(
    robot_pos: torch.Tensor,
    robot_extent: torch.Tensor,
    plate_pos: torch.Tensor,
    plate_extent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return covered-body count, planar overlap score, and planar clearance."""

    overlap = (
        robot_extent[..., :2]
        + plate_extent[:, None, :2]
        - (robot_pos[..., :2] - plate_pos[:, None, :2]).abs()
    )
    covered = (overlap > 0.0).all(dim=-1)
    score = torch.where(covered, overlap.clamp_min(0.0).amin(dim=-1), 0.0).sum(dim=-1)
    clearance = torch.where(
        covered,
        0.0,
        (-overlap).clamp_min(0.0).norm(dim=-1),
    ).amin(dim=-1)
    return covered.sum(dim=-1), score, clearance


def a6_path_geometry(
    env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure horizontal path progress without the completion height exemption."""

    pos, extent = _robot_collision_bounds(env)
    free_scene = env._a6_reset_scene == 2
    rows = torch.arange(env.num_envs, device=env.device)
    plate_ids = torch.stack(env._a6_plate_geom_ids)[free_scene.long()]
    plate_pos = env.sim.data.geom_xpos[rows, plate_ids]
    plate_rot = env.sim.data.geom_xmat[rows, plate_ids]
    plate_size = env.sim.model.geom_size[rows, plate_ids]
    plate_extent = torch.einsum("nij,nj->ni", plate_rot.abs(), plate_size)
    return path_footprint(pos, extent, plate_pos, plate_extent)


def update_plate_flags(
    *,
    active: torch.Tensor,
    contact: torch.Tensor,
    force: torch.Tensor,
    depth: torch.Tensor,
    clearance: torch.Tensor,
    ever_contact: torch.Tensor,
    clear_hold: torch.Tensor,
    episode_steps: torch.Tensor,
    min_planar_clearance: float = A6_MIN_PLANAR_CLEARANCE,
    clear_hold_steps: int = A6_CLEAR_HOLD_STEPS,
    invalid_force_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure tensor implementation of A6 contact/escape/invalid transitions."""

    if min_planar_clearance < 0.0:
        raise ValueError("min_planar_clearance must be nonnegative")
    if clear_hold_steps <= 0:
        raise ValueError("clear_hold_steps must be positive")

    next_ever_contact = ever_contact | (contact & active)
    clear = (
        active & next_ever_contact & ~contact & (clearance >= min_planar_clearance)
    )
    next_clear_hold = torch.where(clear, clear_hold + 1, 0)
    escaped = next_clear_hold >= clear_hold_steps
    invalid_reasons = plate_invalid_reasons(
        active=active,
        force=force,
        depth=depth,
        ever_contact=next_ever_contact,
        episode_steps=episode_steps,
    )
    if invalid_force_override is not None:
        if invalid_force_override.shape != active.shape:
            raise ValueError("invalid force override must match active shape")
        invalid_reasons = (
            invalid_force_override,
            invalid_reasons[1],
            invalid_reasons[2],
        )
    invalid = invalid_reasons[0] | invalid_reasons[1] | invalid_reasons[2]
    return next_ever_contact, next_clear_hold, escaped, invalid


def initialize_a6_plate_state(env: ManagerBasedRlEnv) -> None:
    """Allocate per-environment state once after scene/reset initialization."""

    if hasattr(env, "_a6_plate_ever_contact"):
        return
    n = env.num_envs
    device = env.device
    env._a6_plate_ever_contact = torch.zeros(n, dtype=torch.bool, device=device)
    env._a6_plate_clear_hold = torch.zeros(n, dtype=torch.int64, device=device)
    env._a6_plate_escaped = torch.zeros(n, dtype=torch.bool, device=device)
    env._a6_plate_invalid = torch.zeros(n, dtype=torch.bool, device=device)
    env._a6_plate_force_over_limit_steps = torch.zeros(
        n, dtype=torch.int64, device=device
    )
    env._a6_plate_initial_root_xy = torch.zeros((n, 2), device=device)
    env._a6_plate_initial_lateral_axis_xy = torch.zeros((n, 2), device=device)
    for reason in ("force", "penetration", "no_contact"):
        setattr(
            env,
            f"_a6_plate_last_invalid_{reason}",
            torch.zeros(n, dtype=torch.bool, device=device),
        )
        # The environment resets terminated worlds inside ``env.step``.  Keep
        # a separate snapshot written by the termination term so diagnostics
        # can still recover the cause after that automatic reset has run.
        setattr(
            env,
            f"_a6_plate_terminal_invalid_{reason}",
            torch.zeros(n, dtype=torch.bool, device=device),
        )
    for name in (
        "best_score",
        "progress",
        "clearance",
        "force",
        "initial_overlap",
        "best_distance",
        "best_lateral_distance",
        "lateral_progress",
        "separation_progress",
        "path_best_score",
        "path_best_clearance",
        "path_initial_count",
        "path_current_count",
        "path_current_clearance",
        "path_coverage_delta",
        "path_clearance_delta",
        "path_hand_support",
        "head_height",
        "no_progress_time",
    ):
        setattr(env, f"_a6_plate_{name}", torch.zeros(n, device=device))
    env._a6_plate_tick = -1


def reset_a6_plate_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Clear temporal state for the environments that have just reset."""

    initialize_a6_plate_state(env)
    score, clearance = a6_plate_geometry(env)
    path_count, path_score, path_clearance = a6_path_geometry(env)
    env._a6_plate_best_score[env_ids] = score[env_ids]
    env._a6_plate_initial_overlap[env_ids] = (-score[env_ids]).clamp_min(0.01)
    env._a6_plate_clearance[env_ids] = clearance[env_ids]
    env._a6_plate_path_best_score[env_ids] = path_score[env_ids]
    env._a6_plate_path_best_clearance[env_ids] = path_clearance[env_ids]
    env._a6_plate_path_initial_count[env_ids] = path_count[env_ids].float()
    env._a6_plate_path_current_count[env_ids] = path_count[env_ids].float()
    env._a6_plate_path_current_clearance[env_ids] = path_clearance[env_ids]
    for name in (
        "progress",
        "force",
        "best_distance",
        "best_lateral_distance",
        "lateral_progress",
        "separation_progress",
        "path_coverage_delta",
        "path_clearance_delta",
        "path_hand_support",
        "head_height",
        "no_progress_time",
    ):
        getattr(env, f"_a6_plate_{name}")[env_ids] = 0.0
    env._a6_plate_ever_contact[env_ids] = False
    env._a6_plate_clear_hold[env_ids] = 0
    env._a6_plate_escaped[env_ids] = False
    env._a6_plate_invalid[env_ids] = False
    env._a6_plate_force_over_limit_steps[env_ids] = 0
    robot = env.scene["robot"]
    root_xy = robot.data.root_link_pos_w[env_ids, :2]
    local_lateral = root_xy.new_tensor((0.0, 1.0, 0.0)).expand(len(env_ids), -1)
    lateral_world = quat_apply(robot.data.root_link_quat_w[env_ids], local_lateral)
    lateral_xy = lateral_world[:, :2]
    lateral_xy /= lateral_xy.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    env._a6_plate_initial_root_xy[env_ids] = root_xy
    env._a6_plate_initial_lateral_axis_xy[env_ids] = lateral_xy
    env._a6_plate_tick = -1


def update_a6_plate_state(env: ManagerBasedRlEnv) -> None:
    """Update A6 plate state at most once per policy control step."""

    if not hasattr(env, "_a6_reset_scene"):
        return
    initialize_a6_plate_state(env)
    if env._a6_plate_tick == env.common_step_counter:
        return
    env._a6_plate_tick = env.common_step_counter

    active = env._a6_reset_scene > 0
    free_scene = env._a6_reset_scene == 2
    contacts = []
    forces = []
    depths = []
    for sensor_name in ("guided_contact", "free_contact"):
        data = env.scene[sensor_name].data
        if data.found is None or data.force is None or data.dist is None:
            raise RuntimeError(f"A6 contact sensor {sensor_name!r} is incomplete")
        contacts.append((data.found > 0).any(dim=-1))
        forces.append(data.force.norm(dim=-1).amax(dim=-1))
        depths.append(data.dist.amin(dim=-1))
    contact = torch.where(free_scene, contacts[1], contacts[0])
    force = torch.where(free_scene, forces[1], forces[0])
    depth = torch.where(free_scene, depths[1], depths[0])
    score, clearance = a6_plate_geometry(env)
    path_count, path_score, path_clearance = a6_path_geometry(env)

    env._a6_plate_path_coverage_delta = (
        env._a6_plate_path_best_score - path_score
    ).clamp_min(0.0)
    env._a6_plate_path_clearance_delta = (
        path_clearance - env._a6_plate_path_best_clearance
    ).clamp_min(0.0)
    env._a6_plate_path_best_score = torch.minimum(
        env._a6_plate_path_best_score, path_score
    )
    env._a6_plate_path_best_clearance = torch.maximum(
        env._a6_plate_path_best_clearance, path_clearance
    )
    env._a6_plate_path_current_count = path_count.float()
    env._a6_plate_path_current_clearance = path_clearance
    hand_found = env.scene["path_hands"].data.found
    if hand_found is None:
        raise RuntimeError("A6 path_hands contact sensor is incomplete")
    env._a6_plate_path_hand_support = (hand_found > 0).float().mean(dim=-1)
    robot = env.scene["robot"]
    head_offset = robot.data.root_link_pos_w.new_tensor(A6_HEAD_OFFSET_IN_TORSO).expand(
        env.num_envs, -1
    )
    head_pos = robot.data.root_link_pos_w + quat_apply(
        robot.data.root_link_quat_w, head_offset
    )
    env._a6_plate_head_height = head_pos[:, 2] - env.scene.env_origins[:, 2]

    rows = torch.arange(env.num_envs, device=env.device)
    plate_ids = torch.stack(env._a6_plate_geom_ids)[free_scene.long()]
    plate_xy = env.sim.data.geom_xpos[rows, plate_ids, :2]
    root_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
    (
        env._a6_plate_best_lateral_distance,
        env._a6_plate_lateral_progress,
    ) = lateral_progress_from_state(
        displacement_xy=root_xy - env._a6_plate_initial_root_xy,
        lateral_axis_xy=env._a6_plate_initial_lateral_axis_xy,
        best_distance=env._a6_plate_best_lateral_distance,
    )
    distance = (root_xy - plate_xy).norm(dim=-1)
    # Count the current sample immediately.  Otherwise the first genuine
    # contact would only unlock progress one control step later.
    ever_contact_now = env._a6_plate_ever_contact | (contact & active)
    separation_progress = ((distance - env._a6_plate_best_distance) / 0.025).clamp(
        0.0, 1.0
    )
    env._a6_plate_separation_progress = torch.where(
        active & ever_contact_now,
        separation_progress,
        0.0,
    )
    env._a6_plate_best_distance = torch.maximum(env._a6_plate_best_distance, distance)
    progress = (score - env._a6_plate_best_score).clamp_min(0.0)
    env._a6_plate_best_score = torch.maximum(env._a6_plate_best_score, score)
    env._a6_plate_progress = torch.where(
        active & ever_contact_now,
        (progress / 0.025).clamp_max(1.5),
        0.0,
    )

    raw_invalid_reasons = plate_invalid_reasons(
        active=active,
        force=force,
        depth=depth,
        ever_contact=ever_contact_now,
        episode_steps=env.episode_length_buf,
    )
    force_contract = getattr(
        env,
        "_a6_plate_force_termination_contract",
        (A6_MAX_PLATE_FORCE, 1, None),
    )
    (
        env._a6_plate_force_over_limit_steps,
        invalid_force,
    ) = sustained_force_invalid_from_state(
        active=active,
        force=force,
        consecutive_steps=env._a6_plate_force_over_limit_steps,
        force_limit=force_contract[0],
        hold_steps=force_contract[1],
        catastrophic_force=force_contract[2],
    )
    invalid_reasons = (
        invalid_force,
        raw_invalid_reasons[1],
        raw_invalid_reasons[2],
    )
    (
        env._a6_plate_last_invalid_force,
        env._a6_plate_last_invalid_penetration,
        env._a6_plate_last_invalid_no_contact,
    ) = invalid_reasons

    (
        env._a6_plate_ever_contact,
        env._a6_plate_clear_hold,
        env._a6_plate_escaped,
        env._a6_plate_invalid,
    ) = update_plate_flags(
        active=active,
        contact=contact,
        force=force,
        depth=depth,
        clearance=clearance,
        ever_contact=ever_contact_now,
        clear_hold=env._a6_plate_clear_hold,
        episode_steps=env.episode_length_buf,
        min_planar_clearance=getattr(
            env, "_a6_escape_min_planar_clearance", A6_MIN_PLANAR_CLEARANCE
        ),
        clear_hold_steps=getattr(
            env, "_a6_escape_clear_hold_steps", A6_CLEAR_HOLD_STEPS
        ),
        invalid_force_override=invalid_force,
    )
    env._a6_plate_force = force
    env._a6_plate_clearance = clearance
    no_progress_eligible = (
        active
        & env._a6_plate_ever_contact
        & ~env._a6_plate_escaped
        & ~env._a6_plate_invalid
    )
    env._a6_plate_no_progress_time = advance_no_progress_time(
        current_time=env._a6_plate_no_progress_time,
        coverage_delta=env._a6_plate_path_coverage_delta,
        clearance_delta=env._a6_plate_path_clearance_delta,
        separation_progress=env._a6_plate_separation_progress,
        eligible=no_progress_eligible,
        step_dt=env.step_dt,
    )


def a6_invalid_plate(
    env: ManagerBasedRlEnv,
    force_limit: float = A6_MAX_PLATE_FORCE,
    force_hold_steps: int = 1,
    catastrophic_force: float | None = None,
) -> torch.Tensor:
    """Termination term for excessive plate force/penetration or missing contact."""

    if not hasattr(env, "_a6_reset_scene"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    contract = (float(force_limit), int(force_hold_steps), catastrophic_force)
    existing_contract = getattr(env, "_a6_plate_force_termination_contract", None)
    if existing_contract is None:
        env._a6_plate_force_termination_contract = contract
        if hasattr(env, "_a6_plate_force_over_limit_steps"):
            env._a6_plate_force_over_limit_steps.zero_()
    elif existing_contract != contract:
        raise ValueError("plate force termination contract changed after initialization")
    update_a6_plate_state(env)
    for reason in ("force", "penetration", "no_contact"):
        getattr(env, f"_a6_plate_terminal_invalid_{reason}").copy_(
            getattr(env, f"_a6_plate_last_invalid_{reason}")
        )
    return env._a6_plate_invalid


__all__ = [
    "A6_CATASTROPHIC_PLATE_FORCE",
    "A6_CLEAR_HOLD_STEPS",
    "A6_HEAD_OFFSET_IN_TORSO",
    "A6_LATERAL_PROGRESS_SCALE",
    "A6_MAX_NO_CONTACT_STEPS",
    "A6_MAX_PLATE_FORCE",
    "A6_MAX_PLATE_PENETRATION",
    "A6_MEANINGFUL_PROGRESS_DELTA",
    "A6_MIN_PLANAR_CLEARANCE",
    "A6_SEPARATION_PROGRESS_SCALE",
    "a6_invalid_plate",
    "a6_path_geometry",
    "a6_plate_geometry",
    "advance_no_progress_time",
    "initialize_a6_plate_state",
    "lateral_progress_from_state",
    "path_footprint",
    "plate_invalid_reasons",
    "reset_a6_plate_state",
    "sustained_force_invalid_from_state",
    "update_a6_plate_state",
    "update_plate_flags",
]
