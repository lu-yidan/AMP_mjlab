"""Reset terms that connect the frozen A6 bank to the AMP G1 model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1recovery_amp.a6_reset_assets import (
    A6_FLAT_STRATUM,
    A6_MULTITERRAIN_TRAIN_SHA256,
    A6_NATURAL_CURRICULUM_TRAIN_SHA256,
    A6_ROOT_QPOS_DIM,
    load_a6_multiterrain_bank,
    load_a6_natural_curriculum_bank,
    natural_stage_sampling_pool,
)
from g1recovery_amp.g1_model import G1_JOINT_NAMES
from g1recovery_amp.tasks.recovery.a6_scene import (
    A6_PLATE_CLEARANCE,
    A6_PLATE_HALF_SIZE,
    A6_ROBOT_COLLISION_PATTERN,
)
from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import (
    A6_CLEAR_HOLD_STEPS,
    A6_MIN_PLANAR_CLEARANCE,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_A6_RESET_SEED_OFFSET = 91_019
_A6_NATURAL_RESET_SEED_OFFSET = 62_019
A6_SCENE_NAMES = ("flat", "guided_plate", "free_plate")


def balanced_flat_reset_groups(num_envs: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign fixed four-direction and 75/25 natural/procedural cohorts."""

    env_index = np.arange(num_envs, dtype=np.int64)
    direction = env_index % 4
    rank_within_direction = env_index // 4
    source = (rank_within_direction % 4 == 0).astype(np.int64)
    return direction, source


def balanced_flat_curriculum_groups(
    num_envs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign fixed flat L/M/H cohorts and low-state direction/source labels.

    Stage IDs are 0=low, 1=middle, 2=late. Only low states have a physical
    four-direction label. Middle and late states are always natural and use
    direction ``-1`` to make the non-applicable label explicit.
    """

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    env_index = np.arange(num_envs, dtype=np.int64)
    cohort_direction = env_index % 4
    stage = np.ones(num_envs, dtype=np.int64)
    direction = np.full(num_envs, -1, dtype=np.int64)
    source = np.zeros(num_envs, dtype=np.int64)

    for direction_id in range(4):
        chosen = env_index[cohort_direction == direction_id]
        if len(chosen) == 0:
            continue
        low_count = max(1, round(len(chosen) * 0.4))
        late_count = round(len(chosen) * 0.2) if len(chosen) >= 3 else 0
        late_count = min(late_count, len(chosen) - low_count)
        middle_end = len(chosen) - late_count

        low_ids = chosen[:low_count]
        middle_ids = chosen[low_count:middle_end]
        late_ids = chosen[middle_end:]
        stage[low_ids] = 0
        stage[middle_ids] = 1
        stage[late_ids] = 2
        direction[low_ids] = direction_id

        procedural_count = round(low_count * 0.25)
        source[low_ids[:procedural_count]] = 1

    return stage, direction, source


def balanced_three_scene_curriculum_groups(
    num_envs: int,
    scene_weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign fixed A6 scene/stage/direction/source cohorts.

    ``scene_weights`` follows ``(flat, guided_plate, free_plate)``.  The
    default exactly reproduces the historical 50/25/25 split.  Non-integral
    quotas are converted with the largest-remainder method so every world is
    assigned exactly once while staying as close as possible to the requested
    proportions.
    """

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    weights = np.asarray(scene_weights, dtype=np.float64)
    if weights.shape != (3,):
        raise ValueError("scene_weights must contain flat/guided/free weights")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("scene_weights must be finite and non-negative")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("scene_weights must have a positive sum")

    quotas = num_envs * weights / weight_sum
    counts = np.floor(quotas).astype(np.int64)
    remaining = num_envs - int(counts.sum())
    if remaining:
        # Stable sorting makes ties deterministic and preserves scene order.
        order = np.argsort(-(quotas - counts), kind="stable")
        counts[order[:remaining]] += 1
    scene_counts = tuple(int(count) for count in counts)

    scene = np.repeat(np.arange(3, dtype=np.int64), scene_counts)
    stage = np.zeros(num_envs, dtype=np.int64)
    direction = np.zeros(num_envs, dtype=np.int64)
    source = np.zeros(num_envs, dtype=np.int64)

    flat_count = scene_counts[0]
    if flat_count:
        flat_stage, flat_direction, flat_source = balanced_flat_curriculum_groups(
            flat_count
        )
        stage[:flat_count] = flat_stage
        direction[:flat_count] = flat_direction
        source[:flat_count] = flat_source

    start = flat_count
    for count in scene_counts[1:]:
        stop = start + count
        local_ids = np.arange(count, dtype=np.int64)
        local_direction = local_ids % 4
        direction[start:stop] = local_direction
        for direction_id in range(4):
            chosen = start + local_ids[local_direction == direction_id]
            procedural_count = round(len(chosen) * 0.25)
            source[chosen[:procedural_count]] = 1
        start = stop

    return scene, stage, direction, source


def balanced_three_scene_reset_groups(
    num_envs: int,
    scene_weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign exact low-state direction/source cohorts for formal evaluation.

    Unlike the training curriculum, every scene uses the frozen low-state bank.
    Within each scene, directions are balanced and each direction is split into
    75% natural and 25% procedural reset states.
    """

    scene, _, _, _ = balanced_three_scene_curriculum_groups(
        num_envs, scene_weights=scene_weights
    )
    stage = np.zeros(num_envs, dtype=np.int64)
    direction = np.zeros(num_envs, dtype=np.int64)
    source = np.zeros(num_envs, dtype=np.int64)
    for scene_id in range(3):
        chosen = np.flatnonzero(scene == scene_id)
        local_direction, local_source = balanced_flat_reset_groups(len(chosen))
        direction[chosen] = local_direction
        source[chosen] = local_source
    return scene, stage, direction, source


def _tensor_env_ids(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
) -> torch.Tensor:
    all_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return env_ids


def _initialize_a6_flat_bank(
    env: ManagerBasedRlEnv,
    bank_path: str,
    expected_sha256: str | None,
) -> None:
    bank = load_a6_multiterrain_bank(Path(bank_path), expected_sha256=expected_sha256)
    robot = env.scene["robot"]
    if tuple(robot.joint_names) != G1_JOINT_NAMES:
        raise ValueError("A6 reset bank joint order does not match the AMP G1 model")

    env._a6_reset_bank_path = str(Path(bank_path).resolve())
    env._a6_reset_qpos = torch.as_tensor(
        bank.qpos, dtype=torch.float32, device=env.device
    )
    direction, source = balanced_flat_reset_groups(env.num_envs)
    env._a6_reset_direction = torch.as_tensor(direction, device=env.device)
    env._a6_reset_source = torch.as_tensor(source, device=env.device)
    env._a6_reset_pools = {}
    for direction_id in range(4):
        for source_id in range(2):
            pool = np.flatnonzero(
                (bank.stratum == A6_FLAT_STRATUM)
                & (bank.direction == direction_id)
                & (bank.source == source_id)
            )
            env._a6_reset_pools[(direction_id, source_id)] = torch.as_tensor(
                pool, dtype=torch.int64, device=env.device
            )
    env._a6_reset_rng = torch.Generator(device=env.device).manual_seed(
        int(env.cfg.seed or 0) + _A6_RESET_SEED_OFFSET
    )
    env._a6_reset_bank_index = torch.full(
        (env.num_envs,), -1, dtype=torch.int64, device=env.device
    )


def _initialize_a6_flat_curriculum(
    env: ManagerBasedRlEnv,
    low_bank_path: str,
    low_expected_sha256: str | None,
    natural_bank_path: str,
    natural_expected_sha256: str | None,
) -> None:
    _initialize_a6_flat_bank(env, low_bank_path, low_expected_sha256)
    natural_bank = load_a6_natural_curriculum_bank(
        Path(natural_bank_path), expected_sha256=natural_expected_sha256
    )
    env._a6_reset_natural_bank_path = str(Path(natural_bank_path).resolve())
    env._a6_reset_natural_qpos = torch.as_tensor(
        natural_bank.qpos, dtype=torch.float32, device=env.device
    )
    stage, direction, source = balanced_flat_curriculum_groups(env.num_envs)
    env._a6_reset_stage = torch.as_tensor(stage, device=env.device)
    env._a6_reset_direction = torch.as_tensor(direction, device=env.device)
    env._a6_reset_source = torch.as_tensor(source, device=env.device)
    env._a6_reset_natural_pools = {}
    for stage_id, stage_name in ((1, "middle"), (2, "late")):
        pool, weights = natural_stage_sampling_pool(natural_bank, stage_name)
        env._a6_reset_natural_pools[stage_id] = (
            torch.as_tensor(pool, dtype=torch.int64, device=env.device),
            torch.as_tensor(weights, dtype=torch.float32, device=env.device),
        )
    env._a6_reset_natural_rng = torch.Generator(device=env.device).manual_seed(
        int(env.cfg.seed or 0) + _A6_NATURAL_RESET_SEED_OFFSET
    )
    env._a6_reset_natural_bank_index = torch.full(
        (env.num_envs,), -1, dtype=torch.int64, device=env.device
    )


def _initialize_a6_three_scene_curriculum(
    env: ManagerBasedRlEnv,
    low_bank_path: str,
    low_expected_sha256: str | None,
    natural_bank_path: str,
    natural_expected_sha256: str | None,
    scene_weights: tuple[float, float, float],
    low_only: bool,
) -> None:
    _initialize_a6_flat_curriculum(
        env,
        low_bank_path,
        low_expected_sha256,
        natural_bank_path,
        natural_expected_sha256,
    )
    group_fn = (
        balanced_three_scene_reset_groups
        if low_only
        else balanced_three_scene_curriculum_groups
    )
    scene, stage, direction, source = group_fn(
        env.num_envs, scene_weights=scene_weights
    )
    env._a6_scene_weights = tuple(float(weight) for weight in scene_weights)
    env._a6_low_only_cohorts = bool(low_only)
    env._a6_reset_scene = torch.as_tensor(scene, device=env.device)
    env._a6_reset_stage = torch.as_tensor(stage, device=env.device)
    env._a6_reset_direction = torch.as_tensor(direction, device=env.device)
    env._a6_reset_source = torch.as_tensor(source, device=env.device)

    bank = load_a6_multiterrain_bank(
        Path(low_bank_path), expected_sha256=low_expected_sha256
    )
    env._a6_scene_reset_pools = {}
    for scene_id in range(3):
        for direction_id in range(4):
            for source_id in range(2):
                pool = np.flatnonzero(
                    (bank.stratum == scene_id)
                    & (bank.direction == direction_id)
                    & (bank.source == source_id)
                )
                if not len(pool):
                    raise ValueError(
                        "A6 reset pool is empty for "
                        f"scene={scene_id}, direction={direction_id}, source={source_id}"
                    )
                env._a6_scene_reset_pools[(scene_id, direction_id, source_id)] = (
                    torch.as_tensor(pool, dtype=torch.int64, device=env.device)
                )

    robot = env.scene["robot"]
    geom_indices, _ = robot.find_geoms(A6_ROBOT_COLLISION_PATTERN)
    env._a6_robot_collision_geom_ids = robot.indexing.geom_ids[
        torch.as_tensor(geom_indices, dtype=torch.int64, device=env.device)
    ].long()
    env._a6_plate_geom_ids = []
    for entity_name, geom_name in (
        ("escape_obstacle", "escape_plate_geom"),
        ("free_obstacle", "plate_geom"),
    ):
        entity = env.scene[entity_name]
        indices, _ = entity.find_geoms((geom_name,))
        env._a6_plate_geom_ids.append(entity.indexing.geom_ids[indices[0]].long())
    env._a6_plate_mass = torch.zeros(env.num_envs, device=env.device)
    env._a6_curriculum_start_counter = int(env.common_step_counter)


def a6_curriculum_progress(
    common_step_counter: int,
    curriculum_start_counter: int,
    curriculum_steps: int,
) -> float:
    """Return adaptation-local curriculum progress, independent of parent age."""

    if curriculum_steps <= 0:
        raise ValueError("curriculum_steps must be positive")
    age = max(common_step_counter - curriculum_start_counter, 0)
    return min(float(age) / curriculum_steps, 1.0)


def _write_a6_qpos(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    qpos: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    robot = env.scene[asset_cfg.name]
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] = qpos[:, :3] + env.scene.env_origins[env_ids]
    root_state[:, 3:7] = qpos[:, 3:7]
    root_state[:, 7:] = 0.0
    joint_pos = qpos[:, A6_ROOT_QPOS_DIM:]
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        joint_pos,
        torch.zeros_like(joint_pos),
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )


def reset_from_a6_flat_bank(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    *,
    bank_path: str,
    expected_sha256: str | None = A6_MULTITERRAIN_TRAIN_SHA256,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Reset robot qpos from balanced A6 flat pools and set all velocities to zero."""

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    resolved_ids = _tensor_env_ids(env, env_ids)
    if not hasattr(env, "_a6_reset_qpos"):
        _initialize_a6_flat_bank(env, bank_path, expected_sha256)
    elif env._a6_reset_bank_path != str(Path(bank_path).resolve()):
        raise ValueError("A6 reset bank path changed after environment initialization")

    selected = torch.empty(len(resolved_ids), dtype=torch.int64, device=env.device)
    for direction_id in range(4):
        for source_id in range(2):
            mask = (env._a6_reset_direction[resolved_ids] == direction_id) & (
                env._a6_reset_source[resolved_ids] == source_id
            )
            count = int(mask.sum())
            if count == 0:
                continue
            pool = env._a6_reset_pools[(direction_id, source_id)]
            draws = torch.randint(
                len(pool),
                (count,),
                generator=env._a6_reset_rng,
                device=env.device,
            )
            selected[mask] = pool[draws]

    qpos = env._a6_reset_qpos[selected]
    _write_a6_qpos(env, resolved_ids, qpos, asset_cfg)
    env._a6_reset_bank_index[resolved_ids] = selected


def reset_from_a6_flat_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    *,
    low_bank_path: str,
    natural_bank_path: str,
    low_expected_sha256: str | None = A6_MULTITERRAIN_TRAIN_SHA256,
    natural_expected_sha256: str | None = A6_NATURAL_CURRICULUM_TRAIN_SHA256,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Reset flat environments with the historical 40/40/20 L/M/H curriculum."""

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    resolved_ids = _tensor_env_ids(env, env_ids)
    if not hasattr(env, "_a6_reset_stage"):
        _initialize_a6_flat_curriculum(
            env,
            low_bank_path,
            low_expected_sha256,
            natural_bank_path,
            natural_expected_sha256,
        )
    elif env._a6_reset_bank_path != str(Path(low_bank_path).resolve()):
        raise ValueError("A6 low reset bank path changed after initialization")
    elif env._a6_reset_natural_bank_path != str(Path(natural_bank_path).resolve()):
        raise ValueError("A6 natural reset bank path changed after initialization")

    qpos = torch.empty(
        (len(resolved_ids), env._a6_reset_qpos.shape[1]),
        dtype=torch.float32,
        device=env.device,
    )
    low_selected = torch.full(
        (len(resolved_ids),), -1, dtype=torch.int64, device=env.device
    )
    natural_selected = torch.full_like(low_selected, -1)
    local_stage = env._a6_reset_stage[resolved_ids]

    for direction_id in range(4):
        for source_id in range(2):
            mask = (
                (local_stage == 0)
                & (env._a6_reset_direction[resolved_ids] == direction_id)
                & (env._a6_reset_source[resolved_ids] == source_id)
            )
            count = int(mask.sum())
            if count == 0:
                continue
            pool = env._a6_reset_pools[(direction_id, source_id)]
            draws = torch.randint(
                len(pool),
                (count,),
                generator=env._a6_reset_rng,
                device=env.device,
            )
            selected = pool[draws]
            low_selected[mask] = selected
            qpos[mask] = env._a6_reset_qpos[selected]

    for stage_id in (1, 2):
        mask = local_stage == stage_id
        count = int(mask.sum())
        if count == 0:
            continue
        pool, weights = env._a6_reset_natural_pools[stage_id]
        draws = torch.multinomial(
            weights,
            count,
            replacement=True,
            generator=env._a6_reset_natural_rng,
        )
        selected = pool[draws]
        natural_selected[mask] = selected
        qpos[mask] = env._a6_reset_natural_qpos[selected]

    if torch.any((low_selected < 0) & (natural_selected < 0)):
        raise RuntimeError("some A6 curriculum environments received no reset state")
    _write_a6_qpos(env, resolved_ids, qpos, asset_cfg)
    env._a6_reset_bank_index[resolved_ids] = low_selected
    env._a6_reset_natural_bank_index[resolved_ids] = natural_selected


def _robot_collision_top(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Return the highest collision point covered by the centered A6 plate."""

    geom_ids = env._a6_robot_collision_geom_ids
    pos = env.sim.data.geom_xpos[env_ids[:, None], geom_ids[None, :]]
    mat = env.sim.data.geom_xmat[env_ids[:, None], geom_ids[None, :]]
    size = env.sim.model.geom_size[env_ids[:, None], geom_ids[None, :]]
    geom_type = env.sim.model.geom_type[geom_ids]
    extent = torch.einsum("ngij,ngj->ngi", mat.abs(), size)
    sphere = geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    capsule = geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    extent = torch.where(sphere[None, :, None], size[:, :, :1], extent)
    capsule_extent = size[:, :, :1] + size[:, :, 1:2] * mat[:, :, :, 2].abs()
    extent = torch.where(capsule[None, :, None], capsule_extent, extent)
    center_xy = env.scene["robot"].data.root_link_pos_w[env_ids, :2]
    half_xy = pos.new_tensor(A6_PLATE_HALF_SIZE[:2])
    covered = (
        (pos[:, :, :2] - center_xy[:, None, :]).abs() < extent[:, :, :2] + half_xy
    ).all(dim=-1)
    top = torch.where(
        covered,
        pos[:, :, 2] + extent[:, :, 2],
        torch.full_like(pos[:, :, 2], -torch.inf),
    ).amax(dim=1)
    if not torch.isfinite(top).all():
        raise RuntimeError("A6 plate does not cover any robot collision geometry")
    return top


def _park_a6_plates(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    park = torch.zeros((len(env_ids), 7), dtype=torch.float32, device=env.device)
    park[:, :3] = env.scene.env_origins[env_ids]
    park[:, 0] += 20.0
    park[:, 1] += 20.0
    park[:, 2] += 0.1
    park[:, 3] = 1.0

    guided = env.scene["escape_obstacle"]
    guided.write_mocap_pose_to_sim(park, env_ids=env_ids)
    guided.write_joint_state_to_sim(
        torch.zeros((len(env_ids), 1), device=env.device),
        torch.zeros((len(env_ids), 1), device=env.device),
        env_ids=env_ids,
    )
    free = env.scene["free_obstacle"]
    free_state = free.data.default_root_state[env_ids].clone()
    free_state[:, :7] = park
    free_state[:, 0] += 2.0
    free_state[:, 7:] = 0.0
    free.write_root_state_to_sim(free_state, env_ids=env_ids)


def _place_a6_plates(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    progress: float,
    plate_mass_override: float | None = None,
) -> None:
    """Place active plates at robot-top+2 mm and apply the A6 mass curriculum."""

    _park_a6_plates(env, env_ids)
    env.sim.forward()
    robot = env.scene["robot"]
    top = _robot_collision_top(env, env_ids)
    board = torch.zeros((len(env_ids), 7), dtype=torch.float32, device=env.device)
    board[:, :2] = robot.data.root_link_pos_w[env_ids, :2]
    board[:, 2] = top + A6_PLATE_HALF_SIZE[2] + A6_PLATE_CLEARANCE
    board[:, 3] = 1.0

    local_scene = env._a6_reset_scene[env_ids]
    max_mass = 6.0 + 6.0 * progress
    if plate_mass_override is not None:
        if plate_mass_override <= 0.0:
            raise ValueError("plate_mass_override must be positive")
        mass_draw = torch.full(
            (len(env_ids),),
            plate_mass_override,
            dtype=torch.float32,
            device=env.device,
        )
    else:
        mass_draw = 4.0 + torch.rand(
            len(env_ids), generator=env._a6_reset_rng, device=env.device
        ) * (max_mass - 4.0)
    env._a6_plate_mass[env_ids] = 0.0

    for scene_id, entity_name in ((1, "escape_obstacle"), (2, "free_obstacle")):
        choose = local_scene == scene_id
        chosen_ids = env_ids[choose]
        if not len(chosen_ids):
            continue
        entity = env.scene[entity_name]
        chosen_board = board[choose]
        if scene_id == 1:
            entity.write_mocap_pose_to_sim(chosen_board, env_ids=chosen_ids)
        else:
            state = entity.data.default_root_state[chosen_ids].clone()
            state[:, :7] = chosen_board
            state[:, 7:] = 0.0
            entity.write_root_state_to_sim(state, env_ids=chosen_ids)

        body_id = entity.indexing.body_ids[-1].long()
        mass = mass_draw[choose]
        base_mass = env.sim.get_default_field("body_mass")[body_id]
        base_inertia = env.sim.get_default_field("body_inertia")[body_id]
        env.sim.model.body_mass[chosen_ids, body_id] = mass
        env.sim.model.body_inertia[chosen_ids, body_id] = (
            base_inertia * mass[:, None] / base_mass
        )
        env._a6_plate_mass[chosen_ids] = mass
    env.sim.forward()


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def reset_from_a6_three_scene_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    *,
    low_bank_path: str,
    natural_bank_path: str,
    low_expected_sha256: str | None = A6_MULTITERRAIN_TRAIN_SHA256,
    natural_expected_sha256: str | None = A6_NATURAL_CURRICULUM_TRAIN_SHA256,
    mass_curriculum_steps: int = 100_000,
    plate_mass_override: float | None = None,
    scene_weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
    low_only: bool = False,
    escape_min_planar_clearance: float = A6_MIN_PLANAR_CLEARANCE,
    escape_clear_hold_steps: int = A6_CLEAR_HOLD_STEPS,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Reset a fixed flat/guided/free A6 scene mixture."""

    if escape_min_planar_clearance < 0.0:
        raise ValueError("escape_min_planar_clearance must be nonnegative")
    if escape_clear_hold_steps <= 0:
        raise ValueError("escape_clear_hold_steps must be positive")
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    resolved_ids = _tensor_env_ids(env, env_ids)
    if not hasattr(env, "_a6_reset_scene"):
        _initialize_a6_three_scene_curriculum(
            env,
            low_bank_path,
            low_expected_sha256,
            natural_bank_path,
            natural_expected_sha256,
            scene_weights,
            low_only,
        )
    elif env._a6_reset_bank_path != str(Path(low_bank_path).resolve()):
        raise ValueError("A6 low reset bank path changed after initialization")
    elif env._a6_reset_natural_bank_path != str(Path(natural_bank_path).resolve()):
        raise ValueError("A6 natural reset bank path changed after initialization")
    elif env._a6_scene_weights != tuple(float(weight) for weight in scene_weights):
        raise ValueError("A6 scene weights changed after initialization")
    elif env._a6_low_only_cohorts != bool(low_only):
        raise ValueError("A6 low-only cohort mode changed after initialization")

    escape_contract = (
        float(escape_min_planar_clearance),
        int(escape_clear_hold_steps),
    )
    if hasattr(env, "_a6_escape_contract"):
        if env._a6_escape_contract != escape_contract:
            raise ValueError("A6 escape condition changed after initialization")
    else:
        env._a6_escape_contract = escape_contract
        env._a6_escape_min_planar_clearance = escape_contract[0]
        env._a6_escape_clear_hold_steps = escape_contract[1]

    qpos = torch.empty(
        (len(resolved_ids), env._a6_reset_qpos.shape[1]),
        dtype=torch.float32,
        device=env.device,
    )
    low_selected = torch.full(
        (len(resolved_ids),), -1, dtype=torch.int64, device=env.device
    )
    natural_selected = torch.full_like(low_selected, -1)
    local_scene = env._a6_reset_scene[resolved_ids]
    local_stage = env._a6_reset_stage[resolved_ids]

    for scene_id in range(3):
        for direction_id in range(4):
            for source_id in range(2):
                mask = (
                    (local_stage == 0)
                    & (local_scene == scene_id)
                    & (env._a6_reset_direction[resolved_ids] == direction_id)
                    & (env._a6_reset_source[resolved_ids] == source_id)
                )
                count = int(mask.sum())
                if not count:
                    continue
                pool = env._a6_scene_reset_pools[(scene_id, direction_id, source_id)]
                draws = torch.randint(
                    len(pool),
                    (count,),
                    generator=env._a6_reset_rng,
                    device=env.device,
                )
                selected = pool[draws]
                low_selected[mask] = selected
                qpos[mask] = env._a6_reset_qpos[selected]

    for stage_id in (1, 2):
        mask = local_stage == stage_id
        count = int(mask.sum())
        if not count:
            continue
        pool, weights = env._a6_reset_natural_pools[stage_id]
        draws = torch.multinomial(
            weights,
            count,
            replacement=True,
            generator=env._a6_reset_natural_rng,
        )
        selected = pool[draws]
        natural_selected[mask] = selected
        qpos[mask] = env._a6_reset_natural_qpos[selected]

    if torch.any((low_selected < 0) & (natural_selected < 0)):
        raise RuntimeError("some A6 three-scene environments received no reset state")
    _write_a6_qpos(env, resolved_ids, qpos, asset_cfg)
    env._a6_reset_bank_index[resolved_ids] = low_selected
    env._a6_reset_natural_bank_index[resolved_ids] = natural_selected

    progress = a6_curriculum_progress(
        env.common_step_counter,
        env._a6_curriculum_start_counter,
        mass_curriculum_steps,
    )
    _place_a6_plates(
        env,
        resolved_ids,
        progress,
        plate_mass_override=plate_mass_override,
    )
    from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import (
        reset_a6_plate_state,
    )
    from g1recovery_amp.tasks.recovery.mdp.a6_shared_costs import (
        reset_a6_shared_cost_state,
    )

    reset_a6_plate_state(env, resolved_ids)
    reset_a6_shared_cost_state(env, resolved_ids)


__all__ = [
    "A6_SCENE_NAMES",
    "a6_curriculum_progress",
    "balanced_flat_curriculum_groups",
    "balanced_flat_reset_groups",
    "balanced_three_scene_curriculum_groups",
    "balanced_three_scene_reset_groups",
    "reset_from_a6_flat_bank",
    "reset_from_a6_flat_curriculum",
    "reset_from_a6_three_scene_curriculum",
]
