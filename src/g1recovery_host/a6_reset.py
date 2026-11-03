"""Frozen A6 scene quotas and reset banks projected onto the HoST G1."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import quat_apply

from .a6_geometry import (
    PLATE_HALF_SIZE,
    footprint,
    plate_footprint_inputs,
    quotas,
    robot_bounds,
)

BANK_ROOT = Path(__file__).resolve().parents[2] / "assets/a6/host29"
MAX_GUIDED_EDGE_OFFSET_M = 0.50


def focused_guided_layout(
    num_envs: int,
    postescape_prone: bool = False,
    mix: str = "focused_v1",
    postescape_stride: int = 4,
) -> np.ndarray:
    """Build a fixed A6 scene/direction allocation for training.

    The original mix has no guided side-lying resets. The diverse variants add
    guided side directions and give free-prone starts four times their original
    quota. V3 also doubles both free-side quotas. V4 rehearses flat and free
    recovery more often while retaining all guided directions. Evaluation
    layouts are supplied separately and are unaffected.
    """
    if num_envs < 32 or num_envs % 32:
        raise ValueError("focused A6 num_envs must be a positive multiple of 32")
    if postescape_stride not in (2, 4):
        raise ValueError("postescape stride must be 2 or 4")
    flat_low_fraction, flat_late_fraction = 0.4, 0.2
    if mix == "focused_v1":
        flat, guided, free = num_envs // 4, 5 * num_envs // 8, num_envs // 8
        guided_directions = (guided // 2, guided // 2, 0, 0)
        free_directions = (free // 4,) * 4
    elif mix == "diverse_v2":
        unit = num_envs // 32
        flat, guided, free = 4 * unit, 20 * unit, 8 * unit
        guided_directions = (6 * unit, 6 * unit, 4 * unit, 4 * unit)
        free_directions = (2 * unit, 4 * unit, unit, unit)
    elif mix == "diverse_v3":
        unit = num_envs // 32
        flat, guided, free = 4 * unit, 18 * unit, 10 * unit
        guided_directions = (6 * unit, 6 * unit, 3 * unit, 3 * unit)
        free_directions = (2 * unit, 4 * unit, 2 * unit, 2 * unit)
    elif mix == "diverse_v4_recovery":
        unit = num_envs // 32
        flat, guided, free = 8 * unit, 14 * unit, 10 * unit
        guided_directions = (5 * unit, 5 * unit, 2 * unit, 2 * unit)
        free_directions = (3 * unit, 3 * unit, 2 * unit, 2 * unit)
        flat_low_fraction, flat_late_fraction = 0.6, 0.1
    elif mix == "free_specialist_v1":
        unit = num_envs // 32
        flat, guided, free = 4 * unit, 4 * unit, 24 * unit
        guided_directions = (unit,) * 4
        free_directions = (6 * unit,) * 4
        flat_low_fraction, flat_late_fraction = 0.6, 0.1
    elif mix == "guided_prone_specialist_v1":
        unit = num_envs // 32
        flat, guided, free = 4 * unit, 20 * unit, 8 * unit
        guided_directions = (unit, 17 * unit, unit, unit)
        free_directions = (2 * unit,) * 4
        flat_low_fraction, flat_late_fraction = 0.6, 0.1
    elif mix == "prone_compact_v1":
        unit = num_envs // 32
        flat, guided, free = 16 * unit, 8 * unit, 8 * unit
        guided_directions = (unit, 5 * unit, unit, unit)
        free_directions = (unit, 5 * unit, unit, unit)
        flat_low_fraction, flat_late_fraction = 0.6, 0.1
    elif mix == "flat_prone_specialist_v1":
        unit = num_envs // 32
        flat, guided, free = 24 * unit, 4 * unit, 4 * unit
        guided_directions = (unit,) * 4
        free_directions = (unit,) * 4
        flat_low_fraction, flat_late_fraction = 0.6, 0.1
    elif mix == "side_specialist_v1":
        unit = num_envs // 32
        flat, guided, free = 4 * unit, 16 * unit, 12 * unit
        guided_directions = (2 * unit, 2 * unit, 6 * unit, 6 * unit)
        free_directions = (unit, unit, 5 * unit, 5 * unit)
        flat_low_fraction, flat_late_fraction = 0.6, 0.1
    elif mix == "guided_side_specialist_v1":
        unit = num_envs // 32
        flat, guided, free = 4 * unit, 24 * unit, 4 * unit
        guided_directions = (2 * unit, 2 * unit, 10 * unit, 10 * unit)
        free_directions = (unit,) * 4
        flat_low_fraction, flat_late_fraction = 0.6, 0.1
    elif mix == "guided_left_side_specialist_v1":
        unit = num_envs // 32
        flat, guided, free = 4 * unit, 24 * unit, 4 * unit
        guided_directions = (2 * unit, 2 * unit, 18 * unit, 2 * unit)
        free_directions = (unit,) * 4
        flat_low_fraction, flat_late_fraction = 0.6, 0.1
    else:
        raise ValueError(f"unknown A6 reset mix: {mix}")
    rows = np.zeros((num_envs, 3), dtype=np.int64)
    rows[flat : flat + guided, 0] = 1
    rows[flat + guided :, 0] = 2
    rows[:flat, 1] = np.repeat(np.arange(4), flat // 4)
    if mix in ("prone_compact_v1", "flat_prone_specialist_v1"):
        rows[:flat, 1] = 1
    rows[flat : flat + guided, 1] = np.repeat(np.arange(4), guided_directions)
    rows[flat + guided :, 1] = np.repeat(np.arange(4), free_directions)
    # Retain all flat start tiers. V4 prioritizes low poses that were rare in
    # the guided-heavy V3 mix, without turning every flat start into a fall.
    flat_ids = np.arange(flat).reshape(4, flat // 4)
    for ids in flat_ids:
        low_end = round(len(ids) * flat_low_fraction)
        late_start = len(ids) - round(len(ids) * flat_late_fraction)
        rows[ids[low_end:late_start], 2] = 1
        rows[ids[late_start:], 2] = 2
    if postescape_prone:
        # Rehearse standing from physically reached escape poses. Stratum 3 is
        # reserved for this replay and never appears in ordinary playback.
        prone = np.flatnonzero((rows[:, 0] == 1) & (rows[:, 1] == 1))
        rows[prone[::postescape_stride], 2] = 3
    return rows


def edge_offset_for_iteration(
    iteration: int,
    anchor_iteration: int,
    max_offset_m: float,
    hold_updates: int,
    anneal_updates: int,
) -> float:
    """Hold an easier near-edge reset, then linearly return the plate to centre."""
    if iteration < anchor_iteration or hold_updates < 0 or anneal_updates <= 0:
        raise ValueError("invalid A6 edge-curriculum iteration schedule")
    if (
        not math.isfinite(max_offset_m)
        or not 0 < max_offset_m <= MAX_GUIDED_EDGE_OFFSET_M
    ):
        raise ValueError("A6 guided edge offset must be in (0, 0.50] metres")
    elapsed_anneal = max(0, iteration - anchor_iteration - hold_updates + 1)
    return max_offset_m * max(0.0, 1.0 - elapsed_anneal / anneal_updates)


def side_bridge_tiers(tier, scene, direction, edge_offset_m):
    """Use the half-offset bridge instead of centred guided-side resets."""
    if edge_offset_m <= 0.05:
        return tier
    return torch.where((scene == 1) & (direction >= 2) & (tier == 3), 2, tier)


def side_center_tiers(tier, scene, direction):
    """Give every guided-side world a centered board without changing others."""
    return torch.where(
        (scene == 1) & (direction >= 2), torch.full_like(tier, 3), tier
    )


def side_staged_offset_scale(tier, scene, direction, edge_offset_m):
    """Use 1.0/0.75/0.5 side difficulty before exposing centred starts."""
    scale = torch.where(tier < 2, 1.0, torch.where(tier == 2, 0.5, 0.0))
    if edge_offset_m <= 0.05:
        return scale
    guided_side = (scene == 1) & (direction >= 2)
    side_scale = torch.where(tier < 2, 1.0, torch.where(tier == 2, 0.75, 0.5))
    return torch.where(guided_side, side_scale, scale)


def planar_escape_axis(root_quat, direction):
    """Stable horizontal body axis for a lying robot's guided-board exit.

    The anatomical left/right axis is nearly vertical in a side-lying pose.
    Its tiny XY projection used to choose a noisy, effectively random plate
    edge. The torso-up axis is horizontal in that pose and supplies the same
    left/right board-exit direction without that singularity.
    """
    left = quat_apply(
        root_quat,
        root_quat.new_tensor((0.0, 1.0, 0.0)).expand(len(root_quat), -1),
    )[:, :2]
    up = quat_apply(
        root_quat,
        root_quat.new_tensor((0.0, 0.0, 1.0)).expand(len(root_quat), -1),
    )[:, :2]
    # A direction label also covers middle/late recovery-bank poses which can
    # already be upright. Use body-up only while its planar projection is
    # reliable; then the ordinary left/right axis is stable again.
    side_up = (direction >= 2) & (up.norm(dim=-1) >= 0.20)
    axis = torch.where(side_up[:, None], up, left)
    norm = axis.norm(dim=-1, keepdim=True)
    if (norm < 0.20).any():
        raise ValueError("A6 planar escape axis is ill-defined for this reset pose")
    return axis / norm


def _bank(path: Path, device: str) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"A6 reset bank missing: {path}; run scripts/prepare_a6_banks.py"
        )
    with np.load(path) as archive:
        return {
            key: torch.as_tensor(archive[key].copy(), device=device)
            for key in archive.files
            if archive[key].dtype.kind in "ifub"
        }


def initialize(env, layout=None) -> None:
    if hasattr(env, "_a6_scene"):
        return
    if layout is None:
        scene, direction, stratum = quotas(env.num_envs)
    else:
        # Explicit layouts are used by both evaluation and training curricula.
        rows = np.asarray(layout, dtype=np.int64)
        if (
            rows.shape != (env.num_envs, 3)
            or not np.isin(rows[:, 0], (0, 1, 2)).all()
            or not np.isin(rows[:, 1], (0, 1, 2, 3)).all()
            or not np.isin(rows[:, 2], (0, 1, 2, 3)).all()
            or np.any((rows[:, 2] == 3) & ((rows[:, 0] != 1) | (rows[:, 1] != 1)))
        ):
            raise ValueError("invalid A6 playback layout")
        scene, direction, stratum = rows.T.copy()
    dev, n = env.device, env.num_envs
    env._a6_scene = torch.as_tensor(scene, device=dev)
    env._a6_direction = torch.as_tensor(direction, device=dev)
    env._a6_stratum = torch.as_tensor(stratum, device=dev)
    env._a6_source = torch.zeros(n, device=dev, dtype=torch.long)
    env._a6_postescape_start = env._a6_stratum == 3
    env._a6_postescape_bank_index = torch.full(
        (n,), -1, dtype=torch.long, device=dev
    )
    for d in range(4):
        ids = torch.where((env._a6_stratum == 0) & (env._a6_direction == d))[0]
        env._a6_source[ids[: round(len(ids) * 0.25)]] = 1
    banks = {
        key: _bank(BANK_ROOT / f"{key}_host29.npz", dev)
        for key in ("multiterrain", "natural_curriculum", "procedural_low")
    }
    env._a6_banks = banks
    if env._a6_postescape_start.any():
        postescape_path = Path(
            getattr(
                env,
                "_a6_postescape_bank_path",
                BANK_ROOT / "escaped_prone_1699_512.npz",
            )
        )
        postescape = _bank(postescape_path, dev)
        if (
            postescape["qpos"].shape[1:] != (36,)
            or postescape["qvel"].shape != (len(postescape["qpos"]), 35)
            or postescape["plate_xyz"].shape != (len(postescape["qpos"]), 3)
            or not all(torch.isfinite(value).all() for value in postescape.values())
        ):
            raise ValueError("invalid 29-DoF postescape prone reset bank")
        env._a6_postescape_bank = postescape
    main = banks["multiterrain"]
    env._a6_pools = {
        (d, source): torch.where(
            (main["stratum"] == 0)
            & (main["direction"] == d)
            & (main["source"] == source)
        )[0]
        for d in range(4)
        for source in range(2)
    }
    # Natural stage labels are fixed-width byte strings and are sampled on CPU.
    with np.load(BANK_ROOT / "natural_curriculum_host29.npz") as archive:
        env._a6_natural_pools = {}
        env._a6_natural_weights = {}
        clip_groups = np.char.replace(archive["clips"].astype(str), "__mirror", "")
        for kind, label in ((1, "middle"), (2, "late")):
            rows = np.flatnonzero(archive["stages"] == label)
            groups = clip_groups[rows]
            unique, inverse, counts = np.unique(
                groups, return_inverse=True, return_counts=True
            )
            weights = 1.0 / (len(unique) * counts[inverse])
            env._a6_natural_pools[kind] = torch.as_tensor(rows, device=dev)
            env._a6_natural_weights[kind] = torch.as_tensor(
                weights / weights.sum(), device=dev, dtype=torch.float32
            )
    if any(
        len(pool) == 0
        for pool in (*env._a6_pools.values(), *env._a6_natural_pools.values())
    ):
        raise ValueError("A6 source bank has an empty reset pool")
    env._a6_rng = torch.Generator(device=dev).manual_seed(env.cfg.seed + 91019)
    robot = env.scene["robot"]
    global_geoms = robot.indexing.geom_ids
    collision = [
        int(g) for g in global_geoms if env.sim.mj_model.geom_contype[int(g)] > 0
    ]
    env._a6_collision_geoms = torch.tensor(collision, device=dev, dtype=torch.long)
    if not collision:
        raise ValueError("Unitree G1 MJCF has no collision geoms")
    core_names_expected = {
        "pelvis_collision", "torso_collision", "head_collision",
        "left_hip_collision", "right_hip_collision",
        "left_thigh_collision", "right_thigh_collision",
    }
    core_local_ids, core_names = robot.find_geoms(
        (r"(pelvis|torso|head|(left|right)_(hip|thigh))_collision",)
    )
    if set(core_names) != core_names_expected:
        raise ValueError(f"Expected seven prone core collision geoms: {core_names}")
    core_globals = {int(robot.indexing.geom_ids[i]) for i in core_local_ids}
    env._a6_prone_core_collision_indices = torch.tensor(
        [i for i, geom_id in enumerate(collision) if geom_id in core_globals],
        device=dev,
        dtype=torch.long,
    )
    if len(env._a6_prone_core_collision_indices) != len(core_names_expected):
        raise ValueError("Prone core collision geoms are not in the robot bounds")
    arm_local_ids, arm_names = robot.find_geoms(
        (r"(left|right)_(shoulder_yaw|elbow_yaw|wrist|hand)_collision",)
    )
    if len(arm_names) != 8:
        raise ValueError(f"Expected eight arm collision geoms: {arm_names}")
    arm_globals = {int(robot.indexing.geom_ids[i]) for i in arm_local_ids}
    env._a6_arm_collision_indices = torch.tensor(
        [i for i, geom_id in enumerate(collision) if geom_id in arm_globals],
        device=dev,
        dtype=torch.long,
    )
    if len(env._a6_arm_collision_indices) != len(arm_names):
        raise ValueError("A6 arm collision geoms are not in the robot bounds")
    leg_local_ids, leg_names = robot.find_geoms(
        (r"(left|right)_(shin|linkage_brace|foot[1-7])_collision",)
    )
    if len(leg_names) != 18:
        raise ValueError(f"Expected 18 lower-leg/foot collision geoms: {leg_names}")
    leg_globals = {int(robot.indexing.geom_ids[i]) for i in leg_local_ids}
    env._a6_leg_collision_indices = torch.tensor(
        [i for i, geom_id in enumerate(collision) if geom_id in leg_globals],
        device=dev,
        dtype=torch.long,
    )
    if len(env._a6_leg_collision_indices) != len(leg_names):
        raise ValueError("A6 lower-leg collision geoms are not in the robot bounds")
    env._a6_plate_geoms = torch.tensor(
        [
            int(env.scene[name].indexing.geom_ids[0])
            for name in ("escape_obstacle", "free_obstacle")
        ],
        device=dev,
    )
    env._a6_plate_bodies = torch.tensor(
        [
            int(env.scene[name].indexing.body_ids[-1])
            for name in ("escape_obstacle", "free_obstacle")
        ],
        device=dev,
    )
    env._a6_initial_root_xy = torch.zeros(n, 2, device=dev)
    env._a6_initial_plate_xy = torch.zeros(n, 2, device=dev)
    env._a6_best_plate_shift = torch.zeros(n, device=dev)
    env._a6_initial_lateral_axis_xy = torch.zeros(n, 2, device=dev)
    env._a6_initial_edge_offset_m = torch.zeros(n, device=dev)
    # Fixed per-world tiers keep curriculum accounting stable across resets.
    env._a6_edge_tier = torch.zeros(n, device=dev, dtype=torch.long)
    env._a6_initial_plate_z = torch.zeros(n, device=dev)
    for name, dtype in (
        ("ever_contact", torch.bool),
        ("escaped", torch.bool),
        ("invalid", torch.bool),
        ("clear_age", torch.long),
        ("initial_count", torch.float32),
        ("best_score", torch.float32),
        ("best_clearance", torch.float32),
        ("best_distance", torch.float32),
        ("initial_leg_count", torch.float32),
        ("best_leg_score", torch.float32),
    ):
        setattr(env, f"_a6_{name}", torch.zeros(n, device=dev, dtype=dtype))
    env._a6_reward_tick = -1


@requires_model_fields(
    "body_pos",
    "geom_size",
    recompute=RecomputeLevel.set_const,
)
def reset_a6_scene(
    env,
    env_ids=None,
    layout=None,
    edge_curriculum=False,
    aligned_edge=False,
    footprint_edge=False,
    footprint_overlap_m=0.25,
    hard_mix=False,
    side_bridge_curriculum=False,
    side_staged_bridge=False,
    side_center_only=False,
) -> None:
    initialize(env, layout=layout)
    env._a6_edge_curriculum = edge_curriculum
    edge_offset_m = (
        float(getattr(env, "_a6_edge_offset_m", 0.0)) if edge_curriculum else 0.0
    )
    if (
        not math.isfinite(edge_offset_m)
        or not 0 <= edge_offset_m <= MAX_GUIDED_EDGE_OFFSET_M
    ):
        raise ValueError("A6 guided edge offset must be in [0, 0.50] metres")
    if not math.isfinite(footprint_overlap_m) or not 0 < footprint_overlap_m <= 0.50:
        raise ValueError("A6 footprint overlap must be in (0, 0.50] metres")
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else env_ids.long()
    )
    # Guided starts: 50% at the curriculum's easy offset, 25% halfway in,
    # and 25% with the plate centred on the robot. The latter two tiers
    # supply hard examples from the first update without blocking the gate.
    tier = ids.remainder(4) if hard_mix else torch.zeros_like(ids)
    if side_center_only:
        # A proprio-only actor cannot know which offset was sampled at reset.
        # Keep every guided-side teacher label on the centred target geometry.
        tier = side_center_tiers(tier, env._a6_scene[ids], env._a6_direction[ids])
    if side_bridge_curriculum and not side_staged_bridge and edge_offset_m > 0.05:
        # Side-lying cannot learn from a centred board while the half-offset
        # starts still fail. Reallocate that quarter to the bridge tier.
        tier = side_bridge_tiers(
            tier, env._a6_scene[ids], env._a6_direction[ids], edge_offset_m
        )
    env._a6_edge_tier[ids] = tier
    offset_scale = (
        side_staged_offset_scale(
            tier, env._a6_scene[ids], env._a6_direction[ids], edge_offset_m
        )
        if side_staged_bridge
        else torch.where(tier < 2, 1.0, torch.where(tier == 2, 0.5, 0.0))
    )
    offsets = offset_scale * edge_offset_m
    # A normal guided-prone world may temporarily enter the postescape
    # controller phase. A new episode must restore its original reset type.
    env._a6_postescape_start[ids] = env._a6_stratum[ids] == 3
    reset_scene_to_default(env, ids)
    q = torch.zeros(len(ids), 36, device=env.device)
    qvel = torch.zeros(len(ids), 35, device=env.device)
    saved_plate = torch.zeros(len(ids), 3, device=env.device)
    low = env._a6_stratum[ids] == 0
    for d in range(4):
        for source in range(2):
            mask = low & (env._a6_direction[ids] == d) & (env._a6_source[ids] == source)
            count = int(mask.sum())
            if count:
                pool = env._a6_pools[d, source]
                index = pool[
                    torch.randint(
                        len(pool), (count,), generator=env._a6_rng, device=env.device
                    )
                ]
                q[mask] = env._a6_banks["multiterrain"]["qpos"][index]
    natural = env._a6_banks["natural_curriculum"]["qpos"]
    for kind in (1, 2):
        mask = env._a6_stratum[ids] == kind
        count = int(mask.sum())
        if count:
            pool = env._a6_natural_pools[kind]
            index = pool[
                torch.multinomial(
                    env._a6_natural_weights[kind],
                    count,
                    replacement=True,
                    generator=env._a6_rng,
                )
            ]
            q[mask] = natural[index]
    postescape = env._a6_postescape_start[ids]
    env._a6_postescape_bank_index[ids] = -1
    if postescape.any():
        bank = env._a6_postescape_bank
        index = torch.randint(
            len(bank["qpos"]),
            (int(postescape.sum()),),
            generator=env._a6_rng,
            device=env.device,
        )
        q[postescape] = bank["qpos"][index]
        env._a6_postescape_bank_index[ids[postescape]] = index
        qvel[postescape] = bank["qvel"][index]
        saved_plate[postescape] = bank["plate_xyz"][index]
    robot = env.scene["robot"]
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] = q[:, :3] + env.scene.env_origins[ids]
    root[:, 3:7] = q[:, 3:7]
    root[:, 7:] = qvel[:, :6]
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(q[:, 7:], qvel[:, 6:], env_ids=ids)
    # Both boards are parked during robot AABB measurement; only one is activated.
    guided = env.scene["escape_obstacle"]
    free = env.scene["free_obstacle"]
    park = root[:, :7].clone()
    park[:, :3] = env.scene.env_origins[ids] + park.new_tensor((20.0, 20.0, 0.5))
    park[:, 3:7] = park.new_tensor((1.0, 0.0, 0.0, 0.0))
    guide_body = env._a6_plate_bodies[0]
    # Entity assembly already offsets the body by each environment origin;
    # model.body_pos is local to that origin, unlike free-root qpos.
    env.sim.model.body_pos[ids, guide_body] = (
        park[:, :3] - env.scene.env_origins[ids] + park.new_tensor((0.0, 0.0, 0.5))
    )
    guided.write_joint_state_to_sim(
        torch.full((len(ids), 1), -0.5, device=env.device),
        torch.zeros(len(ids), 1, device=env.device),
        env_ids=ids,
    )
    free_state = free.data.default_root_state[ids].clone()
    free_state[:, :7] = park
    free_state[:, 0] += 2.0
    free_state[:, 7:] = 0
    free.write_root_state_to_sim(free_state, env_ids=ids)
    env.sim.forward()
    pos, ext = robot_bounds(env)
    lateral = planar_escape_axis(root[:, 3:7], env._a6_direction[ids])
    guided_rows = env._a6_scene[ids] == 1
    if aligned_edge and edge_offset_m > 0:
        # The board is axis-aligned in world coordinates. Moving its centre
        # 25 cm along a yawed body axis can leave the pelvis 20 cm from an
        # edge, although the intended easy start is about 7 cm. Choose the
        # dominant world axis and compensate for the board's unequal sides.
        dominant_x = lateral[:, 0].abs() >= lateral[:, 1].abs()
        exit_axis = torch.zeros_like(lateral)
        exit_axis[:, 0] = torch.where(
            dominant_x, lateral[:, 0].sign(), 0.0
        )
        exit_axis[:, 1] = torch.where(
            ~dominant_x, lateral[:, 1].sign(), 0.0
        )
        if footprint_edge:
            # Give every reset the same *robot-plate overlap* instead of the
            # same pelvis-to-edge distance. A long outstretched limb can need
            # over a metre of travel even when the pelvis starts 7 cm away.
            relative = pos[ids, :, :2] - root[:, None, :2]
            parallel = (relative * exit_axis[:, None, :]).sum(-1)
            parallel_extent = torch.where(
                dominant_x[:, None], ext[ids, :, 0], ext[ids, :, 1]
            )
            other_distance = torch.where(
                dominant_x[:, None], relative[:, :, 1].abs(), relative[:, :, 0].abs()
            )
            other_extent = torch.where(
                dominant_x[:, None], ext[ids, :, 1], ext[ids, :, 0]
            )
            other_half = torch.where(
                dominant_x,
                root.new_full((len(ids),), PLATE_HALF_SIZE[1]),
                root.new_full((len(ids),), PLATE_HALF_SIZE[0]),
            )
            eligible = other_distance < other_extent + other_half[:, None]
            far_side = torch.where(
                eligible, parallel + parallel_extent, -torch.inf
            ).amax(-1)
            if not torch.isfinite(far_side).all():
                raise RuntimeError("No robot geom lies in the guided board's lateral lane")
            parallel_half = torch.where(
                dominant_x,
                root.new_full((len(ids),), PLATE_HALF_SIZE[0]),
                root.new_full((len(ids),), PLATE_HALF_SIZE[1]),
            )
            easy_shift = (far_side - footprint_overlap_m + parallel_half).clamp_min(0.0)
            shift = easy_shift * (offsets / 0.25)
        else:
            shift_x = (offsets * (0.38 / 0.25)).clamp(max=0.60)
            shift = torch.where(
                dominant_x,
                shift_x,
                offsets,
            )
        board_xy = root[:, :2] + guided_rows[:, None] * shift[:, None] * exit_axis
        lateral = exit_axis
    else:
        board_xy = root[:, :2] + guided_rows[:, None] * offsets[:, None] * lateral
    covered = (
        (pos[ids, :, :2] - board_xy[:, None, :]).abs()
        < ext[ids, :, :2] + root.new_tensor(PLATE_HALF_SIZE[:2])
    ).all(-1)
    top = torch.where(covered, pos[ids, :, 2] + ext[ids, :, 2], -torch.inf).amax(-1)
    if not torch.isfinite(top[~postescape]).all():
        raise RuntimeError("No robot geom is under the A6 plate footprint")
    board = park.clone()
    board[:, :2] = board_xy
    board[:, 2] = top + 0.035 + 0.002
    if postescape.any():
        board[postescape, :3] = (
            saved_plate[postescape] + env.scene.env_origins[ids[postescape]]
        )
    for scene_id, entity in ((1, guided), (2, free)):
        selected = env._a6_scene[ids] == scene_id
        chosen = ids[selected]
        if not len(chosen):
            continue
        if scene_id == 1:
            env.sim.model.body_pos[chosen, guide_body] = (
                board[selected, :3]
                - env.scene.env_origins[chosen]
                + board.new_tensor((0.0, 0.0, 0.5))
            )
        else:
            state = free.data.default_root_state[chosen].clone()
            state[:, :7] = board[selected]
            state[:, 7:] = 0
            free.write_root_state_to_sim(state, env_ids=chosen)
    env.sim.forward()
    # Frozen per-episode references: the guided plate has no horizontal DoF,
    # while the robot's initial body frame defines a useful prone escape side.
    env._a6_initial_root_xy[ids] = robot.data.root_link_pos_w[ids, :2]
    env._a6_initial_plate_xy[ids] = board[:, :2]
    env._a6_best_plate_shift[ids] = 0
    env._a6_initial_lateral_axis_xy[ids] = lateral
    env._a6_initial_edge_offset_m[ids] = (
        guided_rows * shift if aligned_edge and edge_offset_m > 0 else guided_rows * offsets
    )
    env._a6_initial_plate_z[ids] = board[:, 2]
    pos, ext, plate_pos, plate_ext = plate_footprint_inputs(env)
    count, score, clearance = footprint(pos, ext, plate_pos, plate_ext)
    if postescape.any() and (
        (count[ids[postescape]] != 0).any()
        or (clearance[ids[postescape]] < 0.035).any()
    ):
        raise RuntimeError("postescape replay pose is not clear of the guided board")
    leg_ids = env._a6_leg_collision_indices
    leg_count, leg_score, _ = footprint(
        pos[:, leg_ids], ext[:, leg_ids], plate_pos, plate_ext
    )
    env._a6_initial_count[ids] = count[ids].float().clamp_min(1)
    env._a6_initial_leg_count[ids] = leg_count[ids].float().clamp_min(1)
    env._a6_best_leg_score[ids] = leg_score[ids]
    env._a6_best_score[ids] = score[ids]
    env._a6_best_clearance[ids] = clearance[ids]
    env._a6_best_distance[ids] = 0
    env._a6_ever_contact[ids] = False
    env._a6_escaped[ids] = False
    env._a6_invalid[ids] = False
    env._a6_clear_age[ids] = 0
    from .a6_reward import reset_reward

    reset_reward(env, ids)
    if postescape.any():
        escaped_ids = ids[postescape]
        env._a6_ever_contact[escaped_ids] = True
        env._a6_escaped[escaped_ids] = True
        env._a6_clear_age[escaped_ids] = 25
        env._a6_knee_reward_done[escaped_ids] = True
