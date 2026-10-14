"""SMP historical A6 shared task/cost and obstacle rewards for HoST four critics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import quat_apply_inverse

from .a6_geometry import (
    PLATE_HALF_SIZE,
    completion_clearance,
    footprint,
    plate_footprint_inputs,
)
from .model import HOST29_JOINT_NAMES, HOST29_TARGET_JOINT_POS

TASK_WEIGHTS = (0.22, 0.18, 0.10, 0.15, 0.08, 0.07, 0.07, 0.06, 0.07)
COST_WEIGHTS = (-0.0015, -0.0012, -5e-8, -1e-6, -0.02, -2e-6, -1.0, -0.05)
NO_PROGRESS_GRACE_S = 0.50
NO_PROGRESS_FULL_S = 1.50
GUIDED_NO_PROGRESS_GRACE_S = 0.20
GUIDED_NO_PROGRESS_FULL_S = 0.90
MEANINGFUL_TORSO_PROGRESS_M = 0.02
GUIDED_KNEE_SOFT_LIMIT_RAD = 0.45
PRE_ESCAPE_KNEE_TOLERANCE_RAD = 0.30
PRE_ESCAPE_KNEE_SCALE_RAD = 0.25
PRE_ESCAPE_KNEE_DEEP_SLOPE = 0.20
STAGED_NEAR_EDGE_START_M = 0.08
STAGED_NEAR_EDGE_FULL_M = 0.18
GUIDED_HIP_SOFT_LIMIT_RAD = 0.75
GUIDED_KNEE_ALIGNMENT_RANGE_RAD = 2.40
GUIDED_HIP_ALIGNMENT_RANGE_RAD = 1.60
HEADWARD_UPPER_ARM_TARGET_COSINE = 0.60
GUIDED_MAX_LIFT_M = 0.05
PLATE_FORCE_INVALID_N = 5000.0
A6_RESET_VARIANT = "plate_force_5000_guided_lift_005_no_depth_real_age_v3"
EFFORT = torch.tensor(
    [
        139.0
        if "knee" in n
        else 50.0
        if "ankle" in n
        else 5.0
        if "wrist_pitch" in n or "wrist_yaw" in n
        else 25.0
        if any(k in n for k in ("shoulder", "elbow", "wrist"))
        else 50.0
        if "waist_roll" in n or "waist_pitch" in n
        else 88.0
        for n in HOST29_JOINT_NAMES
    ]
)


def gate(x, low, high):
    return ((x - low) / (high - low)).clamp(0.0, 1.0)


def path_progress(coverage_delta, clearance_delta, head_height, hand_support, active):
    """Historical A6 G: best XY improvement while hand-supported and low."""
    return (
        active
        * (head_height <= 0.90)
        * hand_support
        * (
            (coverage_delta / 0.025).clamp(0, 1)
            + 0.5 * (clearance_delta / 0.02).clamp(0, 1)
        )
    )


def joint_stall_cost(load_per_joint, motion_span):
    """Keep an isolated stalled joint visible despite other joints moving."""
    return (load_per_joint * (1 - motion_span / 0.10).clamp(0, 1)).amax(-1)


def lateral_travel(displacement_xy, lateral_axis_xy, toward_near_edge=False):
    """Measure pelvis motion toward the near edge, or either edge for legacy A6."""
    projected = (displacement_xy * lateral_axis_xy).sum(-1)
    return (-projected).clamp_min(0) if toward_near_edge else projected.abs()


def new_lateral_progress(
    displacement_xy, lateral_axis_xy, best_distance, toward_near_edge=False
):
    """Reward new pelvis travel along the selected initial lateral direction."""
    distance = lateral_travel(
        displacement_xy, lateral_axis_xy, toward_near_edge=toward_near_edge
    )
    return torch.maximum(best_distance, distance), (
        (distance - best_distance) / 0.025
    ).clamp(0, 1)


def new_torso_progress(distance, previous_best):
    """Credit only new XY distance from the trapped episode's initial position."""
    return torch.maximum(previous_best, distance), (
        (distance - previous_best) / 0.025
    ).clamp(0, 1)


def guided_exit_margin(root_xy, plate_xy):
    """Remaining pelvis travel to the nearest guided-plate XY edge in metres."""
    delta = (root_xy - plate_xy).abs()
    return torch.minimum(
        PLATE_HALF_SIZE[0] - delta[:, 0],
        PLATE_HALF_SIZE[1] - delta[:, 1],
    )


def arm_overlap_depth(pos, ext, plate_pos, plate_ext, arm_indices):
    """Deepest XY plate overlap of an arm geom; zero only when all arms clear."""
    overlap = (
        ext[:, arm_indices, :2]
        + plate_ext[:, None, :2]
        - (pos[:, arm_indices, :2] - plate_pos[:, None, :2]).abs()
    )
    return overlap.amin(-1).clamp_min(0).amax(-1)


def no_progress_ramp(seconds, grace_s=NO_PROGRESS_GRACE_S, full_s=NO_PROGRESS_FULL_S):
    """Penalize stalling after a scene-specific grace period."""
    return ((seconds - grace_s) / (full_s - grace_s)).clamp(0, 1)


def guided_leg_posture(knees, hips, previous_knees, previous_hips):
    """Measure sustained leg extension and signed, step-to-step improvement.

    The old best-ever knee bonus paid for a brief extension even if the knees
    immediately curled again.  Footprint reduction alone also paid for folding
    the shins inside the plate.  Both knees and hips now determine a persistent
    straightness score; bending back cancels an earlier extension step.
    """
    knee_excess = (knees - GUIDED_KNEE_SOFT_LIMIT_RAD).clamp_min(0)
    hip_excess = ((hips + 0.10).abs() - GUIDED_HIP_SOFT_LIMIT_RAD).clamp_min(0)
    previous_knee_excess = (previous_knees - GUIDED_KNEE_SOFT_LIMIT_RAD).clamp_min(0)
    previous_hip_excess = (
        (previous_hips + 0.10).abs() - GUIDED_HIP_SOFT_LIMIT_RAD
    ).clamp_min(0)
    knee_cost = (knee_excess / (knee_excess + 0.8)).amax(-1)
    hip_cost = (hip_excess / (hip_excess + 0.8)).amax(-1)
    straightness = torch.exp(
        -1.2 * knee_excess.mean(-1)
        - 0.8 * hip_excess.mean(-1)
        - 0.4 * knee_excess.amax(-1)
    )
    previous_bend = previous_knee_excess.mean(-1) + 0.5 * previous_hip_excess.mean(-1)
    current_bend = knee_excess.mean(-1) + 0.5 * hip_excess.mean(-1)
    extension_step = ((previous_bend - current_bend) / 0.12).clamp(-2.0, 2.0)
    return knee_cost, hip_cost, straightness, extension_step


def pre_escape_knee_curl(knees):
    """Penalize each knee beyond 0.30 rad from the standing knee pose."""
    stand = knees.new_tensor(
        [HOST29_TARGET_JOINT_POS[f"{side}_knee_joint"] for side in ("left", "right")]
    )
    excess = ((knees - stand).abs() - PRE_ESCAPE_KNEE_TOLERANCE_RAD).clamp_min(0)
    # The saturating part reacts promptly just past the threshold. The linear
    # part keeps a useful gradient when a knee is already deeply curled.
    return (
        excess / (excess + PRE_ESCAPE_KNEE_SCALE_RAD)
        + PRE_ESCAPE_KNEE_DEEP_SLOPE * excess
    ).sum(-1)


def staged_guided_knee_action(knees, directions, near_edge, strict=False):
    """Provide a moderate knee-action target until the torso nears the exit."""
    if strict:
        target = torch.full_like(directions, 0.35, dtype=knees.dtype)
    else:
        phase = gate(near_edge, STAGED_NEAR_EDGE_START_M, STAGED_NEAR_EDGE_FULL_M)
        target = torch.where(directions == 1, 1.50 - 1.05 * phase, 0.45)
    return (2.0 * (target[:, None] - knees)).clamp(-0.45, 0.45)


def guided_linear_leg_alignment(knees, hips):
    """Keep a usable extension gradient even when the legs start tightly curled."""
    knee_bend = (
        (
            (knees - GUIDED_KNEE_SOFT_LIMIT_RAD).clamp_min(0)
            / GUIDED_KNEE_ALIGNMENT_RANGE_RAD
        )
        .clamp_max(1)
        .amax(-1)
    )
    hip_bend = (
        (
            ((hips + 0.10).abs() - GUIDED_HIP_SOFT_LIMIT_RAD).clamp_min(0)
            / GUIDED_HIP_ALIGNMENT_RANGE_RAD
        )
        .clamp_max(1)
        .amax(-1)
    )
    return 1.0 - 0.70 * knee_bend - 0.30 * hip_bend


def headward_upper_arm_score(headward_cosine):
    """Score both upper arms pointing toward the head, in the torso frame."""
    if headward_cosine.shape[-1] != 2:
        raise ValueError("headward_cosine must contain left and right arms")
    # The AMP-style clamp at -target gave zero to an arm pointing farther
    # down than -0.60. Supine resets often start there, leaving no preparatory
    # signal until a very large shoulder rotation happens by chance.
    per_arm = (
        (headward_cosine + 1.0) / (1.0 + HEADWARD_UPPER_ARM_TARGET_COSINE)
    ).clamp(0, 1)
    return per_arm.amin(-1)


def hand_supported_lateral_progress(current_lateral, best_lateral, hand_support):
    """Credit hand contact only when it advances the pelvis to new territory."""
    return hand_support * ((current_lateral - best_lateral) / 0.01).clamp(0, 1)


def guided_overcurl(knees, hips):
    """Bounded cost for the deep knee/hip fold seen in guided-board rollouts."""
    knee = ((knees - 1.2) / 1.5).clamp(0, 1).amax(-1)
    hip = (((hips + 0.1).abs() - 1.2) / 1.0).clamp(0, 1).amax(-1)
    return 0.75 * knee + 0.25 * hip


def outward_hand_drive(
    displacement, velocity, lateral_axis, hand_load, toward_near_edge=False
):
    """Reward loaded hands only when the pelvis moves farther toward an edge."""
    lateral = (displacement * lateral_axis).sum(-1)
    lateral_speed = (velocity * lateral_axis).sum(-1)
    outward_speed = (
        -lateral_speed if toward_near_edge else lateral_speed * lateral.sign()
    )
    return hand_load * gate(outward_speed, 0.01, 0.12)


def _init(env):
    if hasattr(env, "_a6_stage"):
        return
    n, dev = env.num_envs, env.device
    robot = env.scene["robot"]
    env._a6_head_geom_id = robot.find_geoms(("head_collision",))[0][0]
    env._a6_foot_ids = robot.find_bodies(
        ("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True
    )[0]
    env._a6_knee_ids = [
        HOST29_JOINT_NAMES.index(f"{side}_knee_joint") for side in ("left", "right")
    ]
    env._a6_hip_pitch_ids = [
        HOST29_JOINT_NAMES.index(f"{side}_hip_pitch_joint")
        for side in ("left", "right")
    ]
    env._a6_effort_limits = EFFORT.to(dev)
    env._a6_rows = torch.arange(n, device=dev)
    env._a6_stage = torch.zeros(n, device=dev, dtype=torch.long)
    env._a6_hold = torch.zeros(n, device=dev, dtype=torch.long)
    env._a6_last_action = torch.zeros(n, 29, device=dev)
    env._a6_last_delta = torch.zeros(n, 29, device=dev)
    env._a6_timer = torch.zeros(n, 29, device=dev)
    env._a6_motion = torch.zeros(25, n, 29, device=dev)
    env._a6_motion_cursor = 0
    env._a6_subtick = 0
    env._a6_accum = torch.zeros(n, 5, device=dev)
    env._a6_stall_accum = torch.zeros(n, 29, device=dev)
    env._a6_task_values = torch.zeros(n, 9, device=dev)
    env._a6_cost_values = torch.zeros(n, 8, device=dev)
    torso_id = robot.find_bodies("torso_link")[0][0]
    shoulder_ids = robot.find_bodies(
        ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
        preserve_order=True,
    )[0]
    elbow_ids = robot.find_bodies(
        ("left_elbow_link", "right_elbow_link"), preserve_order=True
    )[0]
    env._a6_arm_body_ids = (torso_id, shoulder_ids, elbow_ids)
    env._a6_escape_values = torch.zeros(n, 27, device=dev)
    env._a6_aux_progress_reward = torch.zeros(n, device=dev)
    env._a6_knee_reward_done = torch.zeros(n, device=dev, dtype=torch.bool)
    env._a6_best_lateral = torch.zeros(n, device=dev)
    env._a6_best_torso_distance = torch.zeros(n, device=dev)
    env._a6_initial_exit_margin = torch.zeros(n, device=dev)
    env._a6_previous_exit_margin = torch.zeros(n, device=dev)
    env._a6_best_exit_progress = torch.zeros(n, device=dev)
    env._a6_previous_arm_depth = torch.zeros(n, device=dev)
    env._a6_arm_depth = torch.zeros(n, device=dev)
    env._a6_arm_extraction_reward = torch.zeros(n, device=dev)
    env._a6_near_edge_phase = torch.zeros(n, device=dev)
    env._a6_previous_knees = torch.zeros(n, 2, device=dev)
    env._a6_previous_hips = torch.zeros(n, 2, device=dev)
    env._a6_progress_anchor = torch.zeros(n, device=dev)
    env._a6_no_progress_time = torch.zeros(n, device=dev)
    env._a6_plate_lift_peak = torch.zeros(n, device=dev)
    env._a6_reward_tick = -1
    env._a6_reset_reward = lambda ids: reset_reward(env, ids)


def reset_reward(env, ids):
    _init(env)
    r = env.scene["robot"]
    z = r.data.geom_pos_w[ids, env._a6_head_geom_id, 2] - env.scene.env_origins[ids, 2]
    u = (-r.data.projected_gravity_b[ids, 2]).clamp(0, 1)
    stage = torch.zeros_like(ids)
    stage = torch.where((z >= 0.55) & (u >= 0.55), 1, stage)
    stage = torch.where((z >= 0.78) & (u >= 0.72), 2, stage)
    stage = torch.where((z >= 1.08) & (u >= 0.85), 3, stage)
    env._a6_stage[ids] = stage
    env._a6_hold[ids] = 0
    env._a6_timer[ids] = 0
    env._a6_accum[ids] = 0
    env._a6_stall_accum[ids] = 0
    env._a6_best_lateral[ids] = 0
    env._a6_best_torso_distance[ids] = 0
    plate_xy = env.sim.data.geom_xpos[ids, env._a6_plate_geoms[0], :2]
    initial_margin = guided_exit_margin(r.data.root_link_pos_w[ids, :2], plate_xy)
    env._a6_initial_exit_margin[ids] = initial_margin
    env._a6_previous_exit_margin[ids] = initial_margin
    env._a6_best_exit_progress[ids] = 0
    pos, ext, plate_pos, plate_ext = plate_footprint_inputs(env)
    initial_arm_depth = arm_overlap_depth(
        pos, ext, plate_pos, plate_ext, env._a6_arm_collision_indices
    )
    env._a6_previous_arm_depth[ids] = initial_arm_depth[ids]
    env._a6_arm_depth[ids] = initial_arm_depth[ids]
    env._a6_arm_extraction_reward[ids] = 0
    env._a6_near_edge_phase[ids] = 0
    env._a6_previous_knees[ids] = r.data.joint_pos[ids][:, env._a6_knee_ids]
    env._a6_previous_hips[ids] = r.data.joint_pos[ids][:, env._a6_hip_pitch_ids]
    env._a6_progress_anchor[ids] = 0
    env._a6_no_progress_time[ids] = 0
    env._a6_plate_lift_peak[ids] = 0
    env._a6_knee_reward_done[ids] = False
    env._a6_last_action[ids] = 0
    env._a6_last_delta[ids] = 0
    env._a6_motion[:, ids] = r.data.joint_pos[ids][None]


def a6_substep(env):
    """Sample overload and physical costs at 500 Hz, not only at action steps."""
    _init(env)
    if env._a6_subtick % env.cfg.decimation == 0:
        env._a6_accum.zero_()
        env._a6_stall_accum.zero_()
        env._a6_plate_lift_peak.zero_()
    rows = env._a6_rows
    guided_z = env.sim.data.geom_xpos[rows, env._a6_plate_geoms[0], 2]
    lift = (guided_z - env._a6_initial_plate_z).clamp_min(0)
    env._a6_plate_lift_peak = torch.maximum(
        env._a6_plate_lift_peak, lift * (env._a6_scene == 1)
    )
    r = env.scene["robot"]
    tau, dq = r.data.qfrc_actuator, r.data.joint_vel
    stage = env._a6_stage
    ratio = (tau / env._a6_effort_limits).abs()
    env._a6_timer = torch.where(ratio > 0.7, env._a6_timer + env.physics_dt, 0.0)
    stall = (
        gate(ratio, 0.7, 0.95).square()
        * gate(env._a6_timer, 0.3, 1.0)
        * gate(0.3 - dq.abs(), 0, 0.3)
    )
    speed_limit = dq.new_tensor((6.0, 5.0, 4.0, 3.5))[stage, None]
    power_limit = dq.new_tensor((140.0, 110.0, 90.0, 75.0))[stage, None]
    vz = r.data.geom_lin_vel_w[:, env._a6_head_geom_id, 2]
    up = dq.new_tensor((0.3, 0.3, 0.3, 0.2))[stage]
    over = (vz - up).clamp_min(0).square() + (-vz - 0.2).clamp_min(0).square()
    env._a6_accum += torch.stack(
        (
            r.data.joint_acc.square().sum(-1)
            * dq.new_tensor((0.35, 0.65, 1.0, 1.0))[stage],
            tau.square().sum(-1) * dq.new_tensor((0.5, 0.75, 1.0, 1.0))[stage],
            (dq.abs() - speed_limit).clamp_min(0).square().sum(-1),
            ((tau * dq).abs() - power_limit).clamp_min(0).square().mean(-1),
            over,
        ),
        -1,
    )
    env._a6_stall_accum += stall
    env._a6_subtick += 1
    return stall.amax(-1)


def _update(env):
    _init(env)
    if env._a6_reward_tick == env.common_step_counter:
        return
    env._a6_reward_tick = env.common_step_counter
    r = env.scene["robot"]
    data = r.data
    z = data.geom_pos_w[:, env._a6_head_geom_id, 2] - env.scene.env_origins[:, 2]
    vz = data.geom_lin_vel_w[:, env._a6_head_geom_id, 2]
    # HoST's reset-time assistance curriculum consumes this relative height.
    # The flat task reward normally writes it, so A6 must preserve that state.
    from .task import _head_height

    env._host_head_height = _head_height(env).clone()
    u = (-data.projected_gravity_b[:, 2]).clamp(0, 1)
    stage = env._a6_stage
    fallen = (z < 0.65) & (u < 0.45)
    stage[fallen] = 0
    env._a6_hold[fallen] = 0
    knees = data.joint_pos[:, env._a6_knee_ids]
    hips = data.joint_pos[:, env._a6_hip_pitch_ids]
    foot_force = env.scene["a6_feet"].data.force.norm(dim=-1).amin(-1)
    ready = (
        (
            (stage == 0)
            & (z >= 0.55)
            & (u >= 0.55)
            & (knees.amin(-1) >= 0.8)
            & (vz.abs() <= 0.16)
        )
        | (
            (stage == 1)
            & (z >= 0.78)
            & (u >= 0.72)
            & (knees.amin(-1) >= 0.6)
            & (vz.abs() <= 0.18)
        )
        | (
            (stage == 2)
            & (z >= 1.08)
            & (u >= 0.85)
            & (knees.abs().amax(-1) < 0.8)
            & (foot_force > 20)
            & (vz.abs() <= 0.12)
        )
    )
    env._a6_hold = torch.where(ready, env._a6_hold + 1, 0)
    advance = env._a6_hold >= torch.where(stage == 2, 25, 10)
    stage.copy_(torch.where(advance, (stage + 1).clamp_max(3), stage))
    env._a6_hold[advance] = 0
    target_z = z.new_tensor((0.62, 0.86, 1.15, 1.15))[stage]
    target_u = z.new_tensor((0.60, 0.76, 0.93, 0.93))[stage]
    knee_low = z.new_tensor((0.8, 0.6, 0.0, 0.0))[stage, None]
    knee_high = z.new_tensor((1.8, 1.6, 0.65, 0.65))[stage, None]
    knee_error = (
        (knee_low - knees).clamp_min(0).square()
        + (knees - knee_high).clamp_min(0).square()
    ).mean(-1)
    pose = torch.exp(
        -8 * (target_z - z).clamp_min(0).square()
        - 6 * (target_u - u).clamp_min(0).square()
        - 5 * knee_error
    )
    speed_target = z.new_tensor((0.10, 0.15, 0.20, 0.0))[stage] * (
        (target_z - z) / 0.2
    ).clamp(0, 1)
    up_limit = z.new_tensor((0.25, 0.30, 0.30, 0.12))[stage]
    excess = (vz - up_limit).clamp_min(0).square() + (
        -vz - z.new_tensor((0.16, 0.18, 0.18, 0.12))[stage]
    ).clamp_min(0).square()
    head_velocity = torch.exp(-45 * (vz - speed_target).square() - 140 * excess)
    quiet = gate(z, 0.85, 1.15) * gate(u, 0.7, 0.93)
    foot_motion = (
        data.body_link_lin_vel_w[:, env._a6_foot_ids].square().sum(-1).mean(-1)
    )
    action = env.action_manager.action
    delta = action - env._a6_last_action
    accel = delta - env._a6_last_delta
    # PPO randomizes episode_length_buf to stagger timeouts. Only the action
    # term's counter records actual control steps since this scene was reset.
    real_steps = env.action_manager.get_term("joint_pos").real_steps
    fresh = real_steps <= 1
    delta = torch.where(fresh[:, None], 0.0, delta)
    accel = torch.where(fresh[:, None], 0.0, accel)
    env._a6_task_values = torch.stack(
        (
            pose,
            head_velocity,
            torch.exp(-2 * (1.15 - z).clamp_min(0).square()),
            u.square(),
            1 - quiet + quiet * torch.exp(-20 * foot_motion),
            1
            - quiet
            + quiet * torch.exp(-8 * data.root_link_lin_vel_w.square().sum(-1)),
            torch.exp(-0.8 * data.root_link_ang_vel_w.square().sum(-1)),
            torch.exp(-0.04 * data.joint_vel.square().mean(-1)),
            torch.exp(-4 * delta.square().mean(-1)),
        ),
        -1,
    )
    env._a6_cost_values[:, :2] = torch.stack(
        (
            delta.square().sum(-1) * z.new_tensor((0.5, 0.75, 1.0, 1.0))[stage],
            accel.square().sum(-1) * z.new_tensor((0.3, 0.6, 1.0, 1.0))[stage],
        ),
        -1,
    )
    env._a6_cost_values[:, 2:7] = env._a6_accum[:, :5] / env.cfg.decimation
    env._a6_cost_values[:, 7] = (env._a6_stall_accum / env.cfg.decimation).amax(-1)
    env._a6_last_action.copy_(action)
    env._a6_last_delta.copy_(delta)
    env._a6_motion[env._a6_motion_cursor] = data.joint_pos
    env._a6_motion_cursor = (env._a6_motion_cursor + 1) % 25
    span = env._a6_motion.amax(0) - env._a6_motion.amin(0)
    stall_cost = joint_stall_cost(env._a6_stall_accum / env.cfg.decimation, span)
    contact0 = env.scene["guided_contact"].data
    contact1 = env.scene["free_contact"].data
    free = env._a6_scene == 2
    # The sensor retains all ten 2 ms physics samples in each 20 ms action
    # interval. A single end-of-interval sample misses brief plate overloads.
    contact = torch.where(
        free,
        (contact1.found > 0).any(-1)
        | (contact1.force_history.norm(dim=-1) > 1.0).any(dim=(1, 2)),
        (contact0.found > 0).any(-1)
        | (contact0.force_history.norm(dim=-1) > 1.0).any(dim=(1, 2)),
    )
    force = torch.where(
        free,
        contact1.force_history.norm(dim=-1).amax(dim=(1, 2)),
        contact0.force_history.norm(dim=-1).amax(dim=(1, 2)),
    )
    active = env._a6_scene > 0
    env._a6_ever_contact |= contact & active
    pos, ext, plate_pos_for_footprint, plate_ext = plate_footprint_inputs(env)
    count, score, clearance = footprint(pos, ext, plate_pos_for_footprint, plate_ext)
    leg_ids = env._a6_leg_collision_indices
    leg_count, leg_score, _ = footprint(
        pos[:, leg_ids],
        ext[:, leg_ids],
        plate_pos_for_footprint,
        plate_ext,
    )
    delta_score = (env._a6_best_score - score).clamp_min(0)
    delta_clearance = (clearance - env._a6_best_clearance).clamp_min(0)
    env._a6_best_score = torch.minimum(env._a6_best_score, score)
    env._a6_best_clearance = torch.maximum(env._a6_best_clearance, clearance)
    rows = env._a6_rows
    plate_ids = env._a6_plate_geoms[env._a6_scene.clamp_min(1) - 1]
    plate_pos = env.sim.data.geom_xpos[rows, plate_ids]
    distance = (data.root_link_pos_w[:, :2] - plate_pos[:, :2]).norm(dim=-1)
    separation = ((distance - env._a6_best_distance) / 0.025).clamp(0, 1)
    env._a6_best_distance = torch.maximum(env._a6_best_distance, distance)
    # A guided-board episode is complete only after horizontal escape. The
    # previous 3D clearance test also accepted lifting the board overhead.
    guided = env._a6_scene == 1
    clear = (
        active
        & env._a6_ever_contact
        & ~contact
        & torch.where(guided, clearance >= 0.04, completion_clearance(env) >= 0.025)
    )
    env._a6_clear_age = torch.where(clear, env._a6_clear_age + 1, 0)
    if getattr(env, "_a6_arm_extraction_v1", False):
        # Once the policy has kept the full body clear for 25 control steps,
        # preserve the stand-up phase even if a hand briefly touches the board.
        env._a6_escaped |= env._a6_clear_age >= 25
    else:
        env._a6_escaped = env._a6_clear_age >= 25
    env._a6_knee_reward_done |= env._a6_escaped
    motor_active = real_steps > 30
    if hasattr(env, "_a6_postescape_start"):
        motor_active |= env._a6_postescape_start
    plate_lift = torch.maximum(
        (plate_pos[:, 2] - env._a6_initial_plate_z).clamp_min(0),
        env._a6_plate_lift_peak,
    )
    staged_v5 = bool(getattr(env, "_a6_staged_escape_v5", False))
    force_invalid = force > PLATE_FORCE_INVALID_N
    lift_invalid = (
        guided & motor_active & (count > 0) & (plate_lift > GUIDED_MAX_LIFT_M)
    )
    env._a6_invalid = active & (
        force_invalid
        | lift_invalid
        | ((real_steps > 25) & ~env._a6_ever_contact)
    )
    hand_data = env.scene["a6_hands"].data
    support = (hand_data.found > 0).float().mean(-1)
    hand_load = (hand_data.force.norm(dim=-1) / 60.0).clamp(0, 1).mean(-1)
    g_active = active & env._a6_ever_contact & ~env._a6_escaped & ~env._a6_invalid
    g = path_progress(delta_score, delta_clearance, z, support, g_active)
    dense = (
        active
        * ~env._a6_invalid
        * (
            0.85 * (1 - count / env._a6_initial_count).clamp(0, 1)
            + 0.15 * (clearance / 0.04).clamp(0, 1)
        )
    )
    displacement = data.root_link_pos_w[:, :2] - env._a6_initial_root_xy
    toward_near_edge = getattr(env, "_a6_edge_curriculum", False)
    previous_lateral = env._a6_best_lateral.clone()
    best_lateral, lateral_gain = new_lateral_progress(
        displacement,
        env._a6_initial_lateral_axis_xy,
        previous_lateral,
        toward_near_edge=toward_near_edge,
    )
    # Brief loss of contact does not mean the torso has left the footprint.
    trapped = (
        active
        & env._a6_ever_contact
        & (count > 0)
        & ~env._a6_escaped
        & ~env._a6_invalid
        & motor_active
    )
    guided_trapped = trapped & guided
    arm_depth = arm_overlap_depth(
        pos, ext, plate_pos_for_footprint, plate_ext, env._a6_arm_collision_indices
    )
    previous_arm_depth = env._a6_previous_arm_depth.clone()
    env._a6_previous_arm_depth.copy_(arm_depth)
    env._a6_arm_depth.copy_(arm_depth)
    if getattr(env, "_a6_arm_extraction_v1", False):
        # Signed improvement pays for pulling a wrist/hand out and charges for
        # pushing it back under the plate. The bounded residual makes the last
        # centimetre matter even after the pelvis has already crossed an edge.
        arm_step = ((previous_arm_depth - arm_depth) / 0.025).clamp(-1, 1)
        arm_near_clear = torch.exp(-arm_depth / 0.025)
        guided_preescape = (
            guided & env._a6_ever_contact & ~env._a6_escaped
            & ~env._a6_invalid & motor_active
        )
        env._a6_arm_extraction_reward = guided_preescape * (
            8.0 * arm_step
            + 2.0 * arm_near_clear
            - (arm_depth / 0.15).clamp(0, 1)
        )
    else:
        env._a6_arm_extraction_reward.zero_()
    prone_trapped = guided_trapped & (env._a6_direction == 1)
    supine_trapped = guided_trapped & (env._a6_direction == 0)
    env._a6_best_lateral = torch.where(guided_trapped, best_lateral, previous_lateral)
    previous_leg_score = env._a6_best_leg_score.clone()
    env._a6_best_leg_score = torch.minimum(previous_leg_score, leg_score)
    leg_score_gain = ((previous_leg_score - leg_score) / 0.025).clamp(0, 1)
    leg_release = (
        (env._a6_initial_leg_count - leg_count.float())
        / env._a6_initial_leg_count.clamp_min(1)
    ).clamp(0, 1)
    knee_bend_cost, hip_bend_cost, leg_straightness, extension_step = (
        guided_leg_posture(knees, hips, env._a6_previous_knees, env._a6_previous_hips)
    )
    env._a6_previous_knees.copy_(knees)
    env._a6_previous_hips.copy_(hips)
    # Folding feet out of the XY footprint is not escape.  Only reward their
    # release when the legs are extended and the torso has travelled as well.
    torso_distance = displacement.norm(dim=-1)
    previous_torso_distance = env._a6_best_torso_distance.clone()
    best_torso_distance, torso_gain = new_torso_progress(
        torso_distance, previous_torso_distance
    )
    env._a6_best_torso_distance = torch.where(
        guided_trapped, best_torso_distance, previous_torso_distance
    )
    exit_margin = guided_exit_margin(data.root_link_pos_w[:, :2], plate_pos[:, :2])
    exit_progress = (env._a6_initial_exit_margin - exit_margin).clamp_min(0)
    previous_exit_progress = env._a6_best_exit_progress.clone()
    best_exit_progress, exit_gain = new_torso_progress(
        exit_progress, previous_exit_progress
    )
    env._a6_best_exit_progress = torch.where(
        guided_trapped, best_exit_progress, previous_exit_progress
    )
    safe_exit_gate = (1.0 - gate(plate_lift, 0.01, 0.04))
    exit_gain = exit_gain * safe_exit_gate
    exit_speed = (
        (env._a6_previous_exit_margin - exit_margin) / env.step_dt
    ).clamp_min(0)
    env._a6_previous_exit_margin.copy_(exit_margin)
    leg_release_gate = leg_straightness * gate(torso_distance, 0.02, 0.15)
    current_lateral = lateral_travel(
        displacement,
        env._a6_initial_lateral_axis_xy,
        toward_near_edge=toward_near_edge,
    )
    staged_v3 = bool(
        getattr(env, "_a6_staged_escape_v3", False)
        or getattr(env, "_a6_staged_escape_v4", False)
        or staged_v5
    )
    staged = bool(
        getattr(env, "_a6_staged_escape_v1", False)
        or getattr(env, "_a6_staged_escape_v2", False)
        or staged_v3
    )
    near_edge_phase = (
        gate(-exit_margin, -0.08, 0.0)
        if staged_v5
        else gate(current_lateral, STAGED_NEAR_EDGE_START_M, STAGED_NEAR_EDGE_FULL_M)
    )
    prone_guided = guided & (env._a6_direction == 1)
    if getattr(env, "_a6_knee_straight_prior_v1", False):
        knee_weight = torch.ones_like(current_lateral)
    elif staged:
        knee_weight = torch.where(
            prone_guided, 0.15 + 0.85 * near_edge_phase, 1.0
        )
    else:
        knee_weight = torch.ones_like(current_lateral)
    env._a6_near_edge_phase = near_edge_phase
    env._a6_knee_penalty_weight = knee_weight
    # Early in the staged curriculum, lateral movement must remain worthwhile
    # before the legs can safely straighten under the guided plate.
    guided_progress_gate = torch.where(guided, 0.20 + 0.80 * leg_straightness, 1.0)
    if staged:
        staged_gate = 0.65 + 0.35 * leg_straightness
        guided_progress_gate = torch.where(
            prone_guided,
            staged_gate * (1.0 - near_edge_phase)
            + guided_progress_gate * near_edge_phase,
            guided_progress_gate,
        )
    g *= guided_progress_gate
    dense *= guided_progress_gate
    if staged_v5:
        g *= torch.where(guided, safe_exit_gate, 1.0)
        dense *= torch.where(guided, safe_exit_gate, 1.0)
    # Progress must be a *new* torso displacement; limb flailing alone cannot
    # clear the timer. The 2 cm threshold avoids resetting it for jitter.
    progress_distance = (
        torch.where(guided, best_exit_progress, torso_distance)
        if staged_v5
        else torch.where(guided, best_torso_distance, torso_distance)
        if staged_v3
        else torch.where(
            guided & ((env._a6_direction == 0) | (env._a6_direction == 1)),
            best_lateral,
            torso_distance,
        )
    )
    advanced = (
        progress_distance >= env._a6_progress_anchor + MEANINGFUL_TORSO_PROGRESS_M
    )
    env._a6_progress_anchor = torch.where(
        trapped & advanced, progress_distance, env._a6_progress_anchor
    )
    env._a6_no_progress_time = torch.where(
        trapped,
        torch.where(advanced, 0.0, env._a6_no_progress_time + env.step_dt),
        0.0,
    )
    no_progress = (
        torch.where(
            guided,
            no_progress_ramp(
                env._a6_no_progress_time,
                GUIDED_NO_PROGRESS_GRACE_S,
                GUIDED_NO_PROGRESS_FULL_S,
            ),
            no_progress_ramp(env._a6_no_progress_time),
        )
        * trapped
    )
    # v5's exponential straightness is near zero for a fully curled leg. Keep
    # that conservative gate on geometry rewards, but give G+ an independent
    # linear posture signal so the first extension is worth learning.
    linear_alignment = guided_linear_leg_alignment(knees, hips)
    moving_straight = (
        guided_trapped * linear_alignment * gate(progress_distance, 0.03, 0.25)
    )
    torso_travel = (
        guided_trapped * exit_gain
        if staged_v5
        else guided_trapped * torso_gain
        if staged_v3
        else guided_trapped * gate(current_lateral, 0.02, 0.35)
    )
    torso_id, shoulder_ids, elbow_ids = env._a6_arm_body_ids
    upper_arm_w = (
        data.body_link_pos_w[:, elbow_ids] - data.body_link_pos_w[:, shoulder_ids]
    )
    torso_quat = data.body_link_quat_w[:, torso_id].unsqueeze(1).expand(-1, 2, -1)
    upper_arm_b = quat_apply_inverse(torso_quat, upper_arm_w)
    arm_cosine = upper_arm_b[..., 2] / upper_arm_b.norm(dim=-1).clamp_min(1e-6)
    headward_arms = guided_trapped * headward_upper_arm_score(arm_cosine)
    hand_push = (
        guided_trapped * support * exit_gain
        if staged_v5
        else guided_trapped
        * hand_supported_lateral_progress(current_lateral, previous_lateral, support)
    )
    hand_drive = (
        guided_trapped * hand_load * gate(exit_speed, 0.01, 0.12) * safe_exit_gate
        if staged_v5
        else guided_trapped
        * outward_hand_drive(
            displacement,
            data.root_link_lin_vel_w[:, :2],
            env._a6_initial_lateral_axis_xy,
            hand_load,
            toward_near_edge=toward_near_edge,
        )
    )
    hand_plant = guided_trapped * support
    env._a6_aux_progress_reward = (
        env.step_dt * (10.0 * torso_travel + 3.0 * hand_push + 2.5 * hand_drive)
        if staged_v5
        else torch.zeros_like(torso_travel)
    )
    overcurl = guided_trapped * guided_overcurl(knees, hips)
    lift_cost = guided * motor_active * gate(plate_lift, 0.005, GUIDED_MAX_LIFT_M)
    env._a6_escape_values = torch.stack(
        (
            g,
            dense,
            env._a6_escaped.float(),
            active * env._a6_ever_contact * separation,
            active * ((force - 300).clamp_min(0) / 300).square(),
            quiet * (foot_motion / 0.1**2).clamp_max(10),
            stall_cost,
            prone_trapped * lateral_gain,
            no_progress,
            stall_cost * no_progress,
            lift_cost,
            supine_trapped * lateral_gain,
            guided_trapped * knee_bend_cost,
            guided_trapped * extension_step,
            guided_trapped * leg_score_gain * leg_release_gate,
            guided_trapped * leg_release * leg_release_gate,
            guided_trapped * leg_straightness,
            guided_trapped * hip_bend_cost,
            moving_straight,
            torso_travel,
            headward_arms,
            guided_trapped * linear_alignment,
            hand_push,
            hand_drive,
            hand_plant,
            overcurl,
            (active & ~env._a6_knee_reward_done).float()
            * knee_weight
            * pre_escape_knee_curl(knees),
        ),
        -1,
    )


def _standing_pose_cost(env):
    """Preserve the flat expert's arm/wrist and centered waist after escape."""
    from .stand_guard import stand_pose_ready

    if not hasattr(env, "_a6_stand_pose_cache"):
        names = tuple(env.scene["robot"].joint_names)
        if names != HOST29_JOINT_NAMES:
            raise ValueError("A6 standing pose uses a different 29-axis order")
        env._a6_stand_pose_cache = (
            torch.tensor(
                [HOST29_TARGET_JOINT_POS[n] for n in names], device=env.device
            ),
            [
                i
                for i, n in enumerate(names)
                if "shoulder" in n or "elbow" in n or "wrist" in n
            ],
            [i for i, n in enumerate(names) if "waist" in n],
            [i for i, n in enumerate(names) if "wrist_pitch" in n or "wrist_yaw" in n],
        )
    target, arms, waist, weak_wrists = env._a6_stand_pose_cache
    q = env.scene["robot"].data.joint_pos

    def error(ids):
        return F.smooth_l1_loss(
            q[:, ids], target[ids].expand_as(q[:, ids]), beta=0.25, reduction="none"
        ).sum(-1)

    eligible = stand_pose_ready(env, min_steps=90) & (
        (env._a6_scene == 0) | env._a6_escaped
    )
    return eligible.float() * (
        0.10 * error(arms) + 0.25 * error(waist) + 0.10 * error(weak_wrists)
    )


def _guided_recovery_reward(env, *, include_supine=False):
    """Reward rising after true guided clearance, without rewarding board lifting.

    The usual constant escaped bonus has no gradient between a low, curled
    pose and a stable stand. This score pays progressively for lifting the
    pelvis and becoming upright, then pays a small stable-standing bonus.
    """
    robot = env.scene["robot"].data
    pelvis_z = robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    height = ((pelvis_z - 0.10) / 0.60).clamp(0.0, 1.0)
    upright = ((-robot.projected_gravity_b[:, 2] + 1.0) * 0.5).clamp(0.0, 1.0)
    progress = height * (0.65 + 0.35 * upright)
    stable = (pelvis_z > 0.60) & (upright > 0.90)
    eligible = (env._a6_scene == 1) & env._a6_escaped & ~env._a6_invalid
    if not include_supine:
        eligible &= env._a6_direction == 1
    return eligible.float() * (2.5 * progress + 0.5 * stable.float())


def a6_reward(env, group: str, arm: str, preserve_pose: bool = False):
    _update(env)
    r = env.scene["robot"].data
    if group == "task":
        weighted = env._a6_task_values * r.joint_pos.new_tensor(TASK_WEIGHTS)
        if arm == "G+":
            # A vertical stand or bent-knee stage is not a valid guided-plate
            # escape. Keep the weak quiet/smoothness prior until XY clearance.
            guided_obstructed = (env._a6_scene == 1) & ~env._a6_escaped
            weighted = weighted.clone()
            weighted[:, :4] *= (~guided_obstructed).float()[:, None]
        positive = weighted.sum(-1)
        alpha = (
            torch.where((env._a6_scene > 0) & ~env._a6_escaped, 0.05, 1.0)
            if arm == "G+"
            else 1.0
        )
        return positive * alpha
    if group == "regu":
        soft = r.soft_joint_pos_limits
        outside = (soft[..., 0] - r.joint_pos).clamp_min(0) + (
            r.joint_pos - soft[..., 1]
        ).clamp_min(0)
        return (
            (env._a6_cost_values * r.joint_pos.new_tensor(COST_WEIGHTS)).sum(-1)
            - 0.10 * outside.sum(-1)
            - 0.03 * env._a6_escape_values[:, 5]
            - 0.20 * env._a6_escape_values[:, 6]
        )
    if group == "style":
        # Native HoST style is retained as the method-specific prior; its reward
        # already includes dt, so remove it before A6's reward manager applies dt.
        from .task import style_reward

        style = style_reward(env) / env.step_dt
        if arm == "G+":
            style *= torch.where((env._a6_scene > 0) & ~env._a6_escaped, 0.05, 1.0)
        return style
    if group == "target":
        e = env._a6_escape_values
        guided_scale = (env._a6_scene == 1).float()
        staged = (
            bool(
                getattr(env, "_a6_staged_escape_v1", False)
                or getattr(env, "_a6_staged_escape_v2", False)
                or getattr(env, "_a6_staged_escape_v3", False)
                or getattr(env, "_a6_staged_escape_v4", False)
                or getattr(env, "_a6_staged_escape_v5", False)
            )
            and arm == "G+"
        )
        leg_stage_weight = (
            env._a6_knee_penalty_weight
            if (
                getattr(env, "_a6_staged_escape_v2", False)
                or getattr(env, "_a6_staged_escape_v3", False)
            or getattr(env, "_a6_staged_escape_v4", False)
            or getattr(env, "_a6_staged_escape_v5", False)
            )
            and arm == "G+"
            else 1.0
        )
        staged_v5 = bool(getattr(env, "_a6_staged_escape_v5", False))
        lateral_weight = 0.50 if staged_v5 else 3.0 if staged else 1.0
        torso_weight = (
            10.0
            if staged_v5
            else 15.0
            if getattr(env, "_a6_staged_escape_v4", False) and arm == "G+"
            else 3.0
            if staged
            else 0.50
        )
        hand_push_weight = 3.00 if staged_v5 else 2.50 if staged else 1.50
        hand_drive_weight = 2.50 if staged_v5 else 2.00 if staged else 1.20
        lift_weight = 5.0 if staged_v5 else 0.45
        result = (
            0.60 * e[:, 2]
            + env._a6_arm_extraction_reward
            - 0.03 * e[:, 4]
            - 1.00 * e[:, 26]
            - (100.0 + 650.0 * guided_scale if arm == "G+" else 100.0)
            * env._a6_invalid.float()
            + (
                (0.45 + 0.20 * guided_scale) * e[:, 0]
                + (0.08 + 0.04 * guided_scale) * e[:, 1]
                + (0.10 + 0.05 * guided_scale) * e[:, 3]
                + lateral_weight * e[:, 7]
                - (0.15 + 1.35 * guided_scale) * e[:, 8]
                - (0.10 + 0.02 * guided_scale) * e[:, 9]
                - lift_weight * e[:, 10]
                + lateral_weight * e[:, 11]
                - 0.90 * leg_stage_weight * e[:, 12]
                + 0.50 * leg_stage_weight * e[:, 13]
                + 0.25 * leg_stage_weight * e[:, 14]
                + 0.12 * leg_stage_weight * e[:, 15]
                + 0.80 * leg_stage_weight * e[:, 18]
                - 0.35 * leg_stage_weight * e[:, 17]
                + torso_weight * e[:, 19]
                + 0.60 * e[:, 20]
                + 0.80 * leg_stage_weight * e[:, 21]
                + hand_push_weight * e[:, 22]
                + hand_drive_weight * e[:, 23]
                + 0.15 * e[:, 24]
                - 1.20 * leg_stage_weight * e[:, 25]
                if arm == "G+"
                else 0.0
            )
        )
        if getattr(env, "_a6_prone_recovery_v1", False) and arm == "G+":
            result += _guided_recovery_reward(
                env,
                include_supine=getattr(env, "_a6_bidirectional_recovery_v1", False),
            )
        return result - _standing_pose_cost(env) if preserve_pose else result
    raise ValueError(group)


def a6_invalid(env):
    _update(env)
    return env._a6_invalid


def a6_metric(env, key: str):
    _update(env)
    return getattr(env, f"_a6_{key}").float()
