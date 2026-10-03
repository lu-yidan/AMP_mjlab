"""Historical A6 obstacle-interaction rewards for the AMP adaptation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import (
    update_a6_plate_state,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

A6_NO_PROGRESS_GRACE_S = 0.50
A6_NO_PROGRESS_FULL_S = 1.50
A6_HEADWARD_UPPER_ARM_TARGET_COSINE = 0.60
A6_PRONE_DIRECTION_ID = 1


def headward_upper_arm_score_from_state(
    headward_cosine: torch.Tensor,
    eligible: torch.Tensor,
    *,
    target_cosine: float = A6_HEADWARD_UPPER_ARM_TARGET_COSINE,
) -> torch.Tensor:
    """Score whether both upper arms point from the shoulders toward the head."""

    if headward_cosine.shape[-1] != 2:
        raise ValueError("headward_cosine must contain left and right arms")
    if not 0.0 < target_cosine <= 1.0:
        raise ValueError("target_cosine must be in (0, 1]")
    per_arm = (headward_cosine / target_cosine).clamp(-1.0, 1.0)
    # The worse arm determines the result: one correct arm cannot hide the
    # other arm remaining over the chest/abdomen.
    both_arms = per_arm.amin(dim=-1)
    return eligible * both_arms


def no_progress_penalty_from_state(
    no_progress_time: torch.Tensor,
    eligible: torch.Tensor,
    *,
    grace_s: float = A6_NO_PROGRESS_GRACE_S,
    full_s: float = A6_NO_PROGRESS_FULL_S,
) -> torch.Tensor:
    """Ramp a penalty after an obstructed robot stops making real progress."""

    if grace_s < 0.0 or full_s <= grace_s:
        raise ValueError("no-progress timing must satisfy 0 <= grace_s < full_s")
    ramp = ((no_progress_time - grace_s) / (full_s - grace_s)).clamp(0.0, 1.0)
    return eligible * ramp


def path_progress_from_state(
    coverage_delta: torch.Tensor,
    clearance_delta: torch.Tensor,
    head_height: torch.Tensor,
    hand_support: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    """Reward new horizontal clearance only with low hand-supported motion."""

    return (
        eligible
        * (head_height <= 0.90)
        * hand_support
        * (
            (coverage_delta / 0.025).clamp(0.0, 1.0)
            + 0.5 * (clearance_delta / 0.020).clamp(0.0, 1.0)
        )
    )


def clearance_score_from_state(
    active: torch.Tensor,
    invalid: torch.Tensor,
    covered_count: torch.Tensor,
    initial_count: torch.Tensor,
    clearance: torch.Tensor,
) -> torch.Tensor:
    """Measure how much of the robot shadow has cleared the plate footprint."""

    coverage = (1.0 - covered_count / initial_count.clamp_min(1.0)).clamp(0.0, 1.0)
    clearance_score = (clearance / 0.040).clamp(0.0, 1.0)
    return active * ~invalid * (0.85 * coverage + 0.15 * clearance_score)


def force_excess_l2_from_state(
    active: torch.Tensor, force: torch.Tensor
) -> torch.Tensor:
    """Penalize plate force only above the historical 300 N soft threshold."""

    return active * ((force - 300.0).clamp_min(0.0) / 300.0).square()


def plate_geometry_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    update_a6_plate_state(env)
    active = env._a6_reset_scene > 0
    eligible = (
        active
        & env._a6_plate_ever_contact
        & ~env._a6_plate_escaped
        & ~env._a6_plate_invalid
    )
    return path_progress_from_state(
        env._a6_plate_path_coverage_delta,
        env._a6_plate_path_clearance_delta,
        env._a6_plate_head_height,
        env._a6_plate_path_hand_support,
        eligible,
    )


def plate_clearance(env: ManagerBasedRlEnv) -> torch.Tensor:
    update_a6_plate_state(env)
    return clearance_score_from_state(
        env._a6_reset_scene > 0,
        env._a6_plate_invalid,
        env._a6_plate_path_current_count,
        env._a6_plate_path_initial_count,
        env._a6_plate_path_current_clearance,
    )


def plate_completion(env: ManagerBasedRlEnv) -> torch.Tensor:
    update_a6_plate_state(env)
    return env._a6_plate_escaped.float()


def plate_force(env: ManagerBasedRlEnv) -> torch.Tensor:
    update_a6_plate_state(env)
    return force_excess_l2_from_state(env._a6_reset_scene > 0, env._a6_plate_force)


def plate_separation(env: ManagerBasedRlEnv) -> torch.Tensor:
    update_a6_plate_state(env)
    return env._a6_plate_separation_progress


def prone_lateral_progress_from_state(
    progress: torch.Tensor,
    scene: torch.Tensor,
    direction: torch.Tensor,
    ever_contact: torch.Tensor,
    escaped: torch.Tensor,
    invalid: torch.Tensor,
) -> torch.Tensor:
    """Gate lateral progress to trapped prone guided-plate environments."""

    eligible = (
        (scene == 1)
        & (direction == A6_PRONE_DIRECTION_ID)
        & ever_contact
        & ~escaped
        & ~invalid
    )
    return eligible * progress


def prone_lateral_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reward new left/right displacement for trapped prone guided-plate worlds."""

    update_a6_plate_state(env)
    return prone_lateral_progress_from_state(
        env._a6_plate_lateral_progress,
        env._a6_reset_scene,
        env._a6_reset_direction,
        env._a6_plate_ever_contact,
        env._a6_plate_escaped,
        env._a6_plate_invalid,
    )


def plate_no_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize prolonged obstruction without exposing plate state to the actor."""

    update_a6_plate_state(env)
    eligible = (
        (env._a6_reset_scene > 0)
        & env._a6_plate_ever_contact
        & ~env._a6_plate_escaped
        & ~env._a6_plate_invalid
    )
    return no_progress_penalty_from_state(
        env._a6_plate_no_progress_time,
        eligible,
    )


def trapped_shoulders_headward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Encourage both upper arms toward the head only while still obstructed."""

    update_a6_plate_state(env)
    eligible = (
        (env._a6_reset_scene > 0)
        & env._a6_plate_ever_contact
        & ~env._a6_plate_escaped
        & ~env._a6_plate_invalid
    )
    robot = env.scene["robot"]
    if not hasattr(env, "_a6_headward_arm_body_ids"):
        torso_id = robot.find_bodies("torso_link")[0][0]
        shoulder_ids = robot.find_bodies(
            ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
            preserve_order=True,
        )[0]
        elbow_ids = robot.find_bodies(
            ("left_elbow_link", "right_elbow_link"), preserve_order=True
        )[0]
        env._a6_headward_arm_body_ids = (torso_id, shoulder_ids, elbow_ids)
    torso_id, shoulder_ids, elbow_ids = env._a6_headward_arm_body_ids
    body_pos = robot.data.body_link_pos_w
    upper_arm_w = body_pos[:, elbow_ids] - body_pos[:, shoulder_ids]
    torso_quat = robot.data.body_link_quat_w[:, torso_id].unsqueeze(1).expand(-1, 2, -1)
    upper_arm_b = quat_apply_inverse(torso_quat, upper_arm_w)
    headward_cosine = upper_arm_b[..., 2] / upper_arm_b.norm(dim=-1).clamp_min(1.0e-6)
    return headward_upper_arm_score_from_state(headward_cosine, eligible)


__all__ = [
    "A6_HEADWARD_UPPER_ARM_TARGET_COSINE",
    "A6_PRONE_DIRECTION_ID",
    "clearance_score_from_state",
    "force_excess_l2_from_state",
    "headward_upper_arm_score_from_state",
    "no_progress_penalty_from_state",
    "path_progress_from_state",
    "plate_clearance",
    "plate_completion",
    "plate_force",
    "plate_geometry_progress",
    "plate_no_progress",
    "plate_separation",
    "prone_lateral_progress",
    "prone_lateral_progress_from_state",
    "trapped_shoulders_headward",
]
