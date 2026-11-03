"""Training-only action labels from the successful legacy A6 controller."""

from __future__ import annotations

import torch

from .model import HOST29_JOINT_NAMES

HIP_PITCH_IDS = tuple(
    i for i, name in enumerate(HOST29_JOINT_NAMES) if "hip_pitch" in name
)
if len(HIP_PITCH_IDS) != 2:
    raise RuntimeError("Expected two G1 hip pitch joints")
SIDE_SCALED_IDS = tuple(
    i for i, name in enumerate(HOST29_JOINT_NAMES)
    if "hip_pitch" in name
    or (
        any(part in name for part in ("shoulder", "elbow", "wrist"))
        and "shoulder_roll" not in name
        and "shoulder_yaw" not in name
    )
)
SIDE_LATERAL_IDS = tuple(
    i for i, name in enumerate(HOST29_JOINT_NAMES)
    if any(part in name for part in (
        "hip_roll", "hip_yaw", "shoulder_roll", "shoulder_yaw",
    ))
)
if len(SIDE_LATERAL_IDS) != 8:
    raise RuntimeError("Expected eight lateral hip/shoulder axes")

REQUIRED_TEACHER_FEATURES = (
    "_staged_escape_v5",
    "_safe_actions_v1",
    "_lateral_actions_v1",
    "_prone_full_arms_v1",
    "_supine_full_arms_v1",
)


def validate_legacy_teacher(state: dict, physics_revision: str, reset_variant: str) -> None:
    """Accept only the 612D G+ teacher whose action scaling is known exactly."""
    variant = state.get("reward_variant", "")
    if (
        state.get("task") != "a6"
        or state.get("arm") != "G+"
        or state.get("model_source") != "unitree_g1_29"
        or state.get("physics_revision") != physics_revision
        or state.get("a6_reset_variant") != reset_variant
        or any(feature not in variant for feature in REQUIRED_TEACHER_FEATURES)
        or tuple(state["policy"]["actor.0.weight"].shape) != (512, 612)
        or tuple(state["policy"]["actor.6.weight"].shape) != (29, 128)
    ):
        raise ValueError("distillation requires a matching 612D staged-v5 G+ safe-action teacher")


def effective_guided_teacher_action(raw_action: torch.Tensor) -> torch.Tensor:
    """Convert the legacy actor's output to its executed action on prone/supine boards.

    The validated teacher restores full arm and lateral hip authority in both
    postures. Its safe-action layer only scales left/right hip pitch by 0.25.
    The old plate-lift guard is excluded: an independent ablation showed the
    teacher still escapes when that guard is disabled.
    """
    if raw_action.ndim != 2 or raw_action.shape[1] != len(HOST29_JOINT_NAMES):
        raise ValueError("teacher action must have one value for each of 29 joints")
    target = raw_action.clone()
    target[:, HIP_PITCH_IDS] *= 0.25
    return target


def effective_guided_side_teacher_action(
    raw_action: torch.Tensor, scale: float = 0.25
) -> torch.Tensor:
    """Bake the legacy side-lying safe-action scale into proprio actor labels.

    On guided sides the legacy teacher retains full lateral shoulder/hip
    authority, while hip pitch and nonlateral arm joints run at 0.25 scale.
    The returned 29-D vector needs no scene-dependent runtime action layer.
    """
    if raw_action.ndim != 2 or raw_action.shape[1] != len(HOST29_JOINT_NAMES):
        raise ValueError("teacher action must have one value for each of 29 joints")
    if not 0 < scale <= 1:
        raise ValueError("side-safe joint scale must be in (0, 1]")
    target = raw_action.clone()
    target[:, SIDE_SCALED_IDS] *= scale
    return target


def deployment_matched_guided_side_teacher_actions(
    legacy_effective_action: torch.Tensor, student_runtime_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the v33 raw label and its physical v49 action target.

    The v33 label already contains the legacy 0.25 safe-action scale. The
    runtime side controller scales it once more; the teacher-forced worlds
    must execute that same second scale rather than undoing it in the label.
    """
    return (
        legacy_effective_action,
        effective_guided_side_teacher_action(
            legacy_effective_action, scale=student_runtime_scale,
        ),
    )


def apply_side_lateral_gain(
    actions: torch.Tensor, selected: torch.Tensor, gain: float,
) -> torch.Tensor:
    """Increase only side escape axes for a proprio-gated side action."""
    if actions.ndim != 2 or actions.shape[1] != len(HOST29_JOINT_NAMES):
        raise ValueError("side lateral gain requires 29 joint actions")
    if selected.shape != (actions.shape[0],) or selected.dtype != torch.bool:
        raise ValueError("side lateral selection must be one Boolean per world")
    if not 1.0 <= gain <= 1.5:
        raise ValueError("side lateral gain must lie in [1, 1.5]")
    if gain == 1.0 or not selected.any():
        return actions
    boosted = actions[selected].clone()
    boosted[:, SIDE_LATERAL_IDS] *= gain
    result = actions.clone()
    result[selected] = boosted
    return result


def legacy_teacher_observation(critic_observation: torch.Tensor) -> torch.Tensor:
    """Remove the new torque history before calling the 612D legacy teacher."""
    if critic_observation.ndim != 2:
        raise ValueError("critic observation must be a batch")
    if critic_observation.shape[1] == 612:
        return critic_observation
    if critic_observation.shape[1] == 786:
        return torch.cat(
            (critic_observation[:, :564], critic_observation[:, 738:786]), dim=-1
        )
    raise ValueError("teacher requires a 612D or 786D critic observation")


def guided_side_teacher_force_mask(
    scene: torch.Tensor,
    direction: torch.Tensor,
    escaped: torch.Tensor,
    invalid: torch.Tensor,
) -> torch.Tensor:
    """Select obstructed guided-side worlds for training-only teacher control."""
    return (scene == 1) & (direction >= 2) & ~escaped & ~invalid


def progress_weighted_side_imitation_mask(
    imitation_mask: torch.Tensor,
    guided_side_before_step: torch.Tensor,
    escape_features: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """Emphasize teacher actions followed by safe torso and hand progress.

    The simulator-only escape features shape training labels. They are never
    part of the deployed actor observation or action path.
    """
    if (imitation_mask.ndim != 1 or guided_side_before_step.shape != imitation_mask.shape
            or done.shape != imitation_mask.shape
            or escape_features.shape != (len(imitation_mask), 27)):
        raise ValueError("side imitation weights require aligned rollout features")
    progress = (
        escape_features[:, 19] + 0.3 * escape_features[:, 22]
        + 0.3 * escape_features[:, 23]
    ).clamp(0.0, 1.0)
    safe = 1.0 - 0.8 * escape_features[:, 10].clamp(0.0, 1.0)
    side_weight = (0.1 + 4.0 * progress) * safe
    weighted = imitation_mask * side_weight * (~done).float()
    return torch.where(guided_side_before_step, weighted, imitation_mask)


def frozen_flat_teacher_mask(
    scene: torch.Tensor,
    direction: torch.Tensor,
    escaped: torch.Tensor,
    stand_ready: torch.Tensor,
    active: torch.Tensor,
    *,
    prone_recovery: bool,
    bidirectional_recovery: bool,
    supine_postescape_teacher: bool,
) -> torch.Tensor:
    """Choose training-only flat-expert labels without a simulator switch at runtime.

    The legacy bidirectional branch withheld the flat expert from guided
    supine recovery. A separate switch restores only that missing imitation;
    guided prone remains free to learn its low knee-fold transition with PPO.
    """
    selected = (scene == 0) | escaped
    if prone_recovery:
        recovering_guided = (scene == 1) & escaped & ~stand_ready
        withheld_direction = direction == 1
        if bidirectional_recovery and not supine_postescape_teacher:
            withheld_direction |= direction == 0
        selected &= ~(recovering_guided & withheld_direction)
    return active * selected.float()
