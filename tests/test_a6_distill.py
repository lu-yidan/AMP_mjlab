"""Verify that training labels match the legacy controller's executed actions."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_distill import (
    SIDE_SCALED_IDS,
    SIDE_LATERAL_IDS,
    apply_side_lateral_gain,
    deployment_matched_guided_side_teacher_actions,
    effective_guided_teacher_action,
    effective_guided_side_teacher_action,
    frozen_flat_teacher_mask,
    guided_side_teacher_force_mask,
    legacy_teacher_observation,
    progress_weighted_side_imitation_mask,
)
from g1recovery_host.a6_task import host_a6_env_cfg


def test_side_rollin_teacher_executes_the_same_double_scale_as_v49():
    raw_teacher = torch.ones(2, 29)
    legacy_effective = effective_guided_side_teacher_action(
        raw_teacher, scale=0.25,
    )
    student_label, forced_action = deployment_matched_guided_side_teacher_actions(
        legacy_effective, student_runtime_scale=0.5,
    )
    torch.testing.assert_close(student_label, legacy_effective)
    torch.testing.assert_close(forced_action[:, SIDE_SCALED_IDS],
                               torch.full((2, len(SIDE_SCALED_IDS)), 0.125))
    other_ids = [i for i in range(29) if i not in SIDE_SCALED_IDS]
    torch.testing.assert_close(forced_action[:, other_ids], raw_teacher[:, other_ids])


def test_side_lateral_gain_changes_only_gated_escape_axes():
    actions = torch.arange(58, dtype=torch.float32).reshape(2, 29) / 10
    out = apply_side_lateral_gain(actions, torch.tensor([True, False]), 1.1)
    expected = actions.clone()
    expected[0, SIDE_LATERAL_IDS] *= 1.1
    torch.testing.assert_close(out, expected)
    assert len(SIDE_LATERAL_IDS) == 8


def test_guided_teacher_label_matches_legacy_action_layer():
    cfg = host_a6_env_cfg(
        num_envs=32,
        edge_curriculum=True,
        plate_observation=True,
        focused_guided=True,
        reset_mix="diverse_v3",
        safe_actions=True,
        prone_full_arms=True,
        supine_full_arms=True,
        aligned_edge=True,
        lateral_actions=True,
        footprint_edge=True,
    )
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset()
        term = env.action_manager.get_term("joint_pos")
        term.real_steps.fill_(31)
        actions = torch.linspace(-0.7, 0.7, 29).expand(32, -1).clone()
        term.process_actions(actions)
        rows = (env._a6_scene == 1) & (env._a6_direction < 2)
        expected = effective_guided_teacher_action(actions[rows])
        torch.testing.assert_close(term.raw_action[rows], expected)
        side = (env._a6_scene == 1) & (env._a6_direction >= 2)
        side_expected = effective_guided_side_teacher_action(actions[side])
        torch.testing.assert_close(term.raw_action[side], side_expected)
    finally:
        env.close()


def test_teacher_receives_original_host_and_plate_columns_after_torque_expansion():
    critic = torch.arange(786, dtype=torch.float32).unsqueeze(0)
    teacher = legacy_teacher_observation(critic)
    assert teacher.shape == (1, 612)
    torch.testing.assert_close(teacher[:, :564], critic[:, :564])
    torch.testing.assert_close(teacher[:, 564:], critic[:, 738:])


def test_teacher_forcing_selects_only_obstructed_valid_guided_sides():
    scene = torch.tensor([1, 1, 1, 1, 1, 0, 2])
    direction = torch.tensor([2, 3, 2, 3, 1, 2, 3])
    escaped = torch.tensor([False, False, True, False, False, False, False])
    invalid = torch.tensor([False, False, False, True, False, False, False])
    assert guided_side_teacher_force_mask(
        scene, direction, escaped, invalid,
    ).tolist() == [True, True, False, False, False, False, False]


def test_progress_weighted_side_labels_prefer_safe_motion():
    mask = torch.ones(4)
    side = torch.tensor([True, True, True, False])
    features = torch.zeros(4, 27)
    features[1, 19] = 1.0
    features[2, 19] = 1.0
    features[2, 10] = 1.0
    done = torch.tensor([False, False, False, True])
    weighted = progress_weighted_side_imitation_mask(mask, side, features, done)
    torch.testing.assert_close(weighted, torch.tensor([0.1, 4.1, 0.82, 1.0]))


def test_supine_recovery_teacher_restores_only_missing_guided_supine_labels():
    scene = torch.tensor([0, 1, 1, 2, 1])
    direction = torch.tensor([0, 1, 0, 0, 0])
    escaped = torch.tensor([False, True, True, True, False])
    ready = torch.zeros(5, dtype=torch.bool)
    active = torch.ones(5)
    common = {"prone_recovery": True, "bidirectional_recovery": True}
    old = frozen_flat_teacher_mask(
        scene, direction, escaped, ready, active,
        supine_postescape_teacher=False, **common,
    )
    new = frozen_flat_teacher_mask(
        scene, direction, escaped, ready, active,
        supine_postescape_teacher=True, **common,
    )
    assert old.tolist() == [1, 0, 0, 1, 0]
    assert new.tolist() == [1, 0, 1, 1, 0]
