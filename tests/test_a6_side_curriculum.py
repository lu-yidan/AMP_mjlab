"""The offset curriculum must wait for both side-lying directions."""

import pytest
import torch

from g1recovery_host.a6_curriculum import (
    SuccessGatedEdgeCurriculum, guided_side_target_advantage_scale,
)
from g1recovery_host.a6_reset import (
    planar_escape_axis, side_bridge_tiers, side_center_tiers,
    side_staged_offset_scale,
)
from g1recovery_host.a6_reward import lateral_travel, outward_hand_drive


def test_side_escape_axis_uses_horizontal_torso_up_instead_of_vertical_left():
    half = 2 ** -0.5
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [half, half, 0.0, 0.0]])
    direction = torch.tensor([0, 2])
    axis = planar_escape_axis(quat, direction)
    assert torch.allclose(axis, torch.tensor([[0.0, 1.0], [0.0, -1.0]]), atol=1e-6)


def test_side_label_with_upright_recovery_pose_uses_stable_left_axis():
    axis = planar_escape_axis(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([2])
    )
    assert torch.allclose(axis, torch.tensor([[0.0, 1.0]]))


def test_centered_side_rewards_either_exit_but_offset_side_uses_near_edge():
    displacement = torch.tensor([[0.0, 0.1], [0.0, -0.1]])
    axis = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    toward_near_edge = torch.tensor([False, True])
    assert torch.allclose(
        lateral_travel(displacement, axis, toward_near_edge),
        torch.tensor([0.1, 0.1]),
    )
    speed = outward_hand_drive(
        displacement, 2 * displacement, axis, torch.ones(2), toward_near_edge
    )
    assert torch.allclose(speed, torch.ones(2))


def test_side_gate_holds_easy_offset_until_both_sides_escape():
    curriculum = SuccessGatedEdgeCurriculum(start_offset_m=0.25)
    for _ in range(20):
        advanced, totals = curriculum.observe(
            (7, 3, 0), (7, 3, 0), (0, 10, 0), (7, 3, 0),
        )
    assert not advanced
    assert len(totals) == 4
    assert curriculum.offset_m == 0.25
    curriculum.history.clear()
    for _ in range(20):
        advanced, totals = curriculum.observe(
            (7, 3, 0), (7, 3, 0), (7, 3, 0), (7, 3, 0),
        )
    assert advanced
    assert len(totals) == 4
    assert curriculum.offset_m == 0.225


def test_resume_with_different_posture_gate_requires_restart():
    curriculum = SuccessGatedEdgeCurriculum(start_offset_m=0.25)
    curriculum.observe((7, 3, 0), (7, 3, 0))
    with pytest.raises(ValueError, match="posture count"):
        curriculum.observe((7, 3, 0), (7, 3, 0), (7, 3, 0), (7, 3, 0))


def test_bridge_holds_offset_until_both_half_tier_sides_work():
    curriculum = SuccessGatedEdgeCurriculum(start_offset_m=0.25)
    for _ in range(20):
        advanced, totals = curriculum.observe(
            (7, 3, 0), (7, 3, 0), (7, 3, 0), (7, 3, 0),
            (1, 8, 1), (3, 7, 0),
        )
    assert not advanced
    assert len(totals) == 6
    assert curriculum.offset_m == 0.25
    curriculum.history.clear()
    for _ in range(20):
        advanced, _ = curriculum.observe(
            (7, 3, 0), (7, 3, 0), (7, 3, 0), (7, 3, 0),
            (2, 7, 1), (3, 7, 0),
        )
    assert advanced
    assert curriculum.offset_m == 0.225


def test_bridge_remaps_only_centred_guided_side_tier():
    tiers = torch.tensor([3, 3, 3, 3, 2, 1])
    scenes = torch.tensor([1, 1, 1, 2, 1, 1])
    directions = torch.tensor([0, 1, 2, 3, 3, 2])
    assert side_bridge_tiers(tiers, scenes, directions, 0.25).tolist() == [
        3, 3, 2, 3, 2, 1,
    ]
    assert torch.equal(side_bridge_tiers(tiers, scenes, directions, 0.025), tiers)


def test_center_only_tiers_change_guided_sides_without_touching_other_scenes():
    tier = torch.tensor([0, 1, 2, 0, 1, 2, 3])
    scene = torch.tensor([1, 1, 1, 1, 2, 0, 1])
    direction = torch.tensor([0, 1, 2, 3, 2, 3, 3])
    assert side_center_tiers(tier, scene, direction).tolist() == [
        0, 1, 3, 3, 1, 2, 3,
    ]


def test_staged_bridge_uses_intermediate_side_offsets_only():
    tiers = torch.tensor([0, 1, 2, 3, 2, 3])
    scenes = torch.tensor([1, 1, 1, 1, 1, 2])
    directions = torch.tensor([2, 3, 2, 3, 0, 2])
    scales = side_staged_offset_scale(tiers, scenes, directions, 0.225)
    assert scales.tolist() == [1.0, 1.0, 0.75, 0.5, 0.5, 0.0]
    assert side_staged_offset_scale(tiers, scenes, directions, 0.025).tolist() == [
        1.0, 1.0, 0.5, 0.0, 0.5, 0.0,
    ]


def test_staged_gate_requires_both_intermediate_side_tiers():
    curriculum = SuccessGatedEdgeCurriculum(start_offset_m=0.225)
    good = ((7, 3, 0),) * 4 + ((4, 5, 1),) * 2
    for _ in range(20):
        advanced, totals = curriculum.observe(*good, (1, 7, 2), (2, 7, 1))
    assert not advanced
    assert len(totals) == 8
    curriculum.history.clear()
    for _ in range(20):
        advanced, _ = curriculum.observe(*good, (2, 7, 1), (2, 7, 1))
    assert advanced
    assert curriculum.offset_m == 0.2


def test_target_focus_applies_only_to_trapped_guided_side_tiers():
    scene = torch.tensor([1, 1, 1, 1, 2, 0, 1])
    direction = torch.tensor([2, 3, 2, 3, 2, 3, 0])
    escaped = torch.tensor([False, False, False, True, False, False, False])
    tier = torch.tensor([0, 2, 3, 3, 3, 3, 3])
    scale = guided_side_target_advantage_scale(scene, direction, escaped, tier)
    assert scale.tolist() == [1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0]
