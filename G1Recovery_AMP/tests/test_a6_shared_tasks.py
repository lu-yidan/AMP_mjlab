"""Tests for the nine frozen A6 positive recovery task components."""

import unittest

import torch

from g1recovery_amp.tasks.recovery.mdp.a6_shared_tasks import (
    a6_task_components_from_state,
    a6_task_multiplier_from_state,
    smooth_gate,
)


def _components(
    *,
    phase: torch.Tensor,
    head_height: torch.Tensor,
    head_vertical_velocity: torch.Tensor,
    upright: torch.Tensor,
    knee_position: torch.Tensor,
    foot_linear_velocity: torch.Tensor | None = None,
    base_linear_velocity: torch.Tensor | None = None,
    base_angular_velocity: torch.Tensor | None = None,
    joint_velocity: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    previous_action: torch.Tensor | None = None,
    fresh: torch.Tensor | None = None,
) -> torch.Tensor:
    count = phase.shape[0]
    return a6_task_components_from_state(
        phase=phase,
        head_height=head_height,
        head_vertical_velocity=head_vertical_velocity,
        upright=upright,
        knee_position=knee_position,
        foot_linear_velocity=(
            torch.zeros((count, 2, 3))
            if foot_linear_velocity is None
            else foot_linear_velocity
        ),
        base_linear_velocity=(
            torch.zeros((count, 3))
            if base_linear_velocity is None
            else base_linear_velocity
        ),
        base_angular_velocity=(
            torch.zeros((count, 3))
            if base_angular_velocity is None
            else base_angular_velocity
        ),
        joint_velocity=(
            torch.zeros((count, 29)) if joint_velocity is None else joint_velocity
        ),
        action=torch.zeros((count, 29)) if action is None else action,
        previous_action=(
            torch.zeros((count, 29)) if previous_action is None else previous_action
        ),
        fresh=torch.zeros(count, dtype=torch.bool) if fresh is None else fresh,
    )


class TestA6SharedTasks(unittest.TestCase):
    def test_obstructed_scale_only_applies_before_plate_escape(self) -> None:
        multiplier = a6_task_multiplier_from_state(
            active_plate=torch.tensor([False, True, True]),
            escaped=torch.tensor([False, False, True]),
            obstructed_scale=0.05,
        )
        torch.testing.assert_close(multiplier, torch.tensor([1.0, 0.05, 1.0]))

    def test_obstructed_scale_rejects_values_outside_unit_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be in"):
            a6_task_multiplier_from_state(
                active_plate=torch.tensor([True]),
                escaped=torch.tensor([False]),
                obstructed_scale=1.1,
            )

    def test_smooth_gate_has_frozen_endpoints_and_midpoint(self) -> None:
        value = smooth_gate(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]), 1.0, 3.0)
        torch.testing.assert_close(value, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))

    def test_each_phase_pose_is_one_at_its_target(self) -> None:
        values = _components(
            phase=torch.tensor([0, 1, 2, 3]),
            head_height=torch.tensor([0.62, 0.86, 1.15, 1.15]),
            head_vertical_velocity=torch.zeros(4),
            upright=torch.tensor([0.60, 0.76, 0.93, 0.93]),
            knee_position=torch.tensor(
                [[1.0, 1.0], [0.8, 0.8], [0.3, 0.3], [0.3, 0.3]]
            ),
        )
        torch.testing.assert_close(values[:, 0], torch.ones(4))
        torch.testing.assert_close(values[:, 1], torch.ones(4))

    def test_head_velocity_tracks_upward_target_while_below_stage_height(self) -> None:
        values = _components(
            phase=torch.tensor([0, 1, 2]),
            head_height=torch.tensor([0.42, 0.66, 0.95]),
            head_vertical_velocity=torch.tensor([0.10, 0.15, 0.20]),
            upright=torch.ones(3),
            knee_position=torch.tensor([[1.0, 1.0], [0.8, 0.8], [0.3, 0.3]]),
        )
        torch.testing.assert_close(values[:, 1], torch.ones(3))

    def test_foot_and_base_quietness_only_turn_on_near_standing(self) -> None:
        foot_velocity = torch.zeros((2, 2, 3))
        foot_velocity[:, :, 0] = 0.1
        base_velocity = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        values = _components(
            phase=torch.tensor([0, 3]),
            head_height=torch.tensor([0.50, 1.15]),
            head_vertical_velocity=torch.zeros(2),
            upright=torch.tensor([0.30, 0.93]),
            knee_position=torch.tensor([[1.0, 1.0], [0.3, 0.3]]),
            foot_linear_velocity=foot_velocity,
            base_linear_velocity=base_velocity,
        )
        torch.testing.assert_close(values[0, 4:6], torch.ones(2))
        torch.testing.assert_close(values[1, 4], torch.exp(torch.tensor(-0.2)))
        torch.testing.assert_close(values[1, 5], torch.exp(torch.tensor(-2.0)))

    def test_fresh_episode_masks_action_delta(self) -> None:
        values = _components(
            phase=torch.tensor([0, 0]),
            head_height=torch.full((2,), 0.5),
            head_vertical_velocity=torch.zeros(2),
            upright=torch.full((2,), 0.3),
            knee_position=torch.ones((2, 2)),
            action=torch.ones((2, 29)),
            fresh=torch.tensor([False, True]),
        )
        self.assertLess(float(values[0, 8]), 0.03)
        self.assertEqual(float(values[1, 8]), 1.0)


if __name__ == "__main__":
    unittest.main()
