"""Tensor-level tests for historical A6 obstacle reward equations."""

import unittest

import torch

from g1recovery_amp.tasks.recovery.mdp.a6_rewards import (
    clearance_score_from_state,
    force_excess_l2_from_state,
    headward_upper_arm_score_from_state,
    no_progress_penalty_from_state,
    path_progress_from_state,
    prone_lateral_progress_from_state,
)


class TestA6Rewards(unittest.TestCase):
    def test_prone_lateral_reward_is_limited_to_trapped_guided_prone_worlds(
        self,
    ) -> None:
        reward = prone_lateral_progress_from_state(
            progress=torch.ones(6),
            scene=torch.tensor([1, 1, 2, 0, 1, 1]),
            direction=torch.tensor([1, 0, 1, 1, 1, 1]),
            ever_contact=torch.tensor([True, True, True, True, False, True]),
            escaped=torch.tensor([False, False, False, False, False, True]),
            invalid=torch.tensor([False, False, False, False, False, False]),
        )
        torch.testing.assert_close(
            reward, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

    def test_headward_arm_reward_requires_both_arms_and_trapped_gate(self) -> None:
        reward = headward_upper_arm_score_from_state(
            headward_cosine=torch.tensor(
                [
                    [0.60, 0.60],
                    [0.60, -0.30],
                    [-0.60, -0.60],
                    [0.60, 0.60],
                ]
            ),
            eligible=torch.tensor([True, True, True, False]),
        )

        torch.testing.assert_close(reward, torch.tensor([1.0, -0.5, -1.0, 0.0]))

    def test_headward_arm_reward_validates_shape_and_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "left and right"):
            headward_upper_arm_score_from_state(
                torch.zeros((1, 1)), torch.ones(1, dtype=torch.bool)
            )
        with self.assertRaisesRegex(ValueError, "target_cosine"):
            headward_upper_arm_score_from_state(
                torch.zeros((1, 2)),
                torch.ones(1, dtype=torch.bool),
                target_cosine=0.0,
            )

    def test_no_progress_penalty_has_grace_ramp_and_eligibility_gate(self) -> None:
        penalty = no_progress_penalty_from_state(
            no_progress_time=torch.tensor([0.49, 0.50, 1.00, 1.50, 2.00, 2.00]),
            eligible=torch.tensor([True, True, True, True, True, False]),
        )

        torch.testing.assert_close(
            penalty,
            torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 0.0]),
        )

    def test_no_progress_penalty_rejects_invalid_timing(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 <= grace_s < full_s"):
            no_progress_penalty_from_state(
                no_progress_time=torch.zeros(1),
                eligible=torch.ones(1, dtype=torch.bool),
                grace_s=1.0,
                full_s=1.0,
            )

    def test_path_progress_requires_every_gate(self) -> None:
        coverage = torch.full((5,), 0.025)
        clearance = torch.full((5,), 0.020)
        height = torch.tensor([0.90, 0.901, 0.90, 0.90, 0.90])
        support = torch.tensor([1.0, 1.0, 0.0, 0.5, 1.0])
        eligible = torch.tensor([True, True, True, True, False])

        reward = path_progress_from_state(
            coverage, clearance, height, support, eligible
        )

        torch.testing.assert_close(reward, torch.tensor([1.5, 0.0, 0.0, 0.75, 0.0]))

    def test_path_progress_is_bounded_per_control_step(self) -> None:
        reward = path_progress_from_state(
            coverage_delta=torch.tensor([0.25]),
            clearance_delta=torch.tensor([0.20]),
            head_height=torch.tensor([0.50]),
            hand_support=torch.tensor([1.0]),
            eligible=torch.tensor([True]),
        )
        torch.testing.assert_close(reward, torch.tensor([1.5]))

    def test_clearance_combines_coverage_and_minimum_gap(self) -> None:
        reward = clearance_score_from_state(
            active=torch.tensor([True, True, False]),
            invalid=torch.tensor([False, True, False]),
            covered_count=torch.tensor([2.0, 2.0, 0.0]),
            initial_count=torch.tensor([4.0, 4.0, 0.0]),
            clearance=torch.tensor([0.020, 0.020, 0.040]),
        )
        torch.testing.assert_close(reward, torch.tensor([0.5, 0.0, 0.0]))

    def test_force_penalty_starts_strictly_above_300_newtons(self) -> None:
        reward = force_excess_l2_from_state(
            active=torch.tensor([True, True, True, False]),
            force=torch.tensor([299.0, 300.0, 600.0, 600.0]),
        )
        torch.testing.assert_close(reward, torch.tensor([0.0, 0.0, 1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
