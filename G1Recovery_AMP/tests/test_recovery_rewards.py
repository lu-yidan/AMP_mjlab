"""Tests for the source-compatible G1 recovery task rewards."""

import math
import unittest

import torch

from g1recovery_amp.tasks.recovery.mdp.rewards import (
    RECOVERY_LOW_HEIGHT_FLOOR,
    RECOVERY_STANDUP_HEIGHT,
    RECOVERY_TARGET_BASE_HEIGHT,
    ang_vel_xy_from_state,
    lin_vel_xy_from_state,
    low_height_progress_from_state,
    target_base_height_from_state,
    target_joint_deviation_l2_from_state,
    target_orientation_from_state,
)


class TestRecoveryRewards(unittest.TestCase):
    def test_low_height_progress_rises_before_standing_gate(self) -> None:
        height = torch.tensor([0.05, 0.10, 0.375, 0.65, 0.651, 0.75])
        reward = low_height_progress_from_state(
            height, RECOVERY_LOW_HEIGHT_FLOOR, RECOVERY_STANDUP_HEIGHT
        )
        torch.testing.assert_close(reward, torch.tensor([0.0, 0.0, 0.5, 1.0, 0.0, 0.0]))

        # The configured 0.5 weight must not make stopping below the gate
        # preferable to crossing into the existing standing-height reward.
        just_above_gate = target_base_height_from_state(
            height[4:5], RECOVERY_TARGET_BASE_HEIGHT, RECOVERY_STANDUP_HEIGHT
        )
        self.assertGreater(5.0 * float(just_above_gate[0]), 0.5)

    def test_low_height_progress_requires_ordered_bounds(self) -> None:
        with self.assertRaises(ValueError):
            low_height_progress_from_state(torch.tensor([0.3]), 0.65, 0.65)

    def test_final_standing_rewards_are_off_at_or_below_gate(self) -> None:
        height = torch.tensor([0.60, RECOVERY_STANDUP_HEIGHT])
        zeros_3d = torch.zeros((2, 3))

        torch.testing.assert_close(
            ang_vel_xy_from_state(zeros_3d, height, RECOVERY_STANDUP_HEIGHT),
            torch.zeros(2),
        )
        torch.testing.assert_close(
            lin_vel_xy_from_state(zeros_3d, height, RECOVERY_STANDUP_HEIGHT),
            torch.zeros(2),
        )
        torch.testing.assert_close(
            target_orientation_from_state(zeros_3d, height, RECOVERY_STANDUP_HEIGHT),
            torch.zeros(2),
        )

    def test_ideal_standing_state_gets_unit_raw_rewards(self) -> None:
        height = torch.tensor([RECOVERY_TARGET_BASE_HEIGHT])
        zero_velocity = torch.zeros((1, 3))
        projected_gravity = torch.tensor([[0.0, 0.0, -1.0]])

        torch.testing.assert_close(
            ang_vel_xy_from_state(zero_velocity, height, RECOVERY_STANDUP_HEIGHT),
            torch.ones(1),
        )
        torch.testing.assert_close(
            lin_vel_xy_from_state(zero_velocity, height, RECOVERY_STANDUP_HEIGHT),
            torch.ones(1),
        )
        torch.testing.assert_close(
            target_orientation_from_state(
                projected_gravity, height, RECOVERY_STANDUP_HEIGHT
            ),
            torch.ones(1),
        )
        torch.testing.assert_close(
            target_base_height_from_state(
                height,
                RECOVERY_TARGET_BASE_HEIGHT,
                RECOVERY_STANDUP_HEIGHT,
            ),
            torch.ones(1),
        )

    def test_velocity_and_height_errors_match_source_exponents(self) -> None:
        height = torch.tensor([0.70])
        unit_x_velocity = torch.tensor([[1.0, 0.0, 99.0]])

        angular_reward = ang_vel_xy_from_state(
            unit_x_velocity, height, RECOVERY_STANDUP_HEIGHT
        )
        linear_reward = lin_vel_xy_from_state(
            unit_x_velocity, height, RECOVERY_STANDUP_HEIGHT
        )
        height_reward = target_base_height_from_state(
            height,
            RECOVERY_TARGET_BASE_HEIGHT,
            RECOVERY_STANDUP_HEIGHT,
        )

        torch.testing.assert_close(angular_reward, torch.tensor([math.exp(-2.0)]))
        torch.testing.assert_close(linear_reward, torch.tensor([math.exp(-5.0)]))
        torch.testing.assert_close(height_reward, torch.tensor([math.exp(-1.0)]))

    def test_joint_deviation_sums_squared_angle_errors(self) -> None:
        joint_pos = torch.tensor([[1.0, -2.0, 3.0], [1.0, 2.0, 3.0]])
        default_joint_pos = torch.tensor([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
        height = torch.tensor([0.70, 0.60])

        deviation = target_joint_deviation_l2_from_state(
            joint_pos,
            default_joint_pos,
            height,
            RECOVERY_STANDUP_HEIGHT,
        )

        torch.testing.assert_close(deviation, torch.tensor([9.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
