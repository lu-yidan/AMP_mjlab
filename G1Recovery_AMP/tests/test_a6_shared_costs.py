"""Tests for frozen A6 phase transitions and control-rate costs."""

import unittest

import torch

from g1recovery_amp.tasks.recovery.mdp.a6_shared_costs import (
    A6_ACTION_ACCEL_PHASE_FACTORS,
    A6_ACTION_RATE_PHASE_FACTORS,
    A6_EFFORT_MEMORY_TIME_CONSTANT,
    advance_phase_from_state,
    initial_phase_from_state,
    joint_stall_cost,
    linear_gate,
    phased_action_costs,
    physics_substep_cost_components,
    quiet_foot_motion_cost,
    sustained_effort_cost,
    update_joint_stall_substep,
    update_sustained_effort,
)


class TestA6SharedCosts(unittest.TestCase):
    def test_reset_phase_comes_from_current_pose(self) -> None:
        phase = initial_phase_from_state(
            torch.tensor([0.40, 0.55, 0.78, 1.08, 1.20]),
            torch.tensor([0.90, 0.55, 0.72, 0.85, 0.30]),
        )
        self.assertEqual(phase.tolist(), [0, 1, 2, 3, 0])

    def test_phase_zero_needs_ten_consecutive_ready_steps(self) -> None:
        phase = torch.tensor([0])
        hold = torch.tensor([0])
        kwargs = {
            "head_height": torch.tensor([0.60]),
            "upright": torch.tensor([0.60]),
            "knee_pos": torch.tensor([[0.90, 0.85]]),
            "head_vertical_velocity": torch.tensor([0.10]),
            "minimum_foot_load": torch.tensor([0.0]),
        }
        for expected_hold in range(1, 10):
            phase, hold = advance_phase_from_state(phase=phase, hold=hold, **kwargs)
            self.assertEqual(int(phase[0]), 0)
            self.assertEqual(int(hold[0]), expected_hold)
        phase, hold = advance_phase_from_state(phase=phase, hold=hold, **kwargs)
        self.assertEqual(int(phase[0]), 1)
        self.assertEqual(int(hold[0]), 0)

    def test_phase_two_requires_both_feet_loaded_for_25_steps(self) -> None:
        common = {
            "head_height": torch.tensor([1.10]),
            "upright": torch.tensor([0.90]),
            "knee_pos": torch.tensor([[0.30, 0.40]]),
            "head_vertical_velocity": torch.tensor([0.05]),
        }
        phase, hold = advance_phase_from_state(
            phase=torch.tensor([2]),
            hold=torch.tensor([24]),
            minimum_foot_load=torch.tensor([20.0]),
            **common,
        )
        self.assertEqual((int(phase[0]), int(hold[0])), (2, 0))
        phase, hold = advance_phase_from_state(
            phase=torch.tensor([2]),
            hold=torch.tensor([24]),
            minimum_foot_load=torch.tensor([20.1]),
            **common,
        )
        self.assertEqual((int(phase[0]), int(hold[0])), (3, 0))

    def test_fallen_pose_returns_to_phase_zero(self) -> None:
        phase, hold = advance_phase_from_state(
            phase=torch.tensor([3]),
            hold=torch.tensor([7]),
            head_height=torch.tensor([0.64]),
            upright=torch.tensor([0.44]),
            knee_pos=torch.tensor([[0.0, 0.0]]),
            head_vertical_velocity=torch.tensor([0.0]),
            minimum_foot_load=torch.tensor([0.0]),
        )
        self.assertEqual((int(phase[0]), int(hold[0])), (0, 0))

    def test_action_costs_use_phase_factors_and_fresh_mask(self) -> None:
        action = torch.ones((5, 2))
        previous = torch.zeros_like(action)
        previous_previous = torch.zeros_like(action)
        phase = torch.tensor([0, 1, 2, 3, 3])
        fresh = torch.tensor([False, False, False, False, True])
        rate, acceleration = phased_action_costs(
            action, previous, previous_previous, phase, fresh
        )
        torch.testing.assert_close(
            rate,
            torch.tensor([2.0 * x for x in (*A6_ACTION_RATE_PHASE_FACTORS, 0.0)]),
        )
        torch.testing.assert_close(
            acceleration,
            torch.tensor([2.0 * x for x in (*A6_ACTION_ACCEL_PHASE_FACTORS, 0.0)]),
        )

    def test_sustained_effort_uses_half_second_exponential_memory(self) -> None:
        previous = torch.zeros((1, 3))
        torque = torch.tensor([[10.0, 20.0, 30.0]])
        limits = torque.clone()
        updated = update_sustained_effort(previous, torque, limits, 0.002)
        expected_alpha = 1.0 - torch.exp(
            torch.tensor(-0.002 / A6_EFFORT_MEMORY_TIME_CONSTANT)
        )
        torch.testing.assert_close(updated, torch.full_like(updated, expected_alpha))
        torch.testing.assert_close(
            sustained_effort_cost(torch.ones((1, 3))), torch.ones(1)
        )

    def test_linear_gate_and_quiet_foot_motion_boundaries(self) -> None:
        torch.testing.assert_close(
            linear_gate(torch.tensor([-1.0, 1.0, 2.0, 3.0]), 1.0, 3.0),
            torch.tensor([0.0, 0.0, 0.5, 1.0]),
        )
        foot_velocity = torch.zeros((3, 2, 3))
        foot_velocity[0:2, :, 0] = 0.1
        foot_velocity[2, :, 0] = 1.0
        value = quiet_foot_motion_cost(
            torch.tensor([0.84, 1.15, 1.15]),
            torch.tensor([1.0, 0.69, 0.93]),
            foot_velocity,
        )
        torch.testing.assert_close(value, torch.tensor([0.0, 0.0, 10.0]))

    def test_joint_stall_requires_sustained_high_load_and_low_speed(self) -> None:
        reset_timer, reset_evidence = update_joint_stall_substep(
            timer=torch.tensor([[0.8]]),
            torque=torch.tensor([[7.0]]),
            effort_limit=torch.tensor([[10.0]]),
            joint_velocity=torch.tensor([[0.0]]),
            physics_dt=0.002,
        )
        torch.testing.assert_close(reset_timer, torch.zeros_like(reset_timer))
        torch.testing.assert_close(reset_evidence, torch.zeros_like(reset_evidence))
        full_timer, full_evidence = update_joint_stall_substep(
            timer=torch.tensor([[0.998]]),
            torque=torch.tensor([[9.5]]),
            effort_limit=torch.tensor([[10.0]]),
            joint_velocity=torch.tensor([[0.0]]),
            physics_dt=0.002,
        )
        torch.testing.assert_close(full_timer, torch.ones_like(full_timer))
        torch.testing.assert_close(full_evidence, torch.ones_like(full_evidence))

    def test_joint_stall_uses_worst_joint_not_whole_body_average(self) -> None:
        value = joint_stall_cost(
            averaged_evidence=torch.tensor([[1.0, 0.2, 0.8]]),
            joint_position_span=torch.tensor([[0.0, 0.2, 0.05]]),
        )
        torch.testing.assert_close(value, torch.tensor([1.0]))

    def test_physics_substep_components_match_hand_calculation(self) -> None:
        values = physics_substep_cost_components(
            joint_acceleration=torch.ones((1, 3)),
            torque=torch.full((1, 3), 10.0),
            joint_velocity=torch.full((1, 3), 8.0),
            head_vertical_velocity=torch.tensor([0.30]),
            phase=torch.tensor([3]),
            effort_mean_square=torch.ones((1, 3)),
        )
        torch.testing.assert_close(
            values,
            torch.tensor(
                [
                    [
                        3.0,
                        300.0,
                        3.0 * (8.0 - 3.5) ** 2,
                        (80.0 - 75.0) ** 2,
                        (0.30 - 0.20) ** 2,
                        1.0,
                    ]
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
