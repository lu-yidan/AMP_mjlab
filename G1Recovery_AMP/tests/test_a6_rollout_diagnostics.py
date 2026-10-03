"""Unit tests for the standalone A6 checkpoint diagnostic calculations."""

import unittest

import torch

from scripts.check_a6_flat_rollout import (
    _advance_continuous_hold,
    _advance_post_motion_stall,
    _basic_stand_success_gate,
    _relaxed_invalid_reasons_from_state,
    _stable_standing_gates,
)


class TestA6RolloutDiagnostics(unittest.TestCase):
    def _gates(self, **overrides: torch.Tensor) -> dict[str, torch.Tensor]:
        values = {
            "head_height": torch.tensor([1.16]),
            "upright": torch.tensor([0.94]),
            "knee_position": torch.tensor([[0.2, -0.3]]),
            "base_linear_velocity": torch.tensor([[0.05, 0.0, 0.0]]),
            "base_angular_velocity": torch.tensor([[0.0, 0.0, 0.1]]),
            "joint_velocity": torch.zeros((1, 29)),
            "foot_linear_velocity": torch.zeros((1, 2, 3)),
            "minimum_foot_load": torch.tensor([30.0]),
            "stance_width": torch.tensor([0.25]),
        }
        values.update(overrides)
        return _stable_standing_gates(**values)

    def test_all_available_historical_gates_can_pass(self) -> None:
        gates = self._gates()
        self.assertTrue(bool(gates["stable"][0]))
        self.assertTrue(all(bool(value[0]) for value in gates.values()))

    def test_one_failed_gate_rejects_stable_standing(self) -> None:
        gates = self._gates(head_height=torch.tensor([1.14]))
        self.assertFalse(bool(gates["height"][0]))
        self.assertFalse(bool(gates["stable"][0]))

    def test_hold_must_be_continuous_and_longest_is_retained(self) -> None:
        current = torch.zeros(1)
        longest = torch.zeros(1)
        current, longest = _advance_continuous_hold(
            torch.tensor([True]), current, longest, 0.02
        )
        current, longest = _advance_continuous_hold(
            torch.tensor([True]), current, longest, 0.02
        )
        self.assertAlmostEqual(float(current[0]), 0.04, places=6)
        current, longest = _advance_continuous_hold(
            torch.tensor([False]), current, longest, 0.02
        )
        self.assertEqual(float(current[0]), 0.0)
        self.assertAlmostEqual(float(longest[0]), 0.04, places=6)

    def test_hold_rejects_non_positive_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_dt must be positive"):
            _advance_continuous_hold(
                torch.tensor([True]), torch.zeros(1), torch.zeros(1), 0.0
            )

    def test_basic_stand_does_not_require_quiet_stance_or_plate_escape(self) -> None:
        gates = self._gates(
            head_height=torch.tensor([1.16]),
            joint_velocity=torch.full((1, 29), 2.0),
            minimum_foot_load=torch.tensor([30.0]),
        )
        self.assertFalse(bool(gates["stable"][0]))
        qualified = _basic_stand_success_gate(
            head_height=torch.tensor([1.16]),
            upright=torch.tensor([0.90]),
            minimum_foot_load=torch.tensor([30.0]),
            done=torch.tensor([False]),
            active_trial=torch.tensor([True]),
        )
        self.assertTrue(bool(qualified[0]))
        current = torch.zeros(1)
        longest = torch.zeros(1)
        for _ in range(50):
            current, longest = _advance_continuous_hold(
                qualified, current, longest, 0.02
            )
        self.assertGreaterEqual(float(longest[0]), 0.99)

    def test_basic_stand_requires_all_three_conditions_and_active_trial(self) -> None:
        qualified = _basic_stand_success_gate(
            head_height=torch.tensor([1.15, 1.14, 1.15, 1.15, 1.15, 1.15]),
            upright=torch.tensor([0.90, 0.90, 0.89, 0.90, 0.90, 0.90]),
            minimum_foot_load=torch.tensor([20.1, 20.1, 20.1, 20.0, 20.1, 20.1]),
            done=torch.tensor([False, False, False, False, True, False]),
            active_trial=torch.tensor([True, True, True, True, True, False]),
        )
        self.assertEqual(qualified.tolist(), [True, False, False, False, False, False])

    def test_post_motion_stall_requires_prior_motion_and_continuous_quiet(self) -> None:
        ever_moved = torch.tensor([False])
        quiet_steps = torch.zeros(1, dtype=torch.int64)
        reported = torch.tensor([False])

        for moving, quiet, expected in (
            (False, True, False),
            (True, False, False),
            (False, True, False),
            (False, True, True),
        ):
            ever_moved, quiet_steps, reported, event = _advance_post_motion_stall(
                moving=torch.tensor([moving]),
                quiet=torch.tensor([quiet]),
                eligible=torch.tensor([True]),
                done=torch.tensor([False]),
                ever_moved=ever_moved,
                quiet_steps=quiet_steps,
                reported=reported,
                hold_steps=2,
            )
            self.assertEqual(bool(event[0]), expected)

    def test_post_motion_stall_fires_once_and_reset_rearms_it(self) -> None:
        ever_moved = torch.tensor([True])
        quiet_steps = torch.tensor([1])
        reported = torch.tensor([False])
        ever_moved, quiet_steps, reported, first = _advance_post_motion_stall(
            moving=torch.tensor([False]),
            quiet=torch.tensor([True]),
            eligible=torch.tensor([True]),
            done=torch.tensor([False]),
            ever_moved=ever_moved,
            quiet_steps=quiet_steps,
            reported=reported,
            hold_steps=2,
        )
        ever_moved, quiet_steps, reported, duplicate = _advance_post_motion_stall(
            moving=torch.tensor([False]),
            quiet=torch.tensor([True]),
            eligible=torch.tensor([True]),
            done=torch.tensor([False]),
            ever_moved=ever_moved,
            quiet_steps=quiet_steps,
            reported=reported,
            hold_steps=2,
        )
        ever_moved, quiet_steps, reported, reset_event = _advance_post_motion_stall(
            moving=torch.tensor([False]),
            quiet=torch.tensor([True]),
            eligible=torch.tensor([True]),
            done=torch.tensor([True]),
            ever_moved=ever_moved,
            quiet_steps=quiet_steps,
            reported=reported,
            hold_steps=2,
        )
        self.assertTrue(bool(first[0]))
        self.assertFalse(bool(duplicate[0]))
        self.assertFalse(bool(reset_event[0]))
        self.assertFalse(bool(ever_moved[0]))
        self.assertEqual(int(quiet_steps[0]), 0)
        self.assertFalse(bool(reported[0]))

    def test_post_motion_stall_rejects_non_positive_hold(self) -> None:
        with self.assertRaisesRegex(ValueError, "hold_steps must be positive"):
            _advance_post_motion_stall(
                moving=torch.tensor([False]),
                quiet=torch.tensor([True]),
                eligible=torch.tensor([True]),
                done=torch.tensor([False]),
                ever_moved=torch.tensor([False]),
                quiet_steps=torch.zeros(1, dtype=torch.int64),
                reported=torch.tensor([False]),
                hold_steps=0,
            )

    def test_relaxed_invalid_reasons_only_reject_extreme_states(self) -> None:
        force, penetration, no_contact = _relaxed_invalid_reasons_from_state(
            active=torch.tensor([True, True, True, True, False]),
            force=torch.tensor([4999.0, 5001.0, 0.0, 0.0, 9000.0]),
            penetration=torch.tensor([0.049, 0.0, 0.051, 0.0, 0.2]),
            ever_contact=torch.tensor([True, True, True, False, False]),
            episode_steps=torch.tensor([200, 200, 200, 101, 101]),
            max_force=5000.0,
            max_penetration=0.05,
            max_no_contact_steps=100,
        )
        self.assertEqual(force.tolist(), [False, True, False, False, False])
        self.assertEqual(
            penetration.tolist(), [False, False, True, False, False]
        )
        self.assertEqual(
            no_contact.tolist(), [False, False, False, True, False]
        )

    def test_relaxed_invalid_reasons_keep_boundary_values(self) -> None:
        force, penetration, no_contact = _relaxed_invalid_reasons_from_state(
            active=torch.tensor([True]),
            force=torch.tensor([5000.0]),
            penetration=torch.tensor([0.05]),
            ever_contact=torch.tensor([False]),
            episode_steps=torch.tensor([100]),
            max_force=5000.0,
            max_penetration=0.05,
            max_no_contact_steps=100,
        )
        self.assertFalse(bool(force[0]))
        self.assertFalse(bool(penetration[0]))
        self.assertFalse(bool(no_contact[0]))


if __name__ == "__main__":
    unittest.main()
