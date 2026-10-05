"""Unit tests for the A6 plate-state transition rules."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import (
    A6_CLEAR_HOLD_STEPS,
    A6_MAX_NO_CONTACT_STEPS,
    A6_MAX_PLATE_FORCE,
    A6_MAX_PLATE_PENETRATION,
    A6_MIN_PLANAR_CLEARANCE,
    advance_no_progress_time,
    initialize_a6_plate_state,
    lateral_progress_from_state,
    path_footprint,
    plate_invalid_reasons,
    reset_a6_plate_state,
    sustained_force_invalid_from_state,
    update_plate_flags,
)


def _step(
    *,
    active: bool = True,
    contact: bool = False,
    force: float = 0.0,
    depth: float = 0.0,
    clearance: float = 0.0,
    ever_contact: bool = False,
    clear_hold: int = 0,
    episode_steps: int = 0,
    min_planar_clearance: float = A6_MIN_PLANAR_CLEARANCE,
    clear_hold_steps: int = A6_CLEAR_HOLD_STEPS,
) -> tuple[bool, int, bool, bool]:
    result = update_plate_flags(
        active=torch.tensor([active]),
        contact=torch.tensor([contact]),
        force=torch.tensor([force]),
        depth=torch.tensor([depth]),
        clearance=torch.tensor([clearance]),
        ever_contact=torch.tensor([ever_contact]),
        clear_hold=torch.tensor([clear_hold]),
        episode_steps=torch.tensor([episode_steps]),
        min_planar_clearance=min_planar_clearance,
        clear_hold_steps=clear_hold_steps,
    )
    return bool(result[0][0]), int(result[1][0]), bool(result[2][0]), bool(result[3][0])


class TestA6PlateState(unittest.TestCase):
    def test_reset_clears_lateral_record_only_for_selected_environments(self) -> None:
        root_quat = torch.zeros((2, 4))
        root_quat[:, 0] = 1.0
        env = SimpleNamespace(
            num_envs=2,
            device=torch.device("cpu"),
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_pos_w=torch.zeros((2, 3)),
                        root_link_quat_w=root_quat,
                    )
                )
            },
        )
        initialize_a6_plate_state(env)  # type: ignore[arg-type]
        env._a6_plate_best_lateral_distance[:] = torch.tensor([0.10, 0.20])
        env._a6_plate_lateral_progress[:] = torch.tensor([0.50, 0.75])

        with (
            patch(
                "g1recovery_amp.tasks.recovery.mdp.a6_plate_state."
                "a6_plate_geometry",
                return_value=(torch.zeros(2), torch.zeros(2)),
            ),
            patch(
                "g1recovery_amp.tasks.recovery.mdp.a6_plate_state."
                "a6_path_geometry",
                return_value=(torch.ones(2), torch.zeros(2), torch.zeros(2)),
            ),
        ):
            reset_a6_plate_state(env, torch.tensor([1]))  # type: ignore[arg-type]

        torch.testing.assert_close(
            env._a6_plate_best_lateral_distance, torch.tensor([0.10, 0.0])
        )
        torch.testing.assert_close(
            env._a6_plate_lateral_progress, torch.tensor([0.50, 0.0])
        )

    def test_lateral_progress_accepts_left_or_right_without_backtracking_credit(
        self,
    ) -> None:
        axis = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        best, progress = lateral_progress_from_state(
            displacement_xy=torch.tensor([[0.0, 0.025], [0.0, -0.025]]),
            lateral_axis_xy=axis,
            best_distance=torch.zeros(2),
        )
        torch.testing.assert_close(best, torch.tensor([0.025, 0.025]))
        torch.testing.assert_close(progress, torch.ones(2))

        best, progress = lateral_progress_from_state(
            displacement_xy=torch.tensor([[0.0, 0.010], [0.0, -0.050]]),
            lateral_axis_xy=axis,
            best_distance=best,
        )
        torch.testing.assert_close(best, torch.tensor([0.025, 0.050]))
        torch.testing.assert_close(progress, torch.tensor([0.0, 1.0]))

    def test_force_limit_requires_three_steps_but_emergency_cap_is_immediate(
        self,
    ) -> None:
        active = torch.tensor([True, True, True, False])
        counter = torch.zeros(4, dtype=torch.int64)
        ordinary_force = torch.tensor([1600.0, 2501.0, 1400.0, 3000.0])

        counter, invalid = sustained_force_invalid_from_state(
            active=active,
            force=ordinary_force,
            consecutive_steps=counter,
            force_limit=1500.0,
            hold_steps=3,
            catastrophic_force=2500.0,
        )
        self.assertEqual(counter.tolist(), [1, 1, 0, 0])
        self.assertEqual(invalid.tolist(), [False, True, False, False])

        for expected_count, expected_invalid in ((2, False), (3, True)):
            counter, invalid = sustained_force_invalid_from_state(
                active=active,
                force=torch.tensor([1600.0, 0.0, 0.0, 0.0]),
                consecutive_steps=counter,
                force_limit=1500.0,
                hold_steps=3,
                catastrophic_force=2500.0,
            )
            self.assertEqual(int(counter[0]), expected_count)
            self.assertEqual(bool(invalid[0]), expected_invalid)

        counter, invalid = sustained_force_invalid_from_state(
            active=active,
            force=torch.zeros(4),
            consecutive_steps=counter,
            force_limit=1500.0,
            hold_steps=3,
            catastrophic_force=2500.0,
        )
        self.assertEqual(counter.tolist(), [0, 0, 0, 0])
        self.assertFalse(bool(invalid.any()))

    def test_escape_threshold_and_hold_duration_are_configurable(self) -> None:
        strict_clearance = 0.040
        strict_hold = 25
        _, hold, escaped, _ = _step(
            ever_contact=True,
            clearance=strict_clearance - 0.001,
            min_planar_clearance=strict_clearance,
            clear_hold_steps=strict_hold,
        )
        self.assertEqual(hold, 0)
        self.assertFalse(escaped)

        hold = strict_hold - 1
        _, hold, escaped, _ = _step(
            ever_contact=True,
            clear_hold=hold,
            clearance=strict_clearance,
            min_planar_clearance=strict_clearance,
            clear_hold_steps=strict_hold,
        )
        self.assertEqual(hold, strict_hold)
        self.assertTrue(escaped)

    def test_no_progress_timer_ignores_noise_and_resets_on_real_progress(self) -> None:
        timer = advance_no_progress_time(
            current_time=torch.full((6,), 0.48),
            coverage_delta=torch.tensor([0.0, 0.001, 0.0009, 0.0, 0.0, 0.0]),
            clearance_delta=torch.tensor([0.0, 0.0, 0.0, 0.001, 0.0, 0.0]),
            separation_progress=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.04, 0.0]),
            eligible=torch.tensor([True, True, True, True, True, False]),
            step_dt=0.02,
        )

        torch.testing.assert_close(
            timer,
            torch.tensor([0.50, 0.0, 0.50, 0.0, 0.0, 0.0]),
        )

    def test_no_progress_timer_rejects_nonpositive_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_dt must be positive"):
            advance_no_progress_time(
                current_time=torch.zeros(1),
                coverage_delta=torch.zeros(1),
                clearance_delta=torch.zeros(1),
                separation_progress=torch.zeros(1),
                eligible=torch.ones(1, dtype=torch.bool),
                step_dt=0.0,
            )

    def test_path_footprint_has_no_above_plate_exemption(self) -> None:
        count, score, clearance = path_footprint(
            robot_pos=torch.tensor([[[0.0, 0.0, 2.0], [2.0, 0.0, 0.0]]]),
            robot_extent=torch.tensor([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]),
            plate_pos=torch.tensor([[0.0, 0.0, 0.0]]),
            plate_extent=torch.tensor([[0.5, 0.5, 0.1]]),
        )
        self.assertEqual(count.tolist(), [1])
        torch.testing.assert_close(score, torch.tensor([0.6]))
        torch.testing.assert_close(clearance, torch.tensor([0.0]))

    def test_flat_scene_never_activates_plate_rules(self) -> None:
        state = _step(
            active=False,
            contact=True,
            force=A6_MAX_PLATE_FORCE + 1.0,
            depth=-A6_MAX_PLATE_PENETRATION - 0.001,
            clearance=A6_MIN_PLANAR_CLEARANCE,
            episode_steps=A6_MAX_NO_CONTACT_STEPS + 1,
        )
        self.assertEqual(state, (False, 0, False, False))

    def test_contact_is_latched_but_does_not_mean_escape(self) -> None:
        self.assertEqual(_step(contact=True), (True, 0, False, False))
        self.assertEqual(_step(ever_contact=True), (True, 0, False, False))

    def test_escape_requires_consecutive_clear_samples(self) -> None:
        hold = 0
        for _ in range(A6_CLEAR_HOLD_STEPS - 1):
            _, hold, escaped, invalid = _step(
                ever_contact=True,
                clear_hold=hold,
                clearance=A6_MIN_PLANAR_CLEARANCE,
            )
            self.assertFalse(escaped)
            self.assertFalse(invalid)

        _, hold, escaped, invalid = _step(
            ever_contact=True,
            clear_hold=hold,
            clearance=A6_MIN_PLANAR_CLEARANCE,
        )
        self.assertEqual(hold, A6_CLEAR_HOLD_STEPS)
        self.assertTrue(escaped)
        self.assertFalse(invalid)

    def test_contact_or_insufficient_clearance_resets_hold_counter(self) -> None:
        _, contact_hold, _, _ = _step(
            contact=True,
            ever_contact=True,
            clear_hold=A6_CLEAR_HOLD_STEPS - 1,
            clearance=A6_MIN_PLANAR_CLEARANCE,
        )
        _, close_hold, _, _ = _step(
            ever_contact=True,
            clear_hold=A6_CLEAR_HOLD_STEPS - 1,
            clearance=A6_MIN_PLANAR_CLEARANCE - 0.001,
        )
        self.assertEqual(contact_hold, 0)
        self.assertEqual(close_hold, 0)

    def test_force_and_penetration_limits_are_strict(self) -> None:
        self.assertFalse(_step(force=A6_MAX_PLATE_FORCE)[3])
        self.assertTrue(_step(force=A6_MAX_PLATE_FORCE + 1.0)[3])
        self.assertFalse(_step(depth=-A6_MAX_PLATE_PENETRATION)[3])
        self.assertTrue(_step(depth=-A6_MAX_PLATE_PENETRATION - 0.001)[3])

    def test_missing_contact_timeout_only_applies_before_first_contact(self) -> None:
        self.assertFalse(_step(episode_steps=A6_MAX_NO_CONTACT_STEPS)[3])
        self.assertTrue(_step(episode_steps=A6_MAX_NO_CONTACT_STEPS + 1)[3])
        self.assertFalse(
            _step(
                ever_contact=True,
                episode_steps=A6_MAX_NO_CONTACT_STEPS + 1,
            )[3]
        )

    def test_invalid_reasons_remain_distinguishable(self) -> None:
        force, penetration, no_contact = plate_invalid_reasons(
            active=torch.tensor([True, True, True]),
            force=torch.tensor([A6_MAX_PLATE_FORCE + 1.0, 0.0, 0.0]),
            depth=torch.tensor([0.0, -A6_MAX_PLATE_PENETRATION - 0.001, 0.0]),
            ever_contact=torch.tensor([True, True, False]),
            episode_steps=torch.tensor([0, 0, A6_MAX_NO_CONTACT_STEPS + 1]),
        )
        self.assertEqual(force.tolist(), [True, False, False])
        self.assertEqual(penetration.tolist(), [False, True, False])
        self.assertEqual(no_contact.tolist(), [False, False, True])


if __name__ == "__main__":
    unittest.main()
