"""Checks for the frozen A6 grouped dynamics schedule and sampler."""

import unittest

import torch

from g1recovery_amp.tasks.recovery.mdp import (
    A6_DYNAMICS_FINAL_WIDTH,
    A6_DYNAMICS_INITIAL_WIDTH,
    A6_DYNAMICS_NOMINAL_FRACTION,
    A6_DYNAMICS_RAMP_CONTROL_STEPS,
    A6_MAX_COMMAND_DELAY_STEPS,
    a6_dynamics_width,
    sample_a6_dynamics,
)


class TestA6Dynamics(unittest.TestCase):
    def test_width_uses_adaptation_local_age(self) -> None:
        start = 480_000
        self.assertEqual(a6_dynamics_width(start, start), A6_DYNAMICS_INITIAL_WIDTH)
        self.assertAlmostEqual(
            a6_dynamics_width(
                start + A6_DYNAMICS_RAMP_CONTROL_STEPS // 2, start
            ),
            0.15,
        )
        self.assertEqual(
            a6_dynamics_width(
                start + A6_DYNAMICS_RAMP_CONTROL_STEPS * 2, start
            ),
            A6_DYNAMICS_FINAL_WIDTH,
        )

    def test_disabled_sampler_is_nominal(self) -> None:
        generator = torch.Generator().manual_seed(7)
        body, gains, lag, nominal = sample_a6_dynamics(
            64,
            width=0.2,
            generator=generator,
            device="cpu",
            enabled=False,
        )
        torch.testing.assert_close(body, torch.ones_like(body))
        torch.testing.assert_close(gains, torch.ones_like(gains))
        torch.testing.assert_close(lag, torch.zeros_like(lag))
        self.assertTrue(bool(nominal.all()))

    def test_enabled_sampler_respects_ranges_and_nominal_fraction(self) -> None:
        generator = torch.Generator().manual_seed(20261013)
        body, gains, lag, nominal = sample_a6_dynamics(
            20_000,
            width=0.2,
            generator=generator,
            device="cpu",
        )
        self.assertTrue(bool(((body >= 0.8) & (body <= 1.2)).all()))
        self.assertTrue(bool(((gains >= 0.8) & (gains <= 1.2)).all()))
        self.assertTrue(
            bool(
                ((lag >= 0) & (lag <= A6_MAX_COMMAND_DELAY_STEPS)).all()
            )
        )
        torch.testing.assert_close(body[nominal], torch.ones_like(body[nominal]))
        torch.testing.assert_close(gains[nominal], torch.ones_like(gains[nominal]))
        torch.testing.assert_close(lag[nominal], torch.zeros_like(lag[nominal]))
        self.assertAlmostEqual(
            float(nominal.float().mean()), A6_DYNAMICS_NOMINAL_FRACTION, delta=0.015
        )

    def test_invalid_schedule_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ramp_control_steps"):
            a6_dynamics_width(0, 0, 0)
        with self.assertRaisesRegex(ValueError, "width"):
            sample_a6_dynamics(
                1,
                width=1.0,
                generator=torch.Generator(),
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
