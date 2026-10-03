"""Checks for the A6 physics-substep command-delay buffer."""

import unittest

import torch

from g1recovery_amp.tasks.recovery.mdp import write_and_select_delayed_targets


class TestA6Actions(unittest.TestCase):
    def test_zero_and_five_step_lags(self) -> None:
        buffer = torch.zeros((6, 2, 3))
        target = torch.full((2, 3), 0.123)
        lag = torch.tensor([0, 5])

        outputs = []
        for cursor in range(6):
            outputs.append(
                write_and_select_delayed_targets(buffer, cursor, target, lag).clone()
            )

        for output in outputs:
            torch.testing.assert_close(output[0], target[0])
        for output in outputs[:5]:
            torch.testing.assert_close(output[1], torch.zeros_like(target[1]))
        torch.testing.assert_close(outputs[5][1], target[1])

    def test_invalid_lag_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            write_and_select_delayed_targets(
                torch.zeros((6, 1, 2)),
                0,
                torch.zeros((1, 2)),
                torch.tensor([6]),
            )


if __name__ == "__main__":
    unittest.main()
