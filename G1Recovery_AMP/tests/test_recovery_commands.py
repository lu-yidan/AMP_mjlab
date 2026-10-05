"""Regression checks for recovery-specific motion command behavior."""

import unittest
from unittest.mock import Mock

import torch

from g1recovery_amp.tasks.recovery.mdp.commands import (
    RecoveryMotionCommand,
    _resolve_motion_manifest,
)


class TestRecoveryMotionCommand(unittest.TestCase):
    def test_playback_cli_override_becomes_the_only_loaded_motion(self) -> None:
        files, weights = _resolve_motion_manifest(
            "selected.npz",
            ("training_first.npz", "training_second.npz"),
            (1.0, 1.0),
            "start",
        )

        self.assertEqual(files, ("selected.npz",))
        self.assertEqual(weights, (1.0,))

    def test_training_rejects_a_misaligned_motion_manifest(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "motion_file must be the first entry"
        ):
            _resolve_motion_manifest(
                "selected.npz",
                ("training_first.npz", "training_second.npz"),
                (1.0, 1.0),
                "uniform",
            )

    def test_reference_stops_at_last_frame_without_resampling_robot(self) -> None:
        command = object.__new__(RecoveryMotionCommand)
        command.time_steps = torch.tensor([147, 80, 331])
        command.motion_ids = torch.tensor([0, 1, 2])
        command.motion_lengths = torch.tensor([149, 82, 332])
        command._pending_forward = False
        command._resample_command = Mock(
            side_effect=AssertionError("reference wrap must not reset the robot")
        )
        command.update_relative_body_poses = Mock()

        command._update_command()
        command._update_command()

        torch.testing.assert_close(
            command.time_steps,
            torch.tensor([148, 81, 331]),
        )
        command._resample_command.assert_not_called()
        self.assertEqual(command.update_relative_body_poses.call_count, 2)

    def test_scoped_update_advances_only_selected_environments(self) -> None:
        command = object.__new__(RecoveryMotionCommand)
        command.time_steps = torch.tensor([10, 80])
        command.motion_ids = torch.tensor([0, 1])
        command.motion_lengths = torch.tensor([149, 82])
        command._pending_forward = False
        command.update_relative_body_poses = Mock()

        command._update_command(env_ids=torch.tensor([1]))

        torch.testing.assert_close(command.time_steps, torch.tensor([10, 81]))


if __name__ == "__main__":
    unittest.main()
