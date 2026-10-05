"""Tests for the migrated G1 recovery actor observations."""

import math
import unittest
from types import SimpleNamespace
from typing import cast

import torch

from g1recovery_amp.tasks.recovery.mdp.commands import RecoveryMotionCommand
from g1recovery_amp.tasks.recovery.mdp.observations import (
    AMP_DISCRIMINATOR_FRAME_DIM,
    AMP_DISCRIMINATOR_HISTORY_LENGTH,
    AMP_DISCRIMINATOR_OBS_DIM,
    RECOVERY_CRITIC_FRAME_DIM,
    RECOVERY_CRITIC_HISTORY_LENGTH,
    RECOVERY_CRITIC_OBS_DIM,
    RECOVERY_POLICY_FRAME_DIM,
    RECOVERY_POLICY_HISTORY_LENGTH,
    RECOVERY_POLICY_OBS_DIM,
    amp_discriminator_frame_from_state,
    key_body_pos_b_from_world,
    root_local_rot_tan_norm_from_quat,
)


class TestRecoveryObservations(unittest.TestCase):
    def test_amp_demo_frame_ids_are_chronological_and_in_bounds(self) -> None:
        command = cast(
            RecoveryMotionCommand,
            SimpleNamespace(
                recovery_cfg=SimpleNamespace(sampling_mode="start"),
                motion_lengths=torch.tensor([149]),
                motion_weights=torch.tensor([1.0]),
                motion_ids=torch.tensor([0, 0]),
                time_steps=torch.tensor([0, 138]),
                num_envs=2,
                device="cpu",
            ),
        )
        motion_ids, frame_ids = RecoveryMotionCommand.sample_amp_demo_indices(
            command, 10
        )
        torch.testing.assert_close(motion_ids, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(frame_ids[0], torch.arange(10))
        torch.testing.assert_close(frame_ids[1], torch.arange(138, 148))

        command.recovery_cfg.sampling_mode = "uniform"
        torch.manual_seed(3)
        motion_ids, frame_ids = RecoveryMotionCommand.sample_amp_demo_indices(
            command, 10
        )
        torch.testing.assert_close(motion_ids, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(
            frame_ids[:, 1:] - frame_ids[:, :-1],
            torch.ones((2, 9), dtype=torch.long),
        )
        self.assertGreaterEqual(int(frame_ids[:, 0].min()), 0)
        self.assertLessEqual(int(frame_ids[:, 0].max()), 138)

    def test_amp_demo_sampling_uses_selected_motion_length(self) -> None:
        num_envs = 64
        command = cast(
            RecoveryMotionCommand,
            SimpleNamespace(
                recovery_cfg=SimpleNamespace(sampling_mode="uniform"),
                motion_lengths=torch.tensor([20, 30]),
                motion_weights=torch.tensor([0.0, 1.0]),
                motion_ids=torch.zeros(num_envs, dtype=torch.long),
                time_steps=torch.zeros(num_envs, dtype=torch.long),
                num_envs=num_envs,
                device="cpu",
            ),
        )

        motion_ids, frame_ids = RecoveryMotionCommand.sample_amp_demo_indices(
            command, 10
        )

        torch.testing.assert_close(motion_ids, torch.ones(num_envs, dtype=torch.long))
        torch.testing.assert_close(
            frame_ids[:, 1:] - frame_ids[:, :-1],
            torch.ones((num_envs, 9), dtype=torch.long),
        )
        self.assertLessEqual(int(frame_ids.max()), 28)

    def test_multi_motion_gather_uses_each_environment_selection(self) -> None:
        command = cast(
            RecoveryMotionCommand,
            SimpleNamespace(
                motions=(
                    SimpleNamespace(
                        joint_pos=torch.tensor([[0.0, 1.0], [2.0, 3.0]])
                    ),
                    SimpleNamespace(
                        joint_pos=torch.tensor([[10.0, 11.0], [12.0, 13.0]])
                    ),
                ),
                motion_lengths=torch.tensor([2, 2]),
                device="cpu",
            ),
        )

        gathered = RecoveryMotionCommand.gather_motion_field(
            command,
            "joint_pos",
            motion_ids=torch.tensor([0, 1]),
            frame_ids=torch.tensor([1, 0]),
        )

        torch.testing.assert_close(
            gathered,
            torch.tensor([[2.0, 3.0], [10.0, 11.0]]),
        )
        with self.assertRaisesRegex(IndexError, "frame id"):
            RecoveryMotionCommand.gather_motion_field(
                command,
                "joint_pos",
                motion_ids=torch.tensor([0, 1]),
                frame_ids=torch.tensor([2, 0]),
            )

    def test_amp_frame_has_source_field_order_and_dimension(self) -> None:
        num_envs = 2
        history = AMP_DISCRIMINATOR_HISTORY_LENGTH
        root_pos = torch.zeros((num_envs, history, 3))
        root_quat = torch.zeros((num_envs, history, 4))
        root_quat[..., 0] = 1.0
        root_ang_vel = torch.tensor([1.0, 2.0, 3.0]).expand(num_envs, history, -1)
        joint_pos_value = torch.full((num_envs, history, 29), 4.0)
        joint_vel_value = torch.full((num_envs, history, 29), 5.0)
        key_body_pos = torch.full((num_envs, history, 6, 3), 6.0)

        observation = amp_discriminator_frame_from_state(
            root_pos,
            root_quat,
            root_ang_vel,
            joint_pos_value,
            joint_vel_value,
            key_body_pos,
        )

        self.assertEqual(
            observation.shape,
            (num_envs, history, AMP_DISCRIMINATOR_FRAME_DIM),
        )
        self.assertEqual(
            AMP_DISCRIMINATOR_OBS_DIM,
            history * AMP_DISCRIMINATOR_FRAME_DIM,
        )
        torch.testing.assert_close(
            observation[..., 0:6],
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(num_envs, history, -1),
        )
        torch.testing.assert_close(observation[..., 6:9], root_ang_vel)
        torch.testing.assert_close(observation[..., 9:38], joint_pos_value)
        torch.testing.assert_close(observation[..., 38:67], joint_vel_value)
        torch.testing.assert_close(
            observation[..., 67:85],
            key_body_pos.flatten(start_dim=-2),
        )

    def test_no_key_body_policy_dimensions(self) -> None:
        self.assertEqual(RECOVERY_POLICY_FRAME_DIM, 96)
        self.assertEqual(RECOVERY_POLICY_HISTORY_LENGTH, 5)
        self.assertEqual(RECOVERY_POLICY_OBS_DIM, 480)

    def test_privileged_critic_dimensions(self) -> None:
        self.assertEqual(RECOVERY_CRITIC_FRAME_DIM, 118)
        self.assertEqual(RECOVERY_CRITIC_HISTORY_LENGTH, 5)
        self.assertEqual(RECOVERY_CRITIC_OBS_DIM, 590)

    def test_identity_orientation_encoding(self) -> None:
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        encoded = root_local_rot_tan_norm_from_quat(quat)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        torch.testing.assert_close(encoded, expected)

    def test_encoding_ignores_pure_yaw(self) -> None:
        half_yaw = math.pi / 4.0
        quat = torch.tensor([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
        encoded = root_local_rot_tan_norm_from_quat(quat)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        torch.testing.assert_close(encoded, expected, atol=1e-6, rtol=1e-6)

    def test_key_body_positions_are_relative_to_root(self) -> None:
        root_pos = torch.tensor([[1.0, 2.0, 3.0]])
        root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        body_pos = torch.tensor([[[2.0, 2.0, 3.0], [1.0, 4.0, 2.0]]])

        positions_b = key_body_pos_b_from_world(root_pos, root_quat, body_pos)

        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 2.0, -1.0]])
        torch.testing.assert_close(positions_b, expected)

    def test_key_body_positions_follow_root_yaw(self) -> None:
        half_yaw = math.pi / 4.0
        root_pos = torch.zeros((1, 3))
        root_quat = torch.tensor([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]])
        body_pos = torch.tensor([[[1.0, 0.0, 0.0]]])

        positions_b = key_body_pos_b_from_world(root_pos, root_quat, body_pos)

        expected = torch.tensor([[0.0, -1.0, 0.0]])
        torch.testing.assert_close(positions_b, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
