"""Tensor-level checks for the migrated AMP replay storage."""

import unittest

import torch

from g1recovery_amp.tasks.recovery.rl import (
    AMP_REPLAY_BUFFER_LENGTH,
    AmpReplayBuffer,
    estimate_replay_storage_bytes,
)


class TestAmpReplayBuffer(unittest.TestCase):
    def test_allocation_is_lazy_and_memory_estimate_matches_source_shape(self) -> None:
        buffer = AmpReplayBuffer(max_length=3, batch_size=2, observation_shape=(2, 2))
        self.assertFalse(buffer.is_allocated)
        self.assertEqual(buffer.required_storage_bytes, 3 * 2 * 2 * 2 * 4)

        training_bytes = estimate_replay_storage_bytes(
            max_length=AMP_REPLAY_BUFFER_LENGTH,
            batch_size=4096,
            buffer_count=2,
        )
        self.assertEqual(training_bytes, 6_796_083_200)

    def test_append_wraps_and_keeps_newest_steps_in_chronological_order(self) -> None:
        buffer = AmpReplayBuffer(max_length=3, batch_size=2, observation_shape=(1,))
        for step in range(4):
            buffer.append(torch.tensor([[step * 10.0], [step * 10.0 + 1.0]]))

        expected = torch.tensor([[[10.0], [11.0]], [[20.0], [21.0]], [[30.0], [31.0]]])
        torch.testing.assert_close(buffer.chronological(), expected)
        self.assertEqual(buffer.current_length, 3)

    def test_append_detaches_inputs_and_sample_has_expected_shape(self) -> None:
        buffer = AmpReplayBuffer(max_length=2, batch_size=3, observation_shape=(2, 2))
        observations = torch.arange(12.0, requires_grad=True).reshape(3, 2, 2)
        buffer.append(observations)
        sampled = buffer.sample(5)

        self.assertEqual(sampled.shape, (5, 2, 2))
        self.assertFalse(sampled.requires_grad)
        self.assertTrue(torch.isfinite(sampled).all())

    def test_source_batch_generator_shape_and_epoch_sample_set(self) -> None:
        torch.manual_seed(5)
        buffer = AmpReplayBuffer(max_length=4, batch_size=2, observation_shape=(1,))
        for step in range(4):
            buffer.append(torch.tensor([[step * 2.0], [step * 2.0 + 1.0]]))

        batches = list(
            buffer.mini_batch_generator(
                fetch_length=2,
                num_mini_batches=2,
                num_epochs=3,
            )
        )
        self.assertEqual(len(batches), 6)
        self.assertTrue(all(batch.shape == (2, 1) for batch in batches))
        epoch_values = [
            torch.cat(batches[index : index + 2]).flatten().sort().values
            for index in range(0, len(batches), 2)
        ]
        for values in epoch_values[1:]:
            torch.testing.assert_close(values, epoch_values[0])

    def test_invalid_shapes_and_empty_reads_fail_early(self) -> None:
        buffer = AmpReplayBuffer(max_length=2, batch_size=3, observation_shape=(2, 2))
        with self.assertRaisesRegex(RuntimeError, "empty AMP replay"):
            buffer.sample(1)
        with self.assertRaisesRegex(ValueError, "expects shape"):
            buffer.append(torch.zeros(2, 2, 2))


if __name__ == "__main__":
    unittest.main()
