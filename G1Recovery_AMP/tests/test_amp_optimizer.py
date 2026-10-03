"""Checks for one isolated AMP discriminator optimization step."""

import math
import unittest

import torch

from g1recovery_amp.tasks.recovery.rl import (
    AmpDiscriminator,
    AmpDiscriminatorCfg,
    AmpDiscriminatorOptimizer,
    AmpDiscriminatorOptimizerCfg,
    AmpReplayBuffer,
)


class TestAmpDiscriminatorOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.discriminator = AmpDiscriminator(AmpDiscriminatorCfg(hidden_dims=(16, 8)))
        self.optimizer = AmpDiscriminatorOptimizer(self.discriminator)

    def test_source_optimizer_settings_and_parameter_groups(self) -> None:
        self.assertEqual(self.optimizer.cfg.learning_rate, 1.0e-3)
        self.assertEqual(self.optimizer.cfg.gradient_penalty_scale, 10.0)
        self.assertEqual(self.optimizer.cfg.max_gradient_norm, 1.0)
        groups = self.optimizer.optimizer.param_groups
        self.assertEqual(
            [group["name"] for group in groups], ["disc_trunk", "disc_linear"]
        )
        self.assertEqual([group["weight_decay"] for group in groups], [1.0e-3, 1.0e-1])

    def test_update_changes_parameters_and_normalization(self) -> None:
        policy = torch.randn(4, 10, 85)
        demonstration = torch.randn_like(policy) + 0.5
        before = [
            parameter.detach().clone() for parameter in self.discriminator.parameters()
        ]

        metrics = self.optimizer.update(policy, demonstration)

        after = list(self.discriminator.parameters())
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, after))
        )
        self.assertEqual(int(self.discriminator.disc_obs_normalizer.count), 80)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        gradient_squares = [
            parameter.grad.detach().square().sum()
            for parameter in self.discriminator.parameters()
            if parameter.grad is not None
        ]
        clipped_norm = torch.sqrt(torch.stack(gradient_squares).sum())
        self.assertLessEqual(float(clipped_norm), 1.0 + 1.0e-5)

    def test_replay_batches_feed_the_optimizer(self) -> None:
        policy_buffer = AmpReplayBuffer(max_length=2, batch_size=2)
        demonstration_buffer = AmpReplayBuffer(max_length=2, batch_size=2)
        for _ in range(2):
            policy_buffer.append(torch.randn(2, 10, 85))
            demonstration_buffer.append(torch.randn(2, 10, 85))

        policy_batch = policy_buffer.sample(3)
        demonstration_batch = demonstration_buffer.sample(3)
        metrics = self.optimizer.update(policy_batch, demonstration_batch)

        self.assertEqual(
            set(metrics),
            {
                "amp/disc_loss",
                "amp/disc_grad_penalty",
                "amp/disc_total_loss",
                "amp/disc_score",
                "amp/disc_demo_score",
                "amp/disc_grad_norm_before_clip",
            },
        )

    def test_invalid_configuration_and_batch_shape_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            AmpDiscriminatorOptimizer(
                self.discriminator,
                AmpDiscriminatorOptimizerCfg(learning_rate=0.0),
            )
        with self.assertRaisesRegex(ValueError, "equal shapes"):
            self.optimizer.update(torch.zeros(2, 10, 85), torch.zeros(3, 10, 85))


if __name__ == "__main__":
    unittest.main()
