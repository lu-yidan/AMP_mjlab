"""Tensor-level checks for the migrated AMP discriminator."""

import unittest

import torch
from torch import nn

from g1recovery_amp.tasks.recovery.rl import (
    AmpDiscriminator,
    AmpDiscriminatorCfg,
)


class TestAmpDiscriminator(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.cfg = AmpDiscriminatorCfg()
        self.discriminator = AmpDiscriminator(self.cfg)
        self.policy_observation = torch.randn(
            4, self.cfg.history_length, self.cfg.frame_dim
        )
        self.demo_observation = torch.randn_like(self.policy_observation)

    def test_source_network_shape_and_output(self) -> None:
        self.assertEqual(self.discriminator.input_dim, 850)
        linear_layers = [
            layer
            for layer in self.discriminator.disc_trunk
            if isinstance(layer, nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear_layers],
            [(850, 1024), (1024, 512)],
        )
        self.assertEqual(self.discriminator.disc_linear.in_features, 512)
        self.assertEqual(self.discriminator.score(self.policy_observation).shape, (4,))

    def test_lsgan_loss_penalty_and_gradients_are_finite(self) -> None:
        loss, policy_score, demo_score = self.discriminator.lsgan_loss(
            self.policy_observation, self.demo_observation
        )
        penalty = self.discriminator.gradient_penalty(self.demo_observation)
        total_loss = loss + penalty
        total_loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(penalty))
        self.assertTrue(torch.isfinite(policy_score).all())
        self.assertTrue(torch.isfinite(demo_score).all())
        gradients = [
            parameter.grad
            for parameter in self.discriminator.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(value is not None for value in gradients))
        self.assertTrue(
            all(torch.isfinite(value).all() for value in gradients if value is not None)
        )

    def test_normalizer_learns_all_live_and_demo_frames(self) -> None:
        combined = torch.cat((self.policy_observation, self.demo_observation), dim=0)
        self.discriminator.update_normalization(combined)
        normalized = self.discriminator.normalize_observation(combined)

        self.assertEqual(int(self.discriminator.disc_obs_normalizer.count), 80)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertLessEqual(float(normalized.abs().max()), 10.0)

    def test_source_style_reward_and_half_blend(self) -> None:
        for parameter in self.discriminator.parameters():
            nn.init.zeros_(parameter)

        style_reward, score = self.discriminator.style_reward(
            self.policy_observation, step_dt=0.02
        )
        task_reward = torch.full((4,), 2.0)
        blended = self.discriminator.blend_reward(task_reward, style_reward)

        torch.testing.assert_close(score, torch.zeros(4))
        torch.testing.assert_close(style_reward, torch.full((4,), 0.75))
        torch.testing.assert_close(blended, torch.full((4,), 1.375))

    def test_per_environment_task_weight_can_disable_style_reward(self) -> None:
        task_reward = torch.tensor([2.0, 2.0])
        style_reward = torch.tensor([0.5, 0.5])
        blended = self.discriminator.blend_reward(
            task_reward,
            style_reward,
            task_weight=torch.tensor([0.5, 1.0]),
        )
        torch.testing.assert_close(blended, torch.tensor([1.25, 2.0]))

    def test_invalid_observation_shape_fails_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "AMP observation"):
            self.discriminator.score(torch.zeros((4, 850)))


if __name__ == "__main__":
    unittest.main()
