"""A capped adaptive-KL update must not exceed its configured learning rate."""

import unittest

import torch

from g1recovery_host.ppo import HostPolicy, ppo_update


class AdaptiveKlLearningRateTest(unittest.TestCase):
    def test_resume_learning_rate_is_capped_before_optimizer_step(self):
        torch.manual_seed(7)
        policy = HostPolicy(obs_dim=4, action_dim=2)
        optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)
        obs = torch.randn(8, 4)
        with torch.no_grad():
            dist = policy.distribution(obs)
            actions = dist.sample()
            old_logp = dist.log_prob(actions).sum(-1)
            values = policy.values(obs)
            batch = (
                obs,
                obs.clone(),
                actions,
                old_logp,
                dist.mean.clone(),
                dist.stddev.clone(),
                values,
                values + 0.1,
                torch.ones(8),
                torch.ones(8),
                torch.ones(8),
            )
        ppo_update(
            policy, optimizer, batch, epochs=1, minibatches=2,
            adaptive_kl=True, max_learning_rate=5e-5,
        )
        self.assertLessEqual(optimizer.param_groups[0]["lr"], 5e-5)
        self.assertEqual(
            {int(state["step"]) for state in optimizer.state.values()}, {2}
        )


if __name__ == "__main__":
    unittest.main()
