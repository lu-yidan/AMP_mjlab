"""Teacher-executed transitions train imitation and value, not PPO policy ratio."""

import pytest
import torch

from g1recovery_host.ppo import HostPolicy, ppo_update


def test_teacher_executed_transitions_mask_ppo_but_keep_imitation():
    torch.manual_seed(11)
    policy = HostPolicy(obs_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    obs = torch.randn(8, 4)
    with torch.no_grad():
        dist = policy.distribution(obs)
        actions = torch.zeros(8, 2)
        batch = (
            obs, obs.clone(), actions,
            dist.log_prob(actions).sum(-1), dist.mean.clone(),
            dist.stddev.clone(), policy.values(obs),
            policy.values(obs) + 0.1, torch.ones(8),
            torch.ones(8), torch.ones(8),
        )
    metrics = ppo_update(
        policy, optimizer, batch, epochs=1, minibatches=2,
        policy_active=torch.zeros(8), imitation_actions=actions,
        imitation_mask=torch.ones(8), imitation_weight=0.5,
    )
    assert metrics[0] == 0
    assert metrics[8] > 0


def test_teacher_mask_cannot_enable_inactive_policy_step():
    policy = HostPolicy(obs_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    obs = torch.zeros(4, 4)
    with torch.no_grad():
        dist = policy.distribution(obs)
        actions = dist.mean.clone()
        values = policy.values(obs)
        batch = (
            obs, obs.clone(), actions, dist.log_prob(actions).sum(-1),
            dist.mean.clone(), dist.stddev.clone(), values, values.clone(),
            torch.ones(4), torch.zeros(4), torch.ones(4),
        )
    with pytest.raises(ValueError, match="policy_active"):
        ppo_update(policy, optimizer, batch, policy_active=torch.ones(4))
