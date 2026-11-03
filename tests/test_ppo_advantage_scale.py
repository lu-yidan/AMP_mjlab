"""Bounded sample emphasis must act after per-critic GAE normalization."""

import pytest
import torch

from g1recovery_host.ppo import grouped_gae


def test_hard_target_advantage_is_tripled_without_changing_other_samples():
    rewards = torch.zeros(2, 2, 4)
    rewards[:, :, 3] = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    values = torch.zeros_like(rewards)
    dones = torch.ones(2, 2, dtype=torch.bool)
    last_values = torch.zeros(2, 4)
    scale = torch.ones_like(rewards)
    _, baseline = grouped_gae(rewards, values, dones, last_values, advantage_scales=scale)
    scale[0, 0, 3] = 3.0
    _, focused = grouped_gae(rewards, values, dones, last_values, advantage_scales=scale)
    torch.testing.assert_close(focused[0, 0], 3.0 * baseline[0, 0])
    torch.testing.assert_close(focused.reshape(-1)[1:], baseline.reshape(-1)[1:])
    scale[0, 0, 3] = 3.01
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        grouped_gae(rewards, values, dones, last_values, advantage_scales=scale)
