"""MLP actor-critic for SuperMimic.

The actor is deterministic: SuperMimic differentiates the return through its mean action, and exploration
during collection is the algorithm's fixed Gaussian action noise. The network follows BeyondMimic's PPO policy
(actor and critic MLP(512, 256, 128), ELU) on the env's observation groups; it is not the paper's motion-encoder /
decoder policy. Any policy module works if it provides
the same methods (a transformer, a token policy ...):

    act_inference(obs) -> (N, num_actions)   differentiable in obs
    evaluate(obs)      -> (N,) or (N, 1)     the critic
    actor_parameters(), critic_parameters()
    update_normalization(obs)                optional, called on real observations during collection
    aux_loss(obs)      -> scalar or None     optional extra actor loss at the start observations
                                             (clipped separately from the BPTT gradient)
    critic_obs_keys    optional: the observation keys evaluate() reads (only those are kept per imagined
                                             step for the critic fit; default: all)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .normalizer import RunningNorm, mlp


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, obs: dict[str, torch.Tensor], num_actions: int, actor_obs_group: str = "policy",
                 critic_obs_group: str = "critic", actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128), activation: str = "elu", layer_norm: bool = False,
                 obs_normalization: bool = True, init_last_layer_scale: float = 0.01):  # fmt: skip
        super().__init__()
        self.actor_key = actor_obs_group
        self.critic_key = critic_obs_group if critic_obs_group in obs else actor_obs_group
        na, nc = obs[self.actor_key].shape[-1], obs[self.critic_key].shape[-1]
        self.actor = mlp(na, actor_hidden_dims, num_actions, act=activation, layer_norm=layer_norm)
        self.critic = mlp(nc, critic_hidden_dims, 1, act=activation, layer_norm=layer_norm)
        with torch.no_grad():  # start near the default pose
            self.actor[-1].weight.mul_(init_last_layer_scale)
            self.actor[-1].bias.zero_()
        self.obs_normalization = obs_normalization
        self.actor_norm, self.critic_norm = RunningNorm(na), RunningNorm(nc)

    def _actor_in(self, obs):
        x = obs[self.actor_key]
        return self.actor_norm(x) if self.obs_normalization else x

    def act_inference(self, obs):
        return self.actor(self._actor_in(obs))

    @property
    def critic_obs_keys(self):
        return (self.critic_key,)

    def evaluate(self, obs):
        x = obs[self.critic_key]
        return self.critic(self.critic_norm(x) if self.obs_normalization else x).squeeze(-1)

    @torch.no_grad()
    def update_normalization(self, obs):
        if self.obs_normalization:
            self.actor_norm.update(obs[self.actor_key])
            self.critic_norm.update(obs[self.critic_key])

    def actor_parameters(self):
        return list(self.actor.parameters())

    def critic_parameters(self):
        return list(self.critic.parameters())
