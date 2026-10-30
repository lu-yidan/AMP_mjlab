"""Controlled C ablations: tie discriminator weights or remove recovery sampling strata.

The shared-network control retains C's per-mode normalization, conditional expert
matching and update budget. It is not a reproduction of the legacy AMP baseline.
"""
import torch
from rsl_rl.modules.stable_multi_amp import StableMultiAMP


def require_finite(label, *values):
    for i, value in enumerate(values):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'{label}[{i}]: non-finite tensor, shape={tuple(value.shape)}')


class ControlledAMP(StableMultiAMP):
    def __init__(self, loader, cfg, hidden_dims, reward_coef, task_lerp, device):
        super().__init__(loader, cfg, hidden_dims, reward_coef, task_lerp, device)
        # Shared network uses loco's input width; zero-pad recovery conditions.
        # Dual arms retain the original C architecture exactly.
        if cfg.get('shared_discriminator'):
            self.networks['rec'] = self.networks['loco']
        self.optimizers = {k: torch.optim.Adam(net.parameters(), lr=cfg['learning_rate'])
                           for k, net in self.networks.items()}
        if cfg.get('shared_discriminator'):
            self.optimizers['rec'] = self.optimizers['loco']

    def _input(self, kind, a, b, condition):
        require_finite('discriminator input', a, b, condition)
        x = super()._input(kind, a, b, condition)
        if kind == 'rec' and self.cfg.get('shared_discriminator'):
            x = torch.cat((x, torch.zeros(len(a), 3, device=a.device, dtype=a.dtype)), -1)
        require_finite('normalized discriminator input', x)
        return x

    def _recovery_expert(self, geometry):
        if self.cfg['balance_recovery']:
            return super()._recovery_expert(geometry)
        ids = torch.randint(len(self.rec_a), (len(geometry),), device=self.rec_a.device)
        return self.rec_a[ids], self.rec_b[ids]

    @torch.no_grad()
    def reward_and_store(self, a, b, task_reward, recovery, condition, dones, geometry=None):
        require_finite('rollout', a, b, task_reward, condition, geometry)
        reward = super().reward_and_store(a, b, task_reward, recovery, condition, dones, geometry)
        require_finite('combined reward', reward)
        return reward
