"""State-routed AMP with separate recovery and speed-conditioned locomotion priors.

Routing metadata is captured before stepping. Terminal transitions are excluded
until the environment supplies exact pre-reset observations (never train on resets).
"""
import torch
from torch import nn

from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.storage.replay_buffer import ReplayBuffer


def recovery_gate(gravity_z, head_height, gravity_threshold, height_threshold=None):
    mask = (gravity_z + 1.0).abs() > gravity_threshold
    if height_threshold is not None:
        mask = mask | (head_height < height_threshold)
    return mask


class RunningMoments(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        n = x.shape[0]
        delta = x.mean(0) - self.mean
        total = self.count + n
        var = (self.var * self.count + x.var(0, unbiased=False) * n
               + delta.square() * self.count * n / total) / total
        self.mean.add_(delta * n / total)
        self.var.copy_(var)
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / (self.var + 1e-6).sqrt()).clamp(-10, 10)


class MultiAMP(nn.Module):
    def __init__(self, loader, cfg, hidden_dims, reward_coef, task_lerp, device):
        super().__init__()
        self.cfg = dict(cfg)
        self.dim = loader.observation_dim
        self.vector_condition = cfg.get("condition_mode") == "matched_velocity"
        self.condition_dim = 3 if self.vector_condition else 1
        self.reward_coef, self.task_lerp = reward_coef, task_lerp
        if cfg['speed_max'] <= 0 or cfg['batch_size'] <= 0:
            raise ValueError('speed_max and batch_size must be positive')
        self.networks = nn.ModuleDict({
            'rec': Discriminator(self.dim * 2, reward_coef, hidden_dims, device),
            'loco': Discriminator(self.dim * 2 + self.condition_dim, reward_coef, hidden_dims, device),
        })
        self.norms = nn.ModuleDict({k: RunningMoments(self.dim) for k in self.networks})
        self.to(device)
        self.optimizers = {k: torch.optim.Adam(self.networks[k].parameters(), lr=cfg['learning_rate'])
                           for k in self.networks}
        # Condition is stored with the transition, not recomputed from current commands.
        self.replay = {k: ReplayBuffer(self.dim + self.condition_dim, cfg['replay_size'], device)
                       for k in self.networks}
        self.clips = {'rec': [], 'walk': [], 'run': []}
        loco_a, loco_b, velocities = [], [], []
        for i, name in enumerate(loader.motion_names):
            name = name.lower()
            kind = ('rec' if 'fallandgetup' in name else
                    'run' if name.startswith(('run', 'jog', 'arc_jog')) else
                    'walk' if name.startswith(('walk', 'arc_walk', 'idle', 'step')) else None)
            if kind is None:
                raise ValueError(f'Unclassified AMP reference clip: {name}')
            fields = [getattr(loader, attr)[i].flatten(1) for attr in
                      ('_body_pos_b_list', '_body_ori_b_list', '_body_lin_vel_b_list', '_body_ang_vel_b_list')]
            frames = torch.cat(fields, -1)
            if len(frames) < 2 or not torch.isfinite(frames).all():
                raise ValueError(f'Invalid AMP reference clip: {name}')
            self.clips[kind].append(frames)
            if self.vector_condition and kind != 'rec':
                loco_a.append(frames[:-1]); loco_b.append(frames[1:])
                velocities.append(loader._command_velocity_list[i][:-1])
        if any(not clips for clips in self.clips.values()):
            raise ValueError('MultiAMP requires recovery, walk and run reference clips')

        if self.vector_condition:
            self.expert_a = torch.cat(loco_a).to(device)
            self.expert_b = torch.cat(loco_b).to(device)
            self.expert_velocity = torch.cat(velocities).to(device)
            if not torch.isfinite(self.expert_velocity).all():
                raise ValueError('Non-finite expert velocities')
            self.tolerance = torch.tensor(cfg['match_tolerance'], device=device)
            self.scale = torch.tensor(cfg['condition_scale'], device=device)
            if (self.tolerance <= 0).any() or (self.scale <= 0).any():
                raise ValueError('Condition scales and tolerances must be positive')
            self.coverage = 0.0

    def _matched_expert(self, commands, nearest_only=False):
        # Randomize among the nearest 16 transitions; never fabricate expert labels.
        indices, supported = [], []
        for chunk in commands.split(256):
            distance = ((chunk[:, None] - self.expert_velocity[None]) / self.tolerance).abs().amax(-1)
            values, candidates = distance.topk(1 if nearest_only else min(16, len(self.expert_velocity)), largest=False)
            valid = values <= 1.
            weights = valid.float()
            weights[~valid.any(-1), 0] = 1.
            selected = torch.multinomial(weights, 1)
            indices.append(candidates.gather(1, selected).squeeze(1))
            supported.append(valid.any(-1))
        return torch.cat(indices), torch.cat(supported)

    def _expert(self, kind, count, device):
        clips = self.clips[kind]
        ids = torch.randint(len(clips), (count,), device=device)
        a = torch.empty(count, self.dim, device=device)
        b = torch.empty_like(a)
        for i, clip in enumerate(clips):
            mask = ids == i
            idx = torch.randint(len(clip) - 1, (int(mask.sum()),), device=device)
            a[mask], b[mask] = clip[idx], clip[idx + 1]
        return a, b

    def _input(self, kind, a, b, condition):
        x = torch.cat((self.norms[kind](a), self.norms[kind](b)), -1)
        return torch.cat((x, condition.reshape(len(a), self.condition_dim)), -1) if kind == 'loco' else x

    @torch.no_grad()
    def reward_and_store(self, a, b, task_reward, recovery, condition, dones):
        reward = task_reward.clone()
        for kind, mask in (('rec', recovery), ('loco', ~recovery)):
            # Missing terminal observations: task-only reward and no AMP sample.
            mask = mask & ~dones.bool()
            if not mask.any():
                continue
            c = condition[mask]
            if self.vector_condition and kind == 'loco':
                idx, supported = self._matched_expert(c)
                self.coverage = float(supported.float().mean())
                active = mask.nonzero().flatten()
                mask[active[~supported]] = False
                if not mask.any():
                    continue
                # Snap BOTH classes to the same real expert condition. This prevents
                # the discriminator exploiting command-label distribution mismatch.
                c = self.expert_velocity[idx[supported]] / self.scale
            elif self.vector_condition:
                c = c / self.scale
            c = c.reshape(len(c), self.condition_dim)
            d = self.networks[kind](self._input(kind, a[mask], b[mask], c)).squeeze(-1)
            style = self.reward_coef * (1 - 0.25 * (d - 1).square()).clamp(min=0)
            reward[mask] = self.task_lerp * task_reward[mask] + (1 - self.task_lerp) * style
            self.replay[kind].insert(torch.cat((a[mask], c), -1),
                                     torch.cat((b[mask], c), -1))
        return reward

    def update(self, steps):
        metrics = {}
        for kind, buffer in self.replay.items():
            metrics[f'{kind}_samples'] = float(buffer.num_samples)
            if not buffer.num_samples:
                continue
            total = 0.0
            for a, b in buffer.feed_forward_generator(steps, self.cfg['batch_size']):
                c = a[:, -self.condition_dim:]
                a, b = a[:, :-self.condition_dim], b[:, :-self.condition_dim]
                if not self.vector_condition:
                    c = c.squeeze(-1)
                ea, eb = self._expert('rec' if kind == 'rec' else 'walk', len(a), a.device)
                if kind == 'loco' and self.vector_condition:
                    idx, _ = self._matched_expert(c * self.scale, nearest_only=True)
                    ea, eb = self.expert_a[idx], self.expert_b[idx]
                    c = self.expert_velocity[idx] / self.scale
                elif kind == 'loco':
                    run = torch.rand_like(c) < c
                    ra, rb = self._expert('run', int(run.sum()), a.device)
                    ea[run], eb[run] = ra, rb
                self.norms[kind].update(torch.cat((a, b, ea, eb)))
                px = self._input(kind, a, b, c)
                ex = self._input(kind, ea, eb, c).detach().requires_grad_(True)
                pd, ed = self.networks[kind](px), self.networks[kind](ex)
                classification = 0.5 * ((pd + 1).square().mean() + (ed - 1).square().mean())
                grad = torch.autograd.grad(ed.sum(), ex, create_graph=True)[0]
                # Penalize motion gradients, not the conditioning scalar.
                penalty = 10 * grad[:, :self.dim * 2].square().sum(-1).mean()
                loss = classification + penalty
                self.optimizers[kind].zero_grad()
                loss.backward()
                self.optimizers[kind].step()
                total += float(loss.detach())
            metrics[f'{kind}_loss'] = total / steps
        if self.vector_condition:
            metrics["loco_expert_coverage"] = self.coverage
        return metrics

    def checkpoint(self):
        return {'model': self.state_dict(), 'config': self.cfg,
                'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}}

    def restore(self, state, load_optimizer=True):
        if state['config'] != self.cfg:
            raise ValueError('MultiAMP configuration differs from checkpoint')
        self.load_state_dict(state['model'])
        if load_optimizer:
            for k, optimizer in self.optimizers.items():
                optimizer.load_state_dict(state['optimizers'][k])
