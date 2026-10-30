"""Isolated AMP ablations: fixed task scale, fresh replay, phase-matched recovery."""
import math
import torch
from rsl_rl.modules.multi_amp import MultiAMP


def phase_direction(geometry):
    height, gravity = geometry[:, 0], geometry[:, 1:]
    phase = torch.bucketize(height.contiguous(), height.new_tensor([.6, .9, 1.15]))
    axis = gravity[:, :2].abs().argmax(-1)
    sign = gravity[:, :2].gather(1, axis[:, None]).squeeze(1) > 0
    direction = axis * 2 + sign.long()
    # Once upright, orientation about the vertical is not a fall direction.
    direction = torch.where(gravity[:, 2] < -.9, 4, direction)
    return phase, direction


def recovery_segments(geometry, fps):
    """Keep trough-to-stable segments, excluding preceding falls and long idle tails."""
    h = geometry[:, 0]
    stable = (h > 1.05) & (geometry[:, 3] < -.9)
    hold = max(1, round(.3 * fps))
    # CPU scan happens once at initialization, never in a training step.
    hs, ss = h.cpu().tolist(), stable.cpu().tolist()
    mask = torch.zeros(len(h) - 1, dtype=torch.bool, device=h.device)
    start, count = 0, 0
    for i, ok in enumerate(ss):
        count = count + 1 if ok else 0
        if count == hold:
            end = i - hold + 1
            if end > start and min(hs[start:end + 1]) < .6:
                trough = start + min(range(end + 1 - start), key=lambda j: hs[start + j])
                mask[trough:min(i + 1, len(mask))] = True
            start = i + 1
        elif count > hold:
            start = i + 1
    return mask


class FreshReplay:
    def __init__(self, capacity, dim, device):
        self.capacity, self.step, self.num_samples = capacity, 0, 0
        self.a = torch.zeros(capacity, dim, device=device)
        self.b = torch.zeros_like(self.a)
        self.c = torch.zeros(capacity, 3, device=device)
        self.geometry = torch.zeros(capacity, 4, device=device)
        self.birth = torch.zeros(capacity, dtype=torch.long, device=device)

    def insert(self, a, b, c, geometry, iteration):
        n = min(len(a), self.capacity)
        ids = (torch.arange(n, device=a.device) + self.step) % self.capacity
        self.a[ids], self.b[ids], self.c[ids] = a[-n:], b[-n:], c[-n:]
        self.geometry[ids], self.birth[ids] = geometry[-n:], iteration
        self.step = (self.step + n) % self.capacity
        self.num_samples = min(self.capacity, self.num_samples + n)

    def candidates(self, iteration, max_age):
        ids = torch.arange(self.num_samples, device=self.a.device)
        return ids[iteration - self.birth[:self.num_samples] <= max_age]

    def sample(self, batch, iteration, max_age, balanced=False):
        ids = self.candidates(iteration, max_age)
        if not len(ids):
            return None
        recent = ids[self.birth[ids] >= iteration - 1]
        # Half recent, half history; phase/direction balancing within each pool.
        def choose(pool, n):
            if not len(pool):
                pool = ids
            weights = torch.ones(len(pool), device=pool.device)
            if balanced:
                phase, direction = phase_direction(self.geometry[pool])
                label = phase * 5 + direction
                counts = torch.bincount(label, minlength=20).clamp(min=1)
                weights = 1. / counts[label].float()
            return pool[torch.multinomial(weights, n, replacement=True)]
        selected = torch.cat((choose(recent, batch // 2), choose(ids, batch - batch // 2)))
        return selected


class StableMultiAMP(MultiAMP):
    def __init__(self, loader, cfg, hidden_dims, reward_coef, task_lerp, device):
        super().__init__(loader, cfg, hidden_dims, reward_coef, task_lerp, device)
        if not self.vector_condition:
            raise ValueError('StableMultiAMP requires matched_velocity conditions')
        self.replay = {k: FreshReplay(cfg['replay_size'], self.dim, device) for k in ('rec', 'loco')}
        self.register_buffer('update_iteration', torch.zeros((), dtype=torch.long, device=device))
        self.new_samples = dict(rec=0, loco=0)
        self.rollout_stats = {}
        self.expert_report = []
        self.previous_weights = None
        self.reset_weights = None
        if cfg.get('phase_experts'):
            aa, bb, gg = [], [], []
            for i, name in enumerate(loader.motion_names):
                if 'fallandgetup' not in name.lower():
                    continue
                fields = [getattr(loader, attr)[i].flatten(1) for attr in
                          ('_body_pos_b_list', '_body_ori_b_list', '_body_lin_vel_b_list', '_body_ang_vel_b_list')]
                frames = torch.cat(fields, -1)
                geometry = loader._recovery_geometry_list[i]
                keep = recovery_segments(geometry, float(loader.fps.reshape(-1)[0]))
                self.expert_report.append(dict(name=name, total=len(keep), retained=int(keep.sum())))
                aa.append(frames[:-1][keep]); bb.append(frames[1:][keep]); gg.append(geometry[:-1][keep])
            self.rec_a, self.rec_b, self.rec_geometry = torch.cat(aa), torch.cat(bb), torch.cat(gg)
            if cfg.get('expert_bank'):
                import hashlib
                import numpy as np
                from pathlib import Path
                path = Path(cfg['expert_bank'])
                bank = np.load(path)
                if bank['a'].shape[1] != self.dim or list(bank['body_names']) != loader.body_names:
                    raise ValueError('SMP expert body feature interface mismatch')
                self.rec_a = torch.tensor(bank['a'], device=device)
                self.rec_b = torch.tensor(bank['b'], device=device)
                self.rec_geometry = torch.tensor(bank['geometry'], device=device)
                self.cfg['expert_bank_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                self.expert_report = [dict(source=str(path), retained=len(self.rec_a),
                                          sha256=self.cfg['expert_bank_sha256'])]
            if not len(self.rec_a):
                raise ValueError('No trough-to-stable recovery expert transitions')
            ep, ed = phase_direction(self.rec_geometry)
            self.rec_pools = {}
            for phase in range(4):
                for direction in range(5):
                    ids = ((ep == phase) & (ed == direction)).nonzero().flatten()
                    # Missing direction falls back only within the SAME phase.
                    if not len(ids):
                        ids = (ep == phase).nonzero().flatten()
                    self.rec_pools[phase * 5 + direction] = ids
            self.expert_phase_counts = torch.bincount(ep, minlength=4)
            print('RECOVERY_EXPERT_AUDIT', self.expert_report,
                  'phase_counts', self.expert_phase_counts.tolist(), flush=True)

    def _sum(self, key, values):
        total, count = self.rollout_stats.get(key, (0., 0))
        self.rollout_stats[key] = (total + float(values.sum()), count + values.numel())

    @torch.no_grad()
    def reward_and_store(self, a, b, task_reward, recovery, condition, dones, geometry=None):
        if geometry is None:
            raise ValueError('Pre-step head height and torso gravity required')
        # Preserve baseline .75 task / .25 style scale even for missing experts or terminals.
        reward = self.task_lerp * task_reward
        rec_weight = recovery.float()
        if self.cfg.get('smooth_transition'):
            height_ready = ((geometry[:, 0] - .9) / .25).clamp(0, 1)
            upright_ready = ((-geometry[:, 3] - .4) / .5).clamp(0, 1)
            ready = height_ready * upright_ready
            ready = ready.square() * (3 - 2 * ready)  # smoothstep, bounded at both ends
            target = torch.where(recovery, 1 - ready, 0.)
            if self.previous_weights is None or len(self.previous_weights) != len(target):
                rec_weight = target
            else:
                limit = self.cfg.get('control_dt', .02) / self.cfg.get('blend_seconds', .2)
                rec_weight = self.previous_weights + (target - self.previous_weights).clamp(-limit, limit)
                rec_weight = torch.where(self.reset_weights, target, rec_weight)
            self.previous_weights = rec_weight.clone()
            self.reset_weights = dones.bool().clone()
        self._sum('blend_fraction', ((rec_weight > 0) & (rec_weight < 1)).float())
        self._sum('task_reward', task_reward)
        for kind, weights in (('rec', rec_weight), ('loco', 1 - rec_weight)):
            mask = (weights > 0) & ~dones.bool()
            if not mask.any():
                continue
            active = mask.nonzero().flatten()
            c = condition[active] / self.scale
            if kind == 'loco':
                idx, supported = self._matched_expert(condition[active])
                self._sum('loco_expert_coverage', supported.float())
                active = active[supported]
                c = self.expert_velocity[idx[supported]] / self.scale
            elif self.cfg.get('phase_experts'):
                p, _ = phase_direction(geometry[active])
                supported = self.expert_phase_counts[p] > 0
                self._sum('rec_expert_coverage', supported.float())
                active, c = active[supported], c[supported]
            if not len(active):
                continue
            d = self.networks[kind](self._input(kind, a[active], b[active], c)).squeeze(-1)
            style = self.reward_coef * (1 - .25 * (d - 1).square()).clamp(min=0)
            reward[active] += (1 - self.task_lerp) * weights[active] * style
            self._sum(kind + '_style_reward', style)
            self._sum(kind + '_style_zero_fraction', (style == 0).float())
            self._sum(kind + '_policy_score', d)
            self.replay[kind].insert(a[active], b[active], c, geometry[active], int(self.update_iteration))
            self.new_samples[kind] += len(active)
        return reward

    def _recovery_expert(self, geometry):
        p, d = phase_direction(geometry)
        label = p * 5 + d
        selected = torch.empty(len(label), dtype=torch.long, device=label.device)
        for key in label.unique().tolist():
            mask = label == key
            pool = self.rec_pools[key]
            if not len(pool):
                raise RuntimeError('Unsupported recovery phase entered replay')
            selected[mask] = pool[torch.randint(len(pool), (int(mask.sum()),), device=pool.device)]
        return self.rec_a[selected], self.rec_b[selected]

    def update(self, steps):
        metrics = {key: value / max(count, 1) for key, (value, count) in self.rollout_stats.items()}
        self.rollout_stats.clear()
        iteration = int(self.update_iteration)
        batch = self.cfg['batch_size']
        for kind, buffer in self.replay.items():
            fresh = self.new_samples[kind]
            # No fresh evidence => no discriminator update. Cap reuse per new sample.
            updates = min(steps, math.floor(fresh * self.cfg['max_reuse'] / batch))
            metrics.update({kind + '_samples': buffer.num_samples, kind + '_new_samples': fresh,
                            kind + '_updates': updates})
            losses, scores, ages = [], [], []
            normalized = False
            for _ in range(updates):
                ids = buffer.sample(batch, iteration, self.cfg['max_age'],
                                    balanced=kind == 'rec' and self.cfg.get('balance_recovery', self.cfg.get('phase_experts')))
                if ids is None:
                    break
                a, b, c = buffer.a[ids], buffer.b[ids], buffer.c[ids]
                if kind == 'rec':
                    ea, eb = (self._recovery_expert(buffer.geometry[ids]) if self.cfg.get('phase_experts')
                              else self._expert('rec', batch, a.device))
                else:
                    idx, _ = self._matched_expert(c * self.scale, nearest_only=True)
                    ea, eb = self.expert_a[idx], self.expert_b[idx]
                    c = self.expert_velocity[idx] / self.scale
                # Update moments once per rollout, not once per repeated minibatch.
                if not normalized:
                    self.norms[kind].update(torch.cat((a, b, ea, eb)))
                    normalized = True
                px = self._input(kind, a, b, c)
                ex = self._input(kind, ea, eb, c).detach().requires_grad_(True)
                pd, ed = self.networks[kind](px), self.networks[kind](ex)
                grad = torch.autograd.grad(ed.sum(), ex, create_graph=True)[0]
                loss = .5 * ((pd + 1).square().mean() + (ed - 1).square().mean())
                loss = loss + 10 * grad[:, :self.dim * 2].square().sum(-1).mean()
                if self.cfg.get('fail_on_nonfinite') and not torch.isfinite(loss):
                    raise FloatingPointError(f'{kind} discriminator loss non-finite')
                self.optimizers[kind].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.networks[kind].parameters(), 1.,
                    error_if_nonfinite=self.cfg.get('fail_on_nonfinite', False))
                self.optimizers[kind].step()
                losses.append(float(loss.detach())); scores.append(float(ed.detach().mean()))
                ages.append(float((iteration - buffer.birth[ids]).float().mean()))
            metrics[kind + '_updates'] = len(losses)
            metrics[kind + '_loss'] = sum(losses) / max(len(losses), 1)
            metrics[kind + '_expert_score'] = sum(scores) / max(len(scores), 1)
            metrics[kind + '_replay_age'] = sum(ages) / max(len(ages), 1)
            metrics[kind + '_reuse'] = len(losses) * batch / max(fresh, 1)
            self.new_samples[kind] = 0
        self.update_iteration.add_(1)
        return metrics
